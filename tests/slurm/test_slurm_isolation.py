"""Ownership: which jobs on a shared cluster this scheduler may touch.

What this tests:
    The `instance_id` and job-name filtering that decide whether a queue entry
    belongs to this `SlurmManager` — `_from_job_name`, `get_all_jobs`,
    `cancel_all_owned_jobs`, and the two job-name prefixes (`eval360-` for
    serving, `eval360id-` for imported datasets).

Why this exists:
    `SlurmManager`'s docstring makes an explicit multi-tenancy promise: "Only
    jobs whose name contains this instance_id are considered 'ours', allowing
    multiple schedulers to share the same Slurm account without interference."
    Nothing tested it. It cannot be tested with a fake — `FakeSlurmManager`
    keeps its jobs in a per-instance dict, so isolation is true there by
    construction, for a reason that has nothing to do with the production code.
    On a real cluster every instance queries the same `squeue --me` and the
    filtering has to do the work.

    The failure this guards is severe and asymmetric: a scheduler that
    mis-identifies ownership does not crash, it cancels somebody else's running
    eval.

Corner cases covered:
    A second scheduler's jobs are invisible to the first even though both see
    them in `squeue --me`; `cancel_all_owned_jobs` leaves the other instance's
    jobs running; jobs whose names do not match the eval360 pattern at all are
    ignored; imported-dataset jobs, which use a different prefix and are NOT
    matched by `_JOB_NAME_RE`, are still cancelled as owned; and that same
    imported-dataset sweep spares another instance's imported-dataset job.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from .cluster_harness import (
    CLUSTER_GPUS,
    drain,
    make_model,
    running_jobs,
    scancel,
    slurm_session,
    submit_raw,
    wait_for_running,
    wait_until,
)

pytestmark = pytest.mark.cluster


async def test_another_schedulers_jobs_are_invisible():
    """Two instances, one queue, no cross-visibility.

    Both managers run `squeue --me` as the same Unix user and therefore see
    each other's rows in the raw output. Only `_from_job_name`'s instance_id
    check keeps them apart.
    """
    async with slurm_session(instance_id="aaaaaaaa") as first:
        async with slurm_session(instance_id="bbbbbbbb") as second:
            await first.update_allocation([(make_model("ci-model-first"), 1)], [])
            await second.update_allocation([(make_model("ci-model-second"), 1)], [])

            first_jobs = await wait_for_running(first, 1)
            second_jobs = await wait_for_running(second, 1)

            assert "aaaaaaaa" in first_jobs[0]["name"]
            assert "bbbbbbbb" in second_jobs[0]["name"]
            # The decisive assertion: disjoint views of a shared queue.
            assert first_jobs[0]["job_id"] != second_jobs[0]["job_id"]
            assert len(await first.get_all_jobs()) == 1
            assert len(await second.get_all_jobs()) == 1


async def test_cancel_all_owned_jobs_spares_another_instance():
    """The promise that matters: one scheduler shutting down leaves others alone.

    `cancel_all_owned_jobs` runs on interrupt, and it reads `squeue --me` —
    every job the account owns, including other schedulers'. If its prefix
    filter were wrong it would cancel a colleague's running eval on Ctrl-C.
    """
    async with slurm_session(instance_id="cccccccc") as doomed:
        async with slurm_session(instance_id="dddddddd") as survivor:
            await doomed.update_allocation([(make_model("ci-model-doomed"), 1)], [])
            await survivor.update_allocation([(make_model("ci-model-survivor"), 1)], [])
            await wait_for_running(doomed, 1)
            survivor_jobs = await wait_for_running(survivor, 1)
            survivor_id = survivor_jobs[0]["job_id"]

            await doomed.cancel_all_owned_jobs()
            await drain(doomed, timeout=60)

            assert await doomed.get_all_jobs() == {}
            still_running = await running_jobs(survivor)
            assert [job["job_id"] for job in still_running] == [survivor_id], (
                "cancel_all_owned_jobs cancelled another instance's job"
            )


@pytest.mark.parametrize(
    "foreign_name",
    [
        # Someone else's work entirely.
        "somebody-elses-training-run",
        # Right prefix, wrong shape — no serving key, no replica suffix.
        "eval360-not-a-real-job",
        # Well-formed but a different instance_id.
        "eval360-ffffffff-other-000000000000-r0",
    ],
)
async def test_foreign_job_names_are_ignored(foreign_name):
    """`get_all_jobs` returns only jobs matching this instance's name pattern.

    Parameterised over the three ways a name can fail to match, because they
    exercise different branches: no prefix at all, a prefix the regex rejects,
    and a regex match whose instance_id differs.
    """
    job_id = await submit_raw(foreign_name, "sleep 120")
    try:
        async with slurm_session(instance_id="eeeeeeee") as manager:
            await manager.update_allocation([(make_model(), 1)], [])
            await wait_for_running(manager, 1)

            owned = await manager.get_all_jobs()
            assert foreign_name not in owned
            assert all(job["job_id"] != job_id for job in owned.values())

            # And it survives a full cancel sweep.
            await manager.cancel_all_owned_jobs()
            await drain(manager, timeout=60)

            async def _foreign_still_queued():
                process = await _squeue_names()
                return foreign_name in process

            assert await _foreign_still_queued(), (
                "a job this scheduler does not own was cancelled"
            )
    finally:
        await scancel(job_id)


async def _squeue_names() -> set[str]:
    return set(await _squeue_field("%j"))


async def _squeue_job_ids() -> set[int]:
    """Every job ID the account currently has queued, whoever owns it.

    Read straight from squeue rather than through `SlurmManager`, because the
    thing under test is whether a manager's own view is correctly narrower than
    this one.
    """
    return {int(job_id) for job_id in await _squeue_field("%i")}


async def _squeue_field(fmt: str) -> list[str]:
    process = await asyncio.create_subprocess_exec(
        "squeue", "--noheader", "--me", "--format", fmt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    return [line.strip() for line in stdout.decode().splitlines() if line.strip()]


async def test_imported_dataset_jobs_are_submitted_and_owned():
    """The `eval360id-` prefix: submitted, ledgered, and cancelled as ours.

    Imported-dataset jobs deliberately do NOT match `_JOB_NAME_RE` — they carry
    a runner name and an event UUID instead of a serving key — so they are
    invisible to `get_all_jobs` by design. `cancel_all_owned_jobs` matches them
    with a separate prefix check, and that second code path is the one this
    covers: a miss there leaks a job that holds a GPU until its time limit.
    """
    async with slurm_session(instance_id="12345678") as manager:
        model = make_model("ci-imported-model")
        job_id = await manager.submit_imported_dataset_job(
            event_uuid="abcdef0123456789",
            model_instance=model,
            runner_name="cirunner",
            setup_script="true",
            benchmark_script="true",
            output_dir="/tmp/output",
        )
        assert job_id > 0

        async def _queued():
            names = await _squeue_names()
            return any(name.startswith("eval360id-12345678-") for name in names) or None

        await wait_until(_queued, timeout=60, what="the imported-dataset job to appear")

        # Not a serving job, so `get_all_jobs` correctly does not see it.
        assert await manager.get_all_jobs() == {}

        await manager.cancel_all_owned_jobs()

        async def _gone():
            names = await _squeue_names()
            return not any(name.startswith("eval360id-12345678-") for name in names)

        assert await wait_until(
            _gone, timeout=60, what="the imported-dataset job to be cancelled"
        )


async def test_cancelling_imported_jobs_spares_another_instance(tmp_path):
    """Two imported-dataset jobs, two instances: only the owner's one dies.

    The test above submits a single imported job and watches its owner cancel
    it. That cannot tell a correct prefix filter from a broken one: with only
    one job on the cluster, `eval360id-{instance_id}-`, a bare `eval360id-`, and
    "cancel everything the account owns" all produce the same empty queue. This
    is the serving-job shape from `test_cancel_all_owned_jobs_spares_another_
    instance`, applied to the second prefix in `cancel_all_owned_jobs` — the one
    that is NOT covered by `_from_job_name`, and so gets no protection from the
    instance_id check that every serving-job path goes through.

    BOTH JOBS ARE HELD PENDING behind a blocker that owns every GPU, which does
    three things at once. A queued job cannot exit on its own, so "the survivor
    is still there" cannot be satisfied by a job that simply had not got round
    to finishing — the race the single-job version would have had if its job
    were made long-lived by running `sleep`. Neither job executes
    `imported_dataset_script.sh`, so nothing starts a stub server on the shared
    :8000 or builds a venv under the bind-mounted repo. And PENDING is the state
    `cancel_all_owned_jobs` is least likely to be tested against, since it
    filters on name only and never looks at state.
    """
    blocker = await submit_raw(
        "cluster-blocker", "sleep 300", gres=f"gpu:{CLUSTER_GPUS}"
    )
    try:
        async with slurm_session(instance_id="1a1a1a1a") as doomed:
            async with slurm_session(instance_id="2b2b2b2b") as survivor:
                doomed_job = await doomed.submit_imported_dataset_job(
                    event_uuid="1111aaaa2222bbbb",
                    model_instance=make_model("ci-imported-doomed"),
                    runner_name="cirunner",
                    setup_script="true",
                    benchmark_script="true",
                    output_dir=str(tmp_path / "doomed"),
                )
                survivor_job = await survivor.submit_imported_dataset_job(
                    event_uuid="3333cccc4444dddd",
                    model_instance=make_model("ci-imported-survivor"),
                    runner_name="cirunner",
                    setup_script="true",
                    benchmark_script="true",
                    output_dir=str(tmp_path / "survivor"),
                )
                assert doomed_job != survivor_job

                async def _both_queued():
                    return {doomed_job, survivor_job} <= await _squeue_job_ids() or None

                await wait_until(
                    _both_queued,
                    timeout=60,
                    what=f"jobs {doomed_job} and {survivor_job} to be queued",
                )

                await doomed.cancel_all_owned_jobs()

                async def _doomed_gone():
                    return doomed_job not in await _squeue_job_ids() or None

                await wait_until(
                    _doomed_gone,
                    timeout=60,
                    what=f"imported job {doomed_job} to be cancelled by its owner",
                )

                # Held open rather than read once. A sweep that cancelled both
                # would remove them in the same scancel, and the two rows do not
                # have to leave squeue in the same instant — reading at the
                # moment the doomed job vanishes could catch the survivor still
                # listed and call that a pass.
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    assert survivor_job in await _squeue_job_ids(), (
                        "cancel_all_owned_jobs cancelled another instance's "
                        f"imported-dataset job ({survivor_job})"
                    )
                    await asyncio.sleep(0.5)
    finally:
        await scancel(blocker)


async def test_imported_dataset_job_id_is_parsed_from_sbatch(tmp_path):
    """`submit_imported_dataset_job` returns the real Slurm job ID.

    It parses sbatch's stdout by hand (`int(stdout.split()[-1])`) rather than
    going through `_parse_sbatch_job_id`, so it is a separate parser with its
    own chance to be wrong about what sbatch prints.
    """
    async with slurm_session(instance_id="87654321") as manager:
        job_id = await manager.submit_imported_dataset_job(
            event_uuid="fedcba9876543210",
            model_instance=make_model("ci-imported-id"),
            runner_name="cirunner",
            setup_script="true",
            benchmark_script="true",
            output_dir=str(tmp_path),
        )
        # Wait for a SETTLED row, not merely for a row.
        #
        # slurmdbd publishes a job's accounting record in two stages: nothing at
        # all for roughly the first second, then a placeholder carrying
        # `JobName='allocation'`, then the real name. Waiting only for existence
        # caught the placeholder and failed on `'allocation'.startswith(...)` —
        # which is the documented hazard in docs/SLURM_TEST_CLUSTER.md, hit here
        # for the third time.
        #
        # Skipping the placeholder keeps this test about ID PARSING and nothing
        # else: if `submit_imported_dataset_job` returned the wrong ID, no
        # amount of waiting produces a row at all and this still fails, with the
        # same meaning. It is a wait for an external system to settle, not an
        # assertion weakened to accept bad output.
        async def _settled_row():
            outcomes = await manager.get_job_accounting([job_id])
            outcome = outcomes.get(job_id)
            if outcome is None or outcome.job_name == "allocation":
                return None
            return outcome

        outcome = await wait_until(
            _settled_row,
            timeout=90,
            what=f"a settled accounting row for job {job_id} (sbatch's reported ID)",
        )
        assert outcome.job_name.startswith("eval360id-87654321-")
