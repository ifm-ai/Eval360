"""The scheduler event loop, end to end, against a real Slurm cluster.

What this tests:
    `Scheduler.run_evaluate_now` — the same entry point `eval360 evaluate-now`
    calls — driving a real `sbatch` submission, a real `scheduler/slurm/
    sbatch_script.sh`, a real HTTP serving endpoint, and the real generation and
    grading iterators, until the three documented output files exist on disk
    with the right contents. Model registration → event creation → Slurm launch
    → generation streaming → grading → score writing, with nothing between the
    scheduler and the OpenAI wire replaced.

Why this exists:
    `tests/test_scheduler.py` injects `tests/fake_slurm.py`, which replaces
    `SlurmManager` wholesale. Everything the scheduler believes about Slurm in
    that suite — that a submitted job becomes RUNNING, that a RUNNING job has a
    nodelist, that `http://{nodelist}:8000/health` is reachable, that a
    completed event releases its node — is supplied by the same repository that
    consumes it. So the real event loop had never driven a real Slurm job to a
    written output file, and the seam where the scheduler stops being in control
    (sbatch argv, health discovery, URL registration, node release) had no test
    that could fail if it broke.

    This module is the counterpart to `test_slurm_allocation.py`: that one
    checks `SlurmManager` in isolation against a real queue, this one checks
    that the scheduler on top of it produces the artifacts a user is promised.

Corner cases covered:
    Resumption over pre-existing outputs (documented as built-in, and a past
    source of bugs where row identity mismatched between the input file and the
    generations file, making a resume regenerate everything); two events sharing
    one serving deployment, which is how a multi-dataset run is supposed to use
    the cluster; multi-sample generation (`pass_at > 1`) travelling through
    `n=` to the server and back into a pass@k score; and release of the serving
    node once the last event finishes, which nothing in the mocked suite can
    observe because nothing there holds a node.

What this deliberately does NOT cover:
    Two models served CONCURRENTLY. The cluster is one node and the scheduler
    health-checks a hard-coded `http://{nodelist}:8000`, so a second serving
    job cannot bind the port — `fake_vllm_server.py` exits 0 and says so. A
    two-model test would therefore be measuring the single-node cluster, not
    the scheduler. Multi-model allocation arithmetic is covered against the
    real queue by `test_slurm_allocation.py` instead.

Determinism:
    `scripts/ci/fake_vllm_server.py` answers every completion with the fixed
    string `CI-STUB-RESPONSE`, so datasets here use `exact_match` and choose
    each row's `ground_truth` to be either that string or something the stub
    can never emit. Every expected score below is fixed by the fixture, so a
    wrong number means the plumbing lost, duplicated or misgraded something,
    never that the model sampled badly.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import yaml

from scheduler import openai_interface
from scheduler.scheduler import Scheduler

from .cluster_harness import MODEL_PATH, SERVING_VENV, wait_until

pytestmark = pytest.mark.cluster

# The stub server's fixed reply. A row whose ground_truth is this string is
# graded correct exactly when the pipeline delivered the generation intact; a
# row with any other ground_truth must be graded wrong.
STUB_REPLY = "CI-STUB-RESPONSE"

# A whole evaluate-now run: submit, schedule, deploy, generate, grade, release.
# Generous enough for a loaded CI runner (the scheduler polls Slurm every 5s by
# default, so a few polls of slack is normal), tight enough that a hang fails
# this test instead of the workflow.
RUN_TIMEOUT_SECONDS = int(os.environ.get("E2E_RUN_TIMEOUT_SECONDS", "150"))

# The port `SlurmManager.check_live` health-checks, hard-coded there as
# `http://{nodelist}:8000/health`.
SERVING_PORT = 8000
_LISTENER_PID_RE = re.compile(r"pid=(\d+)")


# ---------------------------------------------------------------------------
# Fixture construction
#
# Configs are written per test rather than checked in, because two of the
# fields that matter most here — output_path and data_path — have to be
# absolute paths inside the container's tmp_path, and a checked-in YAML cannot
# carry those.
# ---------------------------------------------------------------------------


def write_dataset(
    path: Path, rows: int, *, ungradable_rows: frozenset[int] = frozenset()
) -> Path:
    """Write a tiny JSONL dataset in the documented record contract.

    `row` is written as an int on purpose. It is the identity the resume path
    keys on: `load_completed_row_ids` regex-scans the generations file and
    `int()`s what it finds, while the input side compares `record["row"]`
    against that set. A dataset that wrote `"row": "0"` would compare `"0"` to
    `0`, match nothing, and regenerate the entire file on every resume — which
    is a bug this suite should be able to see rather than one it should model.

    `ungradable_rows` get a ground truth the stub can never produce. Without at
    least one of them a run's accuracy is 1.0 whatever the grader does, so
    "accuracy == 1.0" would also hold for a grader that ignored its input and
    returned True. A row the pipeline must score WRONG is what makes the score
    assertions load-bearing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "row": index,
                        "completion_input": f"question {index}",
                        "chat_input": [
                            {"role": "user", "content": f"question {index}"}
                        ],
                        "ground_truth": (
                            "NOT-THE-STUB-REPLY"
                            if index in ungradable_rows
                            else STUB_REPLY
                        ),
                    }
                )
                + "\n"
            )
    return path


def write_model_config(
    path: Path,
    *,
    name: str,
    output_root: Path,
    max_model_len: int,
) -> Path:
    """Write a remote_model config the CI node can actually schedule.

    `serving_slurm_resources` is not optional here. The code defaults are
    12345-GPU/CPU placeholders, so the config states a request that fits
    the test cluster's declared gpu:4 and 16 CPUs (scripts/ci/slurm_up.sh);
    a request beyond the node's definition would stay PENDING forever and
    fail on a timeout that says nothing about the scheduler.

    `max_model_len` is per test for two reasons at once. It is the payload that
    proves `vllm_cli_args` survived the base64 round trip through
    `sbatch_script.sh` (the stub prints the argv it was handed), and it changes
    `serving_key`, which keeps each test's Slurm jobs and its entry in the
    process-global `LOCKED_CONNECTIONS` pool registry distinct from every other
    test's.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "remote_model": {"base_name": name, "path": MODEL_PATH, "revision": None},
        "model_type": "instruct",
        "parser_type": "noop",
        "venv_path": SERVING_VENV,
        "max_simultaneous_requests": 4,
        "max_time_to_deploy": 180,
        "vllm_cli_args": ["--max-model-len", str(max_model_len)],
        "openai_kwargs": {"temperature": 0.0},
        "serving_slurm_resources": {
            "gpus_per_node": 1,
            "cpus_per_task": 1,
            "memory_gb": 1,
            "time_limit": "0:20:00",
        },
        "owner": "eval360-ci",
        "ready": True,
        "output_path": str(output_root),
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def write_data_config(
    path: Path,
    *,
    dataset_name: str,
    data_path: Path,
    pass_at: list[int] | None = None,
) -> Path:
    """Write a dataset config graded by exact match against the stub's reply.

    `num_generations` is deliberately omitted so `Task.parse_yaml` counts the
    file; a hard-coded count that drifted from the fixture would raise during
    parsing and never reach the cluster.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "uuid": f"{dataset_name}-task",
        "dataset_name": dataset_name,
        "data_path": str(data_path),
        "semantic_version": "1.0.0",
        "mode": "instruct",
        "grader": {"type": "exact_match"},
        "average_over": [1],
        "pass_at": pass_at if pass_at is not None else [1],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# Driving the scheduler
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def strict_errors(monkeypatch):
    """Make generation and grading errors raise instead of being recorded.

    With `EVAL360_IGNORE_ERRORS` at its default, a failed generation is written
    to the output file as an `exception` record and grading scores it 0 —
    meaning a run in which nothing worked still produces all three files and
    reaches phase 2. Every assertion below about file contents would hold on a
    completely broken pipeline. `false` is what `evaluate-now` sets unless the
    user passes `--ignore-grader-errors`, and it is the only setting under
    which "the run succeeded" means anything here.
    """
    monkeypatch.setenv("EVAL360_IGNORE_ERRORS", "false")




# NOTE: the orphaned-stub guard that used to live here is now in
# tests/slurm/conftest.py's `drained_cluster` fixture, which runs for EVERY
# test in the suite rather than only this module's. Three modules had grown
# their own copy of it; one implementation is the point of this whole suite.

async def run_evaluate_now(
    *,
    model_config: Path,
    data_configs: list[Path],
    log_dir: Path,
) -> Scheduler:
    """Run one in-process evaluate-now and return the scheduler that ran it.

    WHY IN-PROCESS RATHER THAN THE `eval360` CLI. Everything the CLI adds on
    top of `run_evaluate_now` is argument parsing, logging configuration and
    three environment variables — all of it already covered by
    `tests/test_cli_logging.py`, none of it touching Slurm. What has never been covered
    is the event loop underneath, and running it in-process is what lets this
    module pin the scheduler's `instance_id` (so its Slurm jobs are
    distinguishable and cannot be confused with another test's), read back the
    live `SlurmManager` afterwards to assert the node was released, and fail
    with the scheduler's own traceback instead of a subprocess exit code.

    The environment variables the CLI would set are replicated: the cluster
    harness exports `EVAL360_IN_MEMORY_DB=true`, and `strict_errors` sets
    `EVAL360_IGNORE_ERRORS`.
    """
    # `LOCKED_CONNECTIONS` is module-global and keyed by serving key, so it
    # outlives a Scheduler. A real evaluate-now is a fresh process with an empty
    # registry; clearing here reproduces that rather than letting one test hand
    # the next a pool whose URL points at a job that has since been cancelled.
    openai_interface.LOCKED_CONNECTIONS.clear()

    log_dir.mkdir(parents=True, exist_ok=True)
    scheduler = Scheduler(
        None,
        None,
        # One serving job. Raising this would not buy parallelism on a
        # single-node cluster — it would make `get_desired_allocation` spend
        # the spare node on an extra replica of the same model, and that
        # replica cannot bind :8000, so it would serve nothing.
        max_generation_jobs=1,
        max_grading_parallelism=4,
        log_dir=str(log_dir),
        slurm_partition="ci",
        instance_id=uuid.uuid4().hex[:8],
    )
    await asyncio.wait_for(
        scheduler.run_evaluate_now(
            [str(model_config)],
            [str(config) for config in data_configs],
        ),
        timeout=RUN_TIMEOUT_SECONDS,
    )
    return scheduler


def output_files(output_root: Path, model_name: str, dataset_name: str):
    """The three documented artifacts plus the resume-guard metadata file.

    `EventManager.create_events_for_model_and_tasks` writes
    `<dataset>_scores.yaml`, and its contents are YAML mappings rather than
    JSON lines.
    """
    model_dir = output_root / model_name
    return (
        model_dir / f"{dataset_name}_generations.jsonl",
        model_dir / f"{dataset_name}_grades.jsonl",
        model_dir / f"{dataset_name}_scores.yaml",
        model_dir / f"{dataset_name}_run_metadata.yaml",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_scores(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def job_logs(log_dir: Path) -> list[Path]:
    """One file per sbatch job this scheduler submitted (`slurm-%j.out`)."""
    return sorted(log_dir.glob("slurm-*.out"))


async def wait_for_serving_evidence(log_dir: Path, needle: str) -> str:
    """Wait for a job log that proves the serving job really ran our argv.

    The job writes through slurmstepd's I/O forwarding, so a log that exists is
    not yet a log that contains everything the job has printed. Polling for the
    marker rather than reading once keeps this from being a race.
    """

    async def _found():
        for path in job_logs(log_dir):
            text = path.read_text(errors="replace")
            if needle in text:
                return text
        return None

    return await wait_until(
        _found, timeout=30, what=f"a Slurm job log containing {needle!r}"
    )


async def assert_serving_node_released(scheduler: Scheduler) -> None:
    """A finished evaluate-now must leave no job of its own in the queue.

    This is the half of the lifecycle the mocked suite cannot check at all: in
    `fake_slurm.py` a "cancelled" job is a dict removed from a dict. Here a
    leaked job holds a real GPU until its 20-minute time limit expires, so the
    failure mode this guards against is the scheduler silently squatting on a
    node after every run.
    """

    async def _empty():
        return not await scheduler.slurm_manager.get_all_jobs()

    await wait_until(
        _empty,
        timeout=60,
        what=(
            "evaluate-now to release the serving job owned by "
            f"{scheduler.slurm_manager.instance_id}"
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_evaluate_now_writes_the_documented_outputs(tmp_path):
    """One model, one four-row dataset, start to finish on a real cluster.

    Four rows rather than one because a single row cannot show an off-by-one,
    a lost record or a duplicated one, and small enough that the whole run is
    dominated by Slurm scheduling rather than by generation. One of the four is
    unsatisfiable so the expected accuracy is 0.75 rather than a constant the
    grader could produce without reading anything.
    """
    output_root = tmp_path / "output"
    log_dir = tmp_path / "slurm-logs"
    model_config = write_model_config(
        tmp_path / "model.yaml",
        name="ci-e2e-oneshot",
        output_root=output_root,
        max_model_len=1024,
    )
    data_config = write_data_config(
        tmp_path / "data.yaml",
        dataset_name="cistub",
        data_path=write_dataset(
            tmp_path / "cistub.jsonl", rows=4, ungradable_rows=frozenset({3})
        ),
    )

    scheduler = await run_evaluate_now(
        model_config=model_config, data_configs=[data_config], log_dir=log_dir
    )

    generations, grades, scores, metadata = output_files(
        output_root, "ci-e2e-oneshot", "cistub"
    )
    for path in (generations, grades, scores, metadata):
        assert path.is_file(), f"{path} was not written"

    # ---- generations -----------------------------------------------------
    records = read_jsonl(generations)
    assert len(records) == 4
    assert sorted(record["row"] for record in records) == [0, 1, 2, 3]
    for record in records:
        # One generation because average_over=[1] and pass_at=[1]; the content
        # is the stub's reply, which is what proves the response travelled back
        # through the OpenAI client rather than being synthesised locally.
        assert record["generations"] == [STUB_REPLY]

    # ---- grades ----------------------------------------------------------
    graded = read_jsonl(grades)
    assert len(graded) == 4
    by_row = {record["row"]: record for record in graded}
    assert sorted(by_row) == [0, 1, 2, 3]
    for row in (0, 1, 2):
        assert by_row[row]["correct"] == [True]
        assert by_row[row]["picked"] == [STUB_REPLY]
    # Row 3's ground truth is unreachable for the stub, so the grader has to
    # reach the opposite verdict on a generation that is textually identical to
    # the other three. That is what distinguishes grading from asserting.
    assert by_row[3]["correct"] == [False]
    assert by_row[3]["picked"] == [None]

    # ---- scores ----------------------------------------------------------
    # 3 of 4 correct, computed by the real aggregation path from the real
    # grades — not a constant that would survive any grading bug.
    assert read_scores(scores) == {
        "accuracy (avg over 1)": 0.75,
        "accuracy (pass@1)": 0.75,
    }

    # ---- the Slurm side --------------------------------------------------
    # One serving job, and its log proves the argv survived the base64 round
    # trip in sbatch_script.sh: the stub prints what it was actually handed.
    assert len(job_logs(log_dir)) == 1, (
        f"expected exactly one serving job, found {job_logs(log_dir)}"
    )
    text = await wait_for_serving_evidence(log_dir, "[fake-vllm] serving on")
    assert MODEL_PATH in text, "the model path never reached `vllm serve`"
    assert "--max-model-len" in text and "1024" in text, (
        "vllm_cli_args did not survive the base64 round trip through "
        f"sbatch_script.sh; job log was:\n{text}"
    )

    await assert_serving_node_released(scheduler)


async def test_resuming_over_existing_outputs_generates_nothing_twice(tmp_path):
    """A second identical run must add no generation and no grade.

    Resumption is documented as built-in and is the feature with the worst
    failure mode: it does not raise, it silently re-runs an entire benchmark
    and appends a second copy of every row, which then double-counts in the
    score. Comparing the file bytes before and after is the strongest form of
    the claim — not merely "the count is the same" but "nothing was written".

    The second run still registers the model, creates the event and asks for a
    deployment before it discovers the outputs are complete, so this also
    covers the path where an event finishes without issuing a single request.
    """
    output_root = tmp_path / "output"
    model_config = write_model_config(
        tmp_path / "model.yaml",
        name="ci-e2e-resume",
        output_root=output_root,
        max_model_len=2048,
    )
    data_config = write_data_config(
        tmp_path / "data.yaml",
        dataset_name="resumed",
        data_path=write_dataset(tmp_path / "resumed.jsonl", rows=4),
    )

    await run_evaluate_now(
        model_config=model_config,
        data_configs=[data_config],
        log_dir=tmp_path / "slurm-logs-1",
    )

    generations, grades, scores, _ = output_files(
        output_root, "ci-e2e-resume", "resumed"
    )
    first_generations = generations.read_bytes()
    first_grades = grades.read_bytes()
    assert len(read_jsonl(generations)) == 4
    assert len(read_jsonl(grades)) == 4

    scheduler = await run_evaluate_now(
        model_config=model_config,
        data_configs=[data_config],
        log_dir=tmp_path / "slurm-logs-2",
    )

    assert generations.read_bytes() == first_generations, (
        "the resumed run rewrote or appended to the generations file; "
        f"{len(read_jsonl(generations))} records now exist where 4 were complete"
    )
    assert grades.read_bytes() == first_grades, (
        "the resumed run re-graded rows that were already graded; "
        f"{len(read_jsonl(grades))} grade records now exist for 4 rows"
    )

    # Row identity, stated directly rather than inferred from the byte
    # comparison, because this is the specific thing that breaks: a resume that
    # cannot match input rows against completed rows duplicates row ids.
    #
    # Compared as a multiset, not as a sequence. Generations are written in
    # completion order, not row order — requests for all four rows are in
    # flight at once and whichever returns first is appended first — so the
    # file legitimately holds e.g. [2, 0, 1, 3]. What must hold is that each
    # row appears exactly once.
    rows = [record["row"] for record in read_jsonl(generations)]
    assert sorted(rows) == [0, 1, 2, 3], f"duplicated or lost rows: {rows}"

    # Scores are rewritten (opened "w") from the pre-existing grades, so this
    # asserts the resume path can still aggregate a file it did not produce.
    assert read_scores(scores) == {
        "accuracy (avg over 1)": 1.0,
        "accuracy (pass@1)": 1.0,
    }

    await assert_serving_node_released(scheduler)


async def test_two_datasets_share_one_serving_deployment(tmp_path):
    """Two events, one model: one Slurm job, two complete sets of outputs.

    This is the shape of a normal run — a model evaluated on several
    benchmarks — and the property worth pinning is that the second event
    attaches to the deployment the first one created instead of asking for
    another node. On a real cluster the difference is a doubled footprint; in
    `fake_slurm.py` it is a counter that the fake itself maintains.

    The second dataset uses `pass_at=[2]`, which also carries the multi-sample
    path through the real wire: `max(average_over + pass_at)` becomes the `n`
    of the request, the stub returns two choices for it, and pass@2 has to come
    back out of the grader.
    """
    output_root = tmp_path / "output"
    log_dir = tmp_path / "slurm-logs"
    model_config = write_model_config(
        tmp_path / "model.yaml",
        name="ci-e2e-multi",
        output_root=output_root,
        max_model_len=4096,
    )
    first = write_data_config(
        tmp_path / "first.yaml",
        dataset_name="alpha",
        data_path=write_dataset(tmp_path / "alpha.jsonl", rows=3),
    )
    second = write_data_config(
        tmp_path / "second.yaml",
        dataset_name="beta",
        data_path=write_dataset(tmp_path / "beta.jsonl", rows=3),
        pass_at=[2],
    )

    scheduler = await run_evaluate_now(
        model_config=model_config,
        data_configs=[first, second],
        log_dir=log_dir,
    )

    alpha_generations, alpha_grades, alpha_scores, _ = output_files(
        output_root, "ci-e2e-multi", "alpha"
    )
    beta_generations, beta_grades, beta_scores, _ = output_files(
        output_root, "ci-e2e-multi", "beta"
    )

    assert len(read_jsonl(alpha_generations)) == 3
    assert len(read_jsonl(alpha_grades)) == 3
    assert read_scores(alpha_scores) == {
        "accuracy (avg over 1)": 1.0,
        "accuracy (pass@1)": 1.0,
    }

    beta_records = read_jsonl(beta_generations)
    assert len(beta_records) == 3
    for record in beta_records:
        assert record["generations"] == [STUB_REPLY, STUB_REPLY], (
            "pass_at=[2] did not reach the server as n=2"
        )
    for record in read_jsonl(beta_grades):
        assert record["correct"] == [True, True]
    assert read_scores(beta_scores) == {
        "accuracy (avg over 1)": 1.0,
        "accuracy (pass@2)": 1.0,
    }

    assert len(job_logs(log_dir)) == 1, (
        "the two events did not share a deployment — one sbatch job was "
        f"expected, these were submitted: {job_logs(log_dir)}"
    )
    # And that one job really served: with conftest's `drained_cluster` holding the
    # port free beforehand, this marker can only have been printed by the job
    # this test launched, so both events were served by it.
    await wait_for_serving_evidence(log_dir, "[fake-vllm] serving on")

    await assert_serving_node_released(scheduler)
