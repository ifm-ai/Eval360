"""What the scheduler does when Slurm does not cooperate.

What this tests:
    The non-happy paths of `get_model_state` and `update_allocation` — a job
    that cannot be scheduled, a replica that never serves, a job cancelled by
    somebody else, and a resource request Slurm refuses outright.

Why this exists:
    These states are the whole reason the scheduler has a state machine, and
    they are the hardest to fake honestly. `FakeSlurmManager` decides PENDING,
    deploying and dead by looking at its own counters, so a test written
    against it proves the fake's arithmetic rather than Slurm's behaviour. The
    real shapes are genuinely surprising: a PENDING job's `%N` is EMPTY, not a
    reason string, and its elapsed field is `0:00`, which both flow straight
    into parsers that were only ever shown running jobs.

Corner cases covered:
    A PENDING job with an empty nodelist and a zero elapsed time; a replica
    that runs but never answers /health, which must become `dead` only after
    `max_time_to_deploy` rather than immediately; an externally cancelled job
    vanishing from the queue mid-deployment, which is how preemption presents;
    and sbatch rejecting an impossible GRES request so `update_allocation`
    raises instead of silently allocating nothing.
"""

from __future__ import annotations

import pytest

from .cluster_harness import (
    CLUSTER_GPUS,
    deployment_for,
    drain,
    make_model,
    running_jobs,
    scancel,
    slurm_session,
    submit_raw,
    wait_for_live,
    wait_for_running,
    wait_until,
)

pytestmark = pytest.mark.cluster


async def test_a_job_that_cannot_be_scheduled_is_reported_pending():
    """A blocked replica shows up as PENDING with an empty nodelist.

    Verified against real output — `squeue --format %j|%i|%T|%M|%N` for a
    pending job emits a trailing EMPTY field, not the `(Resources)` reason that
    `%R` would give:

        eval360-...-r0|3|PENDING|0:00|

    So `get_all_jobs` must produce `nodelist=''` and `elapsed_time=0`. Both
    values feed code written with running jobs in mind: `_to_seconds` on
    `'0:00'` and, for RUNNING jobs, an f-string that would otherwise build
    `http://:8000/health`.
    """
    # Occupy every GPU so the scheduler's replica cannot start.
    blocker = await submit_raw(
        "cluster-blocker", "sleep 120", gres=f"gpu:{CLUSTER_GPUS}"
    )
    try:
        async with slurm_session() as manager:
            model = make_model()
            await manager.update_allocation([(model, 1)], [])

            async def _pending_row():
                jobs = await manager.get_all_jobs()
                rows = [job for job in jobs.values() if job["state"] == "PENDING"]
                return rows[0] if rows else None

            row = await wait_until(
                _pending_row, timeout=60, what="a PENDING job row"
            )

            assert row["state"] == "PENDING"
            assert row["nodelist"] == "", (
                f"expected an empty nodelist for a pending job, got {row['nodelist']!r}"
            )
            assert row["elapsed_time"] == 0
            # Identity still resolves while pending — the scheduler has to know
            # which model is waiting, not just that something is.
            assert row["serving_key"] == model.serving_key
            assert row["model_name"] == model.name

            pending, deploying, live, dead, replica_counts = await manager.get_model_state(
                deployment_for(model)
            )
            assert pending == [model.name]
            assert live == []
            assert dead == []
            assert replica_counts == {model.name: 1}
    finally:
        await scancel(blocker)


async def test_a_replica_that_never_serves_becomes_dead_after_its_deadline():
    """Running but unhealthy is `deploying` until `max_time_to_deploy`, then `dead`.

    The replica is made unhealthy by pointing the stub server at a different
    port, so the Slurm job runs normally and only the health check fails —
    which is exactly how a real VLLM that crashes on startup presents.

    Asserting the `deploying` phase first matters: a bug that classified any
    unhealthy replica as dead immediately would make every slow model load look
    like a failure, and the final state alone cannot tell the two apart.
    """
    async with slurm_session() as manager:
        model = make_model(
            "ci-never-healthy",
            # The scheduler health-checks a hard-coded :8000; serving 9001
            # means the job is up and the endpoint is not.
            vllm_cli_args=["--port 9001"],
            max_time_to_deploy=15,
        )
        await manager.update_allocation([(model, 1)], [])
        await wait_for_running(manager, 1)

        async def _deploying():
            _, deploying, _, dead, _ = await manager.get_model_state(deployment_for(model))
            assert not dead, "declared dead before max_time_to_deploy elapsed"
            return deploying or None

        await wait_until(_deploying, timeout=30, what="the replica to be 'deploying'")

        async def _dead():
            _, _, live, dead, _ = await manager.get_model_state(deployment_for(model))
            assert not live, "an endpoint on the wrong port answered the health check"
            return dead or None

        dead = await wait_until(
            _dead, timeout=90, what=f"the replica to be declared dead after "
                                    f"{model.max_time_to_deploy}s"
        )
        assert dead == [model.name]


async def test_an_externally_cancelled_job_disappears_from_model_state():
    """Preemption: the job vanishes from squeue and the model stops being live.

    Slurm can take a job away at any moment, and the scheduler learns about it
    only by its absence from the next `squeue`. This is the path behind
    "Evicting URL for {model}: job no longer in squeue".
    """
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 1)], [])
        jobs = await wait_for_running(manager, 1)
        await wait_for_live(manager, deployment_for(model))

        # Cancelled by something other than this scheduler, as a preemption is.
        await scancel(jobs[0]["job_id"])

        async def _gone():
            pending, deploying, live, dead, replica_counts = await manager.get_model_state(
                deployment_for(model)
            )
            return not (pending or deploying or live or replica_counts) or None

        await wait_until(_gone, timeout=60, what="the cancelled job to leave the queue")
        assert await manager.get_all_jobs() == {}


async def test_sbatch_rejecting_a_request_raises_rather_than_allocating_nothing():
    """An impossible GRES request fails loudly.

    Measured: sbatch exits 1 with "Requested node configuration is not
    available" when more GPUs are requested than the node defines. It is a
    submit-time rejection, not a job that sits PENDING, so `update_allocation`
    has to surface it — a swallowed error here would leave the scheduler
    believing it had deployed a model that does not exist.
    """
    async with slurm_session() as manager:
        # An obviously arbitrary count, far beyond the node's gpu:CLUSTER_GPUS.
        assert CLUSTER_GPUS < 12345
        model = make_model("ci-impossible", gpus_per_node=12345)
        with pytest.raises(RuntimeError, match="sbatch failed"):
            await manager.update_allocation([(model, 1)], [])

        assert await manager.get_all_jobs() == {}


async def test_a_model_is_live_only_while_its_endpoint_answers():
    """`live` reflects the health check, not merely the job being RUNNING.

    The distinction is the entire purpose of `check_live`: a RUNNING Slurm job
    whose server has not finished starting must not be handed generation work.
    """
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 1)], [])
        jobs = await wait_for_running(manager, 1)

        live = await wait_for_live(manager, deployment_for(model))
        assert live == [(model.name, f"http://{jobs[0]['nodelist']}:8000")]

        assert await manager.is_url_healthy(live[0][1])
        # A port nothing is serving must not report healthy, or "live" would
        # mean "a hostname was parsed".
        assert not await manager.is_url_healthy(f"http://{jobs[0]['nodelist']}:9002")


async def test_available_nodes_drops_as_the_cluster_fills():
    """`sinfo` idle-node parsing tracks real occupancy.

    The single node stops being `idle` once a job takes part of it, which is
    what the scheduler reads to decide whether there is room to deploy.
    """
    async with slurm_session() as manager:
        # WAIT for idle rather than asserting it. This is the one test here that
        # reads GLOBAL cluster state instead of this session's own jobs, so a
        # previous test whose jobs have been cancelled but not yet reaped leaves
        # the node `mixed` and would fail this for an unrelated reason. The
        # transition is what is under test, not the starting condition.
        async def _idle():
            return (await manager.get_available_nodes()) >= 1 or None

        await wait_until(_idle, timeout=60, what="the cluster to be idle to begin with")

        await manager.update_allocation([(make_model(), 1)], [])
        await wait_for_running(manager, 1)

        async def _no_longer_idle():
            return (await manager.get_available_nodes()) == 0 or None

        await wait_until(
            _no_longer_idle, timeout=60,
            what="the node to stop reporting idle once allocated",
        )

        await manager.cancel_all_owned_jobs()
        await drain(manager, timeout=60)

        async def _idle_again():
            return (await manager.get_available_nodes()) >= 1 or None

        await wait_until(_idle_again, timeout=60, what="the node to return to idle")
