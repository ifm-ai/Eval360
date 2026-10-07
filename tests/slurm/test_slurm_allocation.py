"""Allocation arithmetic, checked against a real queue instead of a fake one.

What this tests:
    `update_allocation` and `get_unneeded_models` deciding how many jobs to
    submit, skip, or cancel — with the job counts they read coming from real
    `squeue` output rather than from a fixture.

Why this exists:
    This is the logic `tests/fake_slurm.py` reimplements. `FakeSlurmManager`
    has its own copy of replica counting, `max_replica_idx` tracking and the
    unneeded/excess split, so the scheduler tests that use it are comparing one
    implementation against another implementation of the same idea. If both
    drifted the same way the tests would still pass. Here the counts come from
    Slurm.

    The idempotency case matters most in practice: `update_allocation` runs on
    every scheduler poll, so a counting bug does not fail loudly — it quietly
    submits a duplicate replica every few seconds until the account is full.

Corner cases covered:
    Scaling up from an existing deployment adds only the shortfall rather than
    the whole count; replica indices continue from the highest live index
    instead of restarting at zero; models are tracked per serving key so two
    models do not contaminate each other's counts; cancelling an unneeded model
    leaves a coexisting wanted model running; and a model whose config is
    edited in place — same name, new serving key — is redeployed rather than
    left serving the old config.
"""

from __future__ import annotations

import pytest

from .cluster_harness import (
    deployment_for,
    drain,
    make_model,
    running_jobs,
    slurm_session,
    wait_for_running,
    wait_until,
)

pytestmark = pytest.mark.cluster


async def test_a_single_replica_is_submitted_by_default():
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 1)], [])
        jobs = await wait_for_running(manager, 1)
        assert jobs[0]["replica_index"] == 0
        assert jobs[0]["serving_key"] == model.serving_key


async def test_multiple_replicas_get_distinct_indices():
    """Two replicas means two jobs, indexed 0 and 1.

    The index is part of the job name and is how a replica is told apart from
    its siblings; duplicates would make two jobs indistinguishable to
    `_from_job_name` and collapse them into one entry in `get_all_jobs`, whose
    result is keyed by job name.
    """
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 2)], [])
        jobs = await wait_for_running(manager, 2)

        indices = sorted(job["replica_index"] for job in jobs)
        assert indices == [0, 1]
        assert len({job["name"] for job in jobs}) == 2
        assert {job["serving_key"] for job in jobs} == {model.serving_key}


async def test_reallocating_the_same_count_submits_nothing_new():
    """The idempotency guarantee the scheduler's poll loop depends on.

    `update_allocation` is called on every poll with the same desired state. If
    it could not see that the replicas it wants already exist, it would submit
    another set every few seconds — which fails by filling the account, not by
    raising.
    """
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 2)], [])
        await wait_for_running(manager, 2)
        first_round = {job["name"] for job in await running_jobs(manager)}

        # Three more times, exactly as the poll loop would.
        for _ in range(3):
            await manager.update_allocation([(model, 2)], [])

        after = await running_jobs(manager)
        assert len(after) == 2, f"duplicate replicas were submitted: {after}"
        assert {job["name"] for job in after} == first_round
        # The ledger is the other half of the claim: no extra sbatch happened.
        assert len(manager.get_submitted_jobs()) == 2


async def test_scaling_up_adds_only_the_shortfall():
    """Going from 1 to 3 replicas submits 2 jobs, not 3, and indices continue."""
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 1)], [])
        await wait_for_running(manager, 1)
        assert len(manager.get_submitted_jobs()) == 1

        await manager.update_allocation([(model, 3)], [])
        jobs = await wait_for_running(manager, 3)

        assert len(manager.get_submitted_jobs()) == 3, "scaling up resubmitted replica 0"
        # Indices continue from the existing maximum rather than restarting,
        # which is what keeps names unique across a scale-up.
        assert sorted(job["replica_index"] for job in jobs) == [0, 1, 2]


async def test_an_unneeded_model_is_cancelled_and_a_wanted_one_survives():
    """The cancel path, with a second model present to prove it is targeted.

    Cancelling by serving key is easy to get wrong in a way that only shows up
    when more than one model is deployed — which is the normal case in
    production and never the case in a single-model test.
    """
    async with slurm_session() as manager:
        keep = make_model("ci-keep-model")
        drop = make_model("ci-drop-model", vllm_cli_args=["--max-model-len 512"])
        # Different vllm args give a different serving key, which is what the
        # cancel path actually keys on.
        assert keep.serving_key != drop.serving_key

        await manager.update_allocation([(keep, 1), (drop, 1)], [])
        await wait_for_running(manager, 2)

        await manager.update_allocation([(keep, 1)], ["ci-drop-model"])

        async def _only_keep_remains():
            jobs = await running_jobs(manager)
            names = {job["model_name"] for job in jobs}
            return jobs if names == {"ci-keep-model"} else None

        jobs = await wait_until(
            _only_keep_remains, timeout=60, what="only ci-keep-model to remain"
        )
        assert len(jobs) == 1
        assert jobs[0]["serving_key"] == keep.serving_key


async def test_editing_a_models_config_replaces_its_deployment():
    """Same model NAME, new serving key: the old deployment must not survive.

    THE BUG THIS PINS — currently FAILING, and not flaky. `update_allocation`
    counts existing replicas by model NAME (`replicas_per_model[model_name]`)
    but submits and indexes by SERVING KEY. When a model's config is edited in
    place the name is unchanged, so the stale job still counts towards the new
    config's replica target, `existing_count >= replica_count` holds, and the
    replacement is never submitted. Observed on the cluster:

        Skipping ci-edited-model, already has 1/1 replicas

    with the queue still holding only the OLD key's job.

    Why this is the production shape. A model YAML is changed in place —
    different `vllm_cli_args`, a different revision, different
    `serving_slurm_resources` — and re-registered under the name it already
    had. `serving_key` hashes exactly those fields (see
    `ModelInstance.serving_key`), so the key changes while the name does not.
    Nothing rejects that: `_validate_immutable_external_registration` returns
    early for a non-external model, `register_model` upserts, and `FSManager`'s
    `on_any_event` re-registers on modification, not only on creation.

    Why the test above cannot catch it: it varies the NAME as well as the key,
    so it only ever exercises "a name that is no longer desired". It never sees
    a name that stays desired while the key underneath it moves.

    Why the consequence is worse than a missed redeploy. The stale job is
    serving the OLD configuration, and `_serving_key_registry` maps BOTH keys
    to the same name, so `get_model_state` resolves that job back to the
    desired model and reports it live. Measured directly, after asking for key
    6f2cd0acd022 while fca1b3f77f95 was running:

        live = [('ci-edited-model', 'http://<node>:8000')]
        replica_counts = {'ci-edited-model': 1}

    The scheduler then generates against a server running the previous config
    and the eval completes green, with results nobody asked for.

    `unneeded_models` is empty on purpose, and is derived rather than written
    in: the scheduler passes whatever `get_unneeded_models` returned, and with
    the name still desired that list cannot contain this model. Passing the
    name by hand would test a call the scheduler never makes — and would hide
    the bug, because `kill_all` sweeps every serving key the registry has ever
    associated with a name.
    """
    async with slurm_session() as manager:
        before = make_model("ci-edited-model", vllm_cli_args=["--max-model-len 256"])
        after = make_model("ci-edited-model", vllm_cli_args=["--max-model-len 512"])
        assert before.name == after.name, "the point of this test is a stable name"
        assert before.serving_key != after.serving_key

        await manager.update_allocation([(before, 1)], [])
        stale = (await wait_for_running(manager, 1))[0]
        assert stale["serving_key"] == before.serving_key

        # What the scheduler would compute on the poll after the edit.
        unneeded, excess = await manager.get_unneeded_models(
            created_models=[after.name],
            active_models=[after.name],
            desired_models=deployment_for(after),
            maximum_nodes=4,
        )
        assert unneeded == [] and excess == [], (
            "a still-desired name is not unneeded; this test is about the "
            f"serving key changing underneath it (got {unneeded}, {excess})"
        )

        await manager.update_allocation([(after, 1)], unneeded + excess)

        # Checked before the queue settles because it fails for one reason only
        # — no second sbatch happened, i.e. the replacement was skipped as
        # already deployed. Waiting first would report a timeout instead.
        submitted = manager.get_submitted_jobs()
        assert len(submitted) == 2, (
            "the reconfigured model was never submitted: update_allocation "
            f"counted the stale replica as satisfying it. Ledger: "
            f"{[job.job_name for job in submitted]}"
        )
        assert any(after.serving_key in job.job_name for job in submitted)

        async def _only_the_new_key_remains():
            jobs = await running_jobs(manager)
            keys = {job["serving_key"] for job in jobs}
            return jobs if keys == {after.serving_key} else None

        jobs = await wait_until(
            _only_the_new_key_remains,
            timeout=90,
            what="the stale serving key to be released and the new one to run",
        )
        assert len(jobs) == 1
        assert jobs[0]["model_name"] == after.name


async def test_get_unneeded_models_finds_jobs_with_no_desired_model():
    """A deployed model that is no longer wanted is reported as unneeded."""
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 1)], [])
        await wait_for_running(manager, 1)

        unneeded, excess = await manager.get_unneeded_models(
            created_models=[model.name],
            active_models=[],
            desired_models={},          # nothing is wanted any more
            maximum_nodes=4,
        )
        assert unneeded == [model.name]
        assert excess == []


async def test_get_unneeded_models_keeps_a_still_desired_model():
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 1)], [])
        await wait_for_running(manager, 1)

        unneeded, excess = await manager.get_unneeded_models(
            created_models=[model.name],
            active_models=[model.name],
            desired_models=deployment_for(model),
            maximum_nodes=4,
        )
        assert unneeded == []
        assert excess == []


async def test_get_unneeded_models_trims_to_the_node_budget():
    """With more replicas deployed than nodes allowed, the surplus is excess.

    `maximum_nodes` is how the scheduler caps its own footprint on a shared
    cluster. The count it compares against comes from squeue, so an
    off-by-one here is an off-by-one in real occupancy.
    """
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 3)], [])
        await wait_for_running(manager, 3)

        unneeded, excess = await manager.get_unneeded_models(
            created_models=[model.name],
            active_models=[],
            desired_models=deployment_for(model),
            maximum_nodes=1,
        )
        assert unneeded == []
        assert model.name in excess


async def test_replica_counts_are_reported_per_model():
    """`get_model_state`'s replica_counts must total the occupied nodes.

    The scheduler uses this to know how much of the cluster it is holding,
    including replicas in mixed states, so it has to count queue entries rather
    than live endpoints.
    """
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 2)], [])
        await wait_for_running(manager, 2)

        pending, deploying, live, dead, replica_counts = await manager.get_model_state(
            deployment_for(model)
        )
        assert replica_counts == {model.name: 2}
        assert len(pending) + len(deploying) + len(live) == 2


async def test_two_models_are_tracked_independently():
    """Two models, two serving keys, no cross-contamination of counts."""
    async with slurm_session() as manager:
        first = make_model("ci-model-a")
        second = make_model("ci-model-b", vllm_cli_args=["--max-model-len 1024"])

        await manager.update_allocation([(first, 1), (second, 2)], [])
        jobs = await wait_for_running(manager, 3)

        by_model: dict[str, int] = {}
        for job in jobs:
            by_model[job["model_name"]] = by_model.get(job["model_name"], 0) + 1
        assert by_model == {"ci-model-a": 1, "ci-model-b": 2}

        _, _, _, _, replica_counts = await manager.get_model_state(
            deployment_for(first, second)
        )
        assert replica_counts == {"ci-model-a": 1, "ci-model-b": 2}


async def test_kill_all_removes_every_replica_of_a_model():
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 3)], [])
        await wait_for_running(manager, 3)

        await manager.kill_all([model.name])
        await drain(manager, timeout=60)

        assert await manager.get_all_jobs() == {}
