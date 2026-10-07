"""
Edge-case tests for the imported dataset path in handle_imported_dataset_event,
focusing on _active_imported_dataset_jobs counter integrity and sentinel-file
logic (preemption requeue, setup failure, internal failure).
"""
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scheduler.scheduler import Scheduler
from scheduler.event import ImportedDatasetEventInstance
from scheduler.model import ModelInstance
from scheduler.slurm_manager import SlurmJobVanished
from scheduler.task import ImportedDatasetTask, ImportedDatasetConfig
from scheduler.imported_dataset import register
from scheduler.imported_dataset.base import ImportedDatasetRunnerBase
from scheduler.grader.base import Score


# ---------------------------------------------------------------------------
# Fake runner (identical to the one in test_scheduler.py)
# ---------------------------------------------------------------------------

@register("edge-test-runner")
class _FakeImportedRunner(ImportedDatasetRunnerBase):
    def build_setup_script(self, repo_root):
        return "echo setup"

    def build_benchmark_script(self, model_instance, task, output_dir):
        return "echo benchmark"

    def parse_results(self, output_dir, task, model_instance):
        return [Score(name="accuracy", value=0.85)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_instance(name="test-model", output_path=None):
    return ModelInstance(
        name=name,
        path=f"org/{name}",
        revision=None,
        venv_path="/fake/bin/activate",
        max_time_to_deploy=600,
        vllm_cli_args=[],
        output_path=str(output_path or "/tmp/output"),
        parser_type="noop",
        model_type="base",
        openai_kwargs={},
        max_simultaneous_requests=4,
        owner="test",
        tag="any",
    )


def _make_imported_task(uuid="edge-imported-task-uuid"):
    return ImportedDatasetTask(
        uuid=uuid,
        dataset_name="my_benchmark",
        semantic_version="1.0.0",
        imported_dataset=ImportedDatasetConfig(name="edge-test-runner", commit="abc123"),
    )


def _make_scheduler_in_memory():
    os.environ["EVAL360_IN_MEMORY_DB"] = "true"
    return Scheduler(model_directory=None, dataset_directory=None)


def _setup(tmp_path, task_uuid="edge-task", event_uuid="edge-ev"):
    """Create scheduler, model, task, event — returns (scheduler, event, output_dir)."""
    s = _make_scheduler_in_memory()
    model = _make_model_instance(output_path=tmp_path)
    s.db_manager.register_model(model)
    task = _make_imported_task(uuid=task_uuid)
    s.db_manager.register_task(task)

    scores_path = tmp_path / "my_benchmark_scores.yaml"
    event = ImportedDatasetEventInstance(
        uuid=event_uuid,
        parent_uuid="",
        model=model.name,
        task_uuid=task.uuid,
        path_to_scores=str(scores_path),
    )
    s.db_manager.register_event(event)

    # Mock slurm methods
    s.slurm_manager.submit_imported_dataset_job = AsyncMock(return_value=42)
    s.slurm_manager.get_job_node = AsyncMock(return_value="gpu-node-01")
    s.slurm_manager.wait_for_vllm_health = AsyncMock(return_value=True)
    s.slurm_manager.wait_for_job_completion = AsyncMock()
    s.slurm_manager.cancel_job = AsyncMock()

    output_dir = tmp_path / "my_benchmark_output"
    output_dir.mkdir()

    return s, event, output_dir


# ---------------------------------------------------------------------------
# TestImportedDatasetJobCounterIntegrity
# ---------------------------------------------------------------------------

class TestImportedDatasetJobCounterIntegrity:

    @pytest.mark.asyncio
    async def test_counter_decremented_on_success(self, tmp_path):
        s, event, output_dir = _setup(tmp_path, "task-ctr-ok", "ev-ctr-ok")
        (output_dir / ".job_complete").touch()

        assert s._active_imported_dataset_jobs == 0
        await s.handle_imported_dataset_event(event)
        assert s._active_imported_dataset_jobs == 0

    @pytest.mark.asyncio
    async def test_counter_decremented_on_vllm_health_failure(self, tmp_path):
        s, event, output_dir = _setup(tmp_path, "task-ctr-health", "ev-ctr-health")
        s.slurm_manager.wait_for_vllm_health = AsyncMock(return_value=False)

        assert s._active_imported_dataset_jobs == 0
        await s.handle_imported_dataset_event(event)
        assert s._active_imported_dataset_jobs == 0

    @pytest.mark.asyncio
    async def test_counter_decremented_on_preemption_requeue(self, tmp_path):
        s, event, output_dir = _setup(tmp_path, "task-ctr-preempt", "ev-ctr-preempt")
        s.event_manager.enqueue = AsyncMock()

        # Preemption: .setup_complete_job exists but no .job_complete and no .job_failed
        (output_dir / ".setup_complete_job").touch()

        assert s._active_imported_dataset_jobs == 0
        await s.handle_imported_dataset_event(event)
        assert s._active_imported_dataset_jobs == 0


# ---------------------------------------------------------------------------
# TestImportedDatasetJobVanishes
# ---------------------------------------------------------------------------

class TestImportedDatasetJobVanishes:
    """A job that leaves the queue before it is ever allocated a node.

    `get_job_node` raises for this rather than polling forever (see
    `SlurmManager.get_job_node`). The raise has to stop HERE: this coroutine
    runs as a child of the main loop's TaskGroup, and an exception escaping it
    tears the group down and triggers `cancel_all_owned_jobs`, which would
    cancel unrelated evaluations' Slurm jobs. The blast radius itself is pinned
    by `TestVanishedImportedJobDoesNotStopTheScheduler` in test_scheduler.py;
    what is pinned here is the OUTCOME the event is given, and that the raise
    that is caught is narrow.
    """

    @pytest.mark.asyncio
    async def test_vanished_job_fails_the_event_and_does_not_raise(self, tmp_path):
        """Nothing was written, so there is no evidence to requeue on."""
        s, event, output_dir = _setup(tmp_path, "task-vanish", "ev-vanish")
        s.slurm_manager.get_job_node = AsyncMock(
            side_effect=SlurmJobVanished(
                "Slurm job 42 is no longer queued; it left the queue before a "
                "node was allocated"
            )
        )
        health_mock = AsyncMock(return_value=True)
        s.slurm_manager.wait_for_vllm_health = health_mock

        await s.handle_imported_dataset_event(event)

        assert s.db_manager.get_event_phase(event.uuid) == -1
        assert s.progress_manager.get_status(event) == "Failed"
        health_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_counter_decremented_when_the_job_vanishes(self, tmp_path):
        """The active count must come back down, or allocation shrinks forever.

        `get_desired_allocation` subtracts `_active_imported_dataset_jobs` from
        the node budget, so a leaked increment permanently starves every other
        event of serving capacity.
        """
        s, event, _ = _setup(tmp_path, "task-vanish-ctr", "ev-vanish-ctr")
        s.slurm_manager.get_job_node = AsyncMock(
            side_effect=SlurmJobVanished("Slurm job 42 is no longer queued")
        )

        assert s._active_imported_dataset_jobs == 0
        await s.handle_imported_dataset_event(event)
        assert s._active_imported_dataset_jobs == 0

    @pytest.mark.asyncio
    async def test_vanished_job_that_had_completed_setup_is_requeued(self, tmp_path):
        """A mid-benchmark preemption between two polls is still a preemption.

        `get_job_node` polls every 10s by default, so a job CAN start, finish
        setup, and be preempted without this call ever seeing it RUNNING. The
        sentinel it left behind is the same evidence the post-completion path
        requeues on, and it gets the same outcome here — which is the reason
        the vanish case reuses that triage instead of always failing.
        """
        s, event, output_dir = _setup(tmp_path, "task-vanish-rq", "ev-vanish-rq")
        enqueue_mock = AsyncMock()
        s.event_manager.enqueue = enqueue_mock
        s.slurm_manager.get_job_node = AsyncMock(
            side_effect=SlurmJobVanished("Slurm job 42 is no longer queued")
        )
        (output_dir / ".setup_complete_job").touch()

        await s.handle_imported_dataset_event(event)

        enqueue_mock.assert_called_once_with(event)
        phase = s.db_manager.get_event_phase(event.uuid)
        assert phase != -1, "a requeued event must not also be marked failed"
        assert phase != 2

    @pytest.mark.asyncio
    async def test_a_genuine_error_from_get_job_node_still_propagates(self, tmp_path):
        """Only the expected absence is handled; real faults must not be eaten.

        This is why `SlurmJobVanished` exists as a type. Catching `RuntimeError`
        at the boundary would turn a squeue that is timing out, or a bug in the
        parser, into a quietly failed event with a misleading status.
        """
        s, event, _ = _setup(tmp_path, "task-vanish-real", "ev-vanish-real")
        s.slurm_manager.get_job_node = AsyncMock(
            side_effect=RuntimeError("squeue is broken in some new way")
        )

        with pytest.raises(RuntimeError, match="squeue is broken in some new way"):
            await s.handle_imported_dataset_event(event)

        # The counter is still restored by the `finally`, even on the path that
        # deliberately lets the exception out.
        assert s._active_imported_dataset_jobs == 0


# ---------------------------------------------------------------------------
# TestImportedDatasetPreemptionRequeue
# ---------------------------------------------------------------------------

class TestImportedDatasetPreemptionRequeue:

    @pytest.mark.asyncio
    async def test_preempted_job_is_requeued(self, tmp_path):
        s, event, output_dir = _setup(tmp_path, "task-preempt", "ev-preempt")
        enqueue_mock = AsyncMock()
        s.event_manager.enqueue = enqueue_mock

        # Preemption condition: .setup_complete_job exists, no .job_complete, no .job_failed
        (output_dir / ".setup_complete_job").touch()

        await s.handle_imported_dataset_event(event)

        enqueue_mock.assert_called_once_with(event)
        # Event should NOT be marked as failed or complete
        phase = s.db_manager.get_event_phase(event.uuid)
        assert phase != 2, "preempted event should not be marked complete"
        assert phase != -1, "preempted event should not be marked failed"


# ---------------------------------------------------------------------------
# TestImportedDatasetSetupFailure
# ---------------------------------------------------------------------------

class TestImportedDatasetSetupFailure:

    @pytest.mark.asyncio
    async def test_setup_failure_marks_event_failed(self, tmp_path):
        s, event, output_dir = _setup(tmp_path, "task-setup-fail", "ev-setup-fail")

        # Setup failure: no .job_complete, no .setup_complete_job, no .job_failed
        # (output_dir exists but is empty of sentinels)

        await s.handle_imported_dataset_event(event)

        assert s.db_manager.get_event_phase(event.uuid) == -1


# ---------------------------------------------------------------------------
# TestImportedDatasetInternalFailure
# ---------------------------------------------------------------------------

class TestImportedDatasetInternalFailure:

    @pytest.mark.asyncio
    async def test_internal_failure_marks_event_failed(self, tmp_path):
        s, event, output_dir = _setup(tmp_path, "task-int-fail", "ev-int-fail")

        # Internal failure: .setup_complete_job exists and .job_failed exists,
        # but no .job_complete
        (output_dir / ".setup_complete_job").touch()
        (output_dir / ".job_failed").touch()

        await s.handle_imported_dataset_event(event)

        assert s.db_manager.get_event_phase(event.uuid) == -1
