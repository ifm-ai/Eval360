"""
tests/test_e2e.py — End-to-end scheduler tests using TestManager.

TestManager replaces SlurmManager with in-memory fake Slurm state so tests
exercise the full scheduler orchestration (job submission, state transitions,
URL registration, allocation) without subprocess mocking.

Test structure:
  TestManagerUnit       — standalone TestManager correctness
  TestBasicE2E          — scheduler + TestManager: output files, job lifecycle
  TestURLRegistration   — scheduler registers URLs when jobs go live
  TestAllocation        — update_allocation called with correct replica counts
  TestDeadJobs          — dead jobs cause events to fail
  TestLoopE2E           — loop()-based tests with TestManager
"""
import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from scheduler.scheduler import Scheduler
from scheduler.model import ModelInstance
from scheduler.utils import Sentinel

# Import helpers from test_scheduler (all are module-level, importable)
from tests.test_scheduler import (
    _write_model_yaml,
    _write_dataset_yaml,
    _write_jsonl,
    _sample_rows,
    _FakeOpenAIConnection,
    _scheduler,
    _run_loop_until_files_exist,
)
from tests.fake_slurm import FakeSlurmManager, FakeJob


# ---------------------------------------------------------------------------
# Slow fake OpenAI connection (adds per-row delay so slurm ticks can fire)
# ---------------------------------------------------------------------------

class _SlowFakeOpenAIConnection(_FakeOpenAIConnection):
    """Adds a configurable async sleep between row yields.

    Use this so slurm ticks fire during generation, allowing TestManager to
    create and transition jobs before generation completes.
    """
    def __init__(self, canned_answers, row_delay=0.15):
        super().__init__(canned_answers)
        self._row_delay = row_delay

    async def launch_requests(self, requests, offset, completion_hook):
        async for item in requests:
            if item == Sentinel.COMPLETED:
                await completion_hook()
                yield item
                return
            await asyncio.sleep(self._row_delay)
            row = dict(item)
            row["generations"] = [self._answers[0]]
            yield row


# ---------------------------------------------------------------------------
# Helper: create scheduler with TestManager injected
# ---------------------------------------------------------------------------

def _scheduler_with_test_mgr(tmp_path, **test_mgr_kwargs) -> tuple[Scheduler, FakeSlurmManager]:
    """Create a Scheduler + TestManager pair, injecting the manager."""
    s = _scheduler(tmp_path)
    test_mgr = FakeSlurmManager(**test_mgr_kwargs)
    s.slurm_manager = test_mgr
    return s, test_mgr


# ===========================================================================
# TestManagerUnit — standalone correctness tests for TestManager itself
# ===========================================================================

class TestManagerUnit:
    """TestManager operates correctly as a standalone component."""

    @pytest.mark.asyncio
    async def test_jobs_start_pending(self):
        mgr = FakeSlurmManager(auto_run=False)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "abc"
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        jobs = mgr.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].state == "PENDING"

    @pytest.mark.asyncio
    async def test_auto_run_transitions_to_running(self):
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "abc"
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        mgr._tick()
        assert mgr.get_jobs()[0].state == "RUNNING"

    @pytest.mark.asyncio
    async def test_auto_healthy_marks_serving_key(self):
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0, auto_healthy=True, healthy_delay=0.0)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "sk123"
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        mgr._tick()  # PENDING → RUNNING + healthy
        assert "sk123" in mgr._healthy_sks

    @pytest.mark.asyncio
    async def test_preempt_cancels_jobs(self):
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "abc"
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        mgr._tick()
        mgr.preempt("m")
        assert mgr.get_active_job_count() == 0

    @pytest.mark.asyncio
    async def test_set_healthy_controls_health(self):
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0, auto_healthy=False)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "sk-x"
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        mgr._tick()
        assert "sk-x" not in mgr._healthy_sks
        mgr.set_healthy("sk-x")
        assert "sk-x" in mgr._healthy_sks

    @pytest.mark.asyncio
    async def test_cancel_all_owned_jobs(self):
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "abc"
        await mgr.update_allocation([(model, 2)], unneeded_models=[])
        mgr._tick()
        assert mgr.get_active_job_count() == 2
        await mgr.cancel_all_owned_jobs()
        assert mgr.get_active_job_count() == 0

    @pytest.mark.asyncio
    async def test_update_allocation_idempotent(self):
        """Calling update_allocation twice for the same model doesn't double-submit."""
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "abc"
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        mgr._tick()
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        assert mgr.get_active_job_count("m") == 1

    @pytest.mark.asyncio
    async def test_unneeded_models_cancelled(self):
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "abc"
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        mgr._tick()
        # Now submit with m as unneeded
        await mgr.update_allocation([], unneeded_models=["m"])
        assert mgr.get_active_job_count("m") == 0

    @pytest.mark.asyncio
    async def test_get_model_state_live(self):
        """get_model_state returns live entry when job is RUNNING and healthy."""
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0, auto_healthy=True, healthy_delay=0.0)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "sk-live"
        model.max_time_to_deploy = 600

        di = MagicMock()
        di.model = model
        desired = {"m": di}

        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        mgr._tick()

        pending, deploying, live, dead, replica_counts = await mgr.get_model_state(desired)
        assert ("m", mgr._vllm_url) in live
        assert "m" not in dead

    @pytest.mark.asyncio
    async def test_get_model_state_dead(self):
        """get_model_state marks model dead when elapsed > max_time_to_deploy."""
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0, auto_healthy=False)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "sk-dead"
        model.max_time_to_deploy = 0  # any elapsed time exceeds this

        di = MagicMock()
        di.model = model
        desired = {"m": di}

        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        mgr._tick()  # PENDING → RUNNING
        # Manually age the job so it exceeds max_ttd
        for job in mgr._jobs.values():
            job.created_at -= 10  # 10 seconds ago

        pending, deploying, live, dead, replica_counts = await mgr.get_model_state(desired)
        assert "m" in dead
        assert not any(mn == "m" for mn, _ in live)

    @pytest.mark.asyncio
    async def test_async_iterator_ticks(self):
        """__anext__ advances job state."""
        mgr = FakeSlurmManager(auto_run=True, run_delay=0.0, poll_interval=0.02)
        model = MagicMock()
        model.name = "m"
        model.serving_key = "abc"
        await mgr.update_allocation([(model, 1)], unneeded_models=[])
        # First tick via __anext__
        await mgr.__anext__()
        assert mgr.get_jobs()[0].state == "RUNNING"


# ===========================================================================
# TestBasicE2E — run_evaluate_now with TestManager (no subprocess patching)
# ===========================================================================

class TestBasicE2E:
    """Basic end-to-end tests: output files written correctly via TestManager."""

    @pytest.mark.asyncio
    async def test_generations_written(self, tmp_path):
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])
        s, _ = _scheduler_with_test_mgr(tmp_path)

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml)],
                paths_to_datasets=[str(dataset_yaml)],
            )

        gen_file = tmp_path / "output" / "test-model" / "test_dataset_generations.jsonl"
        assert gen_file.exists()
        assert len(gen_file.read_text().strip().splitlines()) == n_rows

    @pytest.mark.asyncio
    async def test_grades_written(self, tmp_path):
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])
        s, _ = _scheduler_with_test_mgr(tmp_path)

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml)],
                paths_to_datasets=[str(dataset_yaml)],
            )

        grades_file = tmp_path / "output" / "test-model" / "test_dataset_grades.jsonl"
        assert grades_file.exists()
        lines = grades_file.read_text().strip().splitlines()
        assert len(lines) == n_rows
        for line in lines:
            assert "correct" in json.loads(line)

    @pytest.mark.asyncio
    async def test_correct_score(self, tmp_path):
        n_rows = 4
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])
        s, _ = _scheduler_with_test_mgr(tmp_path)

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml)],
                paths_to_datasets=[str(dataset_yaml)],
            )

        scores_file = tmp_path / "output" / "test-model" / "test_dataset_scores.yaml"
        assert scores_file.exists()
        content = scores_file.read_text()
        assert "1.0" in content or ": 1\n" in content

    @pytest.mark.asyncio
    async def test_wrong_score(self, tmp_path):
        n_rows = 4
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        fake_conn = _FakeOpenAIConnection(canned_answers=["Z"])  # always wrong
        s, _ = _scheduler_with_test_mgr(tmp_path)

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml)],
                paths_to_datasets=[str(dataset_yaml)],
            )

        scores_file = tmp_path / "output" / "test-model" / "test_dataset_scores.yaml"
        assert scores_file.exists()
        content = scores_file.read_text()
        assert "0.0" in content or ": 0\n" in content

    @pytest.mark.asyncio
    async def test_multi_dataset(self, tmp_path):
        n_rows = 2
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Two datasets in the same directory with a glob
        _write_jsonl(data_dir / "ds1.jsonl", _sample_rows(n_rows, ground_truth="A"))
        _write_jsonl(data_dir / "ds2.jsonl", _sample_rows(n_rows, ground_truth="A"))

        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")

        # Two separate dataset YAMLs
        ds1_yaml = tmp_path / "ds1.yaml"
        ds1_yaml.write_text(yaml.dump({
            "uuid": "task-ds1",
            "grader": {"type": "exact_match"},
            "average_over": [1], "pass_at": [1],
            "dataset_name": "ds1",
            "data_path": str(data_dir / "ds1.jsonl"),
            "semantic_version": "1.0.0",
            "num_generations": None, "meta": {}, "tag": None,
        }))
        ds2_yaml = tmp_path / "ds2.yaml"
        ds2_yaml.write_text(yaml.dump({
            "uuid": "task-ds2",
            "grader": {"type": "exact_match"},
            "average_over": [1], "pass_at": [1],
            "dataset_name": "ds2",
            "data_path": str(data_dir / "ds2.jsonl"),
            "semantic_version": "1.0.0",
            "num_generations": None, "meta": {}, "tag": None,
        }))

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])
        s, _ = _scheduler_with_test_mgr(tmp_path)

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml)],
                paths_to_datasets=[str(ds1_yaml), str(ds2_yaml)],
            )

        out = tmp_path / "output" / "test-model"
        assert (out / "ds1_generations.jsonl").exists()
        assert (out / "ds2_generations.jsonl").exists()
        assert (out / "ds1_scores.yaml").exists()
        assert (out / "ds2_scores.yaml").exists()


# ===========================================================================
# TestURLRegistration — verify URL pool populated when jobs go live
# ===========================================================================

class TestURLRegistration:
    """Scheduler registers URLs in job_manager when TestManager reports live jobs."""

    @pytest.mark.asyncio
    async def test_url_registered_via_loop(self, tmp_path):
        """Running loop() long enough: live jobs cause URL registration."""
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        from scheduler.model import ModelParser
        from scheduler.task import Task

        # Slow conn: allows slurm ticks to fire during generation
        fake_conn = _SlowFakeOpenAIConnection(canned_answers=["A"], row_delay=0.15)
        s, test_mgr = _scheduler_with_test_mgr(tmp_path)

        gen_file = tmp_path / "output" / "test-model" / "test_dataset_generations.jsonl"
        grades_file = tmp_path / "output" / "test-model" / "test_dataset_grades.jsonl"
        scores_file = tmp_path / "output" / "test-model" / "test_dataset_scores.yaml"

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await _run_loop_until_files_exist(
                s, [gen_file, grades_file, scores_file], register_fn=register, timeout=30)

        # After the loop ran long enough for slurm ticks, URL should be registered
        live_urls = s.job_manager.get_live_urls("test-model")
        assert live_urls == [test_mgr._vllm_url], (
            f"Expected URL {test_mgr._vllm_url!r} in job_manager, got {live_urls}"
        )

    @pytest.mark.asyncio
    async def test_jobs_created_during_loop(self, tmp_path):
        """TestManager creates Slurm jobs when the scheduler runs loop()."""
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        from scheduler.model import ModelParser
        from scheduler.task import Task

        fake_conn = _SlowFakeOpenAIConnection(canned_answers=["A"], row_delay=0.15)
        s, test_mgr = _scheduler_with_test_mgr(tmp_path)

        gen_file = tmp_path / "output" / "test-model" / "test_dataset_generations.jsonl"
        grades_file = tmp_path / "output" / "test-model" / "test_dataset_grades.jsonl"
        scores_file = tmp_path / "output" / "test-model" / "test_dataset_scores.yaml"

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await _run_loop_until_files_exist(
                s, [gen_file, grades_file, scores_file], register_fn=register, timeout=30)

        # Jobs must have been created (possibly already completed)
        all_jobs = test_mgr.get_jobs("test-model")
        assert len(all_jobs) > 0, "TestManager should have created at least one job"


# ===========================================================================
# TestDeadJobs — dead jobs cause events to be marked failed
# ===========================================================================

class TestDeadJobs:
    """When TestManager reports dead jobs, the scheduler fails those events."""

    @pytest.mark.asyncio
    async def test_dead_job_fails_event(self, tmp_path):
        """Event is marked failed when job exceeds max_time_to_deploy."""
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 2
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))

        # max_time_to_deploy=1 so the job quickly times out
        model_yaml = tmp_path / "model.yaml"
        model_yaml.write_text(yaml.dump({
            "remote_model": {"base_name": "test-model", "path": "org/test-model", "revision": None},
            "model_type": "base",
            "parser_type": "noop",
            "name_modifier": None,
            "venv_path": "/fake/bin/activate",
            "max_simultaneous_requests": 4,
            "max_time_to_deploy": 1,  # 1-second timeout — will fire quickly
            "vllm_cli_args": [],
            "openai_kwargs": {"temperature": 0.0},
            "owner": "test",
            "ready": True,
            "output_path": str(tmp_path / "output"),
            "tag": None,
        }))
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        from scheduler.model import ModelParser
        from scheduler.task import Task

        # auto_healthy=False: VLLM never becomes healthy → dead job
        s, test_mgr = _scheduler_with_test_mgr(
            tmp_path, auto_run=True, run_delay=0.0, auto_healthy=False)

        # Slow conn: blocks waiting for URL (which never comes because VLLM is dead)
        # We need a connection that actually blocks on URL acquisition.
        # Since _FakeOpenAIConnection bypasses URL acquisition, we use a different
        # approach: just let the scheduler timeout and check event phase.

        # Use a stop condition: event marked failed = completed in db
        def events_failed():
            return s.db_manager.count_completed_events() >= 1

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        # Patch OpenAIConnection to a version that blocks until cancelled
        cancelled = asyncio.Event()

        class BlockingConn(_FakeOpenAIConnection):
            async def launch_requests(self, requests, offset, completion_hook):
                # Block until event is failed (simulates waiting for URL that never comes)
                await asyncio.wait_for(cancelled.wait(), timeout=30)
                return
                yield  # make it a generator

        blocking_conn = BlockingConn(canned_answers=["A"])

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=blocking_conn):
            # Age the job immediately after it starts running so it's "dead"
            original_tick = test_mgr._tick

            def aging_tick():
                original_tick()
                for job in test_mgr._jobs.values():
                    if job.state == "RUNNING":
                        job.created_at -= 100  # age by 100s → exceeds max_time_to_deploy=1
                test_mgr._tick = original_tick  # restore after first aging

            test_mgr._tick = aging_tick

            await _run_loop_until_files_exist(
                s, [], register_fn=register, timeout=15,
                stop_condition=events_failed)
            cancelled.set()  # unblock the connection

        completed = s.db_manager.count_completed_events()
        assert completed >= 1, "Event should be marked complete (failed) after dead job"
        # Check that it was marked as failed (phase=-1), not successful (phase=2)
        successful = s.db_manager.count_successful_events()
        assert successful == 0, f"Event should be failed, not successful (successful={successful})"


# ===========================================================================
# TestLoopE2E — loop()-based e2e using TestManager (no subprocess patching)
# ===========================================================================

class TestLoopE2E:
    """Full scheduler loop() tests with TestManager replacing SlurmManager."""

    @pytest.mark.asyncio
    async def test_basic_end_to_end(self, tmp_path):
        """loop() + TestManager: output files written, scores correct."""
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        from scheduler.model import ModelParser
        from scheduler.task import Task

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])
        s, _ = _scheduler_with_test_mgr(tmp_path)

        gen_file = tmp_path / "output" / "test-model" / "test_dataset_generations.jsonl"
        grades_file = tmp_path / "output" / "test-model" / "test_dataset_grades.jsonl"
        scores_file = tmp_path / "output" / "test-model" / "test_dataset_scores.yaml"

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await _run_loop_until_files_exist(
                s, [gen_file, grades_file, scores_file], register_fn=register, timeout=30)

        assert len(gen_file.read_text().strip().splitlines()) == n_rows
        assert len(grades_file.read_text().strip().splitlines()) == n_rows
        content = scores_file.read_text()
        assert "1.0" in content or ": 1\n" in content

    @pytest.mark.asyncio
    async def test_all_wrong_score(self, tmp_path):
        """loop() + TestManager: wrong answers produce score 0."""
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        from scheduler.model import ModelParser
        from scheduler.task import Task

        fake_conn = _FakeOpenAIConnection(canned_answers=["Z"])
        s, _ = _scheduler_with_test_mgr(tmp_path)
        scores_file = tmp_path / "output" / "test-model" / "test_dataset_scores.yaml"

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await _run_loop_until_files_exist(
                s, [scores_file], register_fn=register, timeout=30)

        assert "0.0" in scores_file.read_text() or ": 0\n" in scores_file.read_text()

    @pytest.mark.asyncio
    async def test_multiple_slurm_ticks_occur(self, tmp_path):
        """During loop(), TestManager __anext__ fires multiple times."""
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        from scheduler.model import ModelParser
        from scheduler.task import Task

        fake_conn = _SlowFakeOpenAIConnection(canned_answers=["A"], row_delay=0.12)
        s, test_mgr = _scheduler_with_test_mgr(tmp_path, poll_interval=0.05)

        tick_count = [0]
        original_anext = test_mgr.__anext__.__func__ if hasattr(test_mgr.__anext__, '__func__') else None

        original_tick = test_mgr._tick
        def counting_tick():
            tick_count[0] += 1
            original_tick()
        test_mgr._tick = counting_tick

        gen_file = tmp_path / "output" / "test-model" / "test_dataset_generations.jsonl"
        grades_file = tmp_path / "output" / "test-model" / "test_dataset_grades.jsonl"
        scores_file = tmp_path / "output" / "test-model" / "test_dataset_scores.yaml"

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await _run_loop_until_files_exist(
                s, [gen_file, grades_file, scores_file], register_fn=register, timeout=30)

        # With 3 rows × 0.12s delay and 0.05s poll interval, at least a few ticks
        assert tick_count[0] >= 3, f"Expected multiple ticks, got {tick_count[0]}"
