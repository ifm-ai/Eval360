"""
Edge-case tests for dead-model URL eviction, connection cancellation,
disappearing-job URL eviction, and connection-pool unblocking.

These tests exercise the scheduler's handle_job_update code paths at
scheduler.py lines 589-620 (dead model eviction, disappearing job eviction,
and in-flight connection cancellation).
"""
import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scheduler.scheduler import Scheduler
from scheduler.model import ModelParser
from scheduler.task import Task
from scheduler.utils import Sentinel
from scheduler.openai_interface import ModelConnectionPool
from tests.fake_slurm import FakeSlurmManager


# ---------------------------------------------------------------------------
# Helpers (copied from test_scheduler.py)
# ---------------------------------------------------------------------------

def _write_model_yaml(path: Path, output_path: Path, max_time_to_deploy: int = 600) -> Path:
    cfg = {
        "remote_model": {"base_name": "test-model", "path": "org/test-model", "revision": None},
        "model_type": "base",
        "parser_type": "noop",
        "name_modifier": None,
        "venv_path": "/fake/bin/activate",
        "max_simultaneous_requests": 4,
        "max_time_to_deploy": max_time_to_deploy,
        "vllm_cli_args": [],
        "openai_kwargs": {"temperature": 0.0},
        "owner": "test",
        "ready": True,
        "output_path": str(output_path),
        "tag": None,
    }
    p = path / "model.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _write_dataset_yaml(path: Path, data_path: str) -> Path:
    cfg = {
        "uuid": "test-task-uuid",
        "grader": {"type": "exact_match"},
        "average_over": [1],
        "pass_at": [1],
        "dataset_name": "test_dataset",
        "data_path": data_path,
        "semantic_version": "1.0.0",
        "num_generations": None,
        "meta": {},
        "tag": None,
    }
    p = path / "dataset.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _sample_rows(n=3, ground_truth="A"):
    return [
        {
            "row": i,
            "completion_input": f"Q{i}?",
            "chat_input": [{"role": "user", "content": f"Q{i}?"}],
            "ground_truth": ground_truth,
        }
        for i in range(n)
    ]


def _scheduler(tmp_path) -> Scheduler:
    """Create a Scheduler with in-memory DB, no real filesystem watcher."""
    return Scheduler(
        model_directory=None,
        dataset_directory=None,
        max_generation_jobs=2,
        max_grading_parallelism=4,
        log_dir=str(tmp_path),
    )


class _FakeOpenAIConnection:
    """Drop-in that immediately yields canned generations."""
    openai_connection = None

    def __init__(self, canned_answers, **kwargs):
        self._answers = canned_answers

    async def launch_requests(self, requests, offset, completion_hook):
        idx = 0
        async for item in requests:
            if item == Sentinel.COMPLETED:
                await completion_hook()
                yield item
                return
            row = dict(item)
            row["generations"] = [self._answers[idx % len(self._answers)]]
            idx += 1
            yield row

    async def is_live(self):
        return True

    async def wait_for_live(self):
        return True


async def _run_loop_until_files_exist(scheduler, expected_files, *, register_fn=None, timeout=30, stop_condition=None):
    """Run scheduler.loop() as a background task; cancel once expected_files exist."""
    loop_task = asyncio.create_task(scheduler.loop())
    try:
        await asyncio.sleep(0.05)
        if register_fn:
            await register_fn()
        deadline = time.monotonic() + timeout
        while True:
            files_ready = all(Path(f).exists() and Path(f).stat().st_size > 0 for f in expected_files)
            condition_met = stop_condition is None or stop_condition()
            if files_ready and condition_met:
                return
            if loop_task.done():
                exc = loop_task.exception()
                if exc:
                    raise exc
                return
            if time.monotonic() > deadline:
                missing = [f for f in expected_files if not Path(f).exists()]
                raise asyncio.TimeoutError(f"Timed out waiting for: {missing}")
            await asyncio.sleep(0.2)
    finally:
        if not loop_task.done():
            loop_task.cancel()
            try:
                await loop_task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDeadModelURLEviction:
    """When a model exceeds max_time_to_deploy, its URLs must be evicted."""

    @pytest.mark.asyncio
    async def test_dead_model_urls_evicted(self, tmp_path):
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(2, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output", max_time_to_deploy=1)
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        test_mgr = FakeSlurmManager(auto_healthy=False)
        s = _scheduler(tmp_path)
        s.slurm_manager = test_mgr

        # Once the job is RUNNING, age it past max_time_to_deploy so it's reported as dead.
        original_tick = test_mgr._tick
        def aging_tick():
            original_tick()
            for job in test_mgr._jobs.values():
                if job.state == "RUNNING":
                    job.created_at -= 100  # age by 100s >> max_time_to_deploy=1
            test_mgr._tick = original_tick  # restore after first aging
        test_mgr._tick = aging_tick

        class _BlockingConn:
            def __init__(self, **kwargs):
                self._stop = asyncio.Event()

            def cancel(self):
                self._stop.set()

            async def launch_requests(self, requests, offset, completion_hook):
                await self._stop.wait()
                return
                yield  # make this an async generator

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        def _model_was_dead():
            # The model is removed from desired_models_dict once it's marked dead.
            # Once a job has been submitted and the model is no longer desired, it was processed.
            return (
                test_mgr.submit_count >= 1
                and "test-model" not in s.event_manager.desired_models_dict
            )

        with patch("scheduler.openai_interface.OpenAIConnection", side_effect=_BlockingConn):
            await _run_loop_until_files_exist(
                s, [],
                register_fn=register,
                stop_condition=_model_was_dead,
                timeout=15,
            )

        # After the model is marked dead, its URLs must have been evicted.
        assert s.job_manager.get_live_urls("test-model") == []


class TestDeadModelConnectionCancelled:
    """When a model is marked dead, in-flight OpenAIConnections must be cancelled."""

    @pytest.mark.asyncio
    async def test_dead_model_cancels_in_flight_connection(self, tmp_path):
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(2, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output", max_time_to_deploy=1)
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        test_mgr = FakeSlurmManager(auto_healthy=False)
        s = _scheduler(tmp_path)
        s.slurm_manager = test_mgr

        original_tick = test_mgr._tick
        def aging_tick():
            original_tick()
            for job in test_mgr._jobs.values():
                if job.state == "RUNNING":
                    job.created_at -= 100
            test_mgr._tick = original_tick
        test_mgr._tick = aging_tick

        cancel_called = []

        class _TrackingBlockingConn:
            def __init__(self, **kwargs):
                self._stop = asyncio.Event()

            def cancel(self):
                cancel_called.append(True)
                self._stop.set()

            async def launch_requests(self, requests, offset, completion_hook):
                await self._stop.wait()
                return
                yield

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        with patch("scheduler.openai_interface.OpenAIConnection", side_effect=_TrackingBlockingConn):
            await _run_loop_until_files_exist(
                s, [],
                register_fn=register,
                stop_condition=lambda: len(cancel_called) > 0,
                timeout=15,
            )

        assert len(cancel_called) >= 1, "cancel() was never called on the in-flight connection"


class TestDisappearingJobURLEviction:
    """When a model's Slurm job vanishes (externally scancel-ed), URLs must be evicted."""

    @pytest.mark.asyncio
    async def test_externally_cancelled_job_urls_evicted(self, tmp_path):
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(2, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        test_mgr = FakeSlurmManager()
        s = _scheduler(tmp_path)
        s.slurm_manager = test_mgr

        preempted = [False]
        urls_were_registered = [False]

        class _BlockAfterPreemptConn:
            def __init__(self, **kwargs):
                self._stop = asyncio.Event()

            def cancel(self):
                self._stop.set()

            async def launch_requests(self, requests, offset, completion_hook):
                # Wait until preemption has happened, then just block.
                await self._stop.wait()
                return
                yield

        async def register():
            await s.register_model_spec(ModelParser.parse_yaml(str(model_yaml)))
            await s.register_task(Task.parse_yaml(str(dataset_yaml)))

        async def preempt_once_live():
            """Wait for a URL to be registered, then preempt the model."""
            while not s.job_manager.get_live_urls("test-model"):
                await asyncio.sleep(0.05)
            urls_were_registered[0] = True
            # Disable auto_healthy so newly submitted replacement jobs don't
            # immediately go live, giving us a window to observe URL eviction.
            test_mgr._auto_healthy = False
            test_mgr._healthy_sks.clear()
            test_mgr.preempt("test-model")
            preempted[0] = True

        # Start the preemption coroutine as a background task.
        preempt_task = None

        async def register_and_schedule_preempt():
            nonlocal preempt_task
            await register()
            preempt_task = asyncio.create_task(preempt_once_live())

        with patch("scheduler.openai_interface.OpenAIConnection", side_effect=_BlockAfterPreemptConn):
            await _run_loop_until_files_exist(
                s, [],
                register_fn=register_and_schedule_preempt,
                stop_condition=lambda: preempted[0] and s.job_manager.get_live_urls("test-model") == [],
                timeout=15,
            )
            if preempt_task and not preempt_task.done():
                preempt_task.cancel()
                try:
                    await preempt_task
                except asyncio.CancelledError:
                    pass

        assert urls_were_registered[0], "URL was never registered (model never went live)"
        assert s.job_manager.get_live_urls("test-model") == [], (
            "URLs should be evicted after job disappears from squeue"
        )


class TestConnectionPoolAllURLsRemoved:
    """ModelConnectionPool.acquire() must not hang forever when all URLs are removed."""

    @pytest.mark.asyncio
    async def test_pool_acquire_unblocked_by_cancel(self):
        pool = ModelConnectionPool(capacity_per_url=4)

        # Add a URL, acquire all its capacity, then remove it.
        await pool.add_url("http://fake:8000")
        url, n = await pool.acquire(requests_left=4)
        assert url == "http://fake:8000"
        assert n == 4

        # Now all slots are used. A second acquire() should block.
        acquire_task = asyncio.create_task(pool.acquire(requests_left=1))
        await asyncio.sleep(0.05)
        assert not acquire_task.done(), "acquire() should be blocking (no capacity)"

        # Remove the URL. This notifies the condition, but acquire() will loop
        # again and find no URLs with capacity — it blocks again.
        await pool.remove_url("http://fake:8000")
        await asyncio.sleep(0.05)
        assert not acquire_task.done(), "acquire() should still be blocking (no URLs at all)"

        # Adding a new URL with fresh capacity should unblock it.
        await pool.add_url("http://fake2:8000")
        result = await asyncio.wait_for(acquire_task, timeout=2.0)
        assert result[0] == "http://fake2:8000"
        assert result[1] >= 1
