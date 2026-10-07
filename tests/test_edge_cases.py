"""
Regression tests for closed issues — edge cases around run_evaluate_now.

- run_evaluate_now hangs when the only event fails (dead job).
- run_evaluate_now doesn't cancel Slurm jobs on interrupt.
- Pre-existing generations cause unnecessary VLLM deployment.
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
from scheduler.utils import Sentinel
from tests.fake_slurm import FakeSlurmManager
from tests.test_scheduler import (
    _write_model_yaml,
    _write_dataset_yaml,
    _write_jsonl,
    _sample_rows,
    _FakeOpenAIConnection,
    _scheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_model_yaml_with_ttd(path: Path, output_path: Path, max_time_to_deploy: int = 600) -> Path:
    """Like _write_model_yaml but with configurable max_time_to_deploy."""
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


def _scheduler_with_fake_slurm(tmp_path, **slurm_kwargs) -> tuple[Scheduler, FakeSlurmManager]:
    s = _scheduler(tmp_path)
    mgr = FakeSlurmManager(**slurm_kwargs)
    s.slurm_manager = mgr
    return s, mgr


# ---------------------------------------------------------------------------
# Test 1: run_evaluate_now exits promptly when the only event fails
# ---------------------------------------------------------------------------

class TestEvaluateNowExitsOnFailedEvent:
    """Regression: run_evaluate_now hung when the only event failed.

    When the only event reaches phase -1 (deployment failure / dead job),
    run_evaluate_now should exit promptly — not hang forever.
    """

    @pytest.mark.asyncio
    async def test_exits_on_failed_event(self, tmp_path):
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(3, ground_truth="A"))
        model_yaml = _write_model_yaml_with_ttd(
            tmp_path, output_path=tmp_path / "output", max_time_to_deploy=1
        )
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        s, test_mgr = _scheduler_with_fake_slurm(
            tmp_path, auto_run=True, run_delay=0.0, auto_healthy=False
        )

        # BlockingConn: blocks forever in launch_requests (simulates waiting
        # for a VLLM URL that never becomes healthy).
        class BlockingConn(_FakeOpenAIConnection):
            async def launch_requests(self, requests, offset, completion_hook):
                await asyncio.Event().wait()  # block forever
                return
                yield  # make it an async generator

        blocking_conn = BlockingConn(canned_answers=["A"])

        # Age jobs past max_time_to_deploy on the first tick so they appear "dead".
        original_tick = test_mgr._tick

        def aging_tick():
            original_tick()
            for job in test_mgr._jobs.values():
                if job.state == "RUNNING":
                    job.created_at -= 100  # exceeds max_time_to_deploy=1
            test_mgr._tick = original_tick  # restore after first aging

        test_mgr._tick = aging_tick

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=blocking_conn):
            # If run_evaluate_now hangs, this will raise asyncio.TimeoutError
            # and fail the test — which is the desired behavior for a regression test.
            with pytest.raises(
                RuntimeError,
                match="1 of 1 selected evaluation events failed",
            ):
                await asyncio.wait_for(
                    s.run_evaluate_now(
                        paths_to_model_specs=[str(model_yaml)],
                        paths_to_datasets=[str(dataset_yaml)],
                    ),
                    timeout=15,
                )

        # The event should have been marked as failed (phase -1).
        assert s.db_manager.count_completed_events() >= 1
        assert s.db_manager.count_successful_events() == 0


# ---------------------------------------------------------------------------
# Test 2: run_evaluate_now cancels jobs on interrupt
# ---------------------------------------------------------------------------

class TestEvaluateNowCancelsJobsOnInterrupt:
    """Regression: run_evaluate_now did not cancel Slurm jobs on interrupt.

    When run_evaluate_now is interrupted (CancelledError), it should call
    cancel_all_owned_jobs() so no Slurm jobs are left running.
    """

    @pytest.mark.asyncio
    async def test_cancels_jobs_on_interrupt(self, tmp_path):
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(3, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        s, test_mgr = _scheduler_with_fake_slurm(tmp_path)

        # SlowConn: yields results very slowly so we have time to cancel.
        class SlowConn(_FakeOpenAIConnection):
            async def launch_requests(self, requests, offset, completion_hook):
                async for item in requests:
                    if item == Sentinel.COMPLETED:
                        await completion_hook()
                        yield item
                        return
                    await asyncio.sleep(60)  # block long enough to be cancelled
                    row = dict(item)
                    row["generations"] = ["A"]
                    yield row

        slow_conn = SlowConn(canned_answers=["A"])

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=slow_conn):
            task = asyncio.create_task(
                s.run_evaluate_now(
                    paths_to_model_specs=[str(model_yaml)],
                    paths_to_datasets=[str(dataset_yaml)],
                )
            )

            # Wait until at least one job has been submitted.
            deadline = time.monotonic() + 10
            while test_mgr.get_active_job_count() == 0:
                if time.monotonic() > deadline:
                    task.cancel()
                    raise AssertionError("No job was submitted within 10 seconds")
                await asyncio.sleep(0.05)

            # Cancel the task (simulating Ctrl-C / interrupt).
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # After cancellation, all owned jobs should have been cancelled.
        assert test_mgr.get_active_job_count() == 0, (
            f"Expected 0 active jobs after interrupt, got {test_mgr.get_active_job_count()}"
        )


# ---------------------------------------------------------------------------
# Test 3: pre-existing generations skip VLLM deployment
# ---------------------------------------------------------------------------

class TestCompletedGenerationsSkipDeployment:
    """Regression: pre-existing generations caused an unnecessary VLLM deployment.

    When all generations already exist on disk, the scheduler should
    complete grading without needing VLLM to become healthy. The
    completion_hook fires immediately, removing the model from the
    desired set, so the Slurm job (if any) is quickly cleaned up.
    """

    @pytest.mark.asyncio
    async def test_completes_grading_with_preexisting_generations(self, tmp_path):
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"

        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        # Pre-populate the generations file with all rows completed.
        output_dir = tmp_path / "output" / "test-model"
        output_dir.mkdir(parents=True)
        gen_rows = []
        for i in range(n_rows):
            gen_rows.append({
                "row": i,
                "completion_input": f"Q{i}?",
                "chat_input": [{"role": "user", "content": f"Q{i}?"}],
                "ground_truth": "A",
                "generations": ["A"],
            })
        _write_jsonl(output_dir / "test_dataset_generations.jsonl", gen_rows)

        s, test_mgr = _scheduler_with_fake_slurm(tmp_path)

        # _FakeOpenAIConnection handles the empty iterator case (COMPLETED
        # is the only item yielded when all generations pre-exist).
        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])

        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await asyncio.wait_for(
                s.run_evaluate_now(
                    paths_to_model_specs=[str(model_yaml)],
                    paths_to_datasets=[str(dataset_yaml)],
                ),
                timeout=15,
            )

        # Grades and scores files should be written correctly from
        # pre-existing generations — no new generation was needed.
        grades_file = output_dir / "test_dataset_grades.jsonl"
        scores_file = output_dir / "test_dataset_scores.yaml"

        assert grades_file.exists(), "grades file was not written"
        grade_lines = grades_file.read_text().strip().splitlines()
        assert len(grade_lines) == n_rows, f"Expected {n_rows} grade lines, got {len(grade_lines)}"
        for line in grade_lines:
            row = json.loads(line)
            assert "correct" in row, f"Grade row missing 'correct' field: {row}"

        assert scores_file.exists(), "scores file was not written"
        scores_content = scores_file.read_text()
        assert len(scores_content.strip()) > 0, "scores file is empty"
