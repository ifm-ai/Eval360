"""Request-native, write-last terminal-result integration tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from scheduler.evaluation_request import EvaluationRequest
from scheduler.terminal_result import (
    bind_regular_file,
    canonical_json,
    publish_terminal_result,
)
from scheduler.utils import ExceptionWrapper, Sentinel
from tests.fake_slurm import FakeSlurmManager
from tests.test_evaluation_request import build_request_fixture
from tests.test_scheduler import _FakeOpenAIConnection, _scheduler


def _load_request(tmp_path: Path, *, result_name: str = "terminal-result.json"):
    request_path, result_path, runner, raw = build_request_fixture(
        tmp_path,
        result_name=result_name,
    )
    request = EvaluationRequest.load(
        request_path=request_path,
        terminal_result_path=result_path,
        actual_runner_entrypoint=runner,
    )
    return request, request_path, result_path, runner, raw


async def _run_request(tmp_path: Path, request: EvaluationRequest, slurm=None):
    scheduler = _scheduler(tmp_path)
    scheduler.slurm_manager = slurm or FakeSlurmManager()
    with patch(
        "scheduler.openai_interface.OpenAIConnection",
        return_value=_FakeOpenAIConnection(canned_answers=["A"]),
    ):
        result = await scheduler.run_evaluate_now(
            None,
            None,
            force=True,
            evaluation_request=request,
        )
    return scheduler, result


def _completed_resume_result(
    request: EvaluationRequest,
    output_binding: dict,
) -> dict:
    events = []
    for ordinal, suite_task in enumerate(request.suite.tasks):
        events.append(
            {
                "controller_units": [
                    {
                        "executor": "none",
                        "job_ids": [],
                        "outcome": "not_required",
                        "required": False,
                        "role": role,
                    }
                    for role in ("generation", "grading", "aggregation")
                ],
                "event_id": f"event-{ordinal}",
                "inputs": {
                    "dataset_files": [
                        item.as_dict()
                        for item in request.dataset_bindings[suite_task.task_id]
                    ],
                    "task_closure_sha256": request.task_closure_sha256[
                        suite_task.task_id
                    ],
                },
                "ordinal": ordinal,
                "outputs": [
                    {"role": role, **output_binding}
                    for role in (
                        "generations",
                        "grades",
                        "scores",
                        "run_metadata",
                    )
                ],
                "task": {
                    "definition": suite_task.as_dict(),
                    "id": suite_task.task_id,
                },
                "terminal_phase": 2,
            }
        )
    return {
        "bindings": request.bindings_dict(),
        "contract": "eval360.evaluate-now-terminal-result",
        "execution": request.raw_execution,
        "jobs": [],
        "runner": {
            "id": request.runner_definition.runner_id,
            "source": request.runner_definition.source.as_dict(),
        },
        "schema_version": "1.0",
        "selection": {
            "events": events,
            "mode": "evaluation_request",
            "suite": {
                "catalog_id": request.suite_catalog.catalog_id,
                "id": request.suite.suite_id,
                "owner": request.suite.owner,
                "source": request.suite.source.as_dict(),
                "version": request.suite.version,
            },
        },
        "status": "succeeded",
    }


@pytest.mark.asyncio
async def test_terminal_result_records_exact_order_outputs_and_sacct_children(
    tmp_path: Path,
):
    request, _, result_path, _, _ = _load_request(tmp_path)
    _, scheduler_result = await _run_request(tmp_path, request)

    result = publish_terminal_result(request, scheduler_result)

    assert result_path.read_text(encoding="utf-8") == canonical_json(result)
    assert result["status"] == "succeeded"
    assert result["controller"] == {"exit_code": 0, "outcome": "succeeded"}
    assert result["bindings"] == request.bindings_dict()
    assert result["runner"] == {
        "id": request.runner_definition.runner_id,
        "source": request.runner_definition.source.as_dict(),
    }
    events = result["selection"]["events"]
    assert [event["ordinal"] for event in events] == [0, 1]
    assert [event["task"]["id"] for event in events] == ["task-a", "task-b"]
    assert [event["task"]["definition"] for event in events] == [
        task.as_dict() for task in request.suite.tasks
    ]
    assert all(event["terminal_phase"] == 2 for event in events)
    assert all(
        [output["role"] for output in event["outputs"]]
        == ["generations", "grades", "scores", "run_metadata"]
        for event in events
    )
    assert all(
        len(output["sha256"]) == 64 and output["size_bytes"] >= 0
        for event in events
        for output in event["outputs"]
    )
    assert result["jobs"]
    for job in result["jobs"]:
        assert job["scheduler"]["source"] == "sacct"
        assert job["scheduler"]["state"] == "CANCELLED"
        assert job["cancellation_intent"] == "scheduler_release"
        assert job["associations"]
        assert job["role_completions"]
    assert {
        completion["event_id"]
        for job in result["jobs"]
        for completion in job["role_completions"]
    } == {event["event_id"] for event in events}

    with pytest.raises(FileExistsError):
        publish_terminal_result(request, scheduler_result)


@pytest.mark.asyncio
async def test_terminal_flow_releases_ledger_before_propagating_job_error(
    tmp_path: Path,
):
    request, _, result_path, _, _ = _load_request(tmp_path)
    scheduler = _scheduler(tmp_path)
    source_stopped = asyncio.Event()

    class ObservableFakeSlurm(FakeSlurmManager):
        async def __anext__(self):
            try:
                return await super().__anext__()
            except asyncio.CancelledError:
                source_stopped.set()
                raise

    slurm = ObservableFakeSlurm()
    scheduler.slurm_manager = slurm
    job_handler_started = asyncio.Event()
    job_handler_finished = asyncio.Event()
    original_handle_job_update = scheduler.handle_job_update
    original_record_role = scheduler._record_serving_role_completion
    original_release = slurm.release_submitted_model_serving_jobs
    injected_job_update = False
    release_calls = 0

    async def handle_job_update(job_state):
        if job_state != "fail-after-source-stop":
            await original_handle_job_update(job_state)
            return
        job_handler_started.set()
        await source_stopped.wait()
        job_handler_finished.set()
        raise RuntimeError("periodic job handler failed after phase completion")

    async def record_role(event_instance, *, is_grading):
        nonlocal injected_job_update
        await original_record_role(event_instance, is_grading=is_grading)
        if not injected_job_update:
            injected_job_update = True
            await scheduler._queue.put(("job", "fail-after-source-stop"))
            await job_handler_started.wait()

    async def release_submitted_model_serving_jobs():
        nonlocal release_calls
        assert job_handler_finished.is_set()
        release_calls += 1
        await original_release()

    scheduler.handle_job_update = handle_job_update
    scheduler._record_serving_role_completion = record_role
    slurm.release_submitted_model_serving_jobs = (
        release_submitted_model_serving_jobs
    )

    with (
        patch(
            "scheduler.openai_interface.OpenAIConnection",
            return_value=_FakeOpenAIConnection(canned_answers=["A"]),
        ),
        patch("scheduler.scheduler._EVALUATE_NOW_QUEUE_POLL_TIMEOUT", 0.01),
        pytest.raises(
            RuntimeError,
            match="periodic job handler failed after phase completion",
        ),
    ):
        await asyncio.wait_for(
            scheduler.run_evaluate_now(
                None,
                None,
                force=True,
                evaluation_request=request,
            ),
            timeout=5,
        )

    assert not result_path.exists()
    assert job_handler_started.is_set()
    assert source_stopped.is_set()
    assert job_handler_finished.is_set()
    assert release_calls == 1
    jobs = slurm.get_submitted_jobs()
    assert jobs
    assert all(
        job.cancellation_intent == "scheduler_release" for job in jobs
    )


@pytest.mark.asyncio
async def test_interrupt_cleanup_failure_does_not_mask_primary_error(
    tmp_path: Path,
):
    request, _, result_path, _, _ = _load_request(tmp_path)
    scheduler = _scheduler(tmp_path)

    class CleanupFailure(FakeSlurmManager):
        async def cancel_all_owned_jobs(self):
            raise RuntimeError("interrupt cleanup failed")

    async def fail_event(_event):
        raise ValueError("primary evaluation failure")

    scheduler.slurm_manager = CleanupFailure()
    scheduler.handle_event = fail_event
    with (
        patch("scheduler.scheduler._EVALUATE_NOW_QUEUE_POLL_TIMEOUT", 0.01),
        pytest.raises(ValueError, match="primary evaluation failure"),
    ):
        await asyncio.wait_for(
            scheduler.run_evaluate_now(
                None,
                None,
                force=True,
                evaluation_request=request,
            ),
            timeout=5,
        )

    assert not result_path.exists()


@pytest.mark.asyncio
async def test_failed_sacct_child_prevents_result_construction(tmp_path: Path):
    request, _, result_path, _, _ = _load_request(tmp_path)

    class FailedAccounting(FakeSlurmManager):
        async def wait_for_terminal_job_outcomes(self, job_ids):
            outcomes = await super().wait_for_terminal_job_outcomes(job_ids)
            first = min(outcomes)
            outcomes[first] = replace(
                outcomes[first],
                raw_state="NODE_FAIL",
                state="NODE_FAIL",
                exit_code=1,
                signal=0,
                reason="Node failure",
            )
            return outcomes

    with pytest.raises(RuntimeError, match="Slurm child job .* unsuccessful"):
        await _run_request(tmp_path, request, FailedAccounting())
    assert not result_path.exists()


@pytest.mark.asyncio
async def test_scheduler_release_without_role_completion_is_not_success(
    tmp_path: Path,
):
    request, _, result_path, _, _ = _load_request(tmp_path)

    class MissingRoleCompletion(FakeSlurmManager):
        def get_submitted_jobs(self):
            return tuple(
                replace(job, completed_event_roles=())
                for job in super().get_submitted_jobs()
            )

    with pytest.raises(RuntimeError, match="unassociated.*Slurm child"):
        await _run_request(tmp_path, request, MissingRoleCompletion())
    assert not result_path.exists()


@pytest.mark.asyncio
async def test_request_mode_propagates_slurm_submission_ambiguity(tmp_path: Path):
    request, _, result_path, _, _ = _load_request(tmp_path)

    class AmbiguousSubmission(FakeSlurmManager):
        def __init__(self):
            super().__init__()
            self._raised = False

        async def update_allocation(
            self,
            desired_allocations,
            unneeded_models,
            sibling_names_by_sk=None,
        ):
            if desired_allocations and not self._raised:
                self._raised = True
                raise RuntimeError(
                    "sbatch succeeded without a parseable submitted job ID"
                )
            return None

    with pytest.raises(RuntimeError, match="parseable submitted job ID"):
        await _run_request(tmp_path, request, AmbiguousSubmission())
    assert not result_path.exists()


@pytest.mark.asyncio
async def test_request_mode_never_ignores_generation_failure(
    tmp_path: Path,
    monkeypatch,
):
    request, _, result_path, _, _ = _load_request(tmp_path)

    class FailedGeneration(_FakeOpenAIConnection):
        async def launch_requests(self, requests, offset, completion_hook):
            items = [item async for item in requests]
            await completion_hook()
            for item in items:
                if item == Sentinel.COMPLETED:
                    yield item
                    return
                yield ExceptionWrapper(
                    exception=RuntimeError("request-native generation failed"),
                    instance=dict(item),
                    trace="",
                )

    scheduler = _scheduler(tmp_path)
    scheduler.slurm_manager = FakeSlurmManager()
    monkeypatch.setenv("EVAL360_IGNORE_ERRORS", "true")
    with (
        patch(
            "scheduler.openai_interface.OpenAIConnection",
            return_value=FailedGeneration(canned_answers=["A"]),
        ),
        pytest.raises(RuntimeError, match="request-native generation failed"),
    ):
        await scheduler.run_evaluate_now(
            None,
            None,
            force=True,
            evaluation_request=request,
        )
    assert not result_path.exists()


@pytest.mark.asyncio
async def test_stale_completed_outputs_cannot_resume_under_a_new_request(
    tmp_path: Path,
):
    request, request_path, _, runner, raw = _load_request(tmp_path)
    _, scheduler_result = await _run_request(tmp_path, request)
    publish_terminal_result(request, scheduler_result)

    changed = json.loads(request_path.read_text(encoding="utf-8"))
    changed["execution"]["owner"] = "different-owner"
    changed_path = tmp_path / "changed-request.json"
    changed_path.write_text(canonical_json(changed), encoding="utf-8")
    changed_request = EvaluationRequest.load(
        request_path=changed_path,
        terminal_result_path=tmp_path / "changed-result.json",
        actual_runner_entrypoint=runner,
    )
    scheduler = _scheduler(tmp_path)
    scheduler.slurm_manager = FakeSlurmManager()

    with pytest.raises(ValueError, match="different evaluation request"):
        await scheduler.run_evaluate_now(
            None,
            None,
            evaluation_request=changed_request,
        )
    assert not changed_request.output_path.exists()


@pytest.mark.asyncio
async def test_manifest_or_payload_drift_prevents_write_last_publication(
    tmp_path: Path,
):
    request, _, result_path, _, _ = _load_request(tmp_path)
    _, scheduler_result = await _run_request(tmp_path, request)
    Path(request.release_payloads[0].resolved_path).write_bytes(b"changed")

    with pytest.raises(ValueError, match="release payload.*changed"):
        publish_terminal_result(request, scheduler_result)
    assert not result_path.exists()


def test_nonfinite_terminal_payload_is_not_published(tmp_path: Path):
    request, _, result_path, _, _ = _load_request(tmp_path)
    output = tmp_path / "output.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    output_binding = bind_regular_file(output, label="output").as_dict()
    scheduler_result = _completed_resume_result(request, output_binding)
    scheduler_result["metric"] = float("nan")

    with pytest.raises(ValueError, match="Out of range float values"):
        publish_terminal_result(request, scheduler_result)
    assert not result_path.exists()


def test_missing_required_controller_unit_is_not_published(tmp_path: Path):
    request, _, result_path, _, _ = _load_request(tmp_path)
    output = tmp_path / "output.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    scheduler_result = _completed_resume_result(
        request,
        bind_regular_file(output, label="output").as_dict(),
    )
    scheduler_result["selection"]["events"][0]["controller_units"].pop()

    with pytest.raises(ValueError, match="incomplete controller units"):
        publish_terminal_result(request, scheduler_result)
    assert not result_path.exists()
