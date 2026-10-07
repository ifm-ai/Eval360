"""Strict owner-manifest and request-native ``evaluate-now`` tests."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scheduler.evaluation_request import (
    EVALUATION_REQUEST_CONTRACT,
    EvaluationRequest,
)
from scheduler.terminal_result import bind_regular_file, canonical_json


def _write_canonical(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")
    return path


def _binding(path: Path) -> dict:
    return bind_regular_file(path, label=path.name).as_dict()


def _source() -> dict:
    return {
        "commit": "a" * 40,
        "repository": "https://github.com/IFM-AI/Eval360.git",
        "tree": "b" * 40,
    }


def build_request_fixture(
    tmp_path: Path,
    *,
    task_order: tuple[str, ...] = ("task-a", "task-b"),
    result_name: str = "terminal-result.json",
) -> tuple[Path, Path, Path, dict]:
    """Create one self-contained canonical request and its immutable inputs."""
    runner = tmp_path / "runner" / "eval360"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    serving_root = tmp_path / "serving"
    serving_entrypoint = serving_root / "bin" / "activate"
    serving_entrypoint.parent.mkdir(parents=True)
    serving_entrypoint.write_text("# serving runtime\n", encoding="utf-8")
    serving_payload = serving_root / "runtime.lock"
    serving_payload.write_text("vllm==1\n", encoding="utf-8")
    serving_manifest = _write_canonical(
        tmp_path / "definitions" / "serving.json",
        {"capability": "serving-v1"},
    )

    release_root = tmp_path / "release"
    release_root.mkdir()
    release_payload = release_root / "model.safetensors"
    release_payload.write_bytes(b"model-weights")
    release_manifest = _write_canonical(
        release_root / "manifest.json",
        {"release": "checkpoint-5000"},
    )

    data_root = tmp_path / "datasets"
    data_root.mkdir()
    rows_by_task = {
        "task-a": [
            {
                "row": 0,
                "completion_input": "A?",
                "chat_input": [{"content": "A?", "role": "user"}],
                "ground_truth": "A",
            },
            {
                "row": 1,
                "completion_input": "B?",
                "chat_input": [{"content": "B?", "role": "user"}],
                "ground_truth": "A",
            },
        ],
        "task-b": [
            {
                "row": 2,
                "completion_input": "C?",
                "chat_input": [{"content": "C?", "role": "user"}],
                "ground_truth": "A",
            }
        ],
    }
    dataset_paths: dict[str, Path] = {}
    for task_id, rows in rows_by_task.items():
        dataset_paths[task_id] = data_root / f"{task_id}.jsonl"
        dataset_paths[task_id].write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    suite_tasks = []
    for task_id in task_order:
        dataset = dataset_paths[task_id]
        dataset_binding = _binding(dataset)
        suite_tasks.append(
            {
                "average_over": [1],
                "dataset_files": [
                    {
                        "path": dataset.name,
                        "sha256": dataset_binding["sha256"],
                        "size_bytes": dataset_binding["size_bytes"],
                    }
                ],
                "dataset_name": task_id,
                "enabled": True,
                "grader": {"llm_as_judge": None, "type": "exact_match"},
                "id": task_id,
                "meta": {},
                "mode": "base",
                "num_generations": len(rows_by_task[task_id]),
                "openai_settings": None,
                "pass_at": [1],
                "semantic_version": "1.0.0",
                "tag": "any",
            }
        )

    suite = {
        "compatibility": {
            "model_family": "k2_horizon",
            "model_type": "base",
        },
        "contract": "eval360.suite",
        "model": {
            "allow_long_max_model_len": True,
            "cache_salt": {"mode": "disabled", "salt": None},
            "max_simultaneous_requests": 4,
            "max_time_to_deploy": 600,
            "openai_kwargs": {"temperature": 0.0},
            "parser_type": "noop",
            "prompt_prefix_instructions": None,
            "vllm_cli_args": [],
            "vllm_logging_level": "WARNING",
        },
        "owner": "LLM360",
        "schema_version": "1.0",
        "serving_runtime": {
            "capability_id": "k2-horizon-serving@1",
            "manifest_sha256": _binding(serving_manifest)["sha256"],
        },
        "source": _source(),
        "suite_id": "eval360/k2-horizon-base",
        "tasks": suite_tasks,
        "version": "1",
    }
    suite_path = _write_canonical(tmp_path / "catalog" / "suite.json", suite)
    suite_binding = _binding(suite_path)
    catalog = {
        "catalog_id": "eval360/default",
        "contract": "eval360.suite-catalog",
        "owner": "LLM360",
        "schema_version": "1.0",
        "source": _source(),
        "suites": [
            {
                "manifest_path": "suite.json",
                "manifest_sha256": suite_binding["sha256"],
                "manifest_size_bytes": suite_binding["size_bytes"],
                "suite_id": suite["suite_id"],
                "version": suite["version"],
            }
        ],
    }
    catalog_path = _write_canonical(tmp_path / "catalog" / "catalog.json", catalog)

    runner_binding = _binding(runner)
    runner_definition = {
        "contract": "eval360.runner-definition",
        "entrypoint": {
            "path": "eval360",
            "sha256": runner_binding["sha256"],
            "size_bytes": runner_binding["size_bytes"],
        },
        "request_contract": {
            "contract": EVALUATION_REQUEST_CONTRACT,
            "schema_version": "1.0",
        },
        "runner_id": "eval360@aaa16a5",
        "schema_version": "1.0",
        "source": _source(),
        "terminal_result_contract": {
            "contract": "eval360.evaluate-now-terminal-result",
            "schema_version": "1.0",
        },
    }
    runner_definition_path = _write_canonical(
        runner.parent / "runner.json", runner_definition
    )

    request_tasks = [
        {
            "dataset_files": [_binding(dataset_paths[task_id])],
            "id": task_id,
        }
        for task_id in task_order
    ]
    request = {
        "bindings": {
            "release_manifest": _binding(release_manifest),
            "release_payloads": [_binding(release_payload)],
            "runner_definition": _binding(runner_definition_path),
            "runner_entrypoint": runner_binding,
            "serving_manifest": _binding(serving_manifest),
            "serving_payloads": [
                _binding(serving_entrypoint),
                _binding(serving_payload),
            ],
            "suite": suite_binding,
            "suite_catalog": _binding(catalog_path),
        },
        "contract": EVALUATION_REQUEST_CONTRACT,
        "execution": {
            "dataset_root": str(data_root),
            "model_family": "k2_horizon",
            "model_name": "checkpoint-5000",
            "model_revision": None,
            "output_path": str(tmp_path / "output"),
            "owner": "test-owner",
            "release_path": str(release_root),
            "serving": {
                "entry_path": "bin/activate",
                "kind": "venv",
                "root_path": str(serving_root),
            },
        },
        "schema_version": "1.0",
        "suite": {
            "id": suite["suite_id"],
            "version": suite["version"],
        },
        "tasks": request_tasks,
    }
    request_path = _write_canonical(tmp_path / "request.json", request)
    return request_path, tmp_path / result_name, runner, request


def test_request_loader_derives_exact_ordered_work_without_yaml(tmp_path: Path):
    request_path, result_path, runner, _ = build_request_fixture(tmp_path)

    request = EvaluationRequest.load(
        request_path=request_path,
        terminal_result_path=result_path,
        actual_runner_entrypoint=runner,
    )

    assert request.request_binding.configured_path == str(request_path)
    assert request.runner_definition.runner_id == "eval360@aaa16a5"
    assert request.runner_definition.source.commit == "a" * 40
    assert request.suite.suite_id == "eval360/k2-horizon-base"
    assert [task.uuid for task in request.tasks] == ["task-a", "task-b"]
    assert [task.num_generations for task in request.tasks] == [2, 1]
    assert request.dataset_paths == {
        "task-a": (str(tmp_path / "datasets" / "task-a.jsonl"),),
        "task-b": (str(tmp_path / "datasets" / "task-b.jsonl"),),
    }
    assert request.model_spec.remote_model.path == str(tmp_path / "release")
    assert request.model_spec.venv_path == str(
        tmp_path / "serving" / "bin" / "activate"
    )
    assert request.model_spec.parser_type == "noop"
    assert request.output_path == result_path


@pytest.mark.parametrize(
    ("manifest_key", "extra_key"),
    [
        ("runner_definition", "unknown_runner_field"),
        ("suite_catalog", "unknown_catalog_field"),
        ("suite", "unknown_suite_field"),
    ],
)
def test_owner_manifest_loaders_forbid_unknown_fields(
    tmp_path: Path,
    manifest_key: str,
    extra_key: str,
):
    request_path, result_path, runner, request = build_request_fixture(tmp_path)
    manifest_path = Path(request["bindings"][manifest_key]["configured_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[extra_key] = True
    _write_canonical(manifest_path, manifest)
    request["bindings"][manifest_key] = _binding(manifest_path)
    if manifest_key == "suite":
        catalog_path = Path(request["bindings"]["suite_catalog"]["configured_path"])
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog["suites"][0]["manifest_sha256"] = request["bindings"]["suite"]["sha256"]
        catalog["suites"][0]["manifest_size_bytes"] = request["bindings"]["suite"]["size_bytes"]
        _write_canonical(catalog_path, catalog)
        request["bindings"]["suite_catalog"] = _binding(catalog_path)
    _write_canonical(request_path, request)

    with pytest.raises(ValueError, match="unknown|fields"):
        EvaluationRequest.load(
            request_path=request_path,
            terminal_result_path=result_path,
            actual_runner_entrypoint=runner,
        )


def test_catalog_must_select_exact_suite_digest(tmp_path: Path):
    request_path, result_path, runner, request = build_request_fixture(tmp_path)
    catalog_path = Path(request["bindings"]["suite_catalog"]["configured_path"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["suites"][0]["manifest_sha256"] = "f" * 64
    _write_canonical(catalog_path, catalog)
    request["bindings"]["suite_catalog"] = _binding(catalog_path)
    _write_canonical(request_path, request)

    with pytest.raises(ValueError, match="catalog.*suite manifest"):
        EvaluationRequest.load(
            request_path=request_path,
            terminal_result_path=result_path,
            actual_runner_entrypoint=runner,
        )


def test_request_task_order_must_exactly_project_suite(tmp_path: Path):
    request_path, result_path, runner, request = build_request_fixture(tmp_path)
    request["tasks"].reverse()
    _write_canonical(request_path, request)

    with pytest.raises(ValueError, match="ordered task closure"):
        EvaluationRequest.load(
            request_path=request_path,
            terminal_result_path=result_path,
            actual_runner_entrypoint=runner,
        )


def test_dataset_payload_bytes_and_declared_total_are_verified(tmp_path: Path):
    request_path, result_path, runner, _ = build_request_fixture(tmp_path)
    dataset_path = tmp_path / "datasets" / "task-a.jsonl"
    dataset_path.write_text(
        dataset_path.read_text(encoding="utf-8")
        + '{"row":99,"completion_input":"stale"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset.*changed"):
        EvaluationRequest.load(
            request_path=request_path,
            terminal_result_path=result_path,
            actual_runner_entrypoint=runner,
        )


def test_actual_runner_entrypoint_must_match_definition_and_request(tmp_path: Path):
    request_path, result_path, _, _ = build_request_fixture(tmp_path)
    other = tmp_path / "other-eval360"
    other.write_text("#!/usr/bin/env false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="actual runner entrypoint"):
        EvaluationRequest.load(
            request_path=request_path,
            terminal_result_path=result_path,
            actual_runner_entrypoint=other,
        )


def test_request_and_owner_manifests_must_be_canonical_json(tmp_path: Path):
    request_path, result_path, runner, request = build_request_fixture(tmp_path)
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical JSON"):
        EvaluationRequest.load(
            request_path=request_path,
            terminal_result_path=result_path,
            actual_runner_entrypoint=runner,
        )


def test_release_and_serving_payload_drift_fails_requery(tmp_path: Path):
    request_path, result_path, runner, _ = build_request_fixture(tmp_path)
    request = EvaluationRequest.load(
        request_path=request_path,
        terminal_result_path=result_path,
        actual_runner_entrypoint=runner,
    )
    (tmp_path / "release" / "model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="release payload.*changed"):
        request.verify_inputs()


def test_post_link_directory_fsync_failure_removes_terminal_name(tmp_path: Path):
    from scheduler.terminal_result import publish_terminal_result

    request_path, result_path, runner, _ = build_request_fixture(tmp_path)
    request = EvaluationRequest.load(
        request_path=request_path,
        terminal_result_path=result_path,
        actual_runner_entrypoint=runner,
    )
    output = tmp_path / "output.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    output_binding = _binding(output)
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
    scheduler_result = {
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
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    with (
        patch("scheduler.terminal_result.os.fsync", side_effect=fail_directory_fsync),
        pytest.raises(OSError, match="directory fsync failed"),
    ):
        publish_terminal_result(request, scheduler_result)

    assert not result_path.exists()
