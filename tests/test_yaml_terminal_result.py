"""Write-last terminal evidence for the ordinary eval-config YAML path."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scheduler.terminal_result import (
    YamlTerminalInvocation,
    bind_regular_file,
    canonical_json,
    publish_yaml_terminal_result,
)
from tests.fake_slurm import FakeSlurmManager
from tests.test_scheduler import _FakeOpenAIConnection, _scheduler


@dataclass(frozen=True)
class _YamlFixture:
    runner: Path
    model_paths: tuple[Path, ...]
    data_paths: tuple[Path, ...]
    eval_paths: tuple[Path, ...]
    datasets: tuple[Path, ...]
    result_path: Path


def _binding(path: Path) -> dict[str, str | int]:
    return bind_regular_file(path, label=path.name).as_dict()


def _write_yaml(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _write_dataset(path: Path, *, row: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "row": row,
                "completion_input": f"Question {row}?",
                "chat_input": [{"role": "user", "content": f"Question {row}?"}],
                "ground_truth": "A",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _build_yaml_fixture(tmp_path: Path) -> _YamlFixture:
    runner = tmp_path / "runner" / "eval360"
    runner.parent.mkdir()
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    runner.chmod(0o755)

    model_path = _write_yaml(
        tmp_path / "configs" / "model.yaml",
        {
            "remote_model": {
                "base_name": "checkpoint-5000",
                "path": "org/checkpoint-5000",
                "revision": "revision-a",
            },
            "model_type": "base",
            "parser_type": "noop",
            "venv_path": "/fake/bin/activate",
            "max_simultaneous_requests": 4,
            "max_time_to_deploy": 600,
            "vllm_cli_args": [],
            "openai_kwargs": {"temperature": 0.0},
            "owner": "model-owner",
            "ready": True,
            "output_path": str(tmp_path / "unused-model-output"),
            "tag": "k2_horizon",
        },
    )

    alpha_dataset = _write_dataset(
        tmp_path / "datasets" / "alpha.jsonl",
        row=0,
    )
    beta_dataset = _write_dataset(
        tmp_path / "datasets" / "beta.jsonl",
        row=1,
    )
    alpha_config = _write_yaml(
        tmp_path / "configs" / "alpha.yaml",
        {
            "uuid": "task-alpha",
            "grader": {"type": "exact_match"},
            "average_over": [1],
            "pass_at": [1],
            "dataset_name": "alpha",
            "data_path": str(alpha_dataset),
            "semantic_version": "1.0.0",
            "num_generations": 1,
            "meta": {},
            "tag": "full-suite",
        },
    )
    beta_config = _write_yaml(
        tmp_path / "configs" / "beta.yaml",
        {
            "uuid": "task-beta",
            "grader": {"type": "exact_match"},
            "average_over": [1],
            "pass_at": [1],
            "dataset_name": "beta",
            "data_path": str(beta_dataset),
            "semantic_version": "1.0.0",
            "num_generations": 1,
            "meta": {},
            "tag": "full-suite",
        },
    )
    eval_path = _write_yaml(
        tmp_path / "configs" / "eval.yaml",
        {
            "version": 1,
            "owner": "eval-owner",
            "groups": [
                {
                    "name": "k2-horizon-full",
                    "model_tag": "k2_horizon",
                    "data_tag": "full-suite",
                    "parser_type": "noop",
                    "output_root": str(tmp_path / "outputs"),
                }
            ],
        },
    )

    # Deliberately request beta before alpha. Evidence must retain invocation
    # order instead of sorting paths, task IDs, or resolved events.
    return _YamlFixture(
        runner=runner,
        model_paths=(model_path,),
        data_paths=(beta_config, alpha_config),
        eval_paths=(eval_path,),
        datasets=(beta_dataset, alpha_dataset),
        result_path=tmp_path / "terminal-result.json",
    )


def _bind_invocation(fixture: _YamlFixture) -> YamlTerminalInvocation:
    return YamlTerminalInvocation.bind(
        model_paths=[str(path) for path in fixture.model_paths],
        data_paths=[str(path) for path in fixture.data_paths],
        eval_paths=[str(path) for path in fixture.eval_paths],
        terminal_result_path=fixture.result_path,
        actual_runner_entrypoint=fixture.runner,
    )


def test_yaml_invocation_rejects_different_parser_order(tmp_path: Path):
    fixture = _build_yaml_fixture(tmp_path)
    invocation = _bind_invocation(fixture)

    with pytest.raises(ValueError, match="data parser paths differ"):
        invocation.verify_config_paths(
            model_paths=[str(path) for path in fixture.model_paths],
            data_paths=[str(path) for path in reversed(fixture.data_paths)],
            eval_paths=[str(path) for path in fixture.eval_paths],
        )


async def _run_yaml(
    tmp_path: Path,
    fixture: _YamlFixture,
    invocation: YamlTerminalInvocation,
    slurm: FakeSlurmManager | None = None,
    connection: _FakeOpenAIConnection | None = None,
    force: bool = True,
):
    scheduler = _scheduler(tmp_path)
    scheduler.slurm_manager = slurm or FakeSlurmManager()
    with patch(
        "scheduler.openai_interface.OpenAIConnection",
        return_value=connection or _FakeOpenAIConnection(canned_answers=["A"]),
    ):
        scheduler_result = await scheduler.run_evaluate_now(
            [str(path) for path in fixture.model_paths],
            [str(path) for path in fixture.data_paths],
            force=force,
            eval_paths=[str(path) for path in fixture.eval_paths],
            yaml_terminal_invocation=invocation,
        )
    return scheduler_result


@pytest.mark.asyncio
async def test_eval_config_yaml_result_binds_ordered_inputs_outputs_and_children(
    tmp_path: Path,
):
    fixture = _build_yaml_fixture(tmp_path)
    invocation = _bind_invocation(fixture)

    scheduler_result = await _run_yaml(tmp_path, fixture, invocation)
    result = publish_yaml_terminal_result(invocation, scheduler_result)

    assert fixture.result_path.read_text(encoding="utf-8") == canonical_json(result)
    assert result["status"] == "succeeded"
    assert result["controller"] == {"exit_code": 0, "outcome": "succeeded"}
    assert result["bindings"] == {
        "runner_entrypoint": _binding(fixture.runner),
        "model_configs": [_binding(path) for path in fixture.model_paths],
        "data_configs": [_binding(path) for path in fixture.data_paths],
        "eval_configs": [_binding(path) for path in fixture.eval_paths],
    }

    selection = result["selection"]
    assert selection["mode"] == "eval_configs"
    events = selection["events"]
    assert [event["ordinal"] for event in events] == [0, 1]
    assert [event["event_type"] for event in events] == [
        "generation_grading",
        "generation_grading",
    ]
    assert [event["task"]["dataset_name"] for event in events] == [
        "beta",
        "alpha",
    ]
    assert [event["terminal_phase"] for event in events] == [2, 2]

    for ordinal, event in enumerate(events):
        assert event["model"]["name"].startswith(
            "checkpoint-5000-revision-a-eval-"
        )
        assert event["model"]["parser_type"] == "noop"
        assert len(event["model"]["definition_sha256"]) == 64
        assert event["task"]["id"].startswith(
            f"task-{event['task']['dataset_name']}-eval-"
        )
        assert event["task"]["mode"] == "base"
        assert event["task"]["semantic_version"] == "1.0.0"
        assert event["task"]["grader_type"] == "exact_match"
        assert len(event["task"]["definition_sha256"]) == 64
        assert event["source"] == {
            "model_config": _binding(fixture.model_paths[0]),
            "data_config": _binding(fixture.data_paths[ordinal]),
            "eval_config": _binding(fixture.eval_paths[0]),
            "eval_group": "k2-horizon-full",
        }
        assert event["inputs"] == {
            "dataset_files": [_binding(fixture.datasets[ordinal])]
        }
        assert [output["role"] for output in event["outputs"]] == [
            "generations",
            "grades",
            "scores",
            "run_metadata",
        ]
        for output in event["outputs"]:
            assert _binding(Path(output["configured_path"])) == {
                key: output[key]
                for key in (
                    "configured_path",
                    "resolved_path",
                    "sha256",
                    "size_bytes",
                )
            }
        assert [unit["role"] for unit in event["controller_units"]] == [
            "generation",
            "grading",
            "aggregation",
        ]

    assert result["jobs"]
    completed_roles = set()
    for job in result["jobs"]:
        assert job["scheduler"]["source"] == "sacct"
        assert job["scheduler"]["state"] == "CANCELLED"
        assert job["cancellation_intent"] == "scheduler_release"
        assert job["outcome"] == "succeeded"
        completed_roles.update(
            (completion["event_id"], completion["role"])
            for completion in job["role_completions"]
        )
    assert completed_roles == {
        (event["event_id"], "generation") for event in events
    }
    for event in events:
        [generation] = [
            unit for unit in event["controller_units"] if unit["role"] == "generation"
        ]
        assert generation["job_ids"]
        assert all(
            job_id in {job["job_id"] for job in result["jobs"]}
            for job_id in generation["job_ids"]
        )

    with pytest.raises(FileExistsError):
        publish_yaml_terminal_result(invocation, scheduler_result)


@pytest.mark.asyncio
async def test_failed_sacct_child_prevents_yaml_terminal_result(
    tmp_path: Path,
):
    fixture = _build_yaml_fixture(tmp_path)
    invocation = _bind_invocation(fixture)

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
        await _run_yaml(
            tmp_path,
            fixture,
            invocation,
            slurm=FailedAccounting(),
        )
    assert not fixture.result_path.exists()


@pytest.mark.asyncio
async def test_result_cannot_relabel_resolved_yaml_event(tmp_path: Path):
    fixture = _build_yaml_fixture(tmp_path)
    invocation = _bind_invocation(fixture)
    scheduler_result = await _run_yaml(tmp_path, fixture, invocation)

    relabelled = deepcopy(scheduler_result)
    relabelled["selection"]["events"][0]["source"]["eval_group"] = (
        "invented-group"
    )
    with pytest.raises(ValueError, match="differs from its bound run metadata"):
        publish_yaml_terminal_result(invocation, relabelled)

    malformed = deepcopy(scheduler_result)
    event = malformed["selection"]["events"][0]
    event["model"] = {"definition_sha256": "0" * 64}
    event["task"] = {"definition_sha256": "1" * 64}

    with pytest.raises(ValueError, match="identity is invalid"):
        publish_yaml_terminal_result(invocation, malformed)

    false_accounting = deepcopy(scheduler_result)
    false_accounting["jobs"][0]["scheduler"]["job_id_raw"] = "999999"
    false_accounting["jobs"][0]["scheduler"]["raw_state"] = "NODE_FAIL"
    with pytest.raises(ValueError, match="inconsistent sacct evidence"):
        publish_yaml_terminal_result(invocation, false_accounting)
    assert not fixture.result_path.exists()


@pytest.mark.asyncio
async def test_dataset_glob_is_frozen_before_generation(tmp_path: Path):
    fixture = _build_yaml_fixture(tmp_path)
    shard_dir = tmp_path / "datasets" / "beta-shards"
    initial_shard = _write_dataset(shard_dir / "initial.jsonl", row=1)
    beta_config = yaml.safe_load(fixture.data_paths[0].read_text(encoding="utf-8"))
    beta_config["data_path"] = str(shard_dir / "*.jsonl")
    _write_yaml(fixture.data_paths[0], beta_config)
    fixture = replace(
        fixture,
        datasets=(initial_shard, fixture.datasets[1]),
    )
    invocation = _bind_invocation(fixture)
    late_shard = shard_dir / "late.jsonl"

    class AddsLateShard(_FakeOpenAIConnection):
        async def launch_requests(self, requests, offset, completion_hook):
            if not late_shard.exists():
                _write_dataset(late_shard, row=99)
            async for item in super().launch_requests(
                requests,
                offset,
                completion_hook,
            ):
                yield item

    scheduler_result = await _run_yaml(
        tmp_path,
        fixture,
        invocation,
        connection=AddsLateShard(canned_answers=["A"]),
    )
    result = publish_yaml_terminal_result(invocation, scheduler_result)

    beta_event = result["selection"]["events"][0]
    assert beta_event["inputs"]["dataset_files"] == [_binding(initial_shard)]
    generations = Path(beta_event["outputs"][0]["configured_path"])
    assert len(generations.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.asyncio
async def test_same_yaml_outputs_resume_with_actual_child_evidence(tmp_path: Path):
    fixture = _build_yaml_fixture(tmp_path)
    first_invocation = _bind_invocation(fixture)
    first_result = await _run_yaml(tmp_path, fixture, first_invocation)
    publish_yaml_terminal_result(first_invocation, first_result)

    resumed_fixture = replace(
        fixture,
        result_path=tmp_path / "resumed-terminal-result.json",
    )
    resumed_invocation = _bind_invocation(resumed_fixture)
    resumed_result = await _run_yaml(
        tmp_path,
        resumed_fixture,
        resumed_invocation,
        force=False,
    )
    published = publish_yaml_terminal_result(
        resumed_invocation,
        resumed_result,
    )

    assert published["status"] == "succeeded"
    assert resumed_fixture.result_path.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutated_binding", ["model_config", "dataset", "output"])
async def test_changed_bound_bytes_prevent_yaml_terminal_result(
    tmp_path: Path,
    mutated_binding: str,
):
    fixture = _build_yaml_fixture(tmp_path)
    invocation = _bind_invocation(fixture)
    scheduler_result = await _run_yaml(tmp_path, fixture, invocation)

    if mutated_binding == "model_config":
        target = fixture.model_paths[0]
    elif mutated_binding == "dataset":
        target = fixture.datasets[0]
    else:
        target = Path(
            scheduler_result["selection"]["events"][0]["outputs"][0][
                "configured_path"
            ]
        )
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="changed"):
        publish_yaml_terminal_result(invocation, scheduler_result)
    assert not fixture.result_path.exists()
