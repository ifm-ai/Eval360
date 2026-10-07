import base64
import json
from unittest.mock import patch

import pydantic
import pytest
import yaml

from scheduler.eval_config import (
    EvalConfig,
    EvalConfigParser,
    LoadedEvalConfig,
    LoadedModelSpec,
    LoadedTask,
)
from scheduler.model import ModelParser, ModelSpec
from scheduler.slurm_manager import SlurmManager
from scheduler.task import AsyncGenerationTask, GraderConfig


def _remote_spec(*, base_name: str, tag: str, openai_kwargs=None, vllm_cli_args=None):
    return ModelSpec.model_validate({
        "remote_model": {"path": f"org/{base_name}", "base_name": base_name, "revision": None},
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "vllm_cli_args": vllm_cli_args or [],
        "openai_kwargs": openai_kwargs or {},
        "model_type": "instruct",
        "owner": "base-owner",
        "ready": True,
        "output_path": "/base-output",
        "parser_type": "noop",
        "tag": tag,
    })


def _external_spec(*, base_name: str, tag: str):
    return ModelSpec.model_validate({
        "external_model": {
            "base_name": base_name,
            "base_url": "https://api.example.com/v1",
            "api_key_env": "FAKE_API_KEY_ENV",
        },
        "max_simultaneous_requests": 4,
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "base-owner",
        "ready": True,
        "output_path": "/base-output",
        "parser_type": "noop",
        "tag": tag,
    })


def _local_spec(*, path_glob: str, tag: str):
    return ModelSpec.model_validate({
        "local_model": {
            "path_glob": path_glob,
            "model_family_name": "mid2",
            "version_level": -1,
            "enqueue_existing": True,
        },
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "base-owner",
        "ready": True,
        "output_path": "/base-output",
        "parser_type": "noop",
        "tag": tag,
    })


def _task(*, uuid: str, dataset_name: str, tag: str, openai_settings=None, grader_type="multiple_choice"):
    return AsyncGenerationTask(
        uuid=uuid,
        average_over=[1],
        pass_at=[1],
        openai_settings=openai_settings,
        mode="base",
        dataset_name=dataset_name,
        data_path="/tmp/data.jsonl",
        semantic_version="1.0.0",
        meta={},
        tag=tag,
        num_generations=1,
        grader=GraderConfig(type=grader_type),
    )


def _eval_config(*, owner="eval-owner", groups):
    return EvalConfig.model_validate({
        "version": 1,
        "owner": owner,
        "groups": groups,
    })


def test_parse_yaml_rejects_empty_eval_config(tmp_path):
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text("# intentionally empty\n")

    with pytest.raises(ValueError, match="Empty eval config"):
        EvalConfigParser.parse_yaml(str(eval_yaml))


def test_parse_yaml_rejects_non_mapping_eval_config(tmp_path):
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="must contain a mapping"):
        EvalConfigParser.parse_yaml(str(eval_yaml))


def test_eval_config_rejects_unsupported_version():
    with pytest.raises(pydantic.ValidationError, match="Unsupported eval config version"):
        EvalConfig.model_validate({
            "version": 2,
            "owner": "eval-owner",
            "groups": [{
                "name": "g1",
                "model_tag": "reasoning",
                "data_tag": "aime",
                "parser_type": "boxed",
                "output_root": "/out",
            }],
        })


@pytest.mark.parametrize("output_root", ["", "   ", "relative/out", "./out"])
def test_eval_group_rejects_invalid_output_root(output_root):
    with pytest.raises(pydantic.ValidationError, match="output_root"):
        _eval_config(groups=[{
            "name": "g1",
            "model_tag": "reasoning",
            "data_tag": "aime",
            "parser_type": "boxed",
            "output_root": output_root,
        }])


def test_eval_group_accepts_absolute_output_root():
    config = _eval_config(groups=[{
        "name": "g1",
        "model_tag": "reasoning",
        "data_tag": "aime",
        "parser_type": "boxed",
        "output_root": "/out",
    }])

    assert config.groups[0].output_root == "/out"


def test_eval_group_expands_home_output_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = _eval_config(groups=[{
        "name": "g1",
        "model_tag": "reasoning",
        "data_tag": "aime",
        "parser_type": "boxed",
        "output_root": "~/eval-results",
    }])

    assert config.groups[0].output_root == str(tmp_path / "eval-results")


def test_pair_token_is_stable_and_distinguishes_inputs():
    args = {
        "eval_config_path": "/evals/reasoning.yaml",
        "group_name": "reasoning-aime",
        "model_path": "/models/reasoner.yaml",
        "task_path": "/data/aime.yaml",
    }

    token = EvalConfigParser._pair_token(**args)

    assert EvalConfigParser._pair_token(**args) == token
    assert len(token) == 12
    assert token == token.lower()
    int(token, 16)

    for key, value in {
        "eval_config_path": "/evals/other.yaml",
        "group_name": "reasoning-mmlu",
        "model_path": "/models/other.yaml",
        "task_path": "/data/mmlu.yaml",
    }.items():
        changed = dict(args)
        changed[key] = value
        assert EvalConfigParser._pair_token(**changed) != token


def test_build_resolved_pairs_applies_eval_overrides():
    loaded_models = [
        LoadedModelSpec(
            path="/configs/model.yaml",
            spec=_remote_spec(
                base_name="reasoner",
                tag="reasoning",
                openai_kwargs={
                    "max_tokens": 256,
                    "temperature": 0.2,
                    "extra_body": {"model_only": "m", "shared": "model"},
                },
                vllm_cli_args=["--tensor-parallel-size", "1"],
            ),
        )
    ]
    loaded_tasks = [
        LoadedTask(
            path="/configs/data.yaml",
            task=_task(
                uuid="task-aime",
                dataset_name="aime",
                tag="aime",
                openai_settings={
                    "max_tokens": 128,
                    "top_p": 0.9,
                    "extra_body": {"task_only": "t", "shared": "task"},
                },
                grader_type="exact_match",
            ),
        )
    ]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/configs/eval.yaml",
            config=_eval_config(groups=[{
                "name": "reasoning-aime",
                "model_tag": "reasoning",
                "data_tag": "aime",
                "parser_type": "boxed",
                "output_root": "/eval-output",
                "grader": {"type": "multiple_choice"},
                "openai_overrides": {
                    "temperature": 0.7,
                    "extra_body": {"eval_only": "e", "shared": "eval"},
                },
                "vllm_cli_args": ["--tensor-parallel-size", "8"],
            }]),
        )
    ]

    [resolved] = EvalConfigParser.build_resolved_pairs(
        loaded_models=loaded_models,
        loaded_tasks=loaded_tasks,
        loaded_eval_configs=loaded_eval_configs,
    )

    assert resolved.model.name.startswith("reasoner-eval-")
    assert resolved.model.api_model_name == "reasoner"
    # output_path is keyed by the per-pair token (the same suffix used in the
    # model variant name) so pairs sharing a dataset_name cannot collide on disk.
    pair_token = resolved.model.name.removeprefix("reasoner-eval-")
    assert resolved.model.output_path == f"/eval-output/reasoner/{pair_token}"
    assert resolved.model.parser_type == "boxed"
    assert resolved.model.owner == "eval-owner"
    assert resolved.model.vllm_cli_args == ["--tensor-parallel-size", "8"]
    assert resolved.model.openai_kwargs == {}

    assert resolved.task.uuid.startswith("task-aime-eval-")
    assert resolved.task.grader.type == "multiple_choice"
    assert resolved.task.tag == resolved.model.tag
    assert resolved.task.openai_settings == {
        "max_tokens": 256,
        "top_p": 0.9,
        "temperature": 0.7,
        "extra_body": {
            "task_only": "t",
            "model_only": "m",
            "eval_only": "e",
            "shared": "eval",
        },
    }


@pytest.mark.asyncio
async def test_first_allocation_advertises_later_eval_pair_request_name(tmp_path):
    loaded_models = [
        LoadedModelSpec(
            path="/configs/model.yaml",
            spec=_remote_spec(base_name="reasoner", tag="reasoning"),
        )
    ]
    loaded_tasks = [
        LoadedTask(
            path=f"/configs/{dataset_name}.yaml",
            task=_task(
                uuid=f"task-{dataset_name}",
                dataset_name=dataset_name,
                tag="reasoning-task",
            ),
        )
        for dataset_name in ("first", "later")
    ]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/configs/eval.yaml",
            config=_eval_config(
                groups=[
                    {
                        "name": "reasoning-suite",
                        "model_tag": "reasoning",
                        "data_tag": "reasoning-task",
                        "parser_type": "boxed",
                        "output_root": "/eval-output",
                    }
                ]
            ),
        )
    ]
    first, later = EvalConfigParser.build_resolved_pairs(
        loaded_models=loaded_models,
        loaded_tasks=loaded_tasks,
        loaded_eval_configs=loaded_eval_configs,
    )
    assert first.model.serving_key == later.model.serving_key

    class FakeProcess:
        returncode = 0

        def __init__(self, stdout: str):
            self.stdout = stdout.encode()

        async def communicate(self):
            return self.stdout, b""

    commands = []

    async def fake_exec(*args, **kwargs):
        commands.append(args)
        if args[0] == "squeue":
            return FakeProcess("")
        return FakeProcess("Submitted batch job 1")

    manager = SlurmManager(
        log_dir=str(tmp_path),
        instance_id="deadbeef",
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await manager.update_allocation(
            [(first.model, 1)],
            unneeded_models=[],
            sibling_names_by_sk={
                first.model.serving_key: [first.model.name]
            },
        )

    sbatch_command = next(command for command in commands if command[0] == "sbatch")
    export_arg = next(
        argument
        for argument in sbatch_command
        if argument.startswith("--export=")
    )
    encoded_args = next(
        field.split("=", 1)[1]
        for field in export_arg.split(",")
        if field.startswith("vllm_args=")
    )
    vllm_args = json.loads(base64.b64decode(encoded_args).decode())
    served_name_index = vllm_args.index("--served-model-name")
    advertised_names = vllm_args[served_name_index + 1 :]

    assert first.model.api_model_name == "reasoner"
    assert later.model.api_model_name == "reasoner"
    assert later.model.api_model_name in advertised_names


def test_build_resolved_pairs_isolate_output_paths_for_shared_dataset_name():
    # Regression for M2: two *different* data configs that happen to share the
    # same dataset_name (e.g. data_zoo/mmlu.yaml and examples/data/mmlu.yaml,
    # both dataset_name "mmlu") must resolve to DIFFERENT output directories.
    # Output files in event.py are named "<dataset_name>_generations.jsonl" /
    # "_grades.jsonl" / "_scores.yaml" under model.output_path, so if two pairs
    # shared an output_path the second pair would treat the first pair's rows as
    # already-completed on resume and mix incompatible generations/grades.
    loaded_models = [LoadedModelSpec(path="/m.yaml", spec=_remote_spec(base_name="m1", tag="reasoning"))]
    loaded_tasks = [
        LoadedTask(path="/data_zoo/mmlu.yaml", task=_task(uuid="task-a", dataset_name="mmlu", tag="mmlu")),
        LoadedTask(path="/examples/data/mmlu.yaml", task=_task(uuid="task-b", dataset_name="mmlu", tag="mmlu")),
    ]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[{
                "name": "g1",
                "model_tag": "reasoning",
                "data_tag": "mmlu",
                "parser_type": "boxed",
                "output_root": "/out",
            }]),
        )
    ]

    resolved = EvalConfigParser.build_resolved_pairs(
        loaded_models=loaded_models,
        loaded_tasks=loaded_tasks,
        loaded_eval_configs=loaded_eval_configs,
    )

    assert len(resolved) == 2
    # The dataset_name (which drives the output file names) is intentionally
    # identical for both pairs; isolation must therefore come from output_path.
    assert {r.task.dataset_name for r in resolved} == {"mmlu"}

    output_paths = [r.model.output_path for r in resolved]
    assert len(set(output_paths)) == 2, output_paths

    # Mirror event.py's filename contract and assert the concrete files differ.
    def _files(pair):
        base = pair.model.output_path
        name = pair.task.dataset_name
        return {
            f"{base}/{name}_generations.jsonl",
            f"{base}/{name}_grades.jsonl",
            f"{base}/{name}_scores.yaml",
        }

    assert _files(resolved[0]).isdisjoint(_files(resolved[1]))


def test_build_resolved_pairs_detects_overlap():
    loaded_models = [LoadedModelSpec(path="/m.yaml", spec=_remote_spec(base_name="m1", tag="reasoning"))]
    loaded_tasks = [LoadedTask(path="/d.yaml", task=_task(uuid="task-1", dataset_name="aime", tag="aime"))]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[
                {
                    "name": "g1",
                    "model_tag": "reasoning",
                    "data_tag": "aime",
                    "parser_type": "boxed",
                    "output_root": "/out1",
                },
                {
                    "name": "g2",
                    "model_tag": "reasoning",
                    "data_tag": "aime",
                    "parser_type": "noop",
                    "output_root": "/out2",
                },
            ]),
        )
    ]

    with pytest.raises(ValueError, match="overlap"):
        EvalConfigParser.build_resolved_pairs(
            loaded_models=loaded_models,
            loaded_tasks=loaded_tasks,
            loaded_eval_configs=loaded_eval_configs,
        )


def test_build_resolved_pairs_treats_any_as_literal():
    loaded_models = [
        LoadedModelSpec(path="/m-any.yaml", spec=_remote_spec(base_name="m-any", tag="any")),
        LoadedModelSpec(path="/m-vision.yaml", spec=_remote_spec(base_name="m-vision", tag="vision")),
    ]
    loaded_tasks = [LoadedTask(path="/d.yaml", task=_task(uuid="task-1", dataset_name="bench", tag="bench"))]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[{
                "name": "g1",
                "model_tag": "any",
                "data_tag": "bench",
                "parser_type": "boxed",
                "output_root": "/out",
            }]),
        )
    ]

    resolved = EvalConfigParser.build_resolved_pairs(
        loaded_models=loaded_models,
        loaded_tasks=loaded_tasks,
        loaded_eval_configs=loaded_eval_configs,
    )

    assert len(resolved) == 1
    assert resolved[0].model.path == "org/m-any"


def test_build_resolved_pairs_fails_when_group_matches_zero_models():
    loaded_models = [LoadedModelSpec(path="/m.yaml", spec=_remote_spec(base_name="m1", tag="reasoning"))]
    loaded_tasks = [LoadedTask(path="/d.yaml", task=_task(uuid="task-1", dataset_name="aime", tag="aime"))]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[{
                "name": "g1",
                "model_tag": "missing",
                "data_tag": "aime",
                "parser_type": "boxed",
                "output_root": "/out",
            }]),
        )
    ]

    with pytest.raises(ValueError, match="matched zero models"):
        EvalConfigParser.build_resolved_pairs(
            loaded_models=loaded_models,
            loaded_tasks=loaded_tasks,
            loaded_eval_configs=loaded_eval_configs,
        )


def test_build_resolved_pairs_fails_when_group_matches_zero_data_configs():
    loaded_models = [LoadedModelSpec(path="/m.yaml", spec=_remote_spec(base_name="m1", tag="reasoning"))]
    loaded_tasks = [LoadedTask(path="/d.yaml", task=_task(uuid="task-1", dataset_name="bench", tag="other"))]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[{
                "name": "g1",
                "model_tag": "reasoning",
                "data_tag": "aime",
                "parser_type": "boxed",
                "output_root": "/out",
            }]),
        )
    ]

    with pytest.raises(ValueError, match="matched zero data configs"):
        EvalConfigParser.build_resolved_pairs(
            loaded_models=loaded_models,
            loaded_tasks=loaded_tasks,
            loaded_eval_configs=loaded_eval_configs,
        )


def test_external_model_cannot_take_eval_vllm_override():
    loaded_models = [LoadedModelSpec(path="/m.yaml", spec=_external_spec(base_name="ext-model", tag="reasoning"))]
    loaded_tasks = [LoadedTask(path="/d.yaml", task=_task(uuid="task-1", dataset_name="aime", tag="aime"))]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[{
                "name": "g1",
                "model_tag": "reasoning",
                "data_tag": "aime",
                "parser_type": "boxed",
                "output_root": "/out",
                "vllm_cli_args": ["--tensor-parallel-size", "8"],
            }]),
        )
    ]

    with pytest.raises(ValueError, match="external_model"):
        EvalConfigParser.build_resolved_pairs(
            loaded_models=loaded_models,
            loaded_tasks=loaded_tasks,
            loaded_eval_configs=loaded_eval_configs,
        )


def test_different_eval_vllm_args_produce_distinct_deployment_variants():
    loaded_models = [LoadedModelSpec(path="/m.yaml", spec=_remote_spec(base_name="reasoner", tag="reasoning"))]
    loaded_tasks = [
        LoadedTask(path="/aime.yaml", task=_task(uuid="task-aime", dataset_name="aime", tag="aime")),
        LoadedTask(path="/mmlu.yaml", task=_task(uuid="task-mmlu", dataset_name="mmlu", tag="mmlu")),
    ]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[
                {
                    "name": "g1",
                    "model_tag": "reasoning",
                    "data_tag": "aime",
                    "parser_type": "boxed",
                    "output_root": "/out",
                    "vllm_cli_args": ["--tensor-parallel-size", "4"],
                },
                {
                    "name": "g2",
                    "model_tag": "reasoning",
                    "data_tag": "mmlu",
                    "parser_type": "boxed",
                    "output_root": "/out",
                    "vllm_cli_args": ["--tensor-parallel-size", "8"],
                },
            ]),
        )
    ]

    resolved = EvalConfigParser.build_resolved_pairs(
        loaded_models=loaded_models,
        loaded_tasks=loaded_tasks,
        loaded_eval_configs=loaded_eval_configs,
    )

    assert len(resolved) == 2
    assert resolved[0].model.serving_key != resolved[1].model.serving_key


def test_local_model_eval_group_expands_all_matching_checkpoints(tmp_path):
    checkpoint_a = tmp_path / "checkpoint_0001"
    checkpoint_b = tmp_path / "checkpoint_0002"
    checkpoint_a.mkdir()
    checkpoint_b.mkdir()

    loaded_models = [
        LoadedModelSpec(
            path="/m.yaml",
            spec=_local_spec(path_glob=str(tmp_path / "checkpoint_*"), tag="reasoning"),
        )
    ]
    loaded_tasks = [LoadedTask(path="/d.yaml", task=_task(uuid="task-1", dataset_name="aime", tag="aime"))]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[{
                "name": "g1",
                "model_tag": "reasoning",
                "data_tag": "aime",
                "parser_type": "boxed",
                "output_root": "/out",
            }]),
        )
    ]

    resolved = EvalConfigParser.build_resolved_pairs(
        loaded_models=loaded_models,
        loaded_tasks=loaded_tasks,
        loaded_eval_configs=loaded_eval_configs,
    )

    assert len(resolved) == 2
    assert {pair.model.path for pair in resolved} == {str(checkpoint_a), str(checkpoint_b)}


def test_local_model_eval_group_fails_when_glob_matches_zero_paths(tmp_path):
    loaded_models = [
        LoadedModelSpec(
            path="/m.yaml",
            spec=_local_spec(path_glob=str(tmp_path / "missing_*"), tag="reasoning"),
        )
    ]
    loaded_tasks = [LoadedTask(path="/d.yaml", task=_task(uuid="task-1", dataset_name="aime", tag="aime"))]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(groups=[{
                "name": "g1",
                "model_tag": "reasoning",
                "data_tag": "aime",
                "parser_type": "boxed",
                "output_root": "/out",
            }]),
        )
    ]

    with pytest.raises(ValueError, match="matched zero paths"):
        EvalConfigParser.build_resolved_pairs(
            loaded_models=loaded_models,
            loaded_tasks=loaded_tasks,
            loaded_eval_configs=loaded_eval_configs,
        )


# Model YAML matching the eval-config docs: owner/output_path/parser_type are
# OMITTED because the eval config is responsible for supplying them. The docs
# (README "Adding a model", IMPLEMENTATION.md "Runtime fields") promise this is
# allowed in eval-config mode.
_EVAL_MODE_MODEL_YAML = {
    "remote_model": {"path": "org/reasoner", "base_name": "reasoner", "revision": None},
    "venv_path": "/venv",
    "max_simultaneous_requests": 4,
    "vllm_cli_args": [],
    "openai_kwargs": {},
    "model_type": "instruct",
    "ready": True,
    "tag": "reasoning",
}


def test_eval_mode_model_yaml_may_omit_runtime_fields(tmp_path):
    """A model YAML that omits owner/output_path/parser_type must parse in eval
    mode and have those fields filled from the eval config / group, per the docs.

    WHY: ModelSpec requires owner/output_path/parser_type, but in eval-config
    mode the eval config supplies owner, the group supplies output_root and
    parser_type. If eval-mode parsing ever reverts to strict ModelSpec
    validation, a user following the eval-config docs hits a Pydantic
    ValidationError at startup before _build_resolved_pair can fill the fields.
    """
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump(_EVAL_MODE_MODEL_YAML))

    # Must not raise (this is the regression: strict parsing raised here).
    spec = ModelParser.parse_yaml(str(model_yaml), eval_mode=True)

    loaded_models = [LoadedModelSpec(path=str(model_yaml), spec=spec)]
    loaded_tasks = [LoadedTask(path="/d.yaml", task=_task(uuid="task-1", dataset_name="aime", tag="aime"))]
    loaded_eval_configs = [
        LoadedEvalConfig(
            path="/eval.yaml",
            config=_eval_config(owner="eval-owner", groups=[{
                "name": "g1",
                "model_tag": "reasoning",
                "data_tag": "aime",
                "parser_type": "boxed",
                "output_root": "/eval-output",
            }]),
        )
    ]

    [resolved] = EvalConfigParser.build_resolved_pairs(
        loaded_models=loaded_models,
        loaded_tasks=loaded_tasks,
        loaded_eval_configs=loaded_eval_configs,
    )

    # The eval config / group, not the placeholders, must drive the runtime fields.
    assert resolved.model.owner == "eval-owner"
    pair_token = resolved.model.name.removeprefix("reasoner-eval-")
    assert resolved.model.output_path == f"/eval-output/reasoner/{pair_token}"
    assert resolved.model.parser_type == "boxed"
    # Placeholders must never leak into the resolved model.
    assert "__eval_config_placeholder__" not in resolved.model.owner
    assert "__eval_config_placeholder__" not in resolved.model.output_path


def test_direct_mode_model_yaml_still_requires_runtime_fields(tmp_path):
    """Without eval_mode, omitting owner/output_path/parser_type must still raise.

    WHY: the eval-mode relaxation must not leak into direct/long-running
    registration, where the model YAML is the only source of these fields.
    """
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump(_EVAL_MODE_MODEL_YAML))

    with pytest.raises(pydantic.ValidationError) as exc_info:
        ModelParser.parse_yaml(str(model_yaml))

    missing = {err["loc"][0] for err in exc_info.value.errors() if err["type"] == "missing"}
    assert {"owner", "output_path", "parser_type"} <= missing
