"""
Tests for ModelParser.model_instance_from_path, focusing on output_path construction.
"""
import os
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError
from unittest.mock import patch

from scheduler.model import (
    ExternalModel,
    ModelParser,
    ModelSpec,
    ServingSlurmResources,
)


def _remote_spec(base_name, output_path, revision=None, name_modifier=None, tag="any"):
    return ModelSpec.model_validate({
        "remote_model": {"path": "some/path", "base_name": base_name, "revision": revision},
        "name_modifier": name_modifier,
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": output_path,
        "parser_type": "noop",
        "tag": tag,
    })


def test_parse_yaml_rejects_empty_eval_mode_model_config(tmp_path):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text("# intentionally empty\n")

    with pytest.raises(ValueError, match="Empty model config"):
        ModelParser.parse_yaml(str(model_yaml), eval_mode=True)


def test_parse_yaml_rejects_non_mapping_eval_mode_model_config(tmp_path):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="must contain a mapping"):
        ModelParser.parse_yaml(str(model_yaml), eval_mode=True)


class TestServingSlurmResources:
    def test_exact_resources_propagate_to_instance_and_serving_key(self):
        spec = _remote_spec(base_name="k2", output_path="/results")
        baseline = ModelParser.model_instance_from_path("some/path", spec)
        resources = ServingSlurmResources(
            # Obviously arbitrary placeholders, distinct from the 12345 defaults.
            gpus_per_node=23456,
            cpus_per_task=34567,
            memory_gb=45678,
            time_limit="12:34:56",
        )
        configured = ModelParser.model_instance_from_path(
            "some/path",
            spec.model_copy(update={"serving_slurm_resources": resources}),
        )

        assert configured.serving_slurm_resources == resources
        assert configured.serving_key != baseline.serving_key

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("gpus_per_node", 0),
            ("cpus_per_task", "8"),
            ("memory_gb", -1),
            ("time_limit", "one-day"),
            ("time_limit", "00:00"),
        ],
    )
    def test_invalid_resource_value_is_rejected(self, field, value):
        with pytest.raises(ValidationError):
            ServingSlurmResources.model_validate({field: value})

    def test_code_defaults_are_obvious_placeholders(self):
        # Deliberate placeholders, not measurements: configs state their own.
        defaults = ServingSlurmResources()
        assert defaults.gpus_per_node == 12345
        assert defaults.cpus_per_task == 12345
        assert defaults.memory_gb is None

    def test_k2_horizon_full_suite_binds_gpus_to_tensor_parallel_size(self):
        config = (
            Path(__file__).parent.parent
            / "model_zoo/k2_horizon_7b_base_full.yaml"
        )
        raw = yaml.safe_load(config.read_text())
        cli_args = [str(arg) for arg in raw["vllm_cli_args"]]
        tp = int(cli_args[cli_args.index("--tensor-parallel-size") + 1])
        spec = ModelParser.parse_yaml(config)

        # The raw block states the GPU count explicitly; the 12345 code
        # default is only a placeholder.
        assert raw["serving_slurm_resources"] == {"gpus_per_node": tp}
        assert spec.serving_slurm_resources == ServingSlurmResources(gpus_per_node=tp)


class TestRemoteModelOutputPath:
    def test_output_path_includes_model_name(self):
        spec = _remote_spec(base_name="my-model", output_path="/results")
        instance = ModelParser.model_instance_from_path("some/path", spec)
        assert instance.output_path == "/results/my-model"

    def test_output_path_includes_revision(self):
        spec = _remote_spec(base_name="my-model", output_path="/results", revision="abc123")
        instance = ModelParser.model_instance_from_path("some/path", spec)
        assert instance.output_path == "/results/my-model-abc123"

    def test_output_path_includes_name_modifier(self):
        spec = _remote_spec(base_name="my-model", output_path="/results", name_modifier="chat")
        instance = ModelParser.model_instance_from_path("some/path", spec)
        assert instance.output_path == "/results/my-model-chat"

    def test_output_path_includes_revision_and_modifier(self):
        spec = _remote_spec(base_name="my-model", output_path="/results", revision="v1", name_modifier="ft")
        instance = ModelParser.model_instance_from_path("some/path", spec)
        assert instance.output_path == "/results/my-model-v1-ft"

    def test_different_models_get_different_output_paths(self):
        spec_a = _remote_spec(base_name="model-a", output_path="/results")
        spec_b = _remote_spec(base_name="model-b", output_path="/results")
        instance_a = ModelParser.model_instance_from_path("some/path", spec_a)
        instance_b = ModelParser.model_instance_from_path("some/path", spec_b)
        assert instance_a.output_path != instance_b.output_path

    def test_same_base_path_no_collision(self):
        """Two remote models sharing the same base output_path must not collide."""
        spec_a = _remote_spec(base_name="qwen-7b", output_path="/shared/results")
        spec_b = _remote_spec(base_name="llama-7b", output_path="/shared/results")
        a = ModelParser.model_instance_from_path("some/path", spec_a)
        b = ModelParser.model_instance_from_path("some/path", spec_b)
        assert a.output_path == "/shared/results/qwen-7b"
        assert b.output_path == "/shared/results/llama-7b"


def test_served_model_name_in_vllm_cli_args_is_rejected():
    invalid_spec = {
        "remote_model": {"path": "/models/qwen", "base_name": "qwen"},
        "venv_path": "/venv",
        "max_simultaneous_requests": 1,
        "vllm_cli_args": ["--tensor-parallel-size", "8", "--served-model-name", "custom-name"],
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "tester",
        "ready": True,
        "output_path": "/tmp/out",
        "parser_type": "noop",
    }
    with pytest.raises(ValidationError) as excinfo:
        ModelSpec.model_validate(invalid_spec)
    assert "--served-model-name" in str(excinfo.value)


class TestPromptPrefixInstructions:
    def test_defaults_to_none(self):
        spec = _remote_spec(base_name="my-model", output_path="/results")
        assert spec.prompt_prefix_instructions is None

    def test_accepts_string_value(self):
        spec = ModelSpec.model_validate({
            "remote_model": {"path": "some/path", "base_name": "my-model"},
            "venv_path": "/venv",
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/results",
            "parser_type": "noop",
            "prompt_prefix_instructions": "Think step by step.",
        })
        assert spec.prompt_prefix_instructions == "Think step by step."

    def test_propagated_to_model_instance(self):
        spec = ModelSpec.model_validate({
            "remote_model": {"path": "some/path", "base_name": "my-model"},
            "venv_path": "/venv",
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/results",
            "parser_type": "noop",
            "prompt_prefix_instructions": "Be concise.",
        })
        instance = ModelParser.model_instance_from_path("some/path", spec)
        assert instance.prompt_prefix_instructions == "Be concise."

    def test_none_propagated_to_model_instance(self):
        spec = _remote_spec(base_name="my-model", output_path="/results")
        instance = ModelParser.model_instance_from_path("some/path", spec)
        assert instance.prompt_prefix_instructions is None


def test_model_spec_invalid_model_type_message():
    invalid_spec = {
        "remote_model": {"path": "/models/qwen", "base_name": "qwen"},
        "venv_path": "/venv",
        "max_simultaneous_requests": 1,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "model_type": "chat",
        "owner": "tester",
        "ready": True,
        "output_path": "/tmp/out",
        "parser_type": "noop",
    }
    with pytest.raises(ValidationError) as excinfo:
        ModelSpec.model_validate(invalid_spec)
    assert 'Invalid model_type string: chat. Expected "base" or "instruct".' in str(excinfo.value)


def test_model_instance_inherits_tag():
    spec = _remote_spec(base_name="tagged-model", output_path="/results", tag="vision")
    instance = ModelParser.model_instance_from_path("some/path", spec)
    assert instance.tag == "vision"


def test_model_spec_tag_string_none_normalizes_to_any():
    spec = _remote_spec(base_name="tagged-model", output_path="/results", tag=None)
    assert spec.tag == "any"


def test_model_spec_blank_tag_normalizes_to_any():
    spec = _remote_spec(base_name="tagged-model", output_path="/results", tag="   ")
    assert spec.tag == "any"


def test_allow_long_max_model_len_defaults_true():
    spec = _remote_spec(base_name="my-model", output_path="/results")
    instance = ModelParser.model_instance_from_path("some/path", spec)
    assert spec.allow_long_max_model_len is True
    assert instance.allow_long_max_model_len is True


def test_allow_long_max_model_len_round_trips_from_yaml():
    spec = ModelSpec.model_validate({
        "remote_model": {"path": "some/path", "base_name": "my-model", "revision": None},
        "name_modifier": None,
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "max_time_to_deploy": 600,
        "allow_long_max_model_len": True,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/results",
        "parser_type": "noop",
    })
    instance = ModelParser.model_instance_from_path("some/path", spec)
    assert spec.allow_long_max_model_len is True
    assert instance.allow_long_max_model_len is True


def test_vllm_logging_level_defaults_to_warning():
    spec = _remote_spec(base_name="my-model", output_path="/results")
    instance = ModelParser.model_instance_from_path("some/path", spec)
    assert spec.vllm_logging_level == "WARNING"
    assert instance.vllm_logging_level == "WARNING"


def test_vllm_logging_level_round_trips_from_yaml():
    spec = ModelSpec.model_validate({
        "remote_model": {"path": "some/path", "base_name": "my-model", "revision": None},
        "name_modifier": None,
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "max_time_to_deploy": 600,
        "vllm_cli_args": [],
        "vllm_logging_level": "DEBUG",
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/results",
        "parser_type": "noop",
    })
    instance = ModelParser.model_instance_from_path("some/path", spec)
    assert spec.vllm_logging_level == "DEBUG"
    assert instance.vllm_logging_level == "DEBUG"


def test_cache_salt_propagated_to_model_instance():
    spec = ModelSpec.model_validate({
        "remote_model": {"path": "some/path", "base_name": "my-model"},
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "cache_salt": {"mode": "static", "salt": "partition-a"},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/results",
        "parser_type": "noop",
    })
    instance = ModelParser.model_instance_from_path("some/path", spec)
    assert instance.cache_salt.mode == "static"
    assert instance.cache_salt.salt == "partition-a"


def test_cache_salt_propagated_to_local_model_instance():
    spec = ModelSpec.model_validate({
        "local_model": {"path_glob": "/checkpoints/**/done", "model_family_name": "local-family"},
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "cache_salt": {"mode": "unique"},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/results",
        "parser_type": "noop",
    })
    instance = ModelParser.model_instance_from_path(Path("/checkpoints/run-1/done"), spec)
    assert instance.cache_salt.mode == "unique"


def test_cache_salt_round_trips_through_yaml(tmp_path):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump({
        "remote_model": {"path": "some/path", "base_name": "my-model"},
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "cache_salt": {"mode": "static", "salt": "partition-a"},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/results",
        "parser_type": "noop",
    }))
    spec = ModelParser.parse_yaml(model_yaml)
    assert spec.cache_salt.mode == "static"
    assert spec.cache_salt.salt == "partition-a"


# ---------------------------------------------------------------------------
# ExternalModel tests
# ---------------------------------------------------------------------------

def _external_spec(**kwargs):
    base = {
        "external_model": {
            "base_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        },
        "max_simultaneous_requests": 8,
        "openai_kwargs": {"temperature": 0.0},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/tmp/out",
        "parser_type": "noop",
    }
    base.update(kwargs)
    return ModelSpec.model_validate(base)


class TestExternalModelClass:
    def test_parses_required_fields(self):
        em = ExternalModel(base_name="gpt-4o", base_url="https://api.openai.com/v1")
        assert em.base_name == "gpt-4o"
        assert em.base_url == "https://api.openai.com/v1"

    def test_api_key_env_defaults_to_openai(self):
        em = ExternalModel(base_name="gpt-4o", base_url="https://api.openai.com/v1")
        assert em.api_key_env == "OPENAI_API_KEY"

    def test_requests_per_minute_optional(self):
        em = ExternalModel(base_name="gpt-4o", base_url="https://api.openai.com/v1")
        assert em.requests_per_minute is None

    def test_requests_per_minute_set(self):
        em = ExternalModel(base_name="gpt-4o", base_url="https://api.openai.com/v1", requests_per_minute=500)
        assert em.requests_per_minute == 500


class TestExternalModelSpec:
    def test_valid_external_spec(self):
        spec = _external_spec()
        assert spec.external_model is not None
        assert spec.external_model.base_name == "gpt-4o"

    def test_venv_path_not_required_for_external(self):
        # Must not raise
        spec = _external_spec()
        assert spec.venv_path is None

    def test_vllm_cli_args_not_required_for_external(self):
        spec = _external_spec()
        assert spec.vllm_cli_args is None

    def test_external_and_remote_model_rejected(self):
        with pytest.raises(ValidationError):
            ModelSpec.model_validate({
                "external_model": {"base_name": "gpt-4o", "base_url": "https://api.openai.com/v1"},
                "remote_model": {"path": "some/path", "base_name": "qwen"},
                "max_simultaneous_requests": 4,
                "openai_kwargs": {},
                "model_type": "instruct",
                "owner": "test",
                "ready": True,
                "output_path": "/tmp/out",
                "parser_type": "noop",
            })

    def test_external_and_local_model_rejected(self):
        with pytest.raises(ValidationError):
            ModelSpec.model_validate({
                "external_model": {"base_name": "gpt-4o", "base_url": "https://api.openai.com/v1"},
                "local_model": {"path_glob": "/models/*", "model_family_name": "qwen"},
                "venv_path": "/venv",
                "vllm_cli_args": [],
                "max_simultaneous_requests": 4,
                "openai_kwargs": {},
                "model_type": "instruct",
                "owner": "test",
                "ready": True,
                "output_path": "/tmp/out",
                "parser_type": "noop",
            })

    def test_external_with_venv_path_rejected(self):
        with pytest.raises(ValidationError, match="venv_path.*external"):
            ModelSpec.model_validate({
                "external_model": {"base_name": "gpt-4o", "base_url": "https://api.openai.com/v1"},
                "venv_path": "/venv",
                "max_simultaneous_requests": 4,
                "openai_kwargs": {},
                "model_type": "instruct",
                "owner": "test",
                "ready": True,
                "output_path": "/tmp/out",
                "parser_type": "noop",
            })

    def test_external_with_vllm_cli_args_rejected(self):
        with pytest.raises(ValidationError, match="vllm_cli_args.*external"):
            ModelSpec.model_validate({
                "external_model": {"base_name": "gpt-4o", "base_url": "https://api.openai.com/v1"},
                "vllm_cli_args": ["--tensor-parallel-size", "8"],
                "max_simultaneous_requests": 4,
                "openai_kwargs": {},
                "model_type": "instruct",
                "owner": "test",
                "ready": True,
                "output_path": "/tmp/out",
                "parser_type": "noop",
            })

    def test_remote_model_still_requires_venv_path(self):
        with pytest.raises(ValidationError, match="venv_path"):
            ModelSpec.model_validate({
                "remote_model": {"path": "some/path", "base_name": "qwen"},
                "vllm_cli_args": [],
                "max_simultaneous_requests": 4,
                "openai_kwargs": {},
                "model_type": "instruct",
                "owner": "test",
                "ready": True,
                "output_path": "/tmp/out",
                "parser_type": "noop",
            })

    def test_remote_model_still_requires_vllm_cli_args(self):
        with pytest.raises(ValidationError, match="vllm_cli_args"):
            ModelSpec.model_validate({
                "remote_model": {"path": "some/path", "base_name": "qwen"},
                "venv_path": "/venv",
                "max_simultaneous_requests": 4,
                "openai_kwargs": {},
                "model_type": "instruct",
                "owner": "test",
                "ready": True,
                "output_path": "/tmp/out",
                "parser_type": "noop",
            })

    def test_no_model_source_rejected(self):
        with pytest.raises(ValidationError):
            ModelSpec.model_validate({
                "max_simultaneous_requests": 4,
                "openai_kwargs": {},
                "model_type": "instruct",
                "owner": "test",
                "ready": True,
                "output_path": "/tmp/out",
                "parser_type": "noop",
            })


class TestExternalModelInstance:
    def test_model_instance_from_path_external(self):
        spec = _external_spec()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            instance = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        assert instance.name == "gpt-4o"
        assert instance.base_url == "https://api.openai.com/v1"
        assert instance.api_key == "sk-test-key"
        assert instance.is_external is True
        assert instance.venv_path is None
        assert instance.vllm_cli_args is None

    def test_api_key_resolved_from_env(self):
        spec = _external_spec()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mykey-123"}):
            instance = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        assert instance.api_key == "sk-mykey-123"

    def test_api_key_none_when_env_not_set(self):
        spec = _external_spec()
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            instance = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        assert instance.api_key is None

    def test_requests_per_minute_propagated(self):
        spec = _external_spec(**{
            "external_model": {
                "base_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "requests_per_minute": 500,
            }
        })
        instance = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        assert instance.requests_per_minute == 500

    def test_cache_salt_propagated(self):
        spec = _external_spec(cache_salt={"mode": "static", "salt": "external-partition"})
        instance = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        assert instance.cache_salt.mode == "static"
        assert instance.cache_salt.salt == "external-partition"

    def test_output_path_uses_base_name(self):
        spec = _external_spec()
        instance = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        assert instance.output_path == "/tmp/out/gpt-4o"

    def test_serving_key_is_stable(self):
        spec = _external_spec()
        i1 = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        i2 = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        assert i1.serving_key == i2.serving_key

    def test_serving_key_differs_from_vllm_model(self):
        external_spec = _external_spec()
        vllm_spec = _remote_spec(base_name="gpt-4o", output_path="/tmp/out")
        external_instance = ModelParser.model_instance_from_path("https://api.openai.com/v1", external_spec)
        vllm_instance = ModelParser.model_instance_from_path("some/path", vllm_spec)
        assert external_instance.serving_key != vllm_instance.serving_key

    def test_serving_key_ignores_runtime_name_and_tag_aliases(self):
        spec = _remote_spec(base_name="qwen", output_path="/results", tag="reasoning")
        instance = ModelParser.model_instance_from_path("some/path", spec)
        alias = instance.model_copy(update={"name": "qwen-eval-alias", "tag": "__eval_pair__123"})
        assert alias.serving_key == instance.serving_key

    def test_is_external_false_for_remote_model(self):
        spec = _remote_spec(base_name="qwen", output_path="/results")
        instance = ModelParser.model_instance_from_path("some/path", spec)
        assert instance.is_external is False

    def test_api_model_name_is_base_name_without_modifier(self):
        """api_model_name should be the original base_name, not suffixed by name_modifier."""
        spec = _external_spec()
        spec.name_modifier = "lcr_medium"
        instance = ModelParser.model_instance_from_path("https://api.openai.com/v1", spec)
        assert instance.name == "gpt-4o-lcr_medium"  # internal name has suffix
        assert instance.api_model_name == "gpt-4o"    # outbound API model is clean

    def test_api_model_name_none_for_vllm_model(self):
        """Non-external models don't have api_model_name."""
        spec = _remote_spec(base_name="qwen", output_path="/results")
        instance = ModelParser.model_instance_from_path("some/path", spec)
        assert instance.api_model_name is None
