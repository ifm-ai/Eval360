"""
Tests for pydantic model validators and field logic in:
  - scheduler/model.py  (ModelType, LocalModel, ModelSpec, ModelInstance)
  - scheduler/event.py  (EventInstance, GradingEventInstance, ImportedDatasetEventInstance)
  - scheduler/task.py   (Task, AsyncGenerationTask, GraderConfig, ImportedDatasetConfig)

These complement test_model_parser.py and test_task.py, which cover higher-level
YAML parsing and output-path construction.  This file tests the Pydantic validators
directly (ValidationError cases, field coercion, defaults, frozen config).
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scheduler.model import CacheSaltConfig, ModelParser, ModelType, LocalModel, ModelSpec, ModelInstance
from scheduler.event import (
    EventInstance, GradingEventInstance, ImportedDatasetEventInstance,
)
from scheduler.task import (
    Task, AsyncGenerationTask, GraderConfig, ImportedDatasetConfig, ImportedDatasetTask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_model_spec_dict(**overrides):
    """Minimal valid ModelSpec dict using a remote model."""
    d = {
        "remote_model": {"path": "org/model", "base_name": "mymodel"},
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/tmp/out",
        "parser_type": "noop",
    }
    d.update(overrides)
    return d


def _make_grading_event(**overrides):
    defaults = dict(
        uuid="ev-1",
        parent_uuid="par-1",
        model="mymodel",
        task_uuid="task-1",
        path_to_generations="/out/gens.jsonl",
        path_to_grades="/out/grades.jsonl",
        path_to_scores="/out/scores.jsonl",
        grader_type="multiple_choice",
        parser_type="noop",
    )
    defaults.update(overrides)
    return GradingEventInstance(**defaults)


def _make_task(**overrides):
    defaults = dict(
        uuid="t-1",
        average_over=[1],
        pass_at=[1],
        mode="base",
        grader=GraderConfig(type="multiple_choice"),
        data_path="/data/*.jsonl",
        dataset_name="mmlu",
        semantic_version="1.0.0",
        num_generations=100,
    )
    defaults.update(overrides)
    return AsyncGenerationTask(**defaults)


# ---------------------------------------------------------------------------
# ModelType.from_string
# ---------------------------------------------------------------------------

class TestModelTypeFromString:

    def test_base_maps_to_base(self):
        assert ModelType.from_string("base") == ModelType.BASE

    def test_instruct_maps_to_chat(self):
        assert ModelType.from_string("instruct") == ModelType.CHAT

    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid model type"):
            ModelType.from_string("chat")

    def test_case_sensitive_base_only(self):
        with pytest.raises(ValueError):
            ModelType.from_string("Base")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            ModelType.from_string("")


# ---------------------------------------------------------------------------
# LocalModel validator
# ---------------------------------------------------------------------------

class TestLocalModel:

    def test_absolute_path_accepted(self):
        m = LocalModel(path_glob="/models/*/checkpoint", model_family_name="llama")
        assert m.path_glob.startswith("/")

    def test_relative_path_raises(self):
        with pytest.raises(ValidationError, match='path_glob must start with "/"'):
            LocalModel(path_glob="models/*/checkpoint", model_family_name="llama")

    def test_bare_filename_raises(self):
        with pytest.raises(ValidationError):
            LocalModel(path_glob="checkpoint.bin", model_family_name="llama")

    def test_version_level_defaults_to_minus_one(self):
        m = LocalModel(path_glob="/models/**", model_family_name="test")
        assert m.version_level == -1

    def test_enqueue_existing_defaults_false(self):
        m = LocalModel(path_glob="/models/**", model_family_name="test")
        assert m.enqueue_existing is False


# ---------------------------------------------------------------------------
# ModelSpec validators
# ---------------------------------------------------------------------------

class TestModelSpecRemoteXorLocal:

    def test_both_remote_and_local_raises(self):
        d = _base_model_spec_dict()
        d["local_model"] = {
            "path_glob": "/checkpoints/**",
            "model_family_name": "llama",
        }
        with pytest.raises(ValidationError, match="Exactly one"):
            ModelSpec.model_validate(d)

    def test_neither_remote_nor_local_raises(self):
        d = _base_model_spec_dict()
        del d["remote_model"]
        with pytest.raises(ValidationError, match="Exactly one"):
            ModelSpec.model_validate(d)

    def test_only_remote_accepted(self):
        spec = ModelSpec.model_validate(_base_model_spec_dict())
        assert spec.remote_model is not None
        assert spec.local_model is None

    def test_only_local_accepted(self):
        d = {
            "local_model": {
                "path_glob": "/checkpoints/**",
                "model_family_name": "llama",
            },
            "venv_path": "/venv",
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "base",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        }
        spec = ModelSpec.model_validate(d)
        assert spec.local_model is not None
        assert spec.remote_model is None


class TestModelSpecSplitExtraBody:

    def test_standard_params_not_moved(self):
        d = _base_model_spec_dict(openai_kwargs={"temperature": 0.7, "max_tokens": 512})
        spec = ModelSpec.model_validate(d)
        assert spec.openai_kwargs["temperature"] == 0.7
        assert "extra_body" not in spec.openai_kwargs

    def test_non_standard_param_moved_to_extra_body(self):
        d = _base_model_spec_dict(openai_kwargs={"guided_choice": ["A", "B"]})
        spec = ModelSpec.model_validate(d)
        assert "guided_choice" not in spec.openai_kwargs
        assert spec.openai_kwargs["extra_body"]["guided_choice"] == ["A", "B"]

    def test_mixed_standard_and_non_standard(self):
        d = _base_model_spec_dict(openai_kwargs={
            "temperature": 0.0,
            "guided_choice": ["A", "B"],
        })
        spec = ModelSpec.model_validate(d)
        assert spec.openai_kwargs["temperature"] == 0.0
        assert spec.openai_kwargs["extra_body"]["guided_choice"] == ["A", "B"]
        assert "guided_choice" not in spec.openai_kwargs

    def test_empty_openai_kwargs_unchanged(self):
        spec = ModelSpec.model_validate(_base_model_spec_dict(openai_kwargs={}))
        assert spec.openai_kwargs == {}

    def test_raw_cache_salt_param_rejected(self):
        d = _base_model_spec_dict(openai_kwargs={"cache_salt": "raw"})
        with pytest.raises(ValidationError, match="cache_salt directly"):
            ModelSpec.model_validate(d)

    def test_raw_extra_body_cache_salt_rejected(self):
        d = _base_model_spec_dict(openai_kwargs={"extra_body": {"cache_salt": "raw"}})
        with pytest.raises(ValidationError, match="extra_body.cache_salt"):
            ModelSpec.model_validate(d)

    def test_non_mapping_extra_body_rejected(self):
        d = _base_model_spec_dict(openai_kwargs={"extra_body": ["bad"]})
        with pytest.raises(ValidationError, match="extra_body must be a mapping"):
            ModelSpec.model_validate(d)

    def test_venv_path_must_be_absolute(self):
        d = _base_model_spec_dict(venv_path="relative/venv")
        with pytest.raises(ValidationError, match='venv_path must be an absolute path'):
            ModelSpec.model_validate(d)


class TestCacheSaltConfig:

    def test_defaults_to_disabled(self):
        config = CacheSaltConfig()
        assert config.mode == "disabled"
        assert config.salt is None

    def test_static_requires_salt(self):
        with pytest.raises(ValidationError, match="salt is required"):
            CacheSaltConfig(mode="static")

    def test_static_accepts_salt(self):
        config = CacheSaltConfig(mode="static", salt="debug-run")
        assert config.mode == "static"
        assert config.salt == "debug-run"

    def test_unique_rejects_salt(self):
        with pytest.raises(ValidationError, match="may only be set"):
            CacheSaltConfig(mode="unique", salt="debug-run")

    def test_disabled_rejects_salt(self):
        with pytest.raises(ValidationError, match="may only be set"):
            CacheSaltConfig(mode="disabled", salt="debug-run")

    def test_model_spec_accepts_cache_salt(self):
        spec = ModelSpec.model_validate(
            _base_model_spec_dict(cache_salt={"mode": "static", "salt": "partition-a"})
        )
        assert spec.cache_salt.mode == "static"
        assert spec.cache_salt.salt == "partition-a"

    def test_model_spec_defaults_cache_salt_to_disabled(self):
        spec = ModelSpec.model_validate(_base_model_spec_dict())
        assert spec.cache_salt == CacheSaltConfig(mode="disabled")

    def test_model_spec_rejects_none_cache_salt(self):
        with pytest.raises(ValidationError):
            ModelSpec.model_validate(_base_model_spec_dict(cache_salt=None))

    def test_model_spec_assignment_rejects_none_cache_salt(self):
        spec = ModelSpec.model_validate(_base_model_spec_dict())
        with pytest.raises(ValidationError):
            spec.cache_salt = None

    def test_model_instance_assignment_rejects_none_cache_salt(self):
        spec = ModelSpec.model_validate(_base_model_spec_dict())
        instance = ModelParser.model_instance_from_path(Path(spec.remote_model.path), spec)
        with pytest.raises(ValidationError):
            instance.cache_salt = None

    def test_unknown_cache_salt_key_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CacheSaltConfig(mode="unique", unexpected="value")


class TestModelSpecModelTypeParsing:

    def test_base_string_parsed(self):
        spec = ModelSpec.model_validate(_base_model_spec_dict(model_type="base"))
        assert spec.model_type == ModelType.BASE

    def test_instruct_string_parsed(self):
        spec = ModelSpec.model_validate(_base_model_spec_dict(model_type="instruct"))
        assert spec.model_type == ModelType.CHAT

    def test_model_type_instance_passthrough(self):
        d = _base_model_spec_dict()
        d["model_type"] = ModelType.CHAT
        spec = ModelSpec.model_validate(d)
        assert spec.model_type == ModelType.CHAT


# ---------------------------------------------------------------------------
# ModelInstance validators
# ---------------------------------------------------------------------------

class TestModelInstanceModelTypeParsing:

    def _make_instance(self, model_type):
        return ModelInstance(
            name="test",
            path="org/model",
            venv_path="/venv",
            max_simultaneous_requests=4,
            vllm_cli_args=[],
            openai_kwargs={},
            parser_type="noop",
            model_type=model_type,
            owner="test",
            output_path="/tmp",
        )

    def test_base_string(self):
        assert self._make_instance("base").model_type == ModelType.BASE

    def test_instruct_string(self):
        assert self._make_instance("instruct").model_type == ModelType.CHAT

    def test_model_type_passthrough(self):
        assert self._make_instance(ModelType.BASE).model_type == ModelType.BASE

    def test_invalid_string_raises(self):
        with pytest.raises(ValidationError):
            self._make_instance("gpt")

    def test_invalid_type_raises_type_error(self):
        """Passing a non-string, non-ModelType value raises TypeError from the validator."""
        with pytest.raises(TypeError):
            self._make_instance(42)

    def test_venv_path_must_be_absolute(self):
        with pytest.raises(ValidationError, match='venv_path must be an absolute path'):
            ModelInstance(
                name="test",
                path="org/model",
                venv_path="relative/venv",
                max_simultaneous_requests=4,
                vllm_cli_args=[],
                openai_kwargs={},
                parser_type="noop",
                model_type="base",
                owner="test",
                output_path="/tmp",
            )

    def test_raw_cache_salt_param_rejected(self):
        with pytest.raises(ValidationError, match="cache_salt directly"):
            ModelInstance(
                name="test",
                path="org/model",
                venv_path="/venv",
                max_simultaneous_requests=4,
                vllm_cli_args=[],
                openai_kwargs={"cache_salt": "raw"},
                parser_type="noop",
                model_type="base",
                owner="test",
                output_path="/tmp",
            )

    def test_raw_extra_body_cache_salt_rejected(self):
        with pytest.raises(ValidationError, match="extra_body.cache_salt"):
            ModelInstance(
                name="test",
                path="org/model",
                venv_path="/venv",
                max_simultaneous_requests=4,
                vllm_cli_args=[],
                openai_kwargs={"extra_body": {"cache_salt": "raw"}},
                parser_type="noop",
                model_type="base",
                owner="test",
                output_path="/tmp",
            )

    def test_non_mapping_extra_body_rejected(self):
        with pytest.raises(ValidationError, match="extra_body must be a mapping"):
            ModelInstance(
                name="test",
                path="org/model",
                venv_path="/venv",
                max_simultaneous_requests=4,
                vllm_cli_args=[],
                openai_kwargs={"extra_body": ["bad"]},
                parser_type="noop",
                model_type="base",
                owner="test",
                output_path="/tmp",
            )


# ---------------------------------------------------------------------------
# GradingEventInstance path validators
# ---------------------------------------------------------------------------

class TestGradingEventInstancePathValidators:

    def test_valid_event_constructs_without_error(self):
        event = _make_grading_event()
        assert event.uuid == "ev-1"

    def test_generations_and_grades_same_path_raises(self):
        with pytest.raises(ValidationError, match="not different"):
            _make_grading_event(
                path_to_generations="/out/same.jsonl",
                path_to_grades="/out/same.jsonl",
                path_to_scores="/out/scores.jsonl",
            )

    def test_generations_and_scores_same_path_raises(self):
        with pytest.raises(ValidationError, match="not different"):
            _make_grading_event(
                path_to_generations="/out/same.jsonl",
                path_to_grades="/out/grades.jsonl",
                path_to_scores="/out/same.jsonl",
            )

    def test_grades_and_scores_same_path_raises(self):
        with pytest.raises(ValidationError, match="not different"):
            _make_grading_event(
                path_to_generations="/out/gens.jsonl",
                path_to_grades="/out/same.jsonl",
                path_to_scores="/out/same.jsonl",
            )

    def test_all_three_paths_same_raises(self):
        with pytest.raises(ValidationError):
            _make_grading_event(
                path_to_generations="/out/same.jsonl",
                path_to_grades="/out/same.jsonl",
                path_to_scores="/out/same.jsonl",
            )

    def test_parser_type_defaults_to_none(self):
        event = GradingEventInstance(
            uuid="e",
            parent_uuid="p",
            model="m",
            task_uuid="t",
            path_to_generations="/g.jsonl",
            path_to_grades="/gr.jsonl",
            path_to_scores="/sc.jsonl",
            grader_type="multiple_choice",
        )
        assert event.parser_type is None


# ---------------------------------------------------------------------------
# EventInstance frozen config
# ---------------------------------------------------------------------------

class TestEventInstanceFrozen:

    def test_grading_event_is_frozen(self):
        event = _make_grading_event()
        with pytest.raises(ValidationError):
            event.model = "other-model"

    def test_imported_dataset_event_is_frozen(self):
        event = ImportedDatasetEventInstance(
            uuid="e",
            parent_uuid="p",
            model="m",
            task_uuid="t",
            path_to_scores="/sc.jsonl",
        )
        with pytest.raises(ValidationError):
            event.model = "other-model"

    def test_grading_event_uuid_immutable(self):
        event = _make_grading_event()
        with pytest.raises(ValidationError):
            event.uuid = "new-uuid"


# ---------------------------------------------------------------------------
# ImportedDatasetEventInstance
# ---------------------------------------------------------------------------

class TestImportedDatasetEventInstance:

    def test_constructs_with_minimal_fields(self):
        event = ImportedDatasetEventInstance(
            uuid="e-1",
            parent_uuid="p-1",
            model="judge-model",
            task_uuid="t-1",
            path_to_scores="/out/scores.jsonl",
        )
        assert event.uuid == "e-1"
        assert event.model == "judge-model"

    def test_is_subclass_of_event_instance(self):
        event = ImportedDatasetEventInstance(
            uuid="e",
            parent_uuid="p",
            model="m",
            task_uuid="t",
            path_to_scores="/sc.jsonl",
        )
        assert isinstance(event, EventInstance)


# ---------------------------------------------------------------------------
# Task.split_extra_body (openai_settings)
# ---------------------------------------------------------------------------

class TestTaskSplitExtraBody:

    def test_standard_openai_settings_unchanged(self):
        task = _make_task(openai_settings={"temperature": 0.0, "max_tokens": 1024})
        assert task.openai_settings["temperature"] == 0.0
        assert "extra_body" not in task.openai_settings

    def test_non_standard_settings_moved_to_extra_body(self):
        task = _make_task(openai_settings={"guided_choice": ["A", "B"]})
        assert "guided_choice" not in task.openai_settings
        assert task.openai_settings["extra_body"]["guided_choice"] == ["A", "B"]

    def test_raw_cache_salt_setting_rejected(self):
        with pytest.raises(ValidationError, match="cache_salt directly"):
            _make_task(openai_settings={"cache_salt": "raw"})

    def test_raw_extra_body_cache_salt_setting_rejected(self):
        with pytest.raises(ValidationError, match="extra_body.cache_salt"):
            _make_task(openai_settings={"extra_body": {"cache_salt": "raw"}})

    def test_non_mapping_extra_body_setting_rejected(self):
        with pytest.raises(ValidationError, match="extra_body must be a mapping"):
            _make_task(openai_settings={"extra_body": ["bad"]})

    def test_none_openai_settings_not_processed(self):
        task = _make_task(openai_settings=None)
        assert task.openai_settings is None


# ---------------------------------------------------------------------------
# AsyncGenerationTask.check_grader_type
# ---------------------------------------------------------------------------

class TestAsyncGenerationTaskGraderType:

    def test_registered_grader_type_accepted(self):
        task = _make_task(grader=GraderConfig(type="multiple_choice"))
        assert task.grader.type == "multiple_choice"

    def test_unknown_grader_type_raises(self):
        with pytest.raises(ValidationError, match="not registered"):
            _make_task(grader=GraderConfig(type="definitely_not_a_real_grader_xyz"))


# ---------------------------------------------------------------------------
# GraderConfig
# ---------------------------------------------------------------------------

class TestGraderConfig:

    def test_no_llm_as_judge_by_default(self):
        cfg = GraderConfig(type="multiple_choice")
        assert cfg.llm_as_judge is None

    def test_none_judge_passthrough(self):
        cfg = GraderConfig(type="multiple_choice", llm_as_judge=None)
        assert cfg.llm_as_judge is None

    def test_local_model_as_judge_raises(self):
        """Only remote models may be used as LLM-as-judge."""
        local_spec = {
            "local_model": {
                "path_glob": "/checkpoints/**",
                "model_family_name": "judge",
            },
            "venv_path": "/venv",
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp",
            "parser_type": "noop",
        }
        spec = ModelSpec.model_validate(local_spec)
        with pytest.raises((TypeError, ValidationError)):
            GraderConfig(type="math_verify_llm_as_judge", llm_as_judge=spec)

    def test_dict_remote_model_as_judge_creates_model_instance(self):
        judge_dict = {
            "remote_model": {"path": "org/judge-model", "base_name": "judge"},
            "venv_path": "/venv",
            "max_simultaneous_requests": 2,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp",
            "parser_type": "noop",
        }
        cfg = GraderConfig(type="math_verify_llm_as_judge", llm_as_judge=judge_dict)
        assert isinstance(cfg.llm_as_judge, ModelInstance)
        assert "judge" in cfg.llm_as_judge.name

    def test_judge_name_modifier_appended(self):
        """When no name_modifier is set, '-judge' suffix is added automatically."""
        judge_dict = {
            "remote_model": {"path": "org/judge-model", "base_name": "mymodel"},
            "venv_path": "/venv",
            "max_simultaneous_requests": 2,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp",
            "parser_type": "noop",
        }
        cfg = GraderConfig(type="math_verify_llm_as_judge", llm_as_judge=judge_dict)
        assert cfg.llm_as_judge.name.endswith("-judge")

    def test_external_model_as_judge_creates_model_instance(self):
        judge_dict = {
            "external_model": {
                "base_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
            "max_simultaneous_requests": 4,
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp",
            "parser_type": "noop",
        }
        cfg = GraderConfig(type="math_verify_llm_as_judge", llm_as_judge=judge_dict)
        assert isinstance(cfg.llm_as_judge, ModelInstance)
        assert cfg.llm_as_judge.is_external is True
        assert cfg.llm_as_judge.base_url == "https://api.openai.com/v1"
        assert "judge" in cfg.llm_as_judge.name

    def test_external_model_as_judge_name_modifier(self):
        judge_dict = {
            "external_model": {
                "base_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
            },
            "max_simultaneous_requests": 4,
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp",
            "parser_type": "noop",
            "name_modifier": "custom",
        }
        cfg = GraderConfig(type="math_verify_llm_as_judge", llm_as_judge=judge_dict)
        assert "custom-judge" in cfg.llm_as_judge.name


# ---------------------------------------------------------------------------
# ImportedDatasetConfig
# ---------------------------------------------------------------------------

class TestImportedDatasetConfig:

    def test_minimal_construction(self):
        cfg = ImportedDatasetConfig(name="bfcl", commit="abc123")
        assert cfg.name == "bfcl"
        assert cfg.commit == "abc123"
        assert cfg.args == {}

    def test_args_populated(self):
        cfg = ImportedDatasetConfig(name="bfcl", commit="abc123", args={"version": 4.0})
        assert cfg.args["version"] == 4.0

    def test_model_dump_json_round_trips(self):
        cfg = ImportedDatasetConfig(name="bfcl", commit="abc123", args={"k": "v"})
        restored = ImportedDatasetConfig.model_validate_json(cfg.model_dump_json())
        assert restored.name == cfg.name
        assert restored.commit == cfg.commit
        assert restored.args == cfg.args
