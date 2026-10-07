import glob
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydantic
import yaml

from .grader.parser_registry import DEFAULT_PARSER, list_parsers
from .model import ModelInstance, ModelParser, ModelSpec
from .openai_interface import _deep_merge_openai_kwargs
from .task import AsyncGenerationTask, GraderConfig
from .utils import normalize_tag, separate_extra_body

logger = logging.getLogger("EvalConfig")


@dataclass(frozen=True)
class LoadedModelSpec:
    path: str
    spec: ModelSpec


@dataclass(frozen=True)
class LoadedTask:
    path: str
    task: AsyncGenerationTask


@dataclass(frozen=True)
class LoadedEvalConfig:
    path: str
    config: "EvalConfig"


@dataclass(frozen=True)
class ResolvedEvalPair:
    model: ModelInstance
    task: AsyncGenerationTask
    model_config_path: str
    data_config_path: str
    eval_group_name: str
    eval_config_path: str


class EvalGroup(pydantic.BaseModel):
    name: str
    model_tag: str
    data_tag: str
    parser_type: str
    output_root: str
    grader: GraderConfig | None = None
    vllm_cli_args: list[str] | None = None
    openai_overrides: dict[str, Any] | None = None

    @pydantic.field_validator("model_tag", "data_tag", mode="before")
    @classmethod
    def normalize_group_tag(cls, value: Any) -> str:
        return normalize_tag(value)

    @pydantic.field_validator("openai_overrides", mode="after")
    @classmethod
    def split_extra_body(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value:
            return separate_extra_body(value)
        return value

    @pydantic.field_validator("output_root")
    @classmethod
    def check_output_root_absolute(cls, value: str) -> str:
        raw = value.strip()
        if not raw:
            raise ValueError("output_root must be a non-empty absolute path")
        try:
            path = Path(raw).expanduser()
        except RuntimeError as exc:
            raise ValueError("output_root must be a non-empty absolute path") from exc
        if not path.is_absolute():
            raise ValueError("output_root must be a non-empty absolute path")
        return str(path)

    @pydantic.model_validator(mode="after")
    def check_parser_registered(self):
        parser_name = self.parser_type or DEFAULT_PARSER
        available = set(list_parsers())
        if parser_name not in available:
            raise ValueError(
                f'Unknown parser_type "{parser_name}". Available parsers: {", ".join(sorted(available))}'
            )
        self.parser_type = parser_name
        return self

    @pydantic.field_validator("vllm_cli_args")
    @classmethod
    def disallow_served_model_name(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and "--served-model-name" in value:
            raise ValueError(
                '"--served-model-name" must not be set in vllm_cli_args; '
                "it is set automatically from the model config"
            )
        return value


class EvalConfig(pydantic.BaseModel):
    version: int
    owner: str
    groups: list[EvalGroup]

    @pydantic.field_validator("version")
    @classmethod
    def check_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"Unsupported eval config version: {value}. Expected 1.")
        return value

    @pydantic.model_validator(mode="after")
    def check_group_names_unique(self):
        seen: set[str] = set()
        duplicates = []
        for group in self.groups:
            if group.name in seen:
                duplicates.append(group.name)
            seen.add(group.name)
        if duplicates:
            dup_list = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Duplicate eval group names are not allowed: {dup_list}")
        return self


class EvalConfigParser:
    @classmethod
    def parse_yaml(cls, path: str) -> EvalConfig:
        logger.info("reading eval config from %s", path)
        with open(path, "r") as handle:
            obj = yaml.safe_load(handle)
        if obj is None:
            raise ValueError(f"Empty eval config file: {path}")
        if not isinstance(obj, dict):
            raise ValueError(f"Eval config file must contain a mapping: {path}")
        return EvalConfig.model_validate(obj)

    @classmethod
    def build_resolved_pairs(
        cls,
        loaded_models: list[LoadedModelSpec],
        loaded_tasks: list[LoadedTask],
        loaded_eval_configs: list[LoadedEvalConfig],
    ) -> list[ResolvedEvalPair]:
        pair_to_group: dict[tuple[str, str], tuple[str, str]] = {}
        resolved: list[ResolvedEvalPair] = []

        for loaded_eval in loaded_eval_configs:
            eval_config = loaded_eval.config
            for group in eval_config.groups:
                matched_model_specs = [
                    loaded_model
                    for loaded_model in loaded_models
                    if loaded_model.spec.tag == group.model_tag
                ]
                matched_tasks = [
                    loaded_task
                    for loaded_task in loaded_tasks
                    if loaded_task.task.tag == group.data_tag
                ]

                if not matched_model_specs:
                    raise ValueError(
                        f"Eval group '{group.name}' in {loaded_eval.path} matched zero models "
                        f"for model_tag={group.model_tag!r}"
                    )
                if not matched_tasks:
                    raise ValueError(
                        f"Eval group '{group.name}' in {loaded_eval.path} matched zero data configs "
                        f"for data_tag={group.data_tag!r}"
                    )

                concrete_models: list[tuple[LoadedModelSpec, ModelInstance]] = []
                for loaded_model in matched_model_specs:
                    expanded = cls._expand_model_instances(loaded_model)
                    for model_instance in expanded:
                        concrete_models.append((loaded_model, model_instance))

                if not concrete_models:
                    raise ValueError(
                        f"Eval group '{group.name}' in {loaded_eval.path} matched zero models "
                        f"for model_tag={group.model_tag!r}"
                    )

                for loaded_model in matched_model_specs:
                    for loaded_task in matched_tasks:
                        pair_key = (loaded_model.path, loaded_task.path)
                        if pair_key in pair_to_group:
                            prev_path, prev_group = pair_to_group[pair_key]
                            raise ValueError(
                                "Eval group overlap detected for "
                                f"model={loaded_model.path!r}, data={loaded_task.path!r}: "
                                f"{prev_path}:{prev_group} and {loaded_eval.path}:{group.name} "
                                "produce the same (model, data) pair"
                            )
                        pair_to_group[pair_key] = (loaded_eval.path, group.name)

                for loaded_model, model_instance in concrete_models:
                    for loaded_task in matched_tasks:
                        resolved.append(
                            cls._build_resolved_pair(
                                loaded_model=loaded_model,
                                base_model=model_instance,
                                loaded_task=loaded_task,
                                loaded_eval=loaded_eval,
                                group=group,
                            )
                        )

        return resolved

    @classmethod
    def _build_resolved_pair(
        cls,
        *,
        loaded_model: LoadedModelSpec,
        base_model: ModelInstance,
        loaded_task: LoadedTask,
        loaded_eval: LoadedEvalConfig,
        group: EvalGroup,
    ) -> ResolvedEvalPair:
        base_task = loaded_task.task

        if base_model.is_external and group.vllm_cli_args is not None:
            raise ValueError(
                f"Eval group '{group.name}' in {loaded_eval.path} sets vllm_cli_args, "
                f"but matched external_model config {loaded_model.path!r}"
            )

        pair_token = cls._pair_token(
            eval_config_path=loaded_eval.path,
            group_name=group.name,
            model_path=loaded_model.path,
            task_path=loaded_task.path,
        )
        pair_tag = f"__eval_pair__{pair_token}"

        effective_request_settings: dict[str, Any] = {}
        if base_task.openai_settings:
            effective_request_settings = _deep_merge_openai_kwargs(
                effective_request_settings,
                base_task.openai_settings,
            )
        if base_model.openai_kwargs:
            effective_request_settings = _deep_merge_openai_kwargs(
                effective_request_settings,
                base_model.openai_kwargs,
            )
        if group.openai_overrides:
            effective_request_settings = _deep_merge_openai_kwargs(
                effective_request_settings,
                group.openai_overrides,
            )

        effective_grader = (
            group.grader.model_copy(deep=True)
            if group.grader is not None
            else base_task.grader.model_copy(deep=True)
        )

        model_variant = base_model.model_copy(update={
            "name": f"{base_model.name}-eval-{pair_token}",
            # All pair-specific scheduler identities for this concrete model
            # use one API-facing name.  A serving job can therefore advertise
            # that stable name before later sibling pairs become active.
            "api_model_name": base_model.api_model_name or base_model.name,
            "parser_type": group.parser_type,
            "owner": loaded_eval.config.owner,
            # Each resolved eval pair gets its own output directory keyed by the
            # pair_token so that two selected tasks sharing a dataset_name (e.g.
            # data_zoo/mmlu.yaml and examples/data/mmlu.yaml) do not write/resume
            # the same <dataset>_generations/grades/scores files. Output filenames
            # are still derived from dataset_name in event.py; isolating the
            # directory keeps that contract intact without touching direct mode.
            "output_path": os.path.join(group.output_root, base_model.name, pair_token),
            "vllm_cli_args": group.vllm_cli_args if group.vllm_cli_args is not None else base_model.vllm_cli_args,
            # Request-time settings are baked into the task variant so eval overrides
            # apply after the existing model/data merge logic.
            "openai_kwargs": {},
            "tag": pair_tag,
        })
        task_variant = base_task.model_copy(update={
            "uuid": f"{base_task.uuid}-eval-{pair_token}",
            "grader": effective_grader,
            "openai_settings": effective_request_settings or None,
            "tag": pair_tag,
        })

        return ResolvedEvalPair(
            model=model_variant,
            task=task_variant,
            model_config_path=loaded_model.path,
            data_config_path=loaded_task.path,
            eval_group_name=group.name,
            eval_config_path=loaded_eval.path,
        )

    @staticmethod
    def _pair_token(*, eval_config_path: str, group_name: str, model_path: str, task_path: str) -> str:
        payload = "\0".join([eval_config_path, group_name, model_path, task_path])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def _expand_model_instances(cls, loaded_model: LoadedModelSpec) -> list[ModelInstance]:
        model_spec = loaded_model.spec
        if model_spec.local_model is not None:
            matches = sorted(set(glob.glob(model_spec.local_model.path_glob, recursive=True)))
            if not matches:
                raise ValueError(
                    f"local_model path_glob {model_spec.local_model.path_glob!r} "
                    f"in {loaded_model.path} matched zero paths"
                )
            instances: list[ModelInstance] = []
            for match in matches:
                matched_path = Path(match)
                sentinel_path = matched_path / "done.txt" if matched_path.is_dir() else matched_path
                instances.append(ModelParser.model_instance_from_path(sentinel_path, model_spec))
            return instances

        if model_spec.remote_model is not None:
            return [
                ModelParser.model_instance_from_path(
                    model_spec.remote_model.path,
                    model_spec,
                )
            ]

        if model_spec.external_model is not None:
            return [
                ModelParser.model_instance_from_path(
                    model_spec.external_model.base_url,
                    model_spec,
                )
            ]

        raise ValueError(f"Unsupported model config in eval-driven mode: {loaded_model.path}")
