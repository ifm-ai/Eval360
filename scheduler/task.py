import pydantic
import logging
from .model import ModelInstance, ModelSpec, ModelParser
from typing import Any
import yaml
from .choice_scoring_schema import is_choice_scoring_grader_type
from .grader.registry import get_grader, list_graders
from .utils import count_jsonl_records, get_unreadable_paths, is_hf_uri, check_hf_file_exists, normalize_tag, separate_extra_body
logger = logging.getLogger("Task")
logging.basicConfig(level=logging.INFO)


class ImportedDatasetConfig(pydantic.BaseModel):
    name: str
    commit: str
    args: dict[str, Any] = {}


class GraderConfig(pydantic.BaseModel):
    type: str
    llm_as_judge: ModelInstance | None = None

    @pydantic.field_validator("llm_as_judge", mode="before")
    @classmethod
    def parse_llm_as_judge(cls, v: Any) -> ModelInstance:
        if isinstance(v, ModelSpec):
            if bool(v.remote_model):
                v.name_modifier = "judge" if not v.name_modifier else v.name_modifier + "-judge"
                model_instance = ModelParser.model_instance_from_path(
                    path=v.remote_model.path, model_spec=v)
                return model_instance
            elif bool(v.external_model):
                v.name_modifier = "judge" if not v.name_modifier else v.name_modifier + "-judge"
                model_instance = ModelParser.model_instance_from_path(
                    path=v.external_model.base_url, model_spec=v)
                return model_instance
            else:
                raise TypeError(f'llm_as_judge model must be remote or external: {v}')
        if isinstance(v, ModelInstance):
            return v
        if isinstance(v, dict):
            if "remote_model" in v and bool(v["remote_model"]):
                if not v.get("name_modifier"):
                    v["name_modifier"] = "judge"
                else:
                    v["name_modifier"] += "-judge"
                v.setdefault("owner", "")
                v.setdefault("ready", True)
                v.setdefault("output_path", "")
                model_spec = ModelSpec.model_validate(v)
                model_instance = ModelParser.model_instance_from_path(
                    path=model_spec.remote_model.path, model_spec=model_spec)
                return model_instance
            elif "external_model" in v and bool(v["external_model"]):
                if not v.get("name_modifier"):
                    v["name_modifier"] = "judge"
                else:
                    v["name_modifier"] += "-judge"
                v.setdefault("owner", "")
                v.setdefault("ready", True)
                v.setdefault("output_path", "")
                model_spec = ModelSpec.model_validate(v)
                model_instance = ModelParser.model_instance_from_path(
                    path=model_spec.external_model.base_url, model_spec=model_spec)
                return model_instance
            else:
                raise TypeError(f'llm_as_judge model must be remote or external: {v}')
        if v is None:
            return v
        raise TypeError(f'Cannot parse llm_as_judge: {v}, {type(v)}')


class Task(pydantic.BaseModel):
    """Base class for all task types. Use ImportedDatasetTask or AsyncGenerationTask directly."""
    uuid: str
    openai_settings: dict | None = None
    # TODO POC: verify this against model type, only create an event if they match
    mode: str = "base"
    dataset_name: str
    # TODO (Post POC): implement
    semantic_version: str
    # TODO (Post POC): implement
    meta: dict | None = None
    enabled: bool = True
    tag: str = "any"

    @pydantic.field_validator("tag", mode="before")
    @classmethod
    def normalize_tag(cls, v: Any) -> str:
        return normalize_tag(v)

    @pydantic.model_validator(mode='after')
    def split_extra_body(self):
        if self.openai_settings:
            self.openai_settings = separate_extra_body(self.openai_settings)
        return self

    @classmethod
    def parse_yaml(cls, path) -> "ImportedDatasetTask | AsyncGenerationTask":
        logger.info(f"reading from {path}")
        with open(path, "r") as f:
            obj = yaml.safe_load(f)
        assert (obj)
        if "imported_dataset" in obj:
            if "data_path" in obj:
                raise ValueError('data_path must not be set for imported_dataset tasks')
            if "grader" in obj:
                raise ValueError('grader must not be set for imported_dataset tasks')
            return ImportedDatasetTask.model_validate(obj)
        task = AsyncGenerationTask.model_validate(obj)
        if is_hf_uri(task.data_path):
            # Phase 1 of HF validation: lightweight existence/access check — no download yet.
            # Phase 2 (download + num_generations count/validation) is deferred to Scheduler.register_task.
            check_hf_file_exists(task.data_path)
        else:
            unreadable = get_unreadable_paths(task.data_path)
            if unreadable and task.num_generations is None:
                raise PermissionError(
                    f"Cannot auto-count records: some JSONL files in {task.data_path!r} are not readable: "
                    f"{', '.join(unreadable)}"
                )
            count = count_jsonl_records(task.data_path)
            # TODO: unify this num_generations count/validation logic with the equivalent
            # block in Scheduler.register_task (HF path) into a shared helper.
            if task.num_generations is None:
                task = task.model_copy(update={"num_generations": count})
                logger.info(f"auto-counted {count} records from {task.data_path}")
            elif task.num_generations != count:
                raise ValueError(
                    f"num_generations={task.num_generations} does not match "
                    f"the actual record count of {count} in {task.data_path!r}"
                )
        return task


class ImportedDatasetTask(Task):
    imported_dataset: ImportedDatasetConfig


class AsyncGenerationTask(Task):
    average_over: list[int] | None
    pass_at: list[int] | None
    data_path: str
    num_generations: int | None = None
    grader: GraderConfig

    @pydantic.model_validator(mode='after')
    def check_grader_type(self):
        try:
            get_grader(self.grader.type)
        except ImportError as exc:
            raise ValueError(str(exc)) from exc
        except ValueError as exc:
            raise ValueError(
                f'Grader type "{self.grader.type}" is not registered. '
                f'Available grader types: {", ".join(list_graders())}'
            ) from exc
        return self

    @pydantic.model_validator(mode='after')
    def check_choice_scoring_runtime_contract(self):
        if not is_choice_scoring_grader_type(self.grader.type):
            return self
        if self.average_over != [1] or self.pass_at != [1]:
            raise ValueError(
                "choice_scoring grader requires average_over=[1] and pass_at=[1]"
            )
        return self
