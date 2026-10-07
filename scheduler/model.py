import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import pydantic
import yaml

from .cache_salt import CacheSaltConfig
from .external_requests import (
    canonicalize_external_endpoint,
    validate_and_canonicalize_external_endpoint,
)
from .grader.parser_registry import DEFAULT_PARSER, list_parsers
from .utils import normalize_tag, separate_extra_body

logger = logging.getLogger("Model")
logging.basicConfig(level=logging.INFO)

StrictPositiveInt = Annotated[
    int,
    pydantic.Field(strict=True, gt=0),
]
CoerciblePositiveInt = Annotated[
    int,
    pydantic.Field(gt=0),
]
StrictRetryAttemptCount = Annotated[
    int,
    pydantic.Field(strict=True, ge=1, le=100),
]
MAX_EXTERNAL_REQUEST_TIMEOUT_SECONDS = 7200
MAX_EXTERNAL_TOTAL_DEADLINE_SECONDS = 7200
MAX_EXTERNAL_BACKOFF_SECONDS = 60
_ALLOW_EXTERNAL_TOTAL_DEADLINE_OVERRIDE = (
    "allow_external_total_deadline_override"
)
_PYDANTIC_BOOL_ADAPTER = pydantic.TypeAdapter(bool)


class ServingSlurmResources(pydantic.BaseModel):
    """Bounded Slurm resources for one model-serving job.

    Eval360 always starts one task on one node for a serving replica.  The
    right values are site-specific, so set them per model.  The GPU and CPU
    defaults (12345) are deliberate placeholders, not measurements: each
    config states its own GPU count, and an omitted memory value leaves
    memory selection to the site configuration.  Individual maintained model
    definitions can bind all four values exactly.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    gpus_per_node: StrictPositiveInt = 12345  # placeholder; configs set their own
    cpus_per_task: StrictPositiveInt = 12345  # placeholder; set per site
    memory_gb: StrictPositiveInt | None = None
    time_limit: str = "1-00:00"

    @pydantic.field_validator("time_limit", mode="before")
    @classmethod
    def validate_time_limit(cls, value: Any) -> str:
        """Require a canonical, positive Slurm day/time duration."""
        if not isinstance(value, str) or not re.fullmatch(
            r"(?:[0-9]+-)?[0-9]+:[0-5][0-9](?::[0-5][0-9])?",
            value,
        ):
            raise ValueError(
                "time_limit must use Slurm [days-]hours:minutes[:seconds] format"
            )
        numeric_parts = [
            int(part)
            for part in value.replace("-", ":").split(":")
        ]
        if not any(numeric_parts):
            raise ValueError("time_limit must be greater than zero")
        return value


def validate_external_total_deadline_override(value: Any) -> float:
    """Validate an explicit override beyond every model-config timing cap."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= MAX_EXTERNAL_TOTAL_DEADLINE_SECONDS
    ):
        raise ValueError(
            "external total deadline override must exceed "
            f"{MAX_EXTERNAL_TOTAL_DEADLINE_SECONDS} seconds"
        )
    return float(value)


def _validate_external_request_concurrency(value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            "external max_simultaneous_requests must be a strict positive "
            "integer"
        )

_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client_secret",
        "credentials",
        "ingest_token",
        "password",
        "secret",
        "service_key",
        "token",
        "x_api_key",
    }
)


def _fingerprint_registration_secrets(
    value: Any,
    *,
    field_name: str | None = None,
) -> Any:
    """Hash secrets while preserving a comparable registration snapshot."""
    normalized_name = (
        field_name.lower().replace("-", "_")
        if field_name is not None
        else None
    )
    if normalized_name in _SECRET_FIELD_NAMES and value is not None:
        canonical_value = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(
            canonical_value.encode("utf-8")
        ).hexdigest()
    if isinstance(value, dict):
        return {
            key: _fingerprint_registration_secrets(
                nested,
                field_name=str(key),
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _fingerprint_registration_secrets(nested)
            for nested in value
        ]
    return value


def _coerces_to_external(value: Any) -> bool:
    """Match the bool semantics Pydantic applies to ``is_external``."""
    try:
        return _PYDANTIC_BOOL_ADAPTER.validate_python(value) is True
    except pydantic.ValidationError:
        return False


def _normalize_venv_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError('venv_path must be an absolute path')
    return str(path)


class ModelType(Enum):
    BASE = 0
    CHAT = 1

    @classmethod
    def from_string(cls, string):
        if string == "base":
            return ModelType.BASE
        elif string == "instruct":
            return ModelType.CHAT
        else:
            raise ValueError(f'Invalid model type "{string}"')


class RemoteModel(pydantic.BaseModel):
    path: str
    revision: str | None = None
    base_name: str | None


class LocalModel(pydantic.BaseModel):
    path_glob: str
    model_family_name: str
    version_level: int | None = -1  # defaults to -1
    enqueue_existing: bool = False

    @pydantic.model_validator(mode='after')
    def check_glob(self):
        if not (self.path_glob.startswith("/")):
            raise ValueError('path_glob must start with "/"')
        return self


class ExternalRetryJitter(str, Enum):
    """Supported randomization strategies for retry delays."""

    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


class ExternalRetryPolicy(pydantic.BaseModel):
    """Limits and spaces retries for one external API operation.

    ``max_attempts`` includes the initial request. Each attempt may run for at
    most ``request_timeout_seconds``, while ``total_deadline_seconds`` limits
    the entire operation, including waits between attempts. After a failed
    attempt, exponential backoff starts at ``initial_backoff_seconds`` and is
    capped by ``max_backoff_seconds``. ``jitter`` randomizes that capped delay:
    full jitter samples from zero to the cap, equal jitter samples from half
    the cap to the cap, and decorrelated jitter uses the previous delay to
    vary the next cap. No retry or wait may extend the total deadline.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    max_attempts: StrictRetryAttemptCount = 4
    request_timeout_seconds: float = pydantic.Field(
        default=MAX_EXTERNAL_REQUEST_TIMEOUT_SECONDS,
        gt=0,
        le=MAX_EXTERNAL_REQUEST_TIMEOUT_SECONDS,
        allow_inf_nan=False,
    )
    total_deadline_seconds: float = pydantic.Field(
        default=MAX_EXTERNAL_TOTAL_DEADLINE_SECONDS,
        gt=0,
        allow_inf_nan=False,
    )
    initial_backoff_seconds: float = pydantic.Field(
        default=1,
        ge=0,
        le=MAX_EXTERNAL_BACKOFF_SECONDS,
        allow_inf_nan=False,
    )
    max_backoff_seconds: float = pydantic.Field(
        default=MAX_EXTERNAL_BACKOFF_SECONDS,
        gt=0,
        le=MAX_EXTERNAL_BACKOFF_SECONDS,
        allow_inf_nan=False,
    )
    jitter: ExternalRetryJitter = ExternalRetryJitter.FULL

    @pydantic.field_validator("max_attempts", mode="before")
    @classmethod
    def validate_strict_attempt_count(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "max_attempts must be an integer; booleans and numeric "
                "strings are not accepted"
            )
        return value

    @pydantic.field_validator(
        "request_timeout_seconds",
        "total_deadline_seconds",
        "initial_backoff_seconds",
        "max_backoff_seconds",
        mode="before",
    )
    @classmethod
    def validate_strict_numeric_budget(cls, value: Any) -> int | float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                "retry timing values must be numeric; booleans and numeric "
                "strings are not accepted"
            )
        return value

    @pydantic.model_validator(mode="after")
    def check_backoff_bounds(self):
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds must be greater than or equal to "
                "initial_backoff_seconds"
            )
        return self


class ExternalModel(pydantic.BaseModel):
    """Pre-existing OpenAI-compatible API endpoint (no Slurm required)."""
    base_name: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    requests_per_minute: StrictPositiveInt | None = None
    retry_policy: ExternalRetryPolicy = pydantic.Field(
        default_factory=ExternalRetryPolicy
    )

    @pydantic.model_validator(mode="after")
    def check_configured_total_deadline_limit(
        self,
        info: pydantic.ValidationInfo,
    ):
        context = info.context or {}
        if (
            self.retry_policy.total_deadline_seconds
            > MAX_EXTERNAL_TOTAL_DEADLINE_SECONDS
            and not context.get(_ALLOW_EXTERNAL_TOTAL_DEADLINE_OVERRIDE, False)
        ):
            raise ValueError(
                "external retry total_deadline_seconds exceeds the model "
                "configuration limit; use "
                "--external-total-deadline-seconds for an explicit override"
            )
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "retry_policy":
            candidate = self.model_dump()
            candidate["retry_policy"] = value
            value = type(self).model_validate(candidate).retry_policy
        super().__setattr__(name, value)

    @pydantic.field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return validate_and_canonicalize_external_endpoint(value)


class ModelSpec(pydantic.BaseModel):
    local_model: LocalModel | None = None
    remote_model: RemoteModel | None = None
    external_model: ExternalModel | None = None
    name_modifier: str | None = None
    venv_path: str | None = None
    conda_env: str | None = None
    container_image: str | None = None
    container_mounts: list[str] | None = None
    serving_slurm_resources: ServingSlurmResources = pydantic.Field(
        default_factory=ServingSlurmResources
    )
    max_simultaneous_requests: CoerciblePositiveInt
    max_time_to_deploy: int | None = 600  # (default 10 minutes)
    allow_long_max_model_len: bool = True
    vllm_cli_args: list[str] | None = None
    vllm_logging_level: str = "WARNING"
    openai_kwargs: dict[str, Any]
    cache_salt: CacheSaltConfig = pydantic.Field(default_factory=CacheSaltConfig)
    model_type: ModelType
    owner: str
    # TODO: make ready meaningful
    ready: bool
    output_path: str  # absolute path
    parser_type: str
    prompt_prefix_instructions: str | None = None
    # TODO (POSTPOC): handle updates to the model spec producing new model specs
    #                 e.g. different prompts/rope scaling, etc
    tag: str = "any"

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_external_request_concurrency(cls, value: Any) -> Any:
        if (
            isinstance(value, Mapping)
            and value.get("external_model") is not None
        ):
            _validate_external_request_concurrency(
                value.get("max_simultaneous_requests")
            )
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "cache_salt":
            value = CacheSaltConfig.model_validate(value)
        super().__setattr__(name, value)

    @pydantic.field_validator("venv_path", mode="before")
    @classmethod
    def validate_venv_path(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError(f"Cannot parse venv_path: {v}")
        return _normalize_venv_path(v)

    @pydantic.model_validator(mode='after')
    def split_extra_body(self):
        if self.openai_kwargs:
            self.openai_kwargs = separate_extra_body(self.openai_kwargs)
        return self

    @pydantic.model_validator(mode='after')
    def check_exactly_one_model_source(self):
        sources = [self.local_model, self.remote_model, self.external_model]
        if sum(bool(s) for s in sources) != 1:
            raise ValueError(
                'Exactly one of "local_model", "remote_model", or "external_model" is required'
            )
        return self

    @pydantic.model_validator(mode='after')
    def check_vllm_fields_with_external(self):
        if self.external_model is not None:
            if self.venv_path is not None:
                raise ValueError(
                    'venv_path must not be set with external_model (external models do not use VLLM)'
                )
            if self.vllm_cli_args is not None:
                raise ValueError(
                    'vllm_cli_args must not be set with external_model (external models do not use VLLM)'
                )
        return self

    @pydantic.model_validator(mode='after')
    def check_vllm_fields_required_for_slurm(self):
        if self.local_model is not None or self.remote_model is not None:
            if self.vllm_cli_args is None:
                raise ValueError(
                    'vllm_cli_args is required for local_model and remote_model'
                )
        return self

    @pydantic.model_validator(mode='after')
    def check_serving_env(self):
        if self.external_model is not None:
            return self  # external models don't use a serving environment
        options = [
            ("venv_path", self.venv_path is not None),
            ("conda_env", self.conda_env is not None),
            ("container_image", self.container_image is not None),
        ]
        set_options = [name for name, is_set in options if is_set]
        if len(set_options) == 0:
            raise ValueError('Exactly one of "venv_path", "conda_env", or "container_image" is required')
        if len(set_options) > 1:
            raise ValueError(f'Only one serving environment allowed, got: {", ".join(set_options)}')
        if self.container_mounts and not self.container_image:
            raise ValueError('"container_mounts" requires "container_image"')
        if self.container_mounts and self.container_image:
            self._reject_model_path_mounts()
        return self

    def _reject_model_path_mounts(self):
        """Reject container_mounts whose source overlaps with the model path.

        The scheduler auto-injects an identity mount for the resolved model
        path, so user mounts that remap it (or a parent/child of it) to a
        different container path will conflict.  We check against the
        remote_model path or the local_model path_glob.
        """
        import fnmatch
        from pathlib import PurePosixPath

        if self.remote_model is not None:
            model_path = self.remote_model.path
        elif self.local_model is not None:
            model_path = self.local_model.path_glob
        else:
            return

        for mount in self.container_mounts:
            src = mount.split(":")[0]
            # Check if mount source matches the glob, is a child of a glob
            # match, or is a parent of the glob pattern.
            if fnmatch.fnmatch(src, model_path):
                raise ValueError(
                    f'container_mounts entry {mount!r} conflicts with the model path '
                    f'({model_path!r}). The scheduler auto-injects an identity mount '
                    f'for the model path — remove this mount.'
                )
            # Child: src is under a path that matches the glob
            src_parts = PurePosixPath(src).parts
            for i in range(len(src_parts), 0, -1):
                parent = str(PurePosixPath(*src_parts[:i]))
                if parent != src and fnmatch.fnmatch(parent, model_path):
                    raise ValueError(
                        f'container_mounts entry {mount!r} conflicts with the model path '
                        f'({model_path!r}). The scheduler auto-injects an identity mount '
                        f'for the model path — remove this mount.'
                    )
            # Parent: the glob pattern is under src
            glob_parts = PurePosixPath(model_path).parts
            for i in range(len(glob_parts), 0, -1):
                parent = str(PurePosixPath(*glob_parts[:i]))
                if parent == src:
                    raise ValueError(
                        f'container_mounts entry {mount!r} conflicts with the model path '
                        f'({model_path!r}). The scheduler auto-injects an identity mount '
                        f'for the model path — remove this mount.'
                    )

    @pydantic.field_validator("vllm_cli_args")
    @classmethod
    def disallow_served_model_name(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if "--served-model-name" in v:
            raise ValueError(
                '"--served-model-name" must not be set in vllm_cli_args; '
                "it is set automatically from the model config"
            )
        return v

    @pydantic.model_validator(mode='after')
    def check_parser_registered(self):
        parser_name = self.parser_type or DEFAULT_PARSER
        available = set(list_parsers())
        if parser_name not in available:
            raise ValueError(
                f'Unknown parser_type "{parser_name}". Available parsers: {", ".join(sorted(available))}'
            )
        self.parser_type = parser_name
        return self

    @pydantic.field_validator("tag", mode="before")
    @classmethod
    def normalize_tag(cls, v: Any) -> str:
        return normalize_tag(v)

    @pydantic.field_validator("model_type", mode="before")
    @classmethod
    def parse_model_type(cls, v: Any) -> ModelType:
        if isinstance(v, ModelType):
            return v
        if isinstance(v, str):
            mapping = {
                "base": ModelType.BASE,
                "instruct": ModelType.CHAT,
            }
            try:
                return mapping[v.lower()]
            except KeyError:
                raise ValueError(f'Invalid model_type string: {v}. Expected "base" or "instruct".')
        raise TypeError(f'Cannot parse model_type: {v}')


class ModelInstance(pydantic.BaseModel):
    name: str
    # family_name groups checkpoints of the same training run under one
    # label in downstream reporting. Must exclude per-checkpoint suffixes
    # (name_modifier, path version). Defaults to `name` when omitted so that
    # non-checkpointed models (and existing test fixtures) still work.
    family_name: str | None = None
    path: str
    revision: str | None = None
    venv_path: str | None = None
    conda_env: str | None = None
    container_image: str | None = None
    container_mounts: list[str] | None = None
    serving_slurm_resources: ServingSlurmResources = pydantic.Field(
        default_factory=ServingSlurmResources
    )
    max_simultaneous_requests: CoerciblePositiveInt
    max_time_to_deploy: int | None = 600  # (default 10 minutes)
    allow_long_max_model_len: bool = True
    vllm_cli_args: list[str] | None = None
    vllm_logging_level: str = "WARNING"
    # TODO: remove openai_kwargs from this instance? It doesnt actually change the deployment
    openai_kwargs: dict[str, Any]
    cache_salt: CacheSaltConfig = pydantic.Field(default_factory=CacheSaltConfig)
    parser_type: str
    model_type: ModelType
    owner: str
    output_path: str  # absolute path
    tag: str = "any"
    prompt_prefix_instructions: str | None = None
    # External model fields
    base_url: str | None = None
    api_key: str | None = None
    requests_per_minute: StrictPositiveInt | None = None
    is_external: bool = False
    api_model_name: str | None = None  # original base_name for outbound API requests
    external_retry_policy: ExternalRetryPolicy = pydantic.Field(
        default_factory=ExternalRetryPolicy
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_external_request_concurrency(cls, value: Any) -> Any:
        if (
            isinstance(value, Mapping)
            and _coerces_to_external(value.get("is_external", False))
        ):
            _validate_external_request_concurrency(
                value.get("max_simultaneous_requests")
            )
        return value

    @pydantic.model_validator(mode="after")
    def validate_external_url(self):
        if not self.is_external:
            return self
        endpoint = validate_and_canonicalize_external_endpoint(
            self.base_url or self.path
        )
        self.base_url = endpoint
        self.path = endpoint
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "cache_salt":
            value = CacheSaltConfig.model_validate(value)
        elif name == "external_retry_policy":
            value = ExternalRetryPolicy.model_validate(value)
        super().__setattr__(name, value)

    def registration_snapshot(self) -> dict[str, Any]:
        """Return every effective field with secret values fingerprinted.

        Building this from ``model_dump`` makes newly added model fields part
        of immutable same-name registration automatically.
        """
        return _fingerprint_registration_secrets(
            self.model_dump(mode="json"),
        )

    @property
    def serving_key(self) -> str:
        """Content-addressed 12-char hex key that identifies a unique deployment config."""
        import hashlib
        import json as _json
        serving_path = (
            canonicalize_external_endpoint(self.base_url or self.path)
            if self.is_external
            else self.path
        )
        payload = _json.dumps({
            "path": serving_path,
            "revision": self.revision,
            "vllm_cli_args": sorted(self.vllm_cli_args) if self.vllm_cli_args is not None else None,
            "venv_path": self.venv_path,
            "conda_env": self.conda_env,
            "container_image": self.container_image,
            "serving_slurm_resources": (
                None
                if self.is_external
                else self.serving_slurm_resources.model_dump(mode="json")
            ),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def uses_container(self) -> bool:
        return self.container_image is not None

    @property
    def uses_conda(self) -> bool:
        return self.conda_env is not None

    @pydantic.field_validator("venv_path", mode="before")
    @classmethod
    def validate_venv_path(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError(f"Cannot parse venv_path: {v}")
        return _normalize_venv_path(v)

    @pydantic.field_validator("tag", mode="before")
    @classmethod
    def normalize_tag(cls, v: Any) -> str:
        return normalize_tag(v)

    @pydantic.model_validator(mode='after')
    def split_extra_body(self):
        if self.openai_kwargs:
            self.openai_kwargs = separate_extra_body(self.openai_kwargs)
        return self

    @pydantic.model_validator(mode="after")
    def default_family_name(self):
        if self.family_name is None:
            self.family_name = self.name
        return self

    @pydantic.field_validator("model_type", mode="before")
    @classmethod
    def parse_model_type(cls, v: Any) -> ModelType:
        if isinstance(v, ModelType):
            return v
        if isinstance(v, str):
            mapping = {
                "base": ModelType.BASE,
                "instruct": ModelType.CHAT,
            }
            try:
                return mapping[v.lower()]
            except KeyError:
                raise ValueError(f'Invalid model_type string: {v}. Expected "base" or "instruct".')
        raise TypeError(f'Cannot parse model_type: {v}')


class ModelParser:
    def __init__(self):
        pass

    @classmethod
    def model_instance_from_path(cls, path, model_spec):
        if model_spec.local_model is not None:
            path = path.parent
            if model_spec.name_modifier:
                name_prefix = model_spec.local_model.model_family_name + model_spec.name_modifier
            else:
                name_prefix = model_spec.local_model.model_family_name
            name = (name_prefix + "-" +
                    path.relative_to(path.anchor).parts[model_spec.local_model.version_level])
            revision = None
            output_path = os.path.join(model_spec.output_path, name)
            return ModelInstance.model_construct(
                name=name,
                family_name=model_spec.local_model.model_family_name,
                path=str(path),
                revision=revision,
                venv_path=model_spec.venv_path,
                conda_env=model_spec.conda_env,
                container_image=model_spec.container_image,
                container_mounts=model_spec.container_mounts,
                serving_slurm_resources=model_spec.serving_slurm_resources,
                max_simultaneous_requests=model_spec.max_simultaneous_requests,
                max_time_to_deploy=model_spec.max_time_to_deploy,
                allow_long_max_model_len=model_spec.allow_long_max_model_len,
                vllm_cli_args=model_spec.vllm_cli_args,
                vllm_logging_level=model_spec.vllm_logging_level,
                openai_kwargs=model_spec.openai_kwargs,
                cache_salt=model_spec.cache_salt,
                parser_type=model_spec.parser_type,
                model_type=model_spec.model_type,
                owner=model_spec.owner,
                output_path=output_path,
                tag=model_spec.tag,
                prompt_prefix_instructions=model_spec.prompt_prefix_instructions,
                is_external=False,

            )
        elif model_spec.remote_model is not None:
            family_name = model_spec.remote_model.base_name or str(path)
            name = family_name
            revision = model_spec.remote_model.revision
            if revision:
                name += f"-{revision}"
            if model_spec.name_modifier:
                name += f"-{model_spec.name_modifier}"
            output_path = os.path.join(model_spec.output_path, name)
            return ModelInstance.model_construct(
                name=name,
                family_name=family_name,
                path=str(path),
                revision=revision,
                venv_path=model_spec.venv_path,
                conda_env=model_spec.conda_env,
                container_image=model_spec.container_image,
                container_mounts=model_spec.container_mounts,
                serving_slurm_resources=model_spec.serving_slurm_resources,
                max_simultaneous_requests=model_spec.max_simultaneous_requests,
                max_time_to_deploy=model_spec.max_time_to_deploy,
                allow_long_max_model_len=model_spec.allow_long_max_model_len,
                vllm_cli_args=model_spec.vllm_cli_args,
                vllm_logging_level=model_spec.vllm_logging_level,
                openai_kwargs=model_spec.openai_kwargs,
                cache_salt=model_spec.cache_salt,
                parser_type=model_spec.parser_type,
                model_type=model_spec.model_type,
                owner=model_spec.owner,
                output_path=output_path,
                tag=model_spec.tag,
                prompt_prefix_instructions=model_spec.prompt_prefix_instructions,
                is_external=False,

            )
        elif model_spec.external_model is not None:
            ext = model_spec.external_model
            family_name = ext.base_name
            name = family_name
            if model_spec.name_modifier:
                name += f"-{model_spec.name_modifier}"
            output_path = os.path.join(model_spec.output_path, name)
            api_key = os.environ.get(ext.api_key_env) if ext.api_key_env else None
            return ModelInstance.model_construct(
                name=name,
                family_name=family_name,
                path=ext.base_url,
                revision=None,
                venv_path=None,
                serving_slurm_resources=model_spec.serving_slurm_resources,
                max_simultaneous_requests=model_spec.max_simultaneous_requests,
                max_time_to_deploy=None,
                allow_long_max_model_len=True,
                vllm_cli_args=None,
                vllm_logging_level="WARNING",
                openai_kwargs=model_spec.openai_kwargs,
                cache_salt=model_spec.cache_salt,
                parser_type=model_spec.parser_type,
                model_type=model_spec.model_type,
                owner=model_spec.owner,
                output_path=output_path,
                tag=model_spec.tag,
                prompt_prefix_instructions=model_spec.prompt_prefix_instructions,
                base_url=ext.base_url,
                api_key=api_key,
                requests_per_minute=ext.requests_per_minute,
                is_external=True,
                api_model_name=ext.base_name,
                external_retry_policy=ext.retry_policy,
            )
        raise ValueError("ModelSpec has no valid model source")

    # Placeholder runtime fields injected for eval-config parsing. These are
    # always overwritten by EvalConfigParser._build_resolved_pair (owner from
    # the eval config, output_path from the group output_root, parser_type from
    # the group), so the sentinels never reach generation/output. They exist only
    # so ModelSpec validation passes when an eval-mode model YAML omits the
    # runtime fields the eval config is responsible for supplying.
    _EVAL_RUNTIME_FIELD_DEFAULTS = {
        "owner": "__eval_config_placeholder__",
        "output_path": "__eval_config_placeholder__",
        "parser_type": DEFAULT_PARSER,
    }

    @classmethod
    def parse_yaml(
        cls,
        path,
        *,
        eval_mode: bool = False,
        external_total_deadline_seconds: float | None = None,
    ):
        if external_total_deadline_seconds is not None:
            external_total_deadline_seconds = (
                validate_external_total_deadline_override(
                    external_total_deadline_seconds
                )
            )
        logger.info(f"reading from {path}")
        with open(path, "r") as f:
            obj = yaml.safe_load(f)
        if obj is None:
            raise ValueError(f"Empty model config file: {path}")
        if not isinstance(obj, dict):
            raise ValueError(f"Model config file must contain a mapping: {path}")
        if eval_mode:
            # In eval-config mode the eval config supplies owner/output_path and
            # the group supplies parser_type, so these may be omitted from the
            # model YAML. Fill any missing ones with placeholders so validation
            # passes; _build_resolved_pair overwrites them per resolved pair.
            for field, default in cls._EVAL_RUNTIME_FIELD_DEFAULTS.items():
                obj.setdefault(field, default)
        external_model = obj.get("external_model")
        if (
            external_total_deadline_seconds is not None
            and isinstance(external_model, Mapping)
        ):
            retry_policy = external_model.get("retry_policy", {})
            if isinstance(retry_policy, Mapping):
                obj = dict(obj)
                external_model = dict(external_model)
                retry_policy = dict(retry_policy)
                retry_policy["total_deadline_seconds"] = (
                    external_total_deadline_seconds
                )
                external_model["retry_policy"] = retry_policy
                obj["external_model"] = external_model
        model = ModelSpec.model_validate(
            obj,
            context={
                _ALLOW_EXTERNAL_TOTAL_DEADLINE_OVERRIDE: (
                    external_total_deadline_seconds is not None
                )
            },
        )
        return model
