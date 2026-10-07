"""Strict request-native input for evaluator-owned ``evaluate-now`` suites.

The request is deliberately self-contained: owner manifests supply immutable
runner and suite semantics, while the request supplies the site-specific model,
runtime, dataset, and output paths.  No model, task, or eval YAML is generated
or parsed on this route.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .model import ModelSpec
from .task import AsyncGenerationTask
from .terminal_result import (
    FileBinding,
    TERMINAL_RESULT_CONTRACT,
    TERMINAL_RESULT_SCHEMA_VERSION,
    bind_canonical_json_object,
    bind_regular_file,
    canonical_json,
    verify_file_binding,
)
from .utils import count_jsonl_records

EVALUATION_REQUEST_CONTRACT = "eval360.evaluate-now-request"
EVALUATION_REQUEST_SCHEMA_VERSION = "1.0"
RUNNER_DEFINITION_CONTRACT = "eval360.runner-definition"
SUITE_CATALOG_CONTRACT = "eval360.suite-catalog"
SUITE_DEFINITION_CONTRACT = "eval360.suite"
OWNER_MANIFEST_SCHEMA_VERSION = "1.0"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        raise ValueError(f"{label} has invalid fields ({'; '.join(details)})")
    return value


def _array(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    if nonempty and not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_object(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_OBJECT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git object ID")
    return value


def _absolute_path(value: Any, label: str) -> str:
    path = Path(_text(value, label)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return str(path)


def _relative_path(value: Any, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or text != path.as_posix() or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise ValueError(f"{label} must be a normalized contained relative path")
    return text


def _unique(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


@dataclass(frozen=True)
class SourceIdentity:
    """Immutable source repository identity declared by an owner manifest."""

    repository: str
    commit: str
    tree: str

    @classmethod
    def from_value(cls, value: Any, label: str) -> SourceIdentity:
        data = _object(value, {"commit", "repository", "tree"}, label)
        return cls(
            repository=_text(data["repository"], f"{label} repository"),
            commit=_git_object(data["commit"], f"{label} commit"),
            tree=_git_object(data["tree"], f"{label} tree"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "commit": self.commit,
            "repository": self.repository,
            "tree": self.tree,
        }


@dataclass(frozen=True)
class ContractReference:
    """One exact machine-readable contract name and schema version."""

    contract: str
    schema_version: str

    @classmethod
    def from_value(cls, value: Any, label: str) -> ContractReference:
        data = _object(value, {"contract", "schema_version"}, label)
        return cls(
            contract=_text(data["contract"], f"{label} contract"),
            schema_version=_text(
                data["schema_version"], f"{label} schema_version"
            ),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "contract": self.contract,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class PayloadFile:
    """Portable owner-declared identity for one dataset payload file."""

    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_value(cls, value: Any, label: str) -> PayloadFile:
        data = _object(value, {"path", "sha256", "size_bytes"}, label)
        return cls(
            path=_relative_path(data["path"], f"{label} path"),
            size_bytes=_integer(
                data["size_bytes"], f"{label} size_bytes", minimum=0
            ),
            sha256=_digest(data["sha256"], f"{label} sha256"),
        )

    def as_dict(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RunnerDefinition:
    """Strict ``eval360.runner-definition/1.0`` owner manifest."""

    runner_id: str
    source: SourceIdentity
    entrypoint: PayloadFile
    request_contract: ContractReference
    terminal_result_contract: ContractReference

    @classmethod
    def from_value(cls, value: Any) -> RunnerDefinition:
        data = _object(
            value,
            {
                "contract",
                "entrypoint",
                "request_contract",
                "runner_id",
                "schema_version",
                "source",
                "terminal_result_contract",
            },
            "runner definition",
        )
        if data["contract"] != RUNNER_DEFINITION_CONTRACT:
            raise ValueError("runner definition contract is unsupported")
        if data["schema_version"] != OWNER_MANIFEST_SCHEMA_VERSION:
            raise ValueError("runner definition schema_version is unsupported")
        request_contract = ContractReference.from_value(
            data["request_contract"], "runner request contract"
        )
        if request_contract != ContractReference(
            EVALUATION_REQUEST_CONTRACT,
            EVALUATION_REQUEST_SCHEMA_VERSION,
        ):
            raise ValueError("runner definition declares the wrong request contract")
        terminal_contract = ContractReference.from_value(
            data["terminal_result_contract"], "runner terminal-result contract"
        )
        if terminal_contract != ContractReference(
            TERMINAL_RESULT_CONTRACT,
            TERMINAL_RESULT_SCHEMA_VERSION,
        ):
            raise ValueError(
                "runner definition declares the wrong terminal-result contract"
            )
        return cls(
            runner_id=_text(data["runner_id"], "runner definition runner_id"),
            source=SourceIdentity.from_value(
                data["source"], "runner definition source"
            ),
            entrypoint=PayloadFile.from_value(
                data["entrypoint"], "runner definition entrypoint"
            ),
            request_contract=request_contract,
            terminal_result_contract=terminal_contract,
        )


@dataclass(frozen=True)
class CatalogSuite:
    """Exact suite manifest reference declared by a catalog."""

    suite_id: str
    version: str
    manifest_path: str
    manifest_size_bytes: int
    manifest_sha256: str

    @classmethod
    def from_value(cls, value: Any, label: str) -> CatalogSuite:
        data = _object(
            value,
            {
                "manifest_path",
                "manifest_sha256",
                "manifest_size_bytes",
                "suite_id",
                "version",
            },
            label,
        )
        return cls(
            suite_id=_text(data["suite_id"], f"{label} suite_id"),
            version=_text(data["version"], f"{label} version"),
            manifest_path=_relative_path(
                data["manifest_path"], f"{label} manifest_path"
            ),
            manifest_size_bytes=_integer(
                data["manifest_size_bytes"],
                f"{label} manifest_size_bytes",
                minimum=0,
            ),
            manifest_sha256=_digest(
                data["manifest_sha256"], f"{label} manifest_sha256"
            ),
        )


@dataclass(frozen=True)
class SuiteCatalog:
    """Strict ``eval360.suite-catalog/1.0`` owner manifest."""

    catalog_id: str
    owner: str
    source: SourceIdentity
    suites: tuple[CatalogSuite, ...]

    @classmethod
    def from_value(cls, value: Any) -> SuiteCatalog:
        data = _object(
            value,
            {
                "catalog_id",
                "contract",
                "owner",
                "schema_version",
                "source",
                "suites",
            },
            "suite catalog",
        )
        if data["contract"] != SUITE_CATALOG_CONTRACT:
            raise ValueError("suite catalog contract is unsupported")
        if data["schema_version"] != OWNER_MANIFEST_SCHEMA_VERSION:
            raise ValueError("suite catalog schema_version is unsupported")
        suites = tuple(
            CatalogSuite.from_value(item, f"suite catalog entry {ordinal}")
            for ordinal, item in enumerate(
                _array(data["suites"], "suite catalog suites", nonempty=True)
            )
        )
        refs = [(item.suite_id, item.version) for item in suites]
        _unique([f"{suite_id}\0{version}" for suite_id, version in refs], "suite catalog refs")
        if refs != sorted(refs):
            raise ValueError("suite catalog entries must use canonical ref order")
        return cls(
            catalog_id=_text(data["catalog_id"], "suite catalog catalog_id"),
            owner=_text(data["owner"], "suite catalog owner"),
            source=SourceIdentity.from_value(data["source"], "suite catalog source"),
            suites=suites,
        )

    def select(self, suite_id: str, version: str) -> CatalogSuite:
        matches = [
            item
            for item in self.suites
            if item.suite_id == suite_id and item.version == version
        ]
        if len(matches) != 1:
            raise ValueError(
                f"suite catalog does not select exactly one {suite_id}@{version}"
            )
        return matches[0]


@dataclass(frozen=True)
class SuiteModelSettings:
    """Suite-owned generation and parsing settings."""

    max_simultaneous_requests: int
    max_time_to_deploy: int
    allow_long_max_model_len: bool
    vllm_cli_args: tuple[str, ...]
    vllm_logging_level: str
    openai_kwargs: dict[str, Any]
    cache_salt: dict[str, Any]
    parser_type: str
    prompt_prefix_instructions: str | None

    @classmethod
    def from_value(cls, value: Any) -> SuiteModelSettings:
        data = _object(
            value,
            {
                "allow_long_max_model_len",
                "cache_salt",
                "max_simultaneous_requests",
                "max_time_to_deploy",
                "openai_kwargs",
                "parser_type",
                "prompt_prefix_instructions",
                "vllm_cli_args",
                "vllm_logging_level",
            },
            "suite model settings",
        )
        raw_args = _array(data["vllm_cli_args"], "suite model vllm_cli_args")
        args = tuple(
            _text(item, f"suite model vllm_cli_args[{index}]")
            for index, item in enumerate(raw_args)
        )
        if not isinstance(data["openai_kwargs"], dict):
            raise TypeError("suite model openai_kwargs must be an object")
        cache_salt = _object(
            data["cache_salt"], {"mode", "salt"}, "suite model cache_salt"
        )
        return cls(
            max_simultaneous_requests=_integer(
                data["max_simultaneous_requests"],
                "suite model max_simultaneous_requests",
                minimum=1,
            ),
            max_time_to_deploy=_integer(
                data["max_time_to_deploy"],
                "suite model max_time_to_deploy",
                minimum=1,
            ),
            allow_long_max_model_len=_boolean(
                data["allow_long_max_model_len"],
                "suite model allow_long_max_model_len",
            ),
            vllm_cli_args=args,
            vllm_logging_level=_text(
                data["vllm_logging_level"], "suite model vllm_logging_level"
            ),
            openai_kwargs=data["openai_kwargs"],
            cache_salt=cache_salt,
            parser_type=_text(data["parser_type"], "suite model parser_type"),
            prompt_prefix_instructions=_optional_text(
                data["prompt_prefix_instructions"],
                "suite model prompt_prefix_instructions",
            ),
        )


@dataclass(frozen=True)
class SuiteTask:
    """One exact ordered suite task and its immutable dataset closure."""

    task_id: str
    dataset_name: str
    semantic_version: str
    mode: str
    openai_settings: dict[str, Any] | None
    meta: dict[str, Any] | None
    enabled: bool
    tag: str
    average_over: tuple[int, ...]
    pass_at: tuple[int, ...]
    num_generations: int
    grader_type: str
    dataset_files: tuple[PayloadFile, ...]

    @classmethod
    def from_value(cls, value: Any, ordinal: int) -> SuiteTask:
        label = f"suite task {ordinal}"
        data = _object(
            value,
            {
                "average_over",
                "dataset_files",
                "dataset_name",
                "enabled",
                "grader",
                "id",
                "meta",
                "mode",
                "num_generations",
                "openai_settings",
                "pass_at",
                "semantic_version",
                "tag",
            },
            label,
        )
        grader = _object(
            data["grader"], {"llm_as_judge", "type"}, f"{label} grader"
        )
        if grader["llm_as_judge"] is not None:
            raise ValueError(
                "suite schema 1.0 does not support an llm_as_judge model; "
                "publish a later owner schema before selecting one"
            )
        for field in ("openai_settings", "meta"):
            if data[field] is not None and not isinstance(data[field], dict):
                raise TypeError(f"{label} {field} must be an object or null")

        def positive_list(field: str) -> tuple[int, ...]:
            raw = _array(data[field], f"{label} {field}", nonempty=True)
            return tuple(
                _integer(item, f"{label} {field}[{index}]", minimum=1)
                for index, item in enumerate(raw)
            )

        files = tuple(
            PayloadFile.from_value(item, f"{label} dataset file {index}")
            for index, item in enumerate(
                _array(data["dataset_files"], f"{label} dataset_files", nonempty=True)
            )
        )
        _unique([item.path for item in files], f"{label} dataset paths")
        return cls(
            task_id=_text(data["id"], f"{label} id"),
            dataset_name=_text(data["dataset_name"], f"{label} dataset_name"),
            semantic_version=_text(
                data["semantic_version"], f"{label} semantic_version"
            ),
            mode=_text(data["mode"], f"{label} mode"),
            openai_settings=data["openai_settings"],
            meta=data["meta"],
            enabled=_boolean(data["enabled"], f"{label} enabled"),
            tag=_text(data["tag"], f"{label} tag"),
            average_over=positive_list("average_over"),
            pass_at=positive_list("pass_at"),
            num_generations=_integer(
                data["num_generations"], f"{label} num_generations", minimum=1
            ),
            grader_type=_text(grader["type"], f"{label} grader type"),
            dataset_files=files,
        )

    def task_mapping(self, data_path: str) -> dict[str, Any]:
        return {
            "average_over": list(self.average_over),
            "data_path": data_path,
            "dataset_name": self.dataset_name,
            "enabled": self.enabled,
            "grader": {"llm_as_judge": None, "type": self.grader_type},
            "meta": self.meta,
            "mode": self.mode,
            "num_generations": self.num_generations,
            "openai_settings": self.openai_settings,
            "pass_at": list(self.pass_at),
            "semantic_version": self.semantic_version,
            "tag": self.tag,
            "uuid": self.task_id,
        }

    def as_dict(self) -> dict[str, Any]:
        value = self.task_mapping("__request_dataset_closure__")
        value.pop("data_path")
        value["id"] = value.pop("uuid")
        value["dataset_files"] = [item.as_dict() for item in self.dataset_files]
        return value


@dataclass(frozen=True)
class SuiteDefinition:
    """Strict ``eval360.suite/1.0`` owner manifest."""

    suite_id: str
    version: str
    owner: str
    source: SourceIdentity
    model_family: str
    model_type: str
    serving_capability_id: str
    serving_manifest_sha256: str
    model: SuiteModelSettings
    tasks: tuple[SuiteTask, ...]

    @classmethod
    def from_value(cls, value: Any) -> SuiteDefinition:
        data = _object(
            value,
            {
                "compatibility",
                "contract",
                "model",
                "owner",
                "schema_version",
                "serving_runtime",
                "source",
                "suite_id",
                "tasks",
                "version",
            },
            "suite definition",
        )
        if data["contract"] != SUITE_DEFINITION_CONTRACT:
            raise ValueError("suite definition contract is unsupported")
        if data["schema_version"] != OWNER_MANIFEST_SCHEMA_VERSION:
            raise ValueError("suite definition schema_version is unsupported")
        compatibility = _object(
            data["compatibility"],
            {"model_family", "model_type"},
            "suite compatibility",
        )
        model_type = _text(
            compatibility["model_type"], "suite compatibility model_type"
        )
        if model_type not in {"base", "instruct"}:
            raise ValueError("suite compatibility model_type is unsupported")
        serving = _object(
            data["serving_runtime"],
            {"capability_id", "manifest_sha256"},
            "suite serving_runtime",
        )
        tasks = tuple(
            SuiteTask.from_value(item, ordinal)
            for ordinal, item in enumerate(
                _array(data["tasks"], "suite tasks", nonempty=True)
            )
        )
        _unique([item.task_id for item in tasks], "suite task IDs")
        return cls(
            suite_id=_text(data["suite_id"], "suite definition suite_id"),
            version=_text(data["version"], "suite definition version"),
            owner=_text(data["owner"], "suite definition owner"),
            source=SourceIdentity.from_value(data["source"], "suite definition source"),
            model_family=_text(
                compatibility["model_family"],
                "suite compatibility model_family",
            ),
            model_type=model_type,
            serving_capability_id=_text(
                serving["capability_id"], "suite serving capability_id"
            ),
            serving_manifest_sha256=_digest(
                serving["manifest_sha256"],
                "suite serving manifest_sha256",
            ),
            model=SuiteModelSettings.from_value(data["model"]),
            tasks=tasks,
        )


def _expected_binding(value: Any, label: str) -> dict[str, str | int]:
    data = _object(
        value,
        {"configured_path", "resolved_path", "sha256", "size_bytes"},
        label,
    )
    return {
        "configured_path": _absolute_path(
            data["configured_path"], f"{label} configured_path"
        ),
        "resolved_path": _absolute_path(
            data["resolved_path"], f"{label} resolved_path"
        ),
        "sha256": _digest(data["sha256"], f"{label} sha256"),
        "size_bytes": _integer(
            data["size_bytes"], f"{label} size_bytes", minimum=0
        ),
    }


def _bind_expected(value: Any, label: str) -> FileBinding:
    expected = _expected_binding(value, label)
    observed = bind_regular_file(expected["configured_path"], label=label)
    if observed.as_dict() != expected:
        raise ValueError(f"{label} changed or differs from its request binding")
    return observed


def _load_expected_manifest(
    value: Any,
    label: str,
) -> tuple[FileBinding, dict[str, Any]]:
    expected = _expected_binding(value, label)
    observed, parsed = bind_canonical_json_object(
        expected["configured_path"], label=label
    )
    if observed.as_dict() != expected:
        raise ValueError(f"{label} changed or differs from its request binding")
    return observed, parsed


def _path_under(root: str, path: str, label: str) -> None:
    try:
        Path(path).relative_to(Path(root))
    except ValueError as exc:
        raise ValueError(f"{label} must be beneath its configured root") from exc


def _resolve_terminal_path(value: str | os.PathLike[str]) -> tuple[str, Path]:
    configured = os.fspath(value)
    expanded = Path(configured).expanduser()
    if not expanded.name:
        raise ValueError("terminal-result path must name a file")
    try:
        parent = expanded.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"terminal-result parent does not exist: {expanded.parent}"
        ) from exc
    if not parent.is_dir():
        raise ValueError(
            f"terminal-result parent is not a directory: {expanded.parent}"
        )
    resolved = parent / expanded.name
    if os.path.lexists(resolved):
        raise FileExistsError(f"terminal-result path already exists: {resolved}")
    return configured, resolved


@dataclass(frozen=True)
class EvaluationRequest:
    """Validated immutable request plus in-memory Eval360 execution objects."""

    configured_output_path: str
    output_path: Path
    request_binding: FileBinding
    runner_entrypoint: FileBinding
    runner_definition_binding: FileBinding
    suite_catalog_binding: FileBinding
    suite_binding: FileBinding
    release_manifest: FileBinding
    serving_manifest: FileBinding
    release_payloads: tuple[FileBinding, ...]
    serving_payloads: tuple[FileBinding, ...]
    dataset_bindings: dict[str, tuple[FileBinding, ...]]
    runner_definition: RunnerDefinition
    suite_catalog: SuiteCatalog
    suite: SuiteDefinition
    model_spec: ModelSpec
    tasks: tuple[AsyncGenerationTask, ...]
    dataset_paths: dict[str, tuple[str, ...]]
    task_closure_sha256: dict[str, str]
    raw_execution: dict[str, Any]

    @classmethod
    def load(
        cls,
        *,
        request_path: str | os.PathLike[str],
        terminal_result_path: str | os.PathLike[str],
        actual_runner_entrypoint: str | os.PathLike[str],
    ) -> EvaluationRequest:
        """Load one canonical request and validate its entire immutable closure."""
        configured_output, output_path = _resolve_terminal_path(terminal_result_path)
        request_binding, raw = bind_canonical_json_object(
            request_path, label="evaluation request"
        )
        request = _object(
            raw,
            {"bindings", "contract", "execution", "schema_version", "suite", "tasks"},
            "evaluation request",
        )
        if request["contract"] != EVALUATION_REQUEST_CONTRACT:
            raise ValueError("evaluation request contract is unsupported")
        if request["schema_version"] != EVALUATION_REQUEST_SCHEMA_VERSION:
            raise ValueError("evaluation request schema_version is unsupported")

        bindings = _object(
            request["bindings"],
            {
                "release_manifest",
                "release_payloads",
                "runner_definition",
                "runner_entrypoint",
                "serving_manifest",
                "serving_payloads",
                "suite",
                "suite_catalog",
            },
            "evaluation request bindings",
        )
        runner_entrypoint = _bind_expected(
            bindings["runner_entrypoint"], "runner entrypoint"
        )
        actual_entrypoint = bind_regular_file(
            actual_runner_entrypoint, label="actual runner entrypoint"
        )
        if actual_entrypoint.as_dict() != runner_entrypoint.as_dict():
            raise ValueError(
                "actual runner entrypoint differs from the evaluation request"
            )

        runner_definition_binding, raw_runner = _load_expected_manifest(
            bindings["runner_definition"], "runner definition"
        )
        runner_definition = RunnerDefinition.from_value(raw_runner)
        declared_entrypoint = runner_definition.entrypoint
        if (
            declared_entrypoint.size_bytes != runner_entrypoint.size_bytes
            or declared_entrypoint.sha256 != runner_entrypoint.sha256
        ):
            raise ValueError(
                "runner definition entrypoint differs from the actual executable"
            )
        expected_entrypoint = (
            Path(runner_definition_binding.configured_path).parent
            / declared_entrypoint.path
        )
        if str(expected_entrypoint) != runner_entrypoint.configured_path:
            raise ValueError(
                "runner definition entrypoint path differs from the actual executable"
            )

        catalog_binding, raw_catalog = _load_expected_manifest(
            bindings["suite_catalog"], "suite catalog"
        )
        catalog = SuiteCatalog.from_value(raw_catalog)
        suite_binding, raw_suite = _load_expected_manifest(
            bindings["suite"], "suite definition"
        )
        suite = SuiteDefinition.from_value(raw_suite)
        suite_ref = _object(
            request["suite"], {"id", "version"}, "evaluation request suite"
        )
        selected_id = _text(suite_ref["id"], "evaluation request suite id")
        selected_version = _text(
            suite_ref["version"], "evaluation request suite version"
        )
        if suite.suite_id != selected_id or suite.version != selected_version:
            raise ValueError("suite definition identity differs from request selection")
        selected_catalog_entry = catalog.select(selected_id, selected_version)
        expected_suite_path = (
            Path(catalog_binding.configured_path).parent
            / selected_catalog_entry.manifest_path
        )
        if (
            str(expected_suite_path) != suite_binding.configured_path
            or selected_catalog_entry.manifest_sha256 != suite_binding.sha256
            or selected_catalog_entry.manifest_size_bytes != suite_binding.size_bytes
        ):
            raise ValueError("suite catalog selected a different suite manifest")

        release_manifest, _ = _load_expected_manifest(
            bindings["release_manifest"], "release manifest"
        )
        serving_manifest, _ = _load_expected_manifest(
            bindings["serving_manifest"], "serving manifest"
        )
        if suite.serving_manifest_sha256 != serving_manifest.sha256:
            raise ValueError(
                "suite serving-runtime requirement differs from request manifest"
            )

        release_payloads = tuple(
            _bind_expected(item, f"release payload {ordinal}")
            for ordinal, item in enumerate(
                _array(
                    bindings["release_payloads"],
                    "evaluation request release_payloads",
                    nonempty=True,
                )
            )
        )
        serving_payloads = tuple(
            _bind_expected(item, f"serving payload {ordinal}")
            for ordinal, item in enumerate(
                _array(
                    bindings["serving_payloads"],
                    "evaluation request serving_payloads",
                    nonempty=True,
                )
            )
        )
        _unique(
            [item.configured_path for item in release_payloads],
            "release payload paths",
        )
        _unique(
            [item.configured_path for item in serving_payloads],
            "serving payload paths",
        )

        execution = _object(
            request["execution"],
            {
                "dataset_root",
                "model_family",
                "model_name",
                "model_revision",
                "output_path",
                "owner",
                "release_path",
                "serving",
            },
            "evaluation request execution",
        )
        dataset_root = _absolute_path(
            execution["dataset_root"], "evaluation request dataset_root"
        )
        release_path = _absolute_path(
            execution["release_path"], "evaluation request release_path"
        )
        output_root = _absolute_path(
            execution["output_path"], "evaluation request output_path"
        )
        model_family = _text(
            execution["model_family"], "evaluation request model_family"
        )
        if model_family != suite.model_family:
            raise ValueError("request model_family is incompatible with suite")
        _path_under(release_path, release_manifest.configured_path, "release manifest")
        for ordinal, item in enumerate(release_payloads):
            _path_under(release_path, item.configured_path, f"release payload {ordinal}")

        serving = _object(
            execution["serving"],
            {"entry_path", "kind", "root_path"},
            "evaluation request serving",
        )
        serving_root = _absolute_path(
            serving["root_path"], "evaluation request serving root_path"
        )
        serving_entry = _relative_path(
            serving["entry_path"], "evaluation request serving entry_path"
        )
        serving_kind = _text(serving["kind"], "evaluation request serving kind")
        if serving_kind not in {"venv", "container"}:
            raise ValueError("evaluation request serving kind is unsupported")
        serving_execution_path = str(Path(serving_root) / serving_entry)
        if serving_execution_path not in {
            item.configured_path for item in serving_payloads
        }:
            raise ValueError(
                "serving execution path is missing from serving payload bindings"
            )
        for ordinal, item in enumerate(serving_payloads):
            _path_under(serving_root, item.configured_path, f"serving payload {ordinal}")

        raw_request_tasks = _array(
            request["tasks"], "evaluation request tasks", nonempty=True
        )
        if len(raw_request_tasks) != len(suite.tasks):
            raise ValueError(
                "evaluation request ordered task closure differs from suite"
            )
        request_task_ids = []
        parsed_tasks: list[AsyncGenerationTask] = []
        dataset_bindings: dict[str, tuple[FileBinding, ...]] = {}
        dataset_paths: dict[str, tuple[str, ...]] = {}
        task_closure_sha256: dict[str, str] = {}
        for ordinal, (raw_task, suite_task) in enumerate(
            zip(raw_request_tasks, suite.tasks, strict=True)
        ):
            selected = _object(
                raw_task,
                {"dataset_files", "id"},
                f"evaluation request task {ordinal}",
            )
            task_id = _text(selected["id"], f"evaluation request task {ordinal} id")
            request_task_ids.append(task_id)
            if task_id != suite_task.task_id:
                raise ValueError(
                    "evaluation request ordered task closure differs from suite"
                )
            raw_files = _array(
                selected["dataset_files"],
                f"evaluation request task {ordinal} dataset_files",
                nonempty=True,
            )
            if len(raw_files) != len(suite_task.dataset_files):
                raise ValueError(
                    f"task {task_id} dataset closure differs from suite"
                )
            observed_files = []
            for file_ordinal, (raw_file, expected_file) in enumerate(
                zip(raw_files, suite_task.dataset_files, strict=True)
            ):
                observed = _bind_expected(
                    raw_file,
                    f"task {task_id} dataset file {file_ordinal}",
                )
                expected_path = str(Path(dataset_root) / expected_file.path)
                if (
                    observed.configured_path != expected_path
                    or observed.size_bytes != expected_file.size_bytes
                    or observed.sha256 != expected_file.sha256
                ):
                    raise ValueError(
                        f"task {task_id} dataset binding differs from suite"
                    )
                observed_files.append(observed)
            task_file_bindings = tuple(observed_files)
            task_paths = tuple(item.configured_path for item in task_file_bindings)
            actual_count = sum(count_jsonl_records(path) for path in task_paths)
            if actual_count != suite_task.num_generations:
                raise ValueError(
                    f"task {task_id} dataset total {actual_count} differs from "
                    f"suite total {suite_task.num_generations}"
                )
            task = AsyncGenerationTask.model_validate(
                suite_task.task_mapping(task_paths[0])
            )
            parsed_tasks.append(task)
            dataset_bindings[task_id] = task_file_bindings
            dataset_paths[task_id] = task_paths
            closure = {
                "dataset_files": [item.as_dict() for item in task_file_bindings],
                "suite_task": suite_task.as_dict(),
            }
            task_closure_sha256[task_id] = hashlib.sha256(
                canonical_json(closure).encode("utf-8")
            ).hexdigest()
        _unique(request_task_ids, "evaluation request task IDs")

        model_name = _text(
            execution["model_name"], "evaluation request model_name"
        )
        model_revision = _optional_text(
            execution["model_revision"], "evaluation request model_revision"
        )
        settings = suite.model
        model_mapping: dict[str, Any] = {
            "allow_long_max_model_len": settings.allow_long_max_model_len,
            "cache_salt": settings.cache_salt,
            "container_image": (
                serving_execution_path if serving_kind == "container" else None
            ),
            "container_mounts": None,
            "max_simultaneous_requests": settings.max_simultaneous_requests,
            "max_time_to_deploy": settings.max_time_to_deploy,
            "model_type": suite.model_type,
            "name_modifier": None,
            "openai_kwargs": settings.openai_kwargs,
            "output_path": output_root,
            "owner": _text(execution["owner"], "evaluation request owner"),
            "parser_type": settings.parser_type,
            "prompt_prefix_instructions": settings.prompt_prefix_instructions,
            "ready": True,
            "remote_model": {
                "base_name": model_name,
                "path": release_path,
                "revision": model_revision,
            },
            "tag": "any",
            "venv_path": serving_execution_path if serving_kind == "venv" else None,
            "vllm_cli_args": list(settings.vllm_cli_args),
            "vllm_logging_level": settings.vllm_logging_level,
        }
        model_spec = ModelSpec.model_validate(model_mapping)

        raw_execution = {
            "dataset_root": dataset_root,
            "model_family": model_family,
            "model_name": model_name,
            "model_revision": model_revision,
            "output_path": output_root,
            "owner": execution["owner"],
            "release_path": release_path,
            "serving": {
                "entry_path": serving_entry,
                "kind": serving_kind,
                "root_path": serving_root,
            },
        }
        result = cls(
            configured_output_path=configured_output,
            output_path=output_path,
            request_binding=request_binding,
            runner_entrypoint=runner_entrypoint,
            runner_definition_binding=runner_definition_binding,
            suite_catalog_binding=catalog_binding,
            suite_binding=suite_binding,
            release_manifest=release_manifest,
            serving_manifest=serving_manifest,
            release_payloads=release_payloads,
            serving_payloads=serving_payloads,
            dataset_bindings=dataset_bindings,
            runner_definition=runner_definition,
            suite_catalog=catalog,
            suite=suite,
            model_spec=model_spec,
            tasks=tuple(parsed_tasks),
            dataset_paths=dataset_paths,
            task_closure_sha256=task_closure_sha256,
            raw_execution=raw_execution,
        )
        result.verify_inputs()
        return result

    def verify_inputs(self) -> None:
        """Re-query every immutable request, manifest, and payload binding."""
        verify_file_binding(
            self.request_binding,
            label="evaluation request",
            require_canonical_json=True,
        )
        verify_file_binding(self.runner_entrypoint, label="runner entrypoint")
        for label, binding in (
            ("runner definition", self.runner_definition_binding),
            ("suite catalog", self.suite_catalog_binding),
            ("suite definition", self.suite_binding),
            ("release manifest", self.release_manifest),
            ("serving manifest", self.serving_manifest),
        ):
            verify_file_binding(binding, label=label, require_canonical_json=True)
        for ordinal, binding in enumerate(self.release_payloads):
            verify_file_binding(binding, label=f"release payload {ordinal}")
        for ordinal, binding in enumerate(self.serving_payloads):
            verify_file_binding(binding, label=f"serving payload {ordinal}")
        for task_id, bindings in self.dataset_bindings.items():
            for ordinal, binding in enumerate(bindings):
                verify_file_binding(
                    binding,
                    label=f"task {task_id} dataset file {ordinal}",
                )

    def bindings_dict(self) -> dict[str, Any]:
        """Return every exact immutable input binding for terminal evidence."""
        return {
            "evaluation_request": self.request_binding.as_dict(),
            "release_manifest": self.release_manifest.as_dict(),
            "release_payloads": [item.as_dict() for item in self.release_payloads],
            "runner_definition": self.runner_definition_binding.as_dict(),
            "runner_entrypoint": self.runner_entrypoint.as_dict(),
            "serving_manifest": self.serving_manifest.as_dict(),
            "serving_payloads": [item.as_dict() for item in self.serving_payloads],
            "suite": self.suite_binding.as_dict(),
            "suite_catalog": self.suite_catalog_binding.as_dict(),
        }

    def event_request_metadata(self, task_id: str) -> dict[str, str]:
        """Return the exact request/task identity persisted for safe resume."""
        return {
            "request_sha256": self.request_binding.sha256,
            "task_closure_sha256": self.task_closure_sha256[task_id],
        }
