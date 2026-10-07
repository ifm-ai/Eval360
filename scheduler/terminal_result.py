"""Write-last evidence helpers for one ``evaluate-now`` invocation.

The terminal result is deliberately opt-in.  It is a success artifact, not an
attempt log: callers get no terminal-result file unless every selected event,
required output, and applicable Slurm child has been reconciled successfully.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from .evaluation_request import EvaluationRequest

TERMINAL_RESULT_CONTRACT = "eval360.evaluate-now-terminal-result"
TERMINAL_RESULT_SCHEMA_VERSION = "1.0"
YAML_TERMINAL_RESULT_CONTRACT = "eval360.evaluate-now-yaml-terminal-result"
YAML_TERMINAL_RESULT_SCHEMA_VERSION = "1.0"
_HASH_CHUNK_BYTES = 1024 * 1024


def canonical_json(value: Any) -> str:
    """Render deterministic UTF-8 JSON with exactly one trailing newline."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"unsupported JSON constant: {token}")


def _parse_canonical_manifest(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 canonical JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be canonical JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a canonical JSON object")
    if text != canonical_json(value):
        raise ValueError(f"{label} must be canonical JSON")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class FileBinding:
    """Stable identity of one exact regular file."""

    configured_path: str
    resolved_path: str
    size_bytes: int
    sha256: str
    _identity: tuple[int, int, int, int, int]

    def as_dict(self) -> dict[str, str | int]:
        """Return the public, machine-readable binding."""
        return {
            "configured_path": self.configured_path,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _read_regular_file(
    configured_path: str | os.PathLike[str],
    *,
    label: str,
    capture_bytes: bool,
) -> tuple[FileBinding, bytes | None]:
    """Hash one stable regular file and optionally return the exact bytes read."""
    configured = os.fspath(configured_path)
    try:
        resolved = Path(configured).expanduser().resolve(strict=True)
        before = resolved.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect {label} {configured!r}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must resolve to a regular file: {configured!r}")

    descriptor = -1
    digest = hashlib.sha256()
    captured = bytearray() if capture_bytes else None
    try:
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} changed type while it was opened")
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            if captured is not None:
                captured.extend(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"cannot read {label} {configured!r}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if _stat_identity(before) != _stat_identity(opened) or _stat_identity(
        opened
    ) != _stat_identity(after):
        raise ValueError(f"{label} changed while it was read")
    binding = FileBinding(
        configured_path=configured,
        resolved_path=str(resolved),
        size_bytes=opened.st_size,
        sha256=digest.hexdigest(),
        _identity=_stat_identity(opened),
    )
    return binding, bytes(captured) if captured is not None else None


def bind_regular_file(
    configured_path: str | os.PathLike[str],
    *,
    label: str,
    require_canonical_json: bool = False,
) -> FileBinding:
    """Hash one stable regular file and optionally require canonical JSON."""
    binding, raw = _read_regular_file(
        configured_path,
        label=label,
        capture_bytes=require_canonical_json,
    )
    if raw is not None:
        _parse_canonical_manifest(raw, label)
    return binding


def bind_canonical_json_object(
    configured_path: str | os.PathLike[str],
    *,
    label: str,
) -> tuple[FileBinding, dict[str, Any]]:
    """Bind and parse one canonical JSON object from the same stable read."""
    binding, raw = _read_regular_file(
        configured_path,
        label=label,
        capture_bytes=True,
    )
    assert raw is not None
    return binding, _parse_canonical_manifest(raw, label)


def verify_file_binding(
    binding: FileBinding,
    *,
    label: str,
    require_canonical_json: bool = False,
) -> None:
    """Fail when a bound input no longer names the exact bytes selected."""
    current = bind_regular_file(
        binding.configured_path,
        label=label,
        require_canonical_json=require_canonical_json,
    )
    if current != binding:
        raise ValueError(f"{label} changed after it was selected")


def _resolve_terminal_path(
    value: str | os.PathLike[str],
) -> tuple[str, Path]:
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


def _runner_entrypoint_path(value: str | os.PathLike[str]) -> str:
    configured = os.fspath(value)
    expanded = Path(configured).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        return str(expanded)
    resolved = shutil.which(configured)
    if resolved is None:
        raise ValueError(f"cannot resolve runner entrypoint: {configured!r}")
    return resolved


@dataclass(frozen=True)
class YamlTerminalInvocation:
    """Bound ordinary-YAML invocation used only to produce terminal evidence."""

    configured_output_path: str
    output_path: Path
    runner_entrypoint: FileBinding
    model_configs: tuple[FileBinding, ...]
    data_configs: tuple[FileBinding, ...]
    eval_configs: tuple[FileBinding, ...]

    @classmethod
    def bind(
        cls,
        *,
        model_paths: list[str] | tuple[str, ...],
        data_paths: list[str] | tuple[str, ...],
        eval_paths: list[str] | tuple[str, ...] | None,
        terminal_result_path: str | os.PathLike[str],
        actual_runner_entrypoint: str | os.PathLike[str],
    ) -> YamlTerminalInvocation:
        """Bind the exact executable and ordered YAML sources before parsing."""
        if not model_paths or not data_paths or not eval_paths:
            raise ValueError(
                "YAML terminal evidence requires model, data, and eval configs"
            )
        configured_output, output_path = _resolve_terminal_path(
            terminal_result_path
        )
        runner_entrypoint = bind_regular_file(
            _runner_entrypoint_path(actual_runner_entrypoint),
            label="runner entrypoint",
        )

        def bind_many(paths, role: str) -> tuple[FileBinding, ...]:
            return tuple(
                bind_regular_file(path, label=f"{role} config {ordinal}")
                for ordinal, path in enumerate(paths)
            )

        return cls(
            configured_output_path=configured_output,
            output_path=output_path,
            runner_entrypoint=runner_entrypoint,
            model_configs=bind_many(model_paths, "model"),
            data_configs=bind_many(data_paths, "data"),
            eval_configs=bind_many(eval_paths, "eval"),
        )

    def verify_inputs(self) -> None:
        """Verify that every executable/config byte remains the selected byte."""
        verify_file_binding(self.runner_entrypoint, label="runner entrypoint")
        for role, bindings in (
            ("model", self.model_configs),
            ("data", self.data_configs),
            ("eval", self.eval_configs),
        ):
            for ordinal, binding in enumerate(bindings):
                verify_file_binding(
                    binding,
                    label=f"{role} config {ordinal}",
                )

    def verify_config_paths(
        self,
        *,
        model_paths: list[str] | tuple[str, ...],
        data_paths: list[str] | tuple[str, ...],
        eval_paths: list[str] | tuple[str, ...],
    ) -> None:
        """Require the parser inputs to be the exact bound ordered paths."""
        for role, actual, expected in (
            ("model", model_paths, self.model_configs),
            ("data", data_paths, self.data_configs),
            ("eval", eval_paths, self.eval_configs),
        ):
            if tuple(os.fspath(path) for path in actual) != tuple(
                item.configured_path for item in expected
            ):
                raise ValueError(
                    f"{role} parser paths differ from the bound YAML invocation"
                )

    def bindings_dict(self) -> dict[str, object]:
        """Return exact ordered public bindings for the output contract."""
        return {
            "data_configs": [item.as_dict() for item in self.data_configs],
            "eval_configs": [item.as_dict() for item in self.eval_configs],
            "model_configs": [item.as_dict() for item in self.model_configs],
            "runner_entrypoint": self.runner_entrypoint.as_dict(),
        }

    def config_binding(self, role: str, path: str) -> FileBinding:
        """Return the unique invocation binding for one configured YAML path."""
        bindings = {
            "model": self.model_configs,
            "data": self.data_configs,
            "eval": self.eval_configs,
        }[role]
        matches = [item for item in bindings if item.configured_path == path]
        if len(matches) != 1:
            raise ValueError(
                f"selected {role} config {path!r} occurs {len(matches)} times "
                "in the YAML invocation"
            )
        return matches[0]


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("terminal-result write made no progress")
        offset += written


def _verify_public_binding(record: Any, *, label: str) -> None:
    if not isinstance(record, dict):
        raise TypeError(f"terminal result {label} binding must be an object")
    keys = ("configured_path", "resolved_path", "sha256", "size_bytes")
    try:
        expected = {key: record[key] for key in keys}
    except KeyError as exc:
        raise ValueError(
            f"terminal result {label} binding is missing {exc.args[0]!r}"
        ) from exc
    current = bind_regular_file(expected["configured_path"], label=label)
    if current.as_dict() != expected:
        raise ValueError(f"terminal result {label} changed after it was recorded")


def _load_public_yaml_binding(record: dict[str, Any], *, label: str) -> Any:
    """Read YAML from the exact stable file identity recorded in a binding."""
    expected = {
        key: record[key]
        for key in ("configured_path", "resolved_path", "sha256", "size_bytes")
    }
    current, raw = _read_regular_file(
        expected["configured_path"],
        label=label,
        capture_bytes=True,
    )
    if current.as_dict() != expected:
        raise ValueError(f"terminal result {label} changed after it was recorded")
    assert raw is not None
    try:
        return yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"terminal result {label} is not valid UTF-8 YAML") from exc


def _verify_scheduler_result_bindings(
    request: EvaluationRequest,
    scheduler_result: dict[str, Any],
) -> None:
    if scheduler_result.get("contract") != TERMINAL_RESULT_CONTRACT:
        raise ValueError("scheduler returned the wrong terminal-result contract")
    if scheduler_result.get("schema_version") != TERMINAL_RESULT_SCHEMA_VERSION:
        raise ValueError("scheduler returned the wrong terminal-result schema version")
    expected_bindings = request.bindings_dict()
    if scheduler_result.get("bindings") != expected_bindings:
        raise ValueError("scheduler returned different request input bindings")
    if scheduler_result.get("execution") != request.raw_execution:
        raise ValueError("scheduler returned different request execution bindings")
    if scheduler_result.get("runner") != {
        "id": request.runner_definition.runner_id,
        "source": request.runner_definition.source.as_dict(),
    }:
        raise ValueError("scheduler returned different runner identity")
    try:
        selection = scheduler_result["selection"]
        events = selection["events"]
    except (KeyError, TypeError) as exc:
        raise ValueError("scheduler returned no explicit event selection") from exc
    if selection.get("mode") != "evaluation_request":
        raise ValueError("scheduler returned the wrong event selection mode")
    if selection.get("suite") != {
        "catalog_id": request.suite_catalog.catalog_id,
        "id": request.suite.suite_id,
        "owner": request.suite.owner,
        "source": request.suite.source.as_dict(),
        "version": request.suite.version,
    }:
        raise ValueError("scheduler returned different suite identity")
    if not isinstance(events, list) or not events:
        raise TypeError("scheduler terminal-result events must be a non-empty list")
    expected_task_ids = [task.uuid for task in request.tasks]
    actual_task_ids = []
    event_ids: set[str] = set()
    required_job_links: set[tuple[int, str, str]] = set()
    for ordinal, event in enumerate(events):
        if not isinstance(event, dict) or event.get("ordinal") != ordinal:
            raise ValueError("scheduler terminal-result event order is invalid")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise ValueError("scheduler terminal-result event identity is invalid")
        event_ids.add(event_id)
        if event.get("terminal_phase") != 2:
            raise ValueError(f"selected event {ordinal} is not successful")
        task = event.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            raise ValueError(f"selected event {ordinal} has no task identity")
        actual_task_ids.append(task["id"])
        if task.get("definition") != request.suite.tasks[ordinal].as_dict():
            raise ValueError(
                f"selected event {ordinal} has a different suite task definition"
            )
        inputs = event.get("inputs")
        if not isinstance(inputs, dict):
            raise TypeError(f"selected event {ordinal} has no input bindings")
        dataset_files = inputs.get("dataset_files")
        expected_datasets = [
            item.as_dict() for item in request.dataset_bindings[task["id"]]
        ]
        if dataset_files != expected_datasets:
            raise ValueError(
                f"selected event {ordinal} has different dataset bindings"
            )
        for file_ordinal, binding in enumerate(dataset_files):
            _verify_public_binding(
                binding,
                label=f"selected dataset {file_ordinal}",
            )
        if inputs.get("task_closure_sha256") != request.task_closure_sha256[
            task["id"]
        ]:
            raise ValueError(
                f"selected event {ordinal} has a different task closure"
            )
        outputs = event.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"selected event {ordinal} has no output bindings")
        if [
            output.get("role") if isinstance(output, dict) else None
            for output in outputs
        ] != [
            "generations",
            "grades",
            "scores",
            "run_metadata",
        ]:
            raise ValueError(f"selected event {ordinal} output roles are incomplete")
        for output in outputs:
            role = output.get("role") if isinstance(output, dict) else None
            if not isinstance(role, str) or not role:
                raise ValueError(f"selected event {ordinal} has an invalid output role")
            _verify_public_binding(output, label=f"required {role} output")

        units = event.get("controller_units")
        if not isinstance(units, list) or [
            unit.get("role") if isinstance(unit, dict) else None for unit in units
        ] != ["generation", "grading", "aggregation"]:
            raise ValueError(f"selected event {ordinal} has incomplete controller units")
        for unit in units:
            role = unit["role"]
            required = unit.get("required")
            job_ids = unit.get("job_ids")
            if not isinstance(required, bool) or not isinstance(job_ids, list):
                raise ValueError(
                    f"selected event {ordinal} has an invalid {role} unit"
                )
            if any(
                isinstance(job_id, bool)
                or not isinstance(job_id, int)
                or job_id <= 0
                for job_id in job_ids
            ):
                raise ValueError(
                    f"selected event {ordinal} has invalid {role} child IDs"
                )
            if required:
                if unit.get("executor") != "controller" or unit.get(
                    "outcome"
                ) != "succeeded":
                    raise ValueError(
                        f"selected event {ordinal} required {role} unit failed"
                    )
            elif (
                unit.get("executor") != "none"
                or unit.get("outcome") != "not_required"
                or job_ids
            ):
                raise ValueError(
                    f"selected event {ordinal} non-required {role} unit is invalid"
                )
            if role == "aggregation" and job_ids:
                raise ValueError("aggregation cannot claim a Slurm child")
            if role == "generation" and required and not job_ids:
                raise ValueError(
                    f"selected event {ordinal} is missing generation-serving children"
                )
            if role == "grading" and job_ids:
                raise ValueError(
                    "suite schema 1.0 non-LLM grading cannot claim a Slurm child"
                )
            for job_id in job_ids:
                required_job_links.add((job_id, event_id, role))
    if actual_task_ids != expected_task_ids:
        raise ValueError("scheduler terminal-result task order differs from request")

    jobs = scheduler_result.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("scheduler terminal-result jobs must be a list")
    actual_job_links: set[tuple[int, str, str]] = set()
    seen_job_ids: set[int] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise TypeError("scheduler terminal-result job must be an object")
        job_id = job.get("job_id")
        if (
            isinstance(job_id, bool)
            or not isinstance(job_id, int)
            or job_id <= 0
            or job_id in seen_job_ids
        ):
            raise ValueError("scheduler terminal-result job ID is invalid")
        seen_job_ids.add(job_id)
        completions = job.get("role_completions")
        associations = job.get("associations")
        if not isinstance(completions, list) or not completions:
            raise ValueError(f"Slurm child {job_id} has no completed role")
        if not isinstance(associations, list) or not associations:
            raise ValueError(f"Slurm child {job_id} has no event association")
        completion_links: set[tuple[int, str, str]] = set()
        for completion in completions:
            if not isinstance(completion, dict) or set(completion) != {
                "event_id",
                "role",
            }:
                raise ValueError(f"Slurm child {job_id} has an invalid completion")
            link = (job_id, completion["event_id"], completion["role"])
            if (
                completion["event_id"] not in event_ids
                or completion["role"] not in {"generation", "grading"}
                or link in completion_links
            ):
                raise ValueError(f"Slurm child {job_id} has an invalid completion")
            completion_links.add(link)
        association_links: set[tuple[int, str, str]] = set()
        for association in associations:
            if not isinstance(association, dict) or set(association) != {
                "event_id",
                "roles",
            }:
                raise ValueError(f"Slurm child {job_id} has an invalid association")
            roles = association["roles"]
            if (
                association["event_id"] not in event_ids
                or not isinstance(roles, list)
                or "serving" not in roles
            ):
                raise ValueError(f"Slurm child {job_id} has an invalid association")
            association_links.update(
                (job_id, association["event_id"], role)
                for role in roles
                if role != "serving"
            )
        if completion_links != association_links:
            raise ValueError(
                f"Slurm child {job_id} completion and association differ"
            )
        scheduler = job.get("scheduler")
        if not isinstance(scheduler, dict) or scheduler.get("source") != "sacct":
            raise ValueError(f"Slurm child {job_id} has no sacct outcome")
        clean = (
            scheduler.get("state") == "COMPLETED"
            and scheduler.get("exit_code") == 0
            and scheduler.get("signal") == 0
        )
        released = (
            scheduler.get("state") == "CANCELLED"
            and job.get("cancellation_intent") == "scheduler_release"
        )
        if job.get("outcome") != "succeeded" or not (clean or released):
            raise ValueError(f"Slurm child {job_id} has an unsuccessful outcome")
        actual_job_links.update(completion_links)
    if actual_job_links != required_job_links:
        raise ValueError("terminal-result required child roles are incomplete")


def _verify_yaml_scheduler_result(
    invocation: YamlTerminalInvocation,
    scheduler_result: dict[str, Any],
) -> None:
    """Verify the output-only ordinary-YAML terminal result."""
    if set(scheduler_result) != {
        "bindings",
        "contract",
        "jobs",
        "schema_version",
        "selection",
        "status",
    }:
        raise ValueError("scheduler returned an invalid YAML result shape")
    if scheduler_result.get("contract") != YAML_TERMINAL_RESULT_CONTRACT:
        raise ValueError("scheduler returned the wrong YAML terminal-result contract")
    if (
        scheduler_result.get("schema_version")
        != YAML_TERMINAL_RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "scheduler returned the wrong YAML terminal-result schema version"
        )
    if scheduler_result.get("bindings") != invocation.bindings_dict():
        raise ValueError("scheduler returned different YAML invocation bindings")

    selection = scheduler_result.get("selection")
    if not isinstance(selection, dict) or set(selection) != {"events", "mode"}:
        raise TypeError("YAML terminal result selection must be an object")
    if selection.get("mode") != "eval_configs":
        raise ValueError("scheduler returned the wrong YAML selection mode")
    events = selection.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("YAML terminal result must select at least one event")

    configured_bindings = {
        role: [item.as_dict() for item in bindings]
        for role, bindings in (
            ("model_config", invocation.model_configs),
            ("data_config", invocation.data_configs),
            ("eval_config", invocation.eval_configs),
        )
    }
    event_ids: set[str] = set()
    required_job_links: set[tuple[int, str, str]] = set()
    for ordinal, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {
            "controller_units",
            "event_id",
            "event_type",
            "inputs",
            "model",
            "ordinal",
            "outputs",
            "source",
            "task",
            "terminal_phase",
        }:
            raise ValueError("YAML terminal-result event shape is invalid")
        if event.get("ordinal") != ordinal:
            raise ValueError("YAML terminal-result event order is invalid")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise ValueError("YAML terminal-result event identity is invalid")
        event_ids.add(event_id)
        if event.get("terminal_phase") != 2:
            raise ValueError(f"selected event {ordinal} is not successful")
        if event.get("event_type") != "generation_grading":
            raise ValueError(
                "YAML terminal evidence supports generation/grading tasks only"
            )

        source = event.get("source")
        if not isinstance(source, dict) or set(source) != {
            "data_config",
            "eval_config",
            "eval_group",
            "model_config",
        }:
            raise ValueError(f"selected event {ordinal} has invalid YAML sources")
        for role in ("model_config", "data_config"):
            binding = source[role]
            if binding not in configured_bindings[role]:
                raise ValueError(
                    f"selected event {ordinal} names an unbound {role}"
                )
            _verify_public_binding(binding, label=f"selected {role}")
        eval_binding = source["eval_config"]
        if eval_binding not in configured_bindings["eval_config"]:
            raise ValueError(
                f"selected event {ordinal} names an unbound eval_config"
            )
        if not isinstance(source["eval_group"], str) or not source[
            "eval_group"
        ]:
            raise ValueError(
                f"selected event {ordinal} has no eval group identity"
            )
        _verify_public_binding(eval_binding, label="selected eval_config")

        model = event.get("model")
        task = event.get("task")
        if not isinstance(model, dict) or set(model) != {
            "definition_sha256",
            "name",
            "parser_type",
        }:
            raise ValueError(f"selected event {ordinal} model identity is invalid")
        if not isinstance(task, dict) or set(task) != {
            "dataset_name",
            "definition_sha256",
            "grader_type",
            "id",
            "mode",
            "semantic_version",
        }:
            raise ValueError(f"selected event {ordinal} task identity is invalid")
        for role, definition in (("model", model), ("task", task)):
            digest = definition.get("definition_sha256")
            if not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", digest
            ) is None:
                raise ValueError(
                    f"selected event {ordinal} has invalid {role} definition digest"
                )

        inputs = event.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != {"dataset_files"}:
            raise TypeError(f"selected event {ordinal} has no input bindings")
        dataset_files = inputs.get("dataset_files")
        if not isinstance(dataset_files, list) or not dataset_files:
            raise ValueError(
                f"selected event {ordinal} has no dataset file bindings"
            )
        for file_ordinal, binding in enumerate(dataset_files):
            _verify_public_binding(
                binding,
                label=f"selected dataset {file_ordinal}",
            )

        outputs = event.get("outputs")
        if not isinstance(outputs, list) or [
            output.get("role") if isinstance(output, dict) else None
            for output in outputs
        ] != ["generations", "grades", "scores", "run_metadata"]:
            raise ValueError(f"selected event {ordinal} output roles are incomplete")
        for output in outputs:
            if not isinstance(output, dict) or set(output) != {
                "configured_path",
                "resolved_path",
                "role",
                "sha256",
                "size_bytes",
            }:
                raise ValueError(
                    f"selected event {ordinal} has an invalid output binding"
                )
            _verify_public_binding(
                output,
                label=f"required {output['role']} output",
            )
        run_metadata = _load_public_yaml_binding(
            outputs[-1],
            label="required run_metadata output",
        )
        expected_yaml_metadata = {
            "dataset_files": dataset_files,
            "model": model,
            "source": source,
            "task": task,
        }
        if (
            not isinstance(run_metadata, dict)
            or run_metadata.get("model") != model["name"]
            or run_metadata.get("dataset") != task["dataset_name"]
            or run_metadata.get("evaluation_yaml") != expected_yaml_metadata
        ):
            raise ValueError(
                f"selected event {ordinal} differs from its bound run metadata"
            )

        units = event.get("controller_units")
        if not isinstance(units, list) or [
            unit.get("role") if isinstance(unit, dict) else None
            for unit in units
        ] != ["generation", "grading", "aggregation"]:
            raise ValueError(
                f"selected event {ordinal} has incomplete controller units"
            )
        for unit in units:
            if not isinstance(unit, dict) or set(unit) != {
                "executor",
                "job_ids",
                "outcome",
                "required",
                "role",
                "serving_required",
            }:
                raise ValueError(
                    f"selected event {ordinal} has an invalid controller unit"
                )
            role = unit["role"]
            required = unit.get("required")
            serving_required = unit.get("serving_required")
            job_ids = unit.get("job_ids")
            if (
                not isinstance(required, bool)
                or not isinstance(serving_required, bool)
                or not isinstance(job_ids, list)
            ):
                raise TypeError(
                    f"selected event {ordinal} has an invalid {role} unit"
                )
            if any(
                isinstance(job_id, bool)
                or not isinstance(job_id, int)
                or job_id <= 0
                for job_id in job_ids
            ):
                raise ValueError(
                    f"selected event {ordinal} has invalid {role} child IDs"
                )
            if required:
                if unit.get("executor") != "controller" or unit.get(
                    "outcome"
                ) != "succeeded":
                    raise ValueError(
                        f"selected event {ordinal} required {role} unit failed"
                    )
            elif (
                unit.get("executor") != "none"
                or unit.get("outcome") != "not_required"
                or job_ids
                or serving_required
            ):
                raise ValueError(
                    f"selected event {ordinal} non-required {role} unit is invalid"
                )
            if role == "aggregation":
                if job_ids or serving_required:
                    raise ValueError("aggregation cannot claim a Slurm child")
            elif serving_required != bool(job_ids):
                raise ValueError(
                    f"selected event {ordinal} {role} serving child is incomplete"
                )
            required_job_links.update(
                (job_id, event_id, role) for job_id in job_ids
            )

    jobs = scheduler_result.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("YAML terminal-result jobs must be a list")
    actual_job_links: set[tuple[int, str, str]] = set()
    seen_job_ids: set[int] = set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {
            "associations",
            "cancellation_intent",
            "event_id",
            "job_id",
            "job_name",
            "kind",
            "model",
            "outcome",
            "role_completions",
            "runner",
            "scheduler",
            "serving_key",
            "submission_origin",
        }:
            raise ValueError("YAML terminal-result job shape is invalid")
        if job.get("kind") != "model_serving":
            raise ValueError("YAML terminal-result job must be model_serving")
        job_id = job.get("job_id")
        if (
            isinstance(job_id, bool)
            or not isinstance(job_id, int)
            or job_id <= 0
            or job_id in seen_job_ids
        ):
            raise ValueError("YAML terminal-result job ID is invalid")
        seen_job_ids.add(job_id)
        completions = job.get("role_completions")
        associations = job.get("associations")
        if not isinstance(completions, list) or not completions:
            raise ValueError(f"Slurm child {job_id} has no completed role")
        if not isinstance(associations, list) or not associations:
            raise ValueError(f"Slurm child {job_id} has no event association")
        completion_links: set[tuple[int, str, str]] = set()
        for completion in completions:
            if not isinstance(completion, dict) or set(completion) != {
                "event_id",
                "role",
            }:
                raise ValueError(f"Slurm child {job_id} has an invalid completion")
            link = (job_id, completion["event_id"], completion["role"])
            if (
                completion["event_id"] not in event_ids
                or completion["role"] not in {"generation", "grading"}
                or link in completion_links
            ):
                raise ValueError(f"Slurm child {job_id} has an invalid completion")
            completion_links.add(link)

        association_links: set[tuple[int, str, str]] = set()
        associated_events: set[str] = set()
        for association in associations:
            if not isinstance(association, dict) or set(association) != {
                "event_id",
                "roles",
            }:
                raise ValueError(f"Slurm child {job_id} has an invalid association")
            associated_event = association["event_id"]
            roles = association["roles"]
            if (
                associated_event not in event_ids
                or associated_event in associated_events
                or not isinstance(roles, list)
                or not roles
                or roles[0] != "serving"
                or len(set(roles)) != len(roles)
                or any(
                    role not in {"serving", "generation", "grading"}
                    for role in roles
                )
            ):
                raise ValueError(f"Slurm child {job_id} has an invalid association")
            associated_events.add(associated_event)
            association_links.update(
                (job_id, associated_event, role)
                for role in roles
                if role != "serving"
            )
        if completion_links != association_links:
            raise ValueError(
                f"Slurm child {job_id} completion and association differ"
            )
        scheduler = job.get("scheduler")
        if not isinstance(scheduler, dict) or set(scheduler) != {
            "exit_code",
            "job_id_raw",
            "job_name",
            "raw_state",
            "reason",
            "signal",
            "source",
            "state",
        }:
            raise ValueError(f"Slurm child {job_id} has invalid scheduler evidence")
        if scheduler.get("source") != "sacct":
            raise ValueError(f"Slurm child {job_id} has no sacct outcome")
        if scheduler.get("job_name") != job.get("job_name"):
            raise ValueError(f"Slurm child {job_id} has a mismatched job name")
        raw_state = scheduler.get("raw_state")
        state = scheduler.get("state")
        if (
            scheduler.get("job_id_raw") != str(job_id)
            or not isinstance(raw_state, str)
            or not raw_state
            or raw_state.split(maxsplit=1)[0].removesuffix("+") != state
            or any(
                isinstance(scheduler.get(field), bool)
                or not isinstance(scheduler.get(field), int)
                or scheduler[field] < 0
                for field in ("exit_code", "signal")
            )
            or not isinstance(scheduler.get("reason"), str)
        ):
            raise ValueError(f"Slurm child {job_id} has inconsistent sacct evidence")
        clean = (
            state == "COMPLETED"
            and scheduler.get("exit_code") == 0
            and scheduler.get("signal") == 0
        )
        released = (
            state == "CANCELLED"
            and job.get("cancellation_intent") == "scheduler_release"
        )
        if job.get("outcome") != "succeeded" or not (clean or released):
            raise ValueError(f"Slurm child {job_id} has an unsuccessful outcome")
        actual_job_links.update(completion_links)
    if actual_job_links != required_job_links:
        raise ValueError("YAML terminal-result required child roles are incomplete")


def _publish_success_result(
    output_path: Path,
    scheduler_result: dict[str, Any],
) -> dict[str, Any]:
    """Publish one already-verified success result exclusively and durably."""
    result = dict(scheduler_result)
    result["controller"] = {
        "exit_code": 0,
        "outcome": "succeeded",
    }
    payload = canonical_json(result).encode("utf-8")

    target = output_path
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    directory_descriptor = -1
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o666,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.link(temporary, target, follow_symlinks=False)
        linked = True
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        except BaseException:
            target.unlink(missing_ok=True)
            linked = False
            raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not linked:
        raise RuntimeError("terminal result was not published")
    return result


def publish_yaml_terminal_result(
    invocation: YamlTerminalInvocation,
    scheduler_result: dict[str, Any],
) -> dict[str, Any]:
    """Verify and write-last publish a successful ordinary-YAML result."""
    if scheduler_result.get("status") != "succeeded":
        raise ValueError(
            "YAML terminal result can publish only a succeeded scheduler result"
        )
    invocation.verify_inputs()
    _verify_yaml_scheduler_result(invocation, scheduler_result)
    return _publish_success_result(invocation.output_path, scheduler_result)


def publish_terminal_result(
    request: EvaluationRequest,
    scheduler_result: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish a successful canonical result without replacement."""
    if scheduler_result.get("status") != "succeeded":
        raise ValueError(
            "terminal result can publish only a succeeded scheduler result"
        )
    request.verify_inputs()
    _verify_scheduler_result_bindings(request, scheduler_result)
    return _publish_success_result(request.output_path, scheduler_result)
