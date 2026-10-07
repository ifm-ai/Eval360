import asyncio
import fnmatch
import glob
import json
import logging
import os
import pydantic
import traceback
from enum import Enum
from typing import Any
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm as _tqdm_base

logger = logging.getLogger(__name__)

STANDARD_OPENAI_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "n",
    "logprobs",
    "top_logprobs",
    "seed",
    "stream",
    "logit_bias",
    "response_format",
    "user",
}


def separate_extra_body(kwargs: dict[str, any]) -> dict[str, any]:
    """Move non-standard OpenAI params into extra_body, warning the user."""
    kwargs = dict(kwargs)
    if "cache_salt" in kwargs:
        raise ValueError(
            "openai kwargs must not set cache_salt directly; use the model-level "
            "cache_salt config instead"
        )
    existing_extra = kwargs.get("extra_body", {})
    if existing_extra is None:
        kwargs.pop("extra_body", None)
        existing_extra = {}
    elif not isinstance(existing_extra, dict):
        raise ValueError("openai kwargs extra_body must be a mapping")
    elif "cache_salt" in existing_extra:
        raise ValueError(
            "openai kwargs must not set extra_body.cache_salt directly; use the "
            "model-level cache_salt config instead"
        )

    non_standard = {
        k: v
        for k, v in kwargs.items()
        if k not in STANDARD_OPENAI_PARAMS and k != "extra_body"
    }
    if not non_standard:
        return kwargs
    logger.warning(
        f"Non-standard OpenAI params {list(non_standard.keys())} will be sent via extra_body. "
        f"Consider placing them under 'extra_body' in your config explicitly."
    )
    standard = {k: v for k, v in kwargs.items() if k in STANDARD_OPENAI_PARAMS}
    existing_extra = dict(existing_extra)
    existing_extra.update(non_standard)
    standard["extra_body"] = existing_extra
    return standard


class _LoggingWriter:
    """File-like object that routes tqdm output to the Python logger."""

    def write(self, msg):
        msg = msg.strip()
        if msg:
            logger.info(msg)

    def flush(self):
        pass


class _LoggingTqdm(_tqdm_base):
    """tqdm subclass that writes progress to the scheduler logger instead of stdout."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("file", _LoggingWriter())
        kwargs.pop(
            "name", None
        )  # huggingface_hub passes `name` but tqdm doesn't accept it
        super().__init__(*args, **kwargs)


class Sentinel(Enum):
    ERROR = 0
    COMPLETED = 1


class ExceptionWrapper(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    trace: str
    exception: Exception
    instance: dict
    error_code: str | None = None
    attempts: int | None = None
    elapsed_seconds: float | None = None
    retriable: bool | None = None
    http_status: int | None = None

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        instance: dict,
    ) -> "ExceptionWrapper":
        """Preserve structured evidence from direct or grouped failures."""
        from .external_requests import ExternalRequestFailure

        evidence_fields = (
            "error_code",
            "attempts",
            "elapsed_seconds",
            "retriable",
            "http_status",
        )

        def find_evidence(
            candidate: BaseException,
        ) -> Exception | None:
            if (
                isinstance(candidate, Exception)
                and all(hasattr(candidate, field) for field in evidence_fields)
            ):
                return candidate
            if isinstance(candidate, BaseExceptionGroup):
                for nested in candidate.exceptions:
                    found = find_evidence(nested)
                    if found is not None:
                        return found
            return None

        evidence = find_evidence(exception)
        if evidence is not None:
            return cls(
                # Keep the caught exception (including an ExceptionGroup) in
                # the legacy artifact fields.  External retry details are
                # supplementary metadata, not a reason to discard sibling
                # failures or group topology.
                exception=exception,
                trace="".join(
                    traceback.TracebackException.from_exception(
                        exception
                    ).format()
                ),
                instance=instance,
                **{
                    field: getattr(evidence, field)
                    for field in evidence_fields
                },
            )

        return cls(
            exception=exception,
            trace="".join(
                traceback.TracebackException.from_exception(exception).format()
            ),
            instance=instance,
        )


def normalize_tag(value: Any) -> str:
    """Normalize a model/dataset tag string."""
    if value is None:
        return "any"
    candidate = str(value).strip()
    if not candidate or candidate.lower() == "any":
        return "any"
    return candidate


def _completion_input_from_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
            if content is None:
                continue
            parts.append(str(content))
        else:
            parts.append(str(message))
    return "\n\n".join(parts)


def normalize_eval_input_record(
    record: dict[str, Any], row_index: int
) -> dict[str, Any]:
    """Normalize legacy JSONL rows into the Eval360 generation row contract.

    New rows already contain `row`, `completion_input`, `chat_input`, and
    `ground_truth`. Older AIME-style rows use `prompt` plus `label`; those are
    accepted here for backwards compatibility.
    """
    normalized = record

    def ensure_copy() -> dict[str, Any]:
        nonlocal normalized
        if normalized is record:
            normalized = dict(record)
        return normalized

    if "row" not in normalized:
        ensure_copy()["row"] = row_index

    if "ground_truth" not in normalized and "label" in normalized:
        ensure_copy()["ground_truth"] = normalized["label"]

    prompt = normalized.get("prompt")
    if "chat_input" not in normalized:
        if isinstance(prompt, list):
            ensure_copy()["chat_input"] = prompt
        elif isinstance(prompt, str):
            ensure_copy()["chat_input"] = [{"role": "user", "content": prompt}]
        elif "completion_input" in normalized:
            ensure_copy()["chat_input"] = [
                {"role": "user", "content": str(normalized["completion_input"])}
            ]

    if "completion_input" not in normalized:
        if isinstance(prompt, str):
            ensure_copy()["completion_input"] = prompt
        elif isinstance(prompt, list):
            ensure_copy()["completion_input"] = _completion_input_from_messages(prompt)
        elif "chat_input" in normalized and isinstance(normalized["chat_input"], list):
            ensure_copy()["completion_input"] = _completion_input_from_messages(
                normalized["chat_input"]
            )

    return normalized


def is_hf_uri(path: str) -> bool:
    """Return True if path is a Hugging Face dataset URI (hf://...)."""
    return path.startswith("hf://")


def _parse_hf_uri(uri: str) -> tuple[str, str, str | None, str]:
    """Parse hf://org/repo/[subfolder/]filename[@revision].

    Returns (repo_id, filename, subfolder_or_None, revision).
    """
    rest = uri[len("hf://") :]
    rest, revision = rest.rsplit("@", 1) if "@" in rest else (rest, "main")
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid HF URI {uri!r}: expected hf://org/repo/path[@revision]"
        )
    repo_id = "/".join(parts[:2])
    file_parts = parts[2:]
    filename = file_parts[-1]
    subfolder = "/".join(file_parts[:-1]) if len(file_parts) > 1 else None
    return repo_id, filename, subfolder, revision


def check_hf_file_exists(uri: str) -> None:
    """Raise FileNotFoundError if the HF file (or glob pattern) doesn't exist or is inaccessible.

    Makes a lightweight API call without downloading the file.
    For glob URIs, verifies that at least one matching file exists.
    """
    repo_id, filename, subfolder, revision = _parse_hf_uri(uri)
    full_path = f"{subfolder}/{filename}" if subfolder else filename
    api = HfApi()
    if "*" in filename:
        matching = [
            f
            for f in api.list_repo_files(
                repo_id, repo_type="dataset", revision=revision
            )
            if fnmatch.fnmatch(f, full_path)
        ]
        if not matching:
            raise FileNotFoundError(f"No HF files matched pattern: {uri!r}")
    else:
        if not api.file_exists(
            repo_id=repo_id, filename=full_path, repo_type="dataset", revision=revision
        ):
            raise FileNotFoundError(f"HF file not found: {uri!r}")


def _is_unsupported_tqdm_class_error(exc: TypeError) -> bool:
    message = str(exc)
    return "tqdm_class" in message and "unexpected keyword argument" in message


def _hf_hub_download_with_logging(**kwargs) -> str:
    try:
        return hf_hub_download(**kwargs, tqdm_class=_LoggingTqdm)
    except TypeError as exc:
        if not _is_unsupported_tqdm_class_error(exc):
            raise
        return hf_hub_download(**kwargs)


def resolve_hf_path(uri: str, cache_dir: str | None = None) -> str:
    """Download an HF dataset file (or glob of files) to the local cache and return a local path.

    For single-file URIs, returns the local path to the downloaded file.
    For glob URIs (containing *), downloads all matching files and returns a local glob
    pattern that expands to all of them via expand_data_path.

    If cache_dir is None, huggingface_hub respects HF_HUB_CACHE / HF_HOME.
    Calling this repeatedly is idempotent — already-cached files are returned immediately.
    """
    repo_id, filename, subfolder, revision = _parse_hf_uri(uri)

    if "*" not in filename:
        return _hf_hub_download_with_logging(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
        )

    # Glob: list all matching files, download each, return a local glob pattern
    full_pattern = f"{subfolder}/{filename}" if subfolder else filename
    api = HfApi()
    matching = sorted(
        f
        for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if fnmatch.fnmatch(f, full_pattern)
    )
    if not matching:
        raise FileNotFoundError(f"No HF files matched pattern: {uri!r}")

    local_paths = []
    for fpath in matching:
        fname = os.path.basename(fpath)
        fsubfolder = os.path.dirname(fpath) or None
        local_paths.append(
            _hf_hub_download_with_logging(
                repo_id=repo_id,
                filename=fname,
                subfolder=fsubfolder,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
            )
        )

    # All files share a parent directory in the HF cache snapshot; return a local glob pattern
    parent = os.path.dirname(local_paths[0])
    return os.path.join(parent, filename)


def expand_data_path(path: str) -> list[str]:
    """Return a sorted list of files matching path (supports globs)."""
    return sorted(glob.glob(path))


def get_unreadable_paths(data_path: str) -> list[str]:
    """Return paths matched by data_path that exist but cannot be read."""
    return [p for p in expand_data_path(data_path) if not os.access(p, os.R_OK)]


def count_jsonl_records(data_path: str) -> int:
    """Count non-blank lines across all JSONL files matched by data_path.

    Uses buffered binary line iteration to avoid per-line JSON parsing
    while correctly skipping blank lines.

    Raises FileNotFoundError if no files match data_path.
    Raises PermissionError if no matched files are readable.
    Skips individual unreadable files (caller checks get_unreadable_paths
    to decide whether partial unreadability is acceptable).
    """
    paths = expand_data_path(data_path)
    if not paths:
        raise FileNotFoundError(f"No JSONL files found at data_path: {data_path!r}")
    unreadable = set(get_unreadable_paths(data_path))
    if len(unreadable) == len(paths):
        raise PermissionError(
            f"No readable JSONL files at {data_path!r}. "
            f"Cannot read: {', '.join(sorted(unreadable))}"
        )
    count = 0
    for path in paths:
        if path in unreadable:
            continue
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    count += 1
    return count


async def dataset_iterator(
    path,
    logger,
    read_only,
    resume_index=0,
    sentinel=False,
    verbose=False,
    skip_rows=None,
    record_transform=None,
):
    logger.info(f"Reading from {path}")
    # Need a canonical order
    paths = expand_data_path(path)
    if not paths:
        logger.info(f"{path} not found, skipping")
        return
    # This function also truncates the file at the first bad line
    index = 0
    for path in paths:
        if not os.access(path, os.R_OK):
            logger.warning(f"Skipping unreadable file: {path!r}")
            continue
        if not read_only:
            # aiofiles can hang indefinitely on line iteration under our current
            # Python/runtime combination. Use buffered reads here as well and
            # yield to the event loop periodically.
            with open(path, mode="r+") as f:
                while True:
                    # Record where this line starts (byte offset)
                    line_start = f.tell()
                    line = f.readline()
                    if verbose and index % 1000 == 0:
                        logger.info(f"Reading line from {path}: {index}")
                    if not line:
                        break
                    if not line.strip():
                        continue
                    if index < resume_index:
                        index += 1
                        if index % 1000 == 0:
                            await asyncio.sleep(0)
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # TODO POC: dont delete this in input dataset, only generation and grading files
                        logger.debug(f"Error parsing line {index} in {path}")
                        # Rewind to start of bad line and truncate file
                        f.seek(line_start)
                        f.truncate()
                        return
                    if "exception" in record:
                        logger.debug(
                            f"Error record at line {index} in {path}, keeping (counts as incorrect)"
                        )
                        yield record
                        index += 1
                        if index % 1000 == 0:
                            await asyncio.sleep(0)
                        continue
                    yield record
                    index += 1
                    if index % 1000 == 0:
                        await asyncio.sleep(0)
        else:
            # aiofiles can hang indefinitely on line iteration under our
            # current Python/runtime combination. Use normal buffered reads
            # here and yield to the event loop periodically instead.
            with open(path) as f:
                while True:
                    line = f.readline()
                    if verbose and index % 1000 == 0:
                        logger.info(f"Reading line from {path}: {index}")
                    if not line:
                        break
                    if not line.strip():
                        continue
                    if index < resume_index:
                        index += 1
                        if index % 1000 == 0:
                            await asyncio.sleep(0)
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # TODO POC: dont delete this in input dataset, only generation and grading files
                        logger.debug(f"Error parsing line {index} in {path}")
                        return
                    if record_transform is not None:
                        record = record_transform(record, index)
                    if skip_rows is not None and record.get("row") in skip_rows:
                        index += 1
                        if index % 1000 == 0:
                            await asyncio.sleep(0)
                        continue
                    yield record
                    index += 1
                    if index % 1000 == 0:
                        await asyncio.sleep(0)
    if sentinel:
        yield Sentinel.COMPLETED


_ROW_RE = __import__("re").compile(rb'"row"\s*:\s*(\d+)')


def load_completed_row_ids(path: str) -> set[int]:
    """Scan a JSONL file and return the set of completed row IDs.

    Uses a regex on raw bytes to extract the "row" field without parsing
    full JSON — O(N) time, O(N integers) memory regardless of record size.
    Corrupt trailing lines are silently ignored (the row wasn't completed).
    """
    rows: set[int] = set()
    try:
        with open(path, "rb") as f:
            for line in f:
                m = _ROW_RE.search(line)
                if m:
                    rows.add(int(m.group(1)))
    except FileNotFoundError:
        pass
    return rows


async def make_eager(async_iterable):
    items = [item async for item in async_iterable]

    # New async generator over the realized values
    async def eager():
        for item in items:
            yield item

    return eager()
