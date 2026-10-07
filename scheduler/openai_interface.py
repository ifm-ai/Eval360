from __future__ import annotations

from openai import AsyncOpenAI
from openai import APIConnectionError, BadRequestError
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar
import copy
import aiohttp
import asyncio
import httpx
import json
import time
from .utils import Sentinel, ExceptionWrapper
from .rate_limiter import RateLimiter
from .external_requests import (
    RATE_LIMITERS as RATE_LIMITERS,
    ExternalRequestFailure,
    ExternalRequestRunner,
    MalformedExternalResponseError,
    canonicalize_external_endpoint,
    external_quota_identity as external_quota_identity,
    get_external_rate_limiter,
    validate_and_canonicalize_external_endpoint as validate_and_canonicalize_external_endpoint,
    validate_external_rate_limiter_registration as validate_external_rate_limiter_registration,
)
from .choice_scoring_schema import (
    GENERATED_TAIL_N_TOKENS_FIELD,
    build_choice_scoring_prompts,
    choice_scoring_suffix_start_chars,
    is_choice_scoring_grader_type,
    is_choice_scoring_row,
    resolve_scoring_completion_token_counts,
    resolve_scoring_completions,
    sum_choice_scoring_suffix_logprobs_with_count,
    validate_choice_scoring_row,
)
from .cache_salt import CacheSaltConfig, request_kwargs_with_cache_salt
import traceback
import logging

if TYPE_CHECKING:
    from .event import EventInstance
    from .job import JobManager
    from .model import ModelInstance
    from .progress import ProgressManager
    from .task import Task

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("OpenAIConnection")
logging.basicConfig(level=logging.INFO)

_VLLM_ACTIVE_REQUEST_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
)
_CHOICE_SCORING_RETRY_DELAYS_SECONDS = (0.25, 1.0)
_CHOICE_SCORING_MALFORMED_MARKER = "MessagePack data is malformed"
_CHOICE_SCORING_GENERATED_TAIL_TOKENS = 1
ResponseT = TypeVar("ResponseT")


@dataclass(frozen=True)
class _ChoiceScoringResponse:
    """Validated, serialized data from one choice-scoring HTTP response."""

    logprobs: list[dict]
    metadata: list[dict]
    usages: list[dict]


class ModelConnectionPool:
    """Per-model connection pool that distributes requests across all live replicas.

    Each registered URL has its own slot count (capacity = max_simultaneous_requests).
    acquire() picks the URL with the most available slots, blocks if all are at
    capacity.  add_url() / remove_url() are called by the scheduler when replicas
    go live or are evicted.
    """

    def __init__(self, capacity_per_url: int):
        self._capacity = capacity_per_url
        # url → [available_slots]  (list-of-one so it's mutable inside the lock)
        self._slots: dict[str, list[int]] = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    async def add_url(self, url: str):
        url = canonicalize_external_endpoint(url)
        async with self._condition:
            if url not in self._slots:
                self._slots[url] = [self._capacity]
                self._condition.notify_all()

    async def remove_url(self, url: str):
        url = canonicalize_external_endpoint(url)
        async with self._condition:
            self._slots.pop(url, None)
            self._condition.notify_all()

    async def acquire(self, requests_left: int) -> tuple[str, int]:
        """Block until a URL has capacity; return (url, n_slots_acquired)."""
        async with self._condition:
            while True:
                best_url, best_avail = None, 0
                for url, slots in self._slots.items():
                    if slots[0] > best_avail:
                        best_url, best_avail = url, slots[0]
                if best_url is not None:
                    n = min(best_avail, requests_left, 4)
                    self._slots[best_url][0] -= n
                    return best_url, n
                await self._condition.wait()

    async def release(self, url: str, n: int):
        async with self._condition:
            if url in self._slots:
                self._slots[url][0] += n
            self._condition.notify_all()

    @property
    def urls(self) -> list[str]:
        """Snapshot of currently registered URLs."""
        return list(self._slots.keys())

    @property
    def capacity_per_url(self) -> int:
        """Configured capacity used to reject first-writer-wins drift."""
        return self._capacity


class _PoolAcquisitionFailure(Exception):
    """A local pool failure that must not be classified as an HTTP attempt."""

    def __init__(self, original_exception: Exception):
        super().__init__(str(original_exception))
        self.original_exception = original_exception


# Global pool registry: model_name → ModelConnectionPool
LOCKED_CONNECTIONS: dict[str, ModelConnectionPool] = {}


def validate_connection_pool_registration(
    serving_key: str,
    capacity_per_url: int,
) -> None:
    """Reject process-scoped capacity drift without mutating the registry."""
    pool = LOCKED_CONNECTIONS.get(serving_key)
    if (
        pool is not None
        and pool.capacity_per_url != capacity_per_url
    ):
        raise ValueError(
            "conflicting max_simultaneous_requests for serving key "
            f"{serving_key!r}: registered={pool.capacity_per_url}, "
            f"requested={capacity_per_url}"
        )


def get_or_create_connection_pool(
    serving_key: str,
    capacity_per_url: int,
) -> ModelConnectionPool:
    """Return a pool or reject a conflicting live capacity registration."""
    validate_connection_pool_registration(
        serving_key,
        capacity_per_url,
    )
    pool = LOCKED_CONNECTIONS.get(serving_key)
    if pool is None:
        pool = ModelConnectionPool(capacity_per_url)
        LOCKED_CONNECTIONS[serving_key] = pool
        return pool
    return pool


def _is_input_too_long(e: Exception) -> bool:
    """Return True if e is a BadRequestError caused by context-length overflow."""
    if not isinstance(e, BadRequestError):
        return False
    if getattr(e, "code", None) == "context_length_exceeded":
        return True
    msg = str(e).lower()
    return "context length" in msg or "maximum context" in msg


def _extra_body_to_chat_template_kwargs(extra_body: dict) -> dict:
    """Normalise VLLM's two extra_body forms to a chat_template_kwargs dict for /tokenize.

    VLLM accepts template variables in generation two ways:
      - nested: {"chat_template_kwargs": {"reasoning_effort": "high"}}
      - flat:   {"reasoning_effort": "high"}  (passed directly as Jinja vars)

    The /tokenize endpoint only accepts chat_template_kwargs, so both forms are
    normalised here.  This is the single source of truth for that mapping.
    """
    if "chat_template_kwargs" in extra_body:
        return dict(extra_body["chat_template_kwargs"])
    return dict(extra_body)


def _serialize_choice_metadata(choice) -> dict:
    return {
        "finish_reason": getattr(choice, "finish_reason", None),
        "stop_reason": getattr(choice, "stop_reason", None),
    }


def _json_mode_mapping(value, *, field_name: str) -> dict:
    """Normalize SDK metadata to a JSON-mode mapping or fail closed."""
    if isinstance(value, dict):
        dumped = copy.deepcopy(value)
    else:
        model_dump = getattr(value, "model_dump", None)
        if not callable(model_dump):
            raise MalformedExternalResponseError(
                f"response returned unsupported {field_name} metadata"
            )
        try:
            dumped = model_dump(mode="json")
        except Exception as error:
            raise MalformedExternalResponseError(
                f"response returned malformed {field_name} metadata: "
                f"{type(error).__name__}: {error}"
            ) from error
    if not isinstance(dumped, dict):
        raise MalformedExternalResponseError(
            f"response returned non-mapping {field_name} metadata"
        )
    return dumped


def _serialize_usage(usage) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return copy.deepcopy(usage)
    return None


def _extract_reasoning_text(choice) -> str | None:
    reasoning = getattr(choice.message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning:
        return reasoning

    extra = getattr(choice.message, "model_extra", None)
    if not isinstance(extra, dict):
        return None

    for key in ("reasoning_content", "reasoning"):
        value = extra.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _deep_merge_openai_kwargs(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_openai_kwargs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class OpenAIConnection:
    @staticmethod
    def _is_base_model_type(model_type) -> bool:
        return (
            getattr(model_type, "name", None) == "BASE"
            or getattr(model_type, "value", None) == 0
            or model_type == "base"
        )

    @staticmethod
    def _is_chat_model_type(model_type) -> bool:
        return (
            getattr(model_type, "name", None) == "CHAT"
            or getattr(model_type, "value", None) == 1
            or model_type in {"chat", "instruct"}
        )

    def __init__(self, model: ModelInstance, task: Task, event_instance: EventInstance, job_manager: JobManager, progress_manager: ProgressManager, new_field_name: str, force_logprobs: bool = False, debug: bool = False):
        self._model = model
        self._task = task
        self._job_manager = job_manager
        self._progress_manager = progress_manager
        self._event_instance = event_instance

        if self._task.openai_settings:
            self._openai_kwargs = copy.deepcopy(self._task.openai_settings)
        else:
            self._openai_kwargs = {}

        if self._model.openai_kwargs:
            self._openai_kwargs = _deep_merge_openai_kwargs(
                self._openai_kwargs,
                self._model.openai_kwargs,
            )

        if force_logprobs and "logprobs" not in self._openai_kwargs:
            if self._is_base_model_type(self._model.model_type):
                self._openai_kwargs["logprobs"] = 5
            else:
                self._openai_kwargs["logprobs"] = True
                self._openai_kwargs["top_logprobs"] = 5

        self._new_field_name = new_field_name
        # Per-URL client cache so we reuse connections to the same node
        self._clients: dict[str, AsyncOpenAI] = {}
        # Count of requests sent but not yet returned; drives Generating↔Queued status
        self._in_flight = 0
        self._completed_requests_count = 0

        _sk = self._model.serving_key
        get_or_create_connection_pool(
            _sk,
            self._model.max_simultaneous_requests,
        )
        self._debug = debug
        self._cancelled = asyncio.Event()
        self._cancel_reason: str | None = None
        self._emit_cancelled_results = False
        # ModelInstance validates this field as a bool. Require the explicit
        # production value so permissive proxy objects cannot accidentally
        # enable external-only quota and request behavior.
        self._is_external = getattr(model, "is_external", False) is True
        self._external_retry_policy = getattr(
            model,
            "external_retry_policy",
            None,
        )
        self._monotonic = time.monotonic
        # For external models, use the original base_name for API requests
        # (not the internal name which may include name_modifier or -judge suffix)
        self._request_model_name = getattr(model, "api_model_name", None) or model.name
        # Set up token bucket rate limiter for external models with RPM configured
        self._rate_limiter: RateLimiter | None = None
        if self._is_external:
            self._rate_limiter = get_external_rate_limiter(model)

    def cancel(self):
        """Stop all in-flight make_request tasks as soon as possible."""
        self._cancelled.set()

    def cancel_with_failures(self, reason: str = "generation cancelled: VLLM server was killed"):
        """Cancel and emit ExceptionWrapper rows for unfinished prompts."""
        self._cancel_reason = reason
        self._emit_cancelled_results = True
        self.cancel()

    def _make_cancelled_result(self, instance: dict) -> ExceptionWrapper:
        reason = self._cancel_reason or "generation cancelled"
        if self._is_choice_scoring_task():
            instance.pop(self._new_field_name, None)
        return ExceptionWrapper(
            exception=RuntimeError(reason),
            trace=reason,
            instance=instance,
        )

    def _completed_generation_count(self, result: dict) -> int:
        generations = result.get(self._new_field_name)
        if isinstance(generations, list):
            return len(generations)
        if (
            self._is_choice_scoring_task()
            and "choice_scoring_full_logprobs" in result
            and "choice_scoring_completion_logprobs" in result
        ):
            return 1
        return 0

    async def _fetch_vllm_num_requests_running(self, url: str) -> int | None:
        """
        Fetch the current active request count from VLLM's Prometheus /metrics endpoint.

        VLLM exposes separate metrics for running, waiting, and swapped requests.
        The scheduler's ``_in_flight`` counter includes all accepted requests that
        have not returned yet, so we treat the sum of those three metrics as the
        server-side active count. Returns None if the endpoint is unreachable or
        none of those metrics are present.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return None
                    text = await resp.text()
            total = 0
            found = False
            for line in text.splitlines():
                for metric_name in _VLLM_ACTIVE_REQUEST_METRICS:
                    if not line.startswith(metric_name):
                        continue
                    parts = line.rsplit(" ", 1)
                    if len(parts) != 2:
                        continue
                    total += int(float(parts[1]))
                    found = True
                    break
            return total if found else None
        except Exception as e:
            logger.debug("Failed to fetch VLLM metrics from %s: %s", url, e)
            return None

    async def _poll_vllm_metrics(self) -> None:
        """
        Background task: every 5 s, compare VLLM's ``num_requests_running`` metric
        against the scheduler's ``_in_flight`` counter and log any discrepancy.

        These counters cover different lifetimes: VLLM stops counting a request
        when engine work finishes, while ``_in_flight`` includes response transfer
        and client-side parsing.  A zero VLLM count is therefore diagnostic only;
        Slurm and health observations own dead-server cancellation.
        """
        pool = LOCKED_CONNECTIONS[self._model.serving_key]
        idle_urls: set[str] = set()
        last_completed_count = self._completed_requests_count
        while True:
            try:
                await asyncio.wait_for(self._cancelled.wait(), timeout=5.0)
                return  # cancelled — exit cleanly
            except asyncio.TimeoutError:
                pass
            if self._in_flight == 0:
                idle_urls.clear()
                continue
            if self._completed_requests_count != last_completed_count:
                idle_urls.clear()
                last_completed_count = self._completed_requests_count
            live_urls = pool.urls
            if not live_urls:
                continue
            for url in live_urls:
                vllm_count = await self._fetch_vllm_num_requests_running(url)
                if vllm_count is None:
                    continue
                logger.info(
                    "VLLM metrics: url=%s vllm_active=%d scheduler_in_flight=%d",
                    url, vllm_count, self._in_flight,
                )
                if vllm_count == 0 and self._in_flight > 0:
                    if url not in idle_urls:
                        idle_urls.add(url)
                        logger.warning(
                            "Discrepancy: VLLM reports 0 active requests but scheduler "
                            "has %d in-flight on %s; retaining requests because the "
                            "metrics cover different lifetimes",
                            self._in_flight, url,
                        )
                else:
                    idle_urls.discard(url)
            idle_urls.intersection_update(live_urls)

    async def _get_templated_prompt(self, url: str, messages: list[dict]) -> str | None:
        """Call VLLM's /tokenize + /detokenize to retrieve the chat-templated prompt text."""
        if self._is_external:
            # External models don't expose VLLM's /tokenize endpoint
            return None
        request_kwargs = request_kwargs_with_cache_salt(
            self._openai_kwargs,
            CacheSaltConfig(),
        )
        try:
            tokenize_body = {"model": self._request_model_name, "messages": messages, "add_generation_prompt": True}
            # Forward template variables so the tokenized prompt reflects the same
            # template settings (e.g. reasoning_effort) as the actual generation call.
            # VLLM accepts them two ways:
            #   - nested: extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}}
            #   - flat:   extra_body={"reasoning_effort": "high"} (VLLM passes these
            #             directly as Jinja template variables, bypassing chat_template_kwargs)
            # The /tokenize endpoint only accepts chat_template_kwargs, so we normalise
            # both forms to that.
            extra_body = request_kwargs.get("extra_body", {})
            if extra_body:
                tokenize_body["chat_template_kwargs"] = _extra_body_to_chat_template_kwargs(extra_body)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/tokenize",
                    json=tokenize_body,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"debug: /tokenize returned {resp.status}")
                        return None
                    tok_data = await resp.json()
                async with session.post(
                    f"{url}/detokenize",
                    json={"model": self._request_model_name, "tokens": tok_data["tokens"]},
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"debug: /detokenize returned {resp.status}")
                        return None
                    return (await resp.json()).get("prompt")
        except Exception as e:
            logger.warning(f"debug: failed to get templated prompt: {e}")
            return None

    def _get_client(self, url: str) -> AsyncOpenAI:
        """Return a cached AsyncOpenAI client for the given base URL."""
        if url not in self._clients:
            if self._is_external:
                # External model: use URL as-is (user provides full base_url including /v1 if needed)
                # and the real API key from the model config
                api_key = getattr(self._model, "api_key", None) or "no-key"
                request_timeout_seconds = getattr(
                    self._external_retry_policy,
                    "request_timeout_seconds",
                    7200,
                )
                self._clients[url] = AsyncOpenAI(
                    base_url=url,
                    api_key=api_key,
                    max_retries=0,
                    timeout=request_timeout_seconds,
                )
            else:
                self._clients[url] = AsyncOpenAI(
                    base_url=f"{url}/v1",
                    api_key="fake key",
                    max_retries=0,
                    timeout=7200,
                )
        return self._clients[url]

    def _is_choice_scoring_task(self) -> bool:
        grader = getattr(self._task, "grader", None)
        return is_choice_scoring_grader_type(getattr(grader, "type", None))

    def _request_kwargs_with_cache_salt(self) -> dict:
        return request_kwargs_with_cache_salt(
            self._openai_kwargs,
            self._model.cache_salt,
        )

    def _choice_scoring_request_kwargs(self) -> dict:
        request_kwargs = self._request_kwargs_with_cache_salt()
        for key in ("stop", "top_logprobs", "max_tokens", "n"):
            request_kwargs.pop(key, None)
        request_kwargs["temperature"] = 0.0
        request_kwargs["echo"] = True
        request_kwargs["max_tokens"] = (
            _CHOICE_SCORING_GENERATED_TAIL_TOKENS
        )
        request_kwargs["logprobs"] = 1

        extra_body = copy.deepcopy(request_kwargs.get("extra_body", {}))
        extra_body.pop("return_token_ids", None)
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        else:
            request_kwargs.pop("extra_body", None)
        return request_kwargs

    @staticmethod
    def _serialize_choice_scoring_logprobs(choice_logprobs):
        if choice_logprobs is None:
            return None
        return _json_mode_mapping(
            choice_logprobs,
            field_name="choice-scoring logprobs",
        )

    @staticmethod
    def _generated_tail_count_from_echoed_text(
        text: str,
        prompt: str,
        *,
        choice_index: int,
        source_label: str,
    ) -> int:
        """Prove a zero- or one-token tail from an ``echo=True`` response."""
        if text == prompt:
            return 0
        if text.startswith(prompt):
            return 1
        raise MalformedExternalResponseError(
            f"{source_label} returned text that does not echo request prompt "
            f"for choice {choice_index}; generated-tail evidence is ambiguous"
        )

    def _normalize_choice_scoring_response(
        self,
        response,
        *,
        expected_choices: int,
        source_label: str,
        prompts: list[str],
        suffix_start_chars: list[int],
        suffix_n_tokens: list[int | None],
    ) -> _ChoiceScoringResponse:
        """Validate and serialize a choice-scoring response transactionally."""
        try:
            if (
                len(prompts) != expected_choices
                or len(suffix_start_chars) != expected_choices
                or len(suffix_n_tokens) != expected_choices
            ):
                raise ValueError(
                    "choice-scoring validation context length mismatch"
                )
            choices = getattr(response, "choices", None)
            if (
                not isinstance(choices, list)
                or len(choices) != expected_choices
            ):
                actual = (
                    len(choices)
                    if isinstance(choices, list)
                    else "missing"
                )
                raise MalformedExternalResponseError(
                    f"{source_label} returned {actual} choices; "
                    f"expected {expected_choices}"
                )

            indices = [getattr(choice, "index", None) for choice in choices]
            if (
                any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in indices
                )
                or sorted(indices) != list(range(expected_choices))
            ):
                raise MalformedExternalResponseError(
                    f"{source_label} returned invalid choice indices: {indices}"
                )

            ordered_choices = sorted(
                choices,
                key=lambda choice: choice.index,
            )
            serialized_logprobs = [
                self._serialize_choice_scoring_logprobs(
                    getattr(choice, "logprobs", None)
                )
                for choice in ordered_choices
            ]
            for choice_index, payload in enumerate(
                serialized_logprobs
            ):
                if not isinstance(payload, dict):
                    raise MalformedExternalResponseError(
                        f"{source_label} returned missing or invalid "
                        f"logprobs for choice {choice_index}"
                    )

            usage = self._serialize_external_usage(
                getattr(response, "usage", None)
            )
            generated_tail_counts = [0] * len(ordered_choices)
            if any(
                payload.get("text_offset") in (None, [])
                for payload in serialized_logprobs
            ):
                completion_tokens = (
                    usage.get("completion_tokens")
                    if usage is not None
                    else None
                )
                if (
                    isinstance(completion_tokens, bool)
                    or not isinstance(completion_tokens, int)
                    or completion_tokens < 0
                ):
                    raise MalformedExternalResponseError(
                        f"{source_label} must report a non-negative integer "
                        "usage.completion_tokens when text offsets are absent"
                    )
                texts = [
                    getattr(choice, "text", None)
                    for choice in ordered_choices
                ]
                if any(not isinstance(text, str) for text in texts):
                    raise MalformedExternalResponseError(
                        f"{source_label} must report string generated text "
                        "when text offsets are absent"
                    )
                generated_tail_counts = [
                    self._generated_tail_count_from_echoed_text(
                        text,
                        prompts[choice_index],
                        choice_index=choice_index,
                        source_label=source_label,
                    )
                    for choice_index, text in enumerate(texts)
                ]
                if completion_tokens != sum(generated_tail_counts):
                    raise MalformedExternalResponseError(
                        f"{source_label} returned ambiguous generated-tail "
                        "evidence: usage.completion_tokens="
                        f"{completion_tokens}, non-empty choices="
                        f"{sum(generated_tail_counts)}"
                    )

            for choice_index, choice in enumerate(ordered_choices):
                payload = serialized_logprobs[choice_index]
                try:
                    sum_choice_scoring_suffix_logprobs_with_count(
                        payload,
                        suffix_n_tokens[choice_index],
                        suffix_start_chars=(
                            suffix_start_chars[choice_index]
                        ),
                        prompt_text=prompts[choice_index],
                        strict=True,
                        generated_tail_n_tokens=(
                            generated_tail_counts[choice_index]
                        ),
                    )
                except ValueError as error:
                    raise MalformedExternalResponseError(
                        f"{source_label} returned unusable suffix logprobs "
                        f"for choice {choice_index}: {error}"
                    ) from error
                if payload.get("text_offset") in (None, []):
                    payload[GENERATED_TAIL_N_TOKENS_FIELD] = (
                        generated_tail_counts[choice_index]
                    )

            normalized = _ChoiceScoringResponse(
                logprobs=serialized_logprobs,
                metadata=[
                    self._serialize_external_choice_metadata(choice)
                    for choice in ordered_choices
                ],
                usages=[usage] if usage is not None else [],
            )
            self._validate_response_delta_json(
                {
                    "logprobs": normalized.logprobs,
                    "metadata": normalized.metadata,
                    "usages": normalized.usages,
                },
                source_label=source_label,
            )
            return normalized
        except MalformedExternalResponseError:
            raise
        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise MalformedExternalResponseError(
                f"{source_label} returned malformed choice-scoring metadata: "
                f"{type(error).__name__}: {error}"
            ) from error

    @staticmethod
    def _is_choice_scoring_malformed_error(exc: Exception) -> bool:
        return _CHOICE_SCORING_MALFORMED_MARKER in str(exc)

    @staticmethod
    def _is_choice_scoring_retryable_error(exc: Exception) -> bool:
        return isinstance(
            exc,
            (httpx.HTTPError, aiohttp.ClientError, APIConnectionError),
        )

    async def _create_choice_scoring_response(
        self,
        client: AsyncOpenAI | None,
        model_name: str,
        prompts: list[str],
        suffix_start_chars: list[int],
        suffix_n_tokens: list[int | None],
        response_factory: (
            Callable[
                [list[str], list[int], list[int | None]],
                Awaitable,
            ]
            | None
        ) = None,
    ):
        if response_factory is not None:
            return await response_factory(
                prompts,
                suffix_start_chars,
                suffix_n_tokens,
            )
        if client is None:
            raise ValueError(
                "choice-scoring client is required without a response factory"
            )

        for attempt, delay in enumerate(
            (*_CHOICE_SCORING_RETRY_DELAYS_SECONDS, None),
            start=1,
        ):
            try:
                return await client.completions.create(
                    model=model_name,
                    prompt=prompts,
                    n=1,
                    **self._choice_scoring_request_kwargs(),
                )
            except Exception as exc:
                if (
                    delay is None
                    or self._is_choice_scoring_malformed_error(exc)
                    or not self._is_choice_scoring_retryable_error(exc)
                ):
                    raise
                logger.warning(
                    "choice_scoring request failed on attempt %s; retrying in %.2fs: %s",
                    attempt,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("unreachable choice_scoring retry state")

    async def _run_choice_scoring_prompt_batch(
        self,
        client: AsyncOpenAI | None,
        model_name: str,
        prompts: list[str],
        *,
        prompt_kind: str,
        suffix_start_chars: list[int],
        suffix_n_tokens: list[int | None],
        response_factory: (
            Callable[
                [list[str], list[int], list[int | None]],
                Awaitable,
            ]
            | None
        ) = None,
    ) -> tuple[_ChoiceScoringResponse, str]:
        try:
            response = await self._create_choice_scoring_response(
                client=client,
                model_name=model_name,
                prompts=prompts,
                suffix_start_chars=suffix_start_chars,
                suffix_n_tokens=suffix_n_tokens,
                response_factory=response_factory,
            )
            normalized = (
                response
                if isinstance(response, _ChoiceScoringResponse)
                else self._normalize_choice_scoring_response(
                    response,
                    expected_choices=len(prompts),
                    source_label=f"choice-scoring {prompt_kind} endpoint",
                    prompts=prompts,
                    suffix_start_chars=suffix_start_chars,
                    suffix_n_tokens=suffix_n_tokens,
                )
            )
            return normalized, "batched"
        except Exception as exc:
            if not self._is_choice_scoring_malformed_error(exc) or len(prompts) <= 1:
                raise

            logger.warning(
                "choice_scoring %s batched request hit malformed MessagePack; "
                "falling back to one request per choice",
                prompt_kind,
            )

        logprobs = []
        metadata = []
        usages = []
        for choice_idx, prompt in enumerate(prompts):
            response = await self._create_choice_scoring_response(
                client=client,
                model_name=model_name,
                prompts=[prompt],
                suffix_start_chars=[
                    suffix_start_chars[choice_idx]
                ],
                suffix_n_tokens=[suffix_n_tokens[choice_idx]],
                response_factory=response_factory,
            )
            normalized = (
                response
                if isinstance(response, _ChoiceScoringResponse)
                else self._normalize_choice_scoring_response(
                    response,
                    expected_choices=1,
                    source_label=(
                        f"choice-scoring {prompt_kind} fallback "
                        f"choice {choice_idx}"
                    ),
                    prompts=[prompt],
                    suffix_start_chars=[
                        suffix_start_chars[choice_idx]
                    ],
                    suffix_n_tokens=[
                        suffix_n_tokens[choice_idx]
                    ],
                )
            )
            logprobs.extend(normalized.logprobs)
            metadata.extend(normalized.metadata)
            usages.extend(normalized.usages)

        return (
            _ChoiceScoringResponse(
                logprobs=logprobs,
                metadata=metadata,
                usages=usages,
            ),
            "single_prompt_fallback",
        )

    async def _run_choice_scoring_requests(
        self,
        client: AsyncOpenAI | None,
        model_name: str,
        elem: dict,
        response_factory: (
            Callable[
                [list[str], list[int], list[int | None]],
                Awaitable,
            ]
            | None
        ) = None,
    ) -> dict:
        validate_choice_scoring_row(elem, phase="request")
        completions = resolve_scoring_completions(elem)
        full_prompts, completion_prompts = build_choice_scoring_prompts(elem)
        prefix = self._model.prompt_prefix_instructions
        if prefix:
            full_prompts = [f"{prefix}\n\n{prompt}" for prompt in full_prompts]
        if (
            len(full_prompts) != len(completions)
            or len(completion_prompts) != len(completions)
        ):
            raise ValueError(
                "unexpected choice-scoring prompt length: "
                f"{len(full_prompts)=} {len(completion_prompts)=} {len(completions)=}"
            )

        completion_token_counts = (
            resolve_scoring_completion_token_counts(
                elem,
                len(completions),
                required=False,
            )
        )
        suffix_n_tokens = (
            completion_token_counts
            if completion_token_counts is not None
            else [None] * len(completions)
        )
        full_suffix_start_chars = [
            choice_scoring_suffix_start_chars(
                prompt,
                completion,
                max(0, len(prompt) - len(str(completion))),
            )
            for prompt, completion in zip(
                full_prompts,
                completions,
            )
        ]
        completion_suffix_start_chars = [
            choice_scoring_suffix_start_chars(
                prompt,
                completion,
                max(0, len(prompt) - len(str(completion))),
            )
            for prompt, completion in zip(
                completion_prompts,
                completions,
            )
        ]

        full_response, full_mode = await self._run_choice_scoring_prompt_batch(
            client=client,
            model_name=model_name,
            prompts=full_prompts,
            prompt_kind="full-prompt",
            suffix_start_chars=full_suffix_start_chars,
            suffix_n_tokens=suffix_n_tokens,
            response_factory=response_factory,
        )
        completion_response, completion_mode = (
            await self._run_choice_scoring_prompt_batch(
                client=client,
                model_name=model_name,
                prompts=completion_prompts,
                prompt_kind="completion-only",
                suffix_start_chars=completion_suffix_start_chars,
                suffix_n_tokens=suffix_n_tokens,
                response_factory=response_factory,
            )
        )
        generation_metadata = (
            full_response.metadata + completion_response.metadata
        )
        choice_scoring_metadata = {
            "full_prompts": full_prompts,
            "completion_prompts": completion_prompts,
        }
        if full_mode != "batched" or completion_mode != "batched":
            choice_scoring_metadata["request_modes"] = {
                "full": full_mode,
                "completion": completion_mode,
            }

        return {
            "choice_scoring_full_logprobs": [
                copy.deepcopy(logprobs)
                for logprobs in full_response.logprobs
            ],
            "choice_scoring_completion_logprobs": [
                copy.deepcopy(logprobs)
                for logprobs in completion_response.logprobs
            ],
            "choice_scoring_metadata": choice_scoring_metadata,
            "generation_metadata": generation_metadata,
            "response_usage": (
                full_response.usages + completion_response.usages
            ),
        }

    @staticmethod
    def _validate_response_choice_count(
        response,
        *,
        expected_choices: int,
        source_label: str,
    ) -> None:
        """Validate complete choice identity, then normalize by index."""
        choices = getattr(response, "choices", None)
        if (
            not isinstance(choices, list)
            or len(choices) != expected_choices
        ):
            actual_choices = (
                len(choices)
                if isinstance(choices, list)
                else "missing"
            )
            raise MalformedExternalResponseError(
                f"{source_label} returned {actual_choices} choices; "
                f"expected {expected_choices}"
            )

        indices = [getattr(choice, "index", None) for choice in choices]
        if any(type(index) is not int for index in indices):
            raise MalformedExternalResponseError(
                f"{source_label} returned a non-integer choice index"
            )
        expected_indices = set(range(expected_choices))
        if set(indices) != expected_indices:
            raise MalformedExternalResponseError(
                f"{source_label} returned choice indices {indices!r}; "
                f"expected exactly {sorted(expected_indices)!r}"
            )
        choices.sort(key=lambda choice: choice.index)

    @staticmethod
    def _validate_parallel_logprobs(
        response,
        *,
        source_label: str,
    ) -> None:
        """Require logprobs to be present for every choice or for none."""
        has_logprobs = [
            getattr(choice, "logprobs", None) is not None
            for choice in response.choices
        ]
        if any(has_logprobs) and not all(has_logprobs):
            raise MalformedExternalResponseError(
                f"{source_label} returned mixed logprobs presence across "
                "choices; expected all choices or none"
            )

    def _validate_completion_response(
        self,
        response,
        *,
        expected_choices: int,
        source_label: str,
    ) -> None:
        self._validate_response_choice_count(
            response,
            expected_choices=expected_choices,
            source_label=source_label,
        )
        invalid_text_count = sum(
            not isinstance(getattr(choice, "text", None), str)
            for choice in response.choices
        )
        if invalid_text_count:
            raise MalformedExternalResponseError(
                f"{source_label} returned non-string text for "
                f"{invalid_text_count}/{len(response.choices)} choices"
            )
        self._validate_parallel_logprobs(
            response,
            source_label=source_label,
        )

    def _validate_chat_response(
        self,
        response,
        *,
        expected_choices: int,
        source_label: str,
    ) -> None:
        self._validate_response_choice_count(
            response,
            expected_choices=expected_choices,
            source_label=source_label,
        )
        for choice in response.choices:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if (
                message is None
                or (content is not None and not isinstance(content, str))
                or (
                    content is None
                    and choice.finish_reason not in ("stop", "length")
                )
            ):
                raise MalformedExternalResponseError(
                    f"{source_label} returned invalid chat content; expected "
                    "string content or null content with a terminal "
                    "stop/length finish reason"
                )
        self._validate_parallel_logprobs(
            response,
            source_label=source_label,
        )

    @staticmethod
    def _serialize_external_choice_metadata(choice) -> dict:
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None and not isinstance(
            finish_reason,
            str,
        ):
            raise MalformedExternalResponseError(
                "external response returned a non-string finish_reason"
            )
        stop_reason = getattr(choice, "stop_reason", None)
        if (
            stop_reason is not None
            and type(stop_reason) not in (str, int)
        ):
            raise MalformedExternalResponseError(
                "external response returned a non-scalar stop_reason"
            )
        return {
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
        }

    @staticmethod
    def _serialize_external_usage(usage) -> dict | None:
        if usage is None:
            return None
        return _json_mode_mapping(usage, field_name="usage")

    @staticmethod
    def _extract_external_reasoning_text(choice) -> str | None:
        reasoning = getattr(
            choice.message,
            "reasoning_content",
            None,
        )
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise MalformedExternalResponseError(
                    "external response returned non-string "
                    "reasoning_content"
                )
            if reasoning:
                return reasoning

        extra = getattr(choice.message, "model_extra", None)
        if extra is None:
            return None
        if not isinstance(extra, dict):
            raise MalformedExternalResponseError(
                "external response returned non-mapping message extras"
            )
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise MalformedExternalResponseError(
                    f"external response returned non-string {key}"
                )
            if value:
                return value
        return None

    def _record_completion_response(
        self,
        result: dict,
        response,
        *,
        source_label: str,
    ) -> None:
        texts = [choice.text for choice in response.choices]
        if any(text is None for text in texts):
            raise APIConnectionError(
                request=None,
                message=(
                    f"{source_label} returned null text for "
                    f"{sum(text is None for text in texts)}/"
                    f"{len(texts)} choices — server may be dying"
                ),
            )
        result[self._new_field_name] += texts
        result.setdefault("generation_metadata", []).extend(
            self._serialize_external_choice_metadata(choice)
            for choice in response.choices
        )
        usage = self._serialize_external_usage(
            getattr(response, "usage", None)
        )
        if usage is not None:
            result.setdefault("response_usage", []).append(usage)
        finish_reasons = [
            choice.finish_reason for choice in response.choices
        ]
        if any(reason is not None for reason in finish_reasons):
            result["finish_reasons"] = finish_reasons
        if all(
            choice.logprobs is not None
            for choice in response.choices
        ):
            result.setdefault("logprobs", []).extend(
                _json_mode_mapping(
                    choice.logprobs,
                    field_name="logprobs",
                )
                for choice in response.choices
            )

    def _record_chat_response(
        self,
        result: dict,
        response,
        *,
        source_label: str,
    ) -> None:
        contents = []
        reasoning_values = []
        tool_call_values = []
        finish_reasons = []
        for choice in response.choices:
            reasoning = self._extract_external_reasoning_text(choice)
            content = choice.message.content
            if content == "" and reasoning:
                content = reasoning
            elif content is None:
                content = ""
            contents.append(content)
            result.setdefault("generation_metadata", []).append(
                self._serialize_external_choice_metadata(choice)
            )
            reasoning_values.append(reasoning or None)
            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls is not None and not isinstance(tool_calls, list):
                raise MalformedExternalResponseError(
                    "response returned non-list tool_calls metadata"
                )
            tool_call_values.append(
                [
                    _json_mode_mapping(
                        tool_call,
                        field_name="tool call",
                    )
                    for tool_call in tool_calls
                ]
                if tool_calls
                else None
            )
            finish_reasons.append(choice.finish_reason)
        if any(value is not None for value in reasoning_values):
            result["reasoning"] = reasoning_values
        if any(value is not None for value in tool_call_values):
            result["tool_calls"] = tool_call_values
        if any(reason is not None for reason in finish_reasons):
            result["finish_reasons"] = finish_reasons
        result[self._new_field_name] += contents
        usage = self._serialize_external_usage(
            getattr(response, "usage", None)
        )
        if usage is not None:
            result.setdefault("response_usage", []).append(usage)
        if all(
            choice.logprobs is not None
            for choice in response.choices
        ):
            result.setdefault("logprobs", []).extend(
                _json_mode_mapping(
                    choice.logprobs,
                    field_name="logprobs",
                )
                for choice in response.choices
            )

    @staticmethod
    def _validate_response_delta_json(
        delta: dict[str, list],
        *,
        source_label: str,
    ) -> None:
        """Prove the complete response delta is strict-JSON serializable."""
        try:
            json.dumps(
                delta,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise MalformedExternalResponseError(
                f"{source_label} returned non-JSON response metadata: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _completion_response_delta(
        self,
        response,
        *,
        expected_choices: int,
        source_label: str,
    ) -> dict[str, list]:
        """Fully parse one completion response before mutating a result."""
        delta = {self._new_field_name: []}
        try:
            self._validate_completion_response(
                response,
                expected_choices=expected_choices,
                source_label=source_label,
            )
            self._record_completion_response(
                delta,
                response,
                source_label=source_label,
            )
            self._validate_response_delta_json(
                delta,
                source_label=source_label,
            )
        except MalformedExternalResponseError:
            raise
        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise MalformedExternalResponseError(
                f"{source_label} returned malformed completion metadata: "
                f"{type(error).__name__}: {error}"
            ) from error
        return delta

    def _chat_response_delta(
        self,
        response,
        *,
        expected_choices: int,
        source_label: str,
    ) -> dict[str, list]:
        """Fully parse one chat response before mutating a result."""
        delta = {self._new_field_name: []}
        try:
            self._validate_chat_response(
                response,
                expected_choices=expected_choices,
                source_label=source_label,
            )
            self._record_chat_response(
                delta,
                response,
                source_label=source_label,
            )
            self._validate_response_delta_json(
                delta,
                source_label=source_label,
            )
        except MalformedExternalResponseError:
            raise
        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise MalformedExternalResponseError(
                f"{source_label} returned malformed chat metadata: "
                f"{type(error).__name__}: {error}"
            ) from error
        return delta

    def _commit_response_delta(
        self,
        result: dict,
        delta: dict[str, list],
    ) -> None:
        """Commit an already-validated response delta without partial writes."""
        committed = copy.deepcopy(result)
        existing_choices = committed.get(self._new_field_name, [])
        new_choices = delta.get(self._new_field_name, [])
        if not isinstance(existing_choices, list):
            raise TypeError(
                f"input field {self._new_field_name!r} conflicts with "
                "generation output"
            )
        if not isinstance(new_choices, list):
            raise TypeError(
                f"response delta field {self._new_field_name!r} must be "
                "a list"
            )
        old_choice_count = len(existing_choices)
        new_choice_count = len(new_choices)
        optional_choice_fields = (
            "finish_reasons",
            "reasoning",
            "tool_calls",
        )
        for key in optional_choice_fields:
            if key not in committed and key not in delta:
                continue
            destination = committed.setdefault(
                key,
                [None] * old_choice_count,
            )
            if not isinstance(destination, list):
                raise TypeError(
                    f"input field {key!r} conflicts with generation output"
                )
            if len(destination) != old_choice_count:
                raise TypeError(
                    f"input field {key!r} is not parallel to generation "
                    "output"
                )
            values = delta.get(key, [None] * new_choice_count)
            if not isinstance(values, list) or len(values) != new_choice_count:
                raise TypeError(
                    f"response delta field {key!r} is not parallel to "
                    "generation output"
                )
            destination.extend(values)
        for key, values in delta.items():
            if key in optional_choice_fields:
                continue
            if not isinstance(values, list):
                raise TypeError(
                    f"response delta field {key!r} must be a list"
                )
            destination = committed.setdefault(key, [])
            if not isinstance(destination, list):
                raise TypeError(
                    f"input field {key!r} conflicts with generation output"
                )
            destination.extend(values)
        result.clear()
        result.update(committed)

    def _validate_response_delta_against_result(
        self,
        result: dict,
        delta: dict[str, list],
        *,
        source_label: str,
    ) -> None:
        """Require one logprobs mode across every batch for a logical row."""
        committed_choices = result.get(self._new_field_name, [])
        if not committed_choices:
            return
        committed_has_logprobs = "logprobs" in result
        delta_has_logprobs = "logprobs" in delta
        if committed_has_logprobs != delta_has_logprobs:
            raise MalformedExternalResponseError(
                f"{source_label} changed logprobs presence across batches "
                "for one logical generation row"
            )

    def _mark_input_too_long(
        self,
        result: dict,
        requests_left: int,
    ) -> None:
        """Record permanent context overflow without assuming result shape."""
        generations = result.get(self._new_field_name)
        if not isinstance(generations, list):
            generations = []
            result[self._new_field_name] = generations
        committed_choices = len(generations)
        for field_name in (
            "finish_reasons",
            "generation_metadata",
            "logprobs",
            "reasoning",
            "tool_calls",
        ):
            values = result.get(field_name)
            if isinstance(values, list) and len(values) == committed_choices:
                values.extend([None] * requests_left)
        generations.extend([""] * requests_left)
        result["eval360_input_too_long"] = True

    async def _acquire_external_pool_lease(
        self,
        pool: ModelConnectionPool,
        requests_left: int,
    ) -> tuple[str, int]:
        """Acquire local capacity without charging an outbound HTTP attempt."""
        try:
            url, connections = await pool.acquire(requests_left)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _PoolAcquisitionFailure(error) from error
        return url, connections

    @staticmethod
    async def _release_external_pool_lease(
        pool: ModelConnectionPool,
        lease: tuple[str, int],
    ) -> None:
        url, connections = lease
        await pool.release(url, connections)

    async def _run_external_pool_attempt(
        self,
        prepared_attempt: tuple[AsyncOpenAI, str, int],
        request_factory: Callable[
            [AsyncOpenAI, str, int],
            Awaitable[ResponseT],
        ],
    ) -> tuple[ResponseT, int]:
        """Run only the outbound portion while a pool lease is held."""
        client, url, connections = prepared_attempt
        self._in_flight += 1
        if self._in_flight == 1:
            self._progress_manager.set_status(
                self._event_instance,
                "Generating",
            )
        try:
            response = await request_factory(client, url, connections)
            return response, connections
        finally:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._progress_manager.set_status(
                    self._event_instance,
                    "Queued",
                )

    async def _prepare_external_pool_attempt(
        self,
        lease: tuple[str, int],
    ) -> tuple[AsyncOpenAI, str, int]:
        """Finish local client setup before provider and attempt accounting."""
        url, connections = lease
        return self._get_client(url), url, connections

    async def _run_external_pool_request(
        self,
        runner: ExternalRequestRunner,
        pool: ModelConnectionPool,
        requests_left: int,
        operation_started_at: float,
        request_factory: Callable[
            [AsyncOpenAI, str, int],
            Awaitable[ResponseT],
        ],
        *,
        passthrough_error: (
            Callable[[Exception], bool] | None
        ) = None,
    ) -> tuple[ResponseT, int]:
        """Apply retry policy after admission and release every pool lease."""
        return await runner.run(
            lambda lease: self._run_external_pool_attempt(
                lease,
                request_factory,
            ),
            acquire_factory=lambda: self._acquire_external_pool_lease(
                pool,
                requests_left,
            ),
            prepare_factory=self._prepare_external_pool_attempt,
            release_factory=lambda lease: (
                self._release_external_pool_lease(pool, lease)
            ),
            started_at=operation_started_at,
            passthrough_error=passthrough_error,
        )

    async def _make_external_request(
        self,
        i: int,
        elem: dict,
        result: dict,
        num_generations: int,
        offset: int,
    ) -> dict | ExceptionWrapper:
        """Run one external prompt with a deadline shared by all HTTP calls."""
        requests_left = num_generations
        pool = LOCKED_CONNECTIONS[self._model.serving_key]
        operation_started_at = self._monotonic()
        runner = ExternalRequestRunner(
            self._external_retry_policy,
            rate_limiter=self._rate_limiter,
            monotonic=self._monotonic,
        )

        while requests_left:
            if self._cancelled.is_set():
                self._progress_manager.update(
                    event=self._event_instance,
                    index=offset + i,
                    completed=num_generations,
                    new_elems=requests_left,
                    mode="Generation",
                )
                return self._make_cancelled_result(result)

            try:
                if self._is_choice_scoring_task():
                    if not self._is_base_model_type(
                        self._model.model_type
                    ):
                        raise ValueError(
                            "choice_scoring requires a base/completions model"
                        )
                    if not is_choice_scoring_row(elem):
                        validate_choice_scoring_row(elem, phase="request")
                    if requests_left != 1:
                        raise ValueError(
                            "choice_scoring only supports average_over=pass_at=1 "
                            f"(got {requests_left=})"
                        )

                    async def create_choice_response(
                        prompts: list[str],
                        suffix_start_chars: list[int],
                        suffix_n_tokens: list[int | None],
                    ):
                        async def issue_request(
                            client: AsyncOpenAI,
                            _url: str,
                            connections: int,
                        ):
                            if connections != 1:
                                raise ValueError(
                                    "choice_scoring requires exactly one "
                                    f"connection (got {connections=})"
                                )
                            try:
                                response = await client.completions.create(
                                    model=self._request_model_name,
                                    prompt=prompts,
                                    n=1,
                                    **self._choice_scoring_request_kwargs(),
                                )
                            except Exception as error:
                                if (
                                    len(prompts) == 1
                                    and self._is_choice_scoring_malformed_error(
                                        error
                                    )
                                ):
                                    raise MalformedExternalResponseError(
                                        "external choice-scoring fallback "
                                        "returned malformed MessagePack"
                                    ) from error
                                raise
                            return self._normalize_choice_scoring_response(
                                response,
                                expected_choices=len(prompts),
                                source_label=(
                                    "external choice-scoring endpoint"
                                ),
                                prompts=prompts,
                                suffix_start_chars=suffix_start_chars,
                                suffix_n_tokens=suffix_n_tokens,
                            )

                        response, _ = await self._run_external_pool_request(
                            runner,
                            pool,
                            1,
                            operation_started_at,
                            issue_request,
                            passthrough_error=(
                                self._is_choice_scoring_malformed_error
                                if len(prompts) > 1
                                else None
                            ),
                        )
                        return response

                    choice_scoring_delta = (
                        await self._run_choice_scoring_requests(
                            client=None,
                            model_name=self._request_model_name,
                            elem=elem,
                            response_factory=create_choice_response,
                        )
                    )
                    self._validate_response_delta_json(
                        choice_scoring_delta,
                        source_label="external choice-scoring transaction",
                    )
                    committed = copy.deepcopy(result)
                    committed.pop(self._new_field_name, None)
                    committed.update(choice_scoring_delta)
                    result.clear()
                    result.update(committed)
                    self._progress_manager.update(
                        event=self._event_instance,
                        index=offset + i,
                        completed=num_generations,
                        new_elems=1,
                        mode="Generation",
                    )
                    requests_left = 0
                    continue

                if self._is_base_model_type(self._model.model_type):
                    prefix = self._model.prompt_prefix_instructions
                    prompt = (
                        prefix + "\n\n" + elem["completion_input"]
                        if prefix
                        else elem["completion_input"]
                    )
                    if self._debug and "raw_text_input" not in result:
                        result["raw_text_input"] = prompt
                    openai_kwargs = (
                        self._request_kwargs_with_cache_salt()
                    )

                    async def issue_request(
                        client: AsyncOpenAI,
                        _url: str,
                        connections: int,
                    ):
                        response = await client.completions.create(
                            model=self._request_model_name,
                            prompt=prompt,
                            n=connections,
                            **openai_kwargs,
                        )
                        delta = self._completion_response_delta(
                            response,
                            expected_choices=connections,
                            source_label="external endpoint",
                        )
                        self._validate_response_delta_against_result(
                            result,
                            delta,
                            source_label="external endpoint",
                        )
                        return delta

                    response_delta, connections = (
                        await self._run_external_pool_request(
                            runner,
                            pool,
                            requests_left,
                            operation_started_at,
                            issue_request,
                        )
                    )
                    self._commit_response_delta(result, response_delta)
                elif self._is_chat_model_type(self._model.model_type):
                    prefix = self._model.prompt_prefix_instructions
                    if prefix:
                        messages = copy.deepcopy(elem["chat_input"])
                        if (
                            messages
                            and messages[0]["role"] == "system"
                        ):
                            messages[0]["content"] = (
                                prefix
                                + "\n\n"
                                + messages[0]["content"]
                            )
                        else:
                            messages.insert(
                                0,
                                {
                                    "role": "system",
                                    "content": prefix,
                                },
                            )
                    else:
                        messages = elem["chat_input"]
                    if self._debug and "raw_text_input" not in result:
                        result["raw_text_input"] = (
                            await self._get_templated_prompt(
                                self._model.base_url,
                                messages,
                            )
                        )
                    openai_kwargs = (
                        self._request_kwargs_with_cache_salt()
                    )

                    async def issue_request(
                        client: AsyncOpenAI,
                        _url: str,
                        connections: int,
                    ):
                        response = await client.chat.completions.create(
                            model=self._request_model_name,
                            messages=messages,
                            n=connections,
                            **openai_kwargs,
                        )
                        delta = self._chat_response_delta(
                            response,
                            expected_choices=connections,
                            source_label="external endpoint",
                        )
                        self._validate_response_delta_against_result(
                            result,
                            delta,
                            source_label="external endpoint",
                        )
                        return delta

                    response_delta, connections = (
                        await self._run_external_pool_request(
                            runner,
                            pool,
                            requests_left,
                            operation_started_at,
                            issue_request,
                        )
                    )
                    self._commit_response_delta(result, response_delta)
                else:
                    raise ValueError(
                        f"unsupported model type: {self._model.model_type}"
                    )

                requests_left -= connections
                self._progress_manager.update(
                    event=self._event_instance,
                    index=offset + i,
                    completed=num_generations - requests_left,
                    new_elems=connections,
                    mode="Generation",
                )
            except _PoolAcquisitionFailure as error:
                raise error.original_exception
            except ExternalRequestFailure as error:
                if _is_input_too_long(error.original_exception):
                    self._mark_input_too_long(result, requests_left)
                else:
                    result = ExceptionWrapper.from_exception(error, result)
                self._progress_manager.update(
                    event=self._event_instance,
                    index=offset + i,
                    completed=num_generations,
                    new_elems=requests_left,
                    mode="Generation",
                )
                requests_left = 0
            except Exception as error:
                if _is_input_too_long(error):
                    self._mark_input_too_long(result, requests_left)
                else:
                    result = ExceptionWrapper.from_exception(error, result)
                self._progress_manager.update(
                    event=self._event_instance,
                    index=offset + i,
                    completed=num_generations,
                    new_elems=requests_left,
                    mode="Generation",
                )
                requests_left = 0

        return result

    async def launch_requests(
            self,
            requests: AsyncIterator,
            offset: int,
            completion_hook) -> AsyncIterator:
        tasks = {}
        pending_results = {}
        queue = asyncio.Queue()
        num_generations = max(self._task.average_over + self._task.pass_at)
        completed_count = 0
        count_lock = asyncio.Lock()

        async def _make_request(i, elem, result, num_generations):
            nonlocal completed_count
            if self._is_external:
                result = await self._make_external_request(
                    i,
                    elem,
                    result,
                    num_generations,
                    offset,
                )
                await queue.put((i, result))
                async with count_lock:
                    completed_count += 1
                    self._completed_requests_count += 1
                return

            requests_left = num_generations
            pool = LOCKED_CONNECTIONS[self._model.serving_key]
            while requests_left:
                if self._rate_limiter is not None:
                    await self._rate_limiter.acquire()
                if self._cancelled.is_set():
                    result = self._make_cancelled_result(result)
                    self._progress_manager.update(event=self._event_instance, index=offset+i, completed=num_generations, new_elems=requests_left, mode="Generation")
                    requests_left = 0
                    break
                retries = 3
                done = False
                while not done:
                    # Acquire a URL + slot count atomically from the pool.
                    # Blocks if no replicas are live yet or all are at capacity.
                    url, connections = await pool.acquire(requests_left)
                    client = self._get_client(url)
                    self._in_flight += 1
                    if self._in_flight == 1:
                        self._progress_manager.set_status(self._event_instance, "Generating")
                    try:
                        if self._is_choice_scoring_task():
                            if not self._is_base_model_type(self._model.model_type):
                                raise ValueError("choice_scoring requires a base/completions model")
                            if not is_choice_scoring_row(elem):
                                validate_choice_scoring_row(elem, phase="request")
                            if requests_left != 1 or connections != 1:
                                raise ValueError(
                                    "choice_scoring only supports average_over=pass_at=1 "
                                    f"(got {requests_left=} {connections=})"
                                )
                            result.pop(self._new_field_name, None)
                            result.update(
                                await self._run_choice_scoring_requests(
                                    client=client,
                                    model_name=self._request_model_name,
                                    elem=elem,
                                )
                            )
                            self._progress_manager.update(
                                event=self._event_instance,
                                index=offset+i,
                                completed=num_generations,
                                new_elems=1,
                                mode="Generation",
                            )
                            requests_left = 0
                            done = True
                            continue
                        elif self._is_base_model_type(self._model.model_type):
                            prefix = self._model.prompt_prefix_instructions
                            prompt = (prefix + "\n\n" + elem["completion_input"]) if prefix else elem["completion_input"]
                            if self._debug and "raw_text_input" not in result:
                                result["raw_text_input"] = prompt
                            openai_kwargs = self._request_kwargs_with_cache_salt()
                            response = await client.completions.create(
                                model=self._request_model_name,
                                prompt=prompt,
                                n=connections,
                                **openai_kwargs
                            )
                            texts = [choice.text for choice in response.choices]
                            if any(t is None for t in texts):
                                raise APIConnectionError(request=None,
                                    message=f"VLLM returned null text for {sum(t is None for t in texts)}/{len(texts)} choices — server may be dying")
                            result[self._new_field_name] += texts
                            result.setdefault("generation_metadata", []).extend(
                                _serialize_choice_metadata(choice) for choice in response.choices
                            )
                            usage = _serialize_usage(getattr(response, "usage", None))
                            if usage is not None:
                                result.setdefault("response_usage", []).append(usage)
                            for choice in response.choices:
                                if choice.finish_reason is not None:
                                    result.setdefault("finish_reasons", []).append(choice.finish_reason)
                            if response.choices[0].logprobs is not None:
                                result.setdefault("logprobs", []).extend(
                                    choice.logprobs.model_dump() for choice in response.choices)
                        elif self._is_chat_model_type(self._model.model_type):
                            prefix = self._model.prompt_prefix_instructions
                            if prefix:
                                messages = copy.deepcopy(elem["chat_input"])
                                if messages and messages[0]["role"] == "system":
                                    messages[0]["content"] = prefix + "\n\n" + messages[0]["content"]
                                else:
                                    messages.insert(0, {"role": "system", "content": prefix})
                            else:
                                messages = elem["chat_input"]
                            if self._debug and "raw_text_input" not in result:
                                result["raw_text_input"] = await self._get_templated_prompt(url, messages)
                            openai_kwargs = self._request_kwargs_with_cache_salt()
                            response = await client.chat.completions.create(
                                model=self._request_model_name,
                                messages=messages,
                                n=connections,
                                **openai_kwargs
                            )
                            contents = []
                            for choice in response.choices:
                                reasoning = _extract_reasoning_text(choice)
                                content = choice.message.content
                                if content == "" and reasoning:
                                    content = reasoning
                                elif content is None:
                                    if choice.finish_reason not in ("stop", "length"):
                                        raise RuntimeError(
                                            "VLLM returned null content for a choice with finish_reason != stop/length"
                                        )
                                    content = ""
                                contents.append(content)
                                result.setdefault("generation_metadata", []).append(
                                    _serialize_choice_metadata(choice)
                                )
                                if reasoning:
                                    result.setdefault("reasoning", []).append(reasoning)
                                tool_calls = getattr(choice.message, "tool_calls", None)
                                if isinstance(tool_calls, list) and tool_calls:
                                    result.setdefault("tool_calls", []).append(
                                        [tc.model_dump() for tc in tool_calls]
                                    )
                                if choice.finish_reason is not None:
                                    result.setdefault("finish_reasons", []).append(choice.finish_reason)
                            if any(c is None for c in contents):
                                if any(choice.finish_reason not in ("stop", "length") for choice in response.choices):
                                    raise RuntimeError(
                                        f"VLLM returned null content for {sum(c is None for c in contents)}/{len(contents)} choices with finish_reason != stop/length")
                                contents = [c if c is not None else "" for c in contents]
                            result[self._new_field_name] += contents
                            usage = _serialize_usage(getattr(response, "usage", None))
                            if usage is not None:
                                result.setdefault("response_usage", []).append(usage)
                            if response.choices[0].logprobs is not None:
                                result.setdefault("logprobs", []).extend(
                                    choice.logprobs.model_dump() for choice in response.choices)
                        requests_left -= connections
                        self._progress_manager.update(event=self._event_instance, index=offset+i, completed=num_generations-requests_left, new_elems=connections, mode="Generation")
                        done = True
                    except (httpx.HTTPError, aiohttp.ClientError, APIConnectionError) as e:
                        if not self._job_manager.is_url_live(url):
                            # Node died — break inner loop; outer loop retries
                            # via get_live_url() which blocks until a new node is healthy.
                            logger.info(f"Node {url} is dead, will retry on a live node")
                            done = True
                        elif isinstance(e, (httpx.ConnectError, aiohttp.ClientConnectorError, APIConnectionError)):
                            # Connection-level error: server unreachable even though URL not yet
                            # evicted. Break to outer loop — get_live_url() will block once the
                            # URL is evicted (within the next handle_job_update cycle).
                            done = True
                            logger.info(f"Connection error to {url}, will retry: {type(e).__name__}: {e}")
                        elif retries:
                            retries -= 1
                            continue
                        else:
                            tb = "".join(traceback.TracebackException.from_exception(e).format())
                            result = ExceptionWrapper(exception=e, trace=tb, instance=result)
                            self._progress_manager.update(event=self._event_instance, index=offset+i, completed=num_generations, new_elems=requests_left, mode="Generation")
                            requests_left = 0
                            done = True
                    except Exception as e:
                        if _is_input_too_long(e):
                            self._mark_input_too_long(
                                result,
                                requests_left,
                            )
                            self._progress_manager.update(event=self._event_instance, index=offset+i, completed=num_generations, new_elems=requests_left, mode="Generation")
                            requests_left = 0
                            done = True
                        else:
                            tb = "".join(traceback.TracebackException.from_exception(e).format())
                            result = ExceptionWrapper(exception=e, trace=tb, instance=result)
                            self._progress_manager.update(event=self._event_instance, index=offset+i, completed=num_generations, new_elems=requests_left, mode="Generation")
                            requests_left = 0
                            done = True
                    finally:
                        self._in_flight -= 1
                        if self._in_flight == 0:
                            self._progress_manager.set_status(self._event_instance, "Queued")
                        await pool.release(url, connections)
            await queue.put((i, result))
            async with count_lock:
                completed_count += 1
                self._completed_requests_count += 1

        async def make_request(i, elem, result, num_generations):
            try:
                await _make_request(i, elem, result, num_generations)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                await queue.put((i, error))

        done_reading = asyncio.Event()

        total = 0
        consumer_error = None
        async def consumer():
            nonlocal total, consumer_error
            index = 0
            first = True
            try:
                async for elem in requests:
                    if elem == Sentinel.COMPLETED:
                        await queue.put((index, elem))
                        continue
                    result = copy.deepcopy(elem)
                    result[self._new_field_name] = []
                    pending_results[index] = result
                    if self._cancelled.is_set():
                        if not self._emit_cancelled_results:
                            break
                        self._progress_manager.update(
                            event=self._event_instance,
                            index=offset+index,
                            completed=num_generations,
                            new_elems=num_generations,
                            mode="Generation",
                        )
                        await queue.put((index, self._make_cancelled_result(result)))
                        index += 1
                        continue
                    if first:
                        first = False
                        self._progress_manager.set_status(self._event_instance, "Queued")
                    task = asyncio.create_task(make_request(index, elem, result, num_generations))
                    tasks[index] = task
                    index += 1
            except Exception as e:
                consumer_error = e
                await queue.put((index, e))
            finally:
                total = index
                done_reading.set()

        consumer_task = asyncio.create_task(consumer())
        # External models don't expose VLLM's /metrics endpoint — skip the poller
        if self._is_external:
            poller_task = asyncio.ensure_future(asyncio.sleep(0))  # no-op placeholder
        else:
            poller_task = asyncio.create_task(self._poll_vllm_metrics())

        completed_hook_called = False
        saw_completed = False
        cancel_watch = asyncio.create_task(self._cancelled.wait())
        get_task = None
        try:
            while not done_reading.is_set() or tasks:
                get_task = asyncio.create_task(queue.get())
                done_set, _ = await asyncio.wait(
                    [get_task, cancel_watch], return_when=asyncio.FIRST_COMPLETED
                )
                if cancel_watch in done_set:
                    if get_task in done_set:
                        i, result = get_task.result()
                        if result is Sentinel.COMPLETED:
                            saw_completed = True
                        elif not isinstance(result, Exception):
                            yield result
                        tasks.pop(i, None)
                    else:
                        get_task.cancel()
                    if not self._emit_cancelled_results:
                        consumer_task.cancel()
                        for t in tasks.values():
                            t.cancel()
                    else:
                        in_flight_indices = set(tasks.keys())
                        for t in tasks.values():
                            t.cancel()
                        if tasks:
                            await asyncio.gather(*tasks.values(), return_exceptions=True)
                        try:
                            await consumer_task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            consumer_error = e
                        # Drain remaining queue results
                        drained_indices = set()
                        while True:
                            try:
                                i, result = queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if result is Sentinel.COMPLETED:
                                pass
                            elif not isinstance(result, Exception):
                                yield result
                            drained_indices.add(i)
                        # Yield cancelled results for prompts that were in-flight
                        # and did not produce a result during the drain
                        unfinished = []
                        for i in in_flight_indices:
                            if i in drained_indices:
                                continue
                            result = pending_results.get(i)
                            if result is None:
                                continue
                            completed = self._completed_generation_count(result)
                            remaining = max(0, num_generations - completed)
                            if remaining:
                                self._progress_manager.update(
                                    event=self._event_instance,
                                    index=offset+i,
                                    completed=num_generations,
                                    new_elems=remaining,
                                    mode="Generation",
                                )
                            yield self._make_cancelled_result(result)
                            unfinished.append(i)
                        if unfinished:
                            logger.warning(
                                "Generation cancelled with %d unfinished prompt(s); "
                                "writing exception rows so the run can complete",
                                len(unfinished),
                            )
                        if not completed_hook_called and consumer_error is None:
                            await completion_hook()
                            completed_hook_called = True
                    if self._emit_cancelled_results:
                        yield Sentinel.COMPLETED
                    return
                i, result = get_task.result()
                tasks.pop(i, None)
                if result is Sentinel.COMPLETED:
                    saw_completed = True
                    continue
                if not completed_hook_called and completed_count == total and consumer_error is None:
                    await completion_hook()
                    completed_hook_called = True
                if isinstance(result, Exception):
                    logger.debug(f"Error {result} encountered: stopping iteration")
                    raise result
                yield result
        finally:
            cleanup_tasks = [
                cancel_watch,
                poller_task,
                consumer_task,
                *tasks.values(),
            ]
            if get_task is not None:
                cleanup_tasks.append(get_task)
            for task in cleanup_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *cleanup_tasks,
                return_exceptions=True,
            )
        # Drain any remaining items from the queue (e.g. Sentinel.COMPLETED that
        # arrived after all tasks completed but before the loop condition checked).
        while True:
            try:
                i, result = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if result is Sentinel.COMPLETED:
                saw_completed = True
            elif isinstance(result, Exception):
                if consumer_error is None:
                    consumer_error = result
            else:
                yield result
        if consumer_error is not None:
            raise consumer_error
        if not completed_hook_called:
            await completion_hook()
        if saw_completed:
            yield Sentinel.COMPLETED
