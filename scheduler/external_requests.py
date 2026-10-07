from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import httpx
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
)

from .rate_limiter import RateLimiter


T = TypeVar("T")


class ExternalRetryPolicyLike(Protocol):
    max_attempts: int
    request_timeout_seconds: float
    total_deadline_seconds: float
    initial_backoff_seconds: float
    max_backoff_seconds: float
    jitter: str


# External generation and judge clients share one limiter per non-secret
# endpoint/credential quota identity. ``None`` records an explicit no-limit
# registration so a later live config cannot silently change the policy.
RATE_LIMITERS: dict[str, RateLimiter | None] = {}


class MalformedExternalResponseError(Exception):
    """An HTTP-success response that cannot satisfy the caller's contract."""


_EXTERNAL_REQUEST_ERRORS = (
    httpx.HTTPError,
    aiohttp.ClientError,
    asyncio.TimeoutError,
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    json.JSONDecodeError,
    MalformedExternalResponseError,
)


def canonicalize_external_endpoint(endpoint: str) -> str:
    """Return one identity for semantically equivalent endpoint URL spellings."""
    candidate = endpoint.strip()
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return candidate.rstrip("/")

    if not parsed.scheme or hostname is None:
        return candidate.rstrip("/")

    scheme = parsed.scheme.lower()
    hostname = hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"

    userinfo = ""
    if "@" in parsed.netloc:
        userinfo = f"{parsed.netloc.rsplit('@', 1)[0]}@"

    default_port = (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    )
    normalized_port = "" if port is None or default_port else f":{port}"
    normalized_path = parsed.path.rstrip("/")
    return urlunsplit(
        (
            scheme,
            f"{userinfo}{hostname}{normalized_port}",
            normalized_path,
            parsed.query,
            "",
        )
    )


def validate_and_canonicalize_external_endpoint(endpoint: str) -> str:
    """Validate an external HTTP(S) endpoint and return its canonical form."""
    if not isinstance(endpoint, str):
        raise TypeError("external endpoint must be a string")

    candidate = endpoint.strip()
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        # Accessing port performs urllib's range and integer validation.
        parsed.port
    except ValueError as error:
        raise ValueError(
            f"invalid external endpoint URL: {endpoint!r}"
        ) from error

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or hostname is None
        or any(character.isspace() for character in parsed.netloc)
    ):
        raise ValueError(
            "external endpoint must be an absolute HTTP(S) URL with a host"
        )

    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "external endpoint must not contain userinfo, a query, "
            "or a fragment"
        )

    return canonicalize_external_endpoint(candidate)


def classify_external_failure(
    error: Exception,
) -> tuple[str, bool, int | None]:
    """Return a stable failure code, retry decision, and HTTP status."""
    if isinstance(
        error,
        (
            APIResponseValidationError,
            json.JSONDecodeError,
            MalformedExternalResponseError,
        ),
    ):
        return "malformed_response", True, None

    if isinstance(
        error,
        (APITimeoutError, httpx.TimeoutException, asyncio.TimeoutError),
    ):
        return "request_timeout", True, None

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(error, "status", None)

    if status == 429:
        return "rate_limited", True, status
    if status == 408:
        return "request_timeout", True, status
    if status is not None and 500 <= status < 600:
        return "backend_5xx", True, status
    if status is not None and 400 <= status < 500:
        return "backend_4xx", False, status

    return "endpoint_unreachable", True, status


class ExternalRequestFailure(Exception):
    """Terminal external request failure with stable retry evidence."""

    def __init__(
        self,
        original_exception: Exception,
        *,
        error_code: str,
        attempts: int,
        elapsed_seconds: float,
        retriable: bool,
        http_status: int | None,
    ):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.error_code = error_code
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds
        self.retriable = retriable
        self.http_status = http_status
        self.trace = "".join(
            traceback.TracebackException.from_exception(
                original_exception
            ).format()
        )


def external_quota_identity(model: Any) -> str:
    """Build an opaque quota identity without retaining the raw credential."""
    endpoint = getattr(model, "base_url", None)
    if not isinstance(endpoint, str):
        endpoint = str(getattr(model, "path", endpoint))
    endpoint = canonicalize_external_endpoint(endpoint)

    credential = getattr(model, "api_key", None)
    if not isinstance(credential, str) or not credential:
        credential = "no-key"
    credential_fingerprint = hashlib.sha256(
        credential.encode()
    ).hexdigest()
    identity = json.dumps(
        {
            "endpoint": endpoint,
            "credential_fingerprint": credential_fingerprint,
        },
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _external_rate_limit_registration(
    model: Any,
) -> tuple[str, int | None]:
    requests_per_minute = getattr(model, "requests_per_minute", None)
    if requests_per_minute is not None and (
        isinstance(requests_per_minute, bool)
        or not isinstance(requests_per_minute, int)
        or requests_per_minute <= 0
    ):
        raise ValueError(
            "requests_per_minute must be a strict positive integer or None"
        )
    return external_quota_identity(model), requests_per_minute


def validate_external_rate_limiter_registration(model: Any) -> None:
    """Reject process-scoped quota drift without mutating the registry."""
    quota_identity, requests_per_minute = (
        _external_rate_limit_registration(model)
    )
    if quota_identity in RATE_LIMITERS:
        limiter = RATE_LIMITERS[quota_identity]
        registered_rpm = (
            limiter.requests_per_minute
            if limiter is not None
            else None
        )
        if registered_rpm != requests_per_minute:
            raise ValueError(
                "conflicting requests_per_minute for the same external "
                f"quota identity: registered={registered_rpm!r}, "
                f"requested={requests_per_minute!r}"
            )


def get_external_rate_limiter(model: Any) -> RateLimiter | None:
    """Return one deterministic limiter per external provider quota."""
    quota_identity, requests_per_minute = (
        _external_rate_limit_registration(model)
    )
    validate_external_rate_limiter_registration(model)
    if quota_identity in RATE_LIMITERS:
        return RATE_LIMITERS[quota_identity]

    limiter = (
        RateLimiter(requests_per_minute)
        if requests_per_minute is not None
        else None
    )
    RATE_LIMITERS[quota_identity] = limiter
    return limiter


def _backoff_cap(
    policy: ExternalRetryPolicyLike,
    failed_attempt: int,
) -> float:
    """Compute capped exponential backoff without an unbounded exponent."""
    cap = policy.initial_backoff_seconds
    for _ in range(max(0, failed_attempt - 1)):
        if cap >= policy.max_backoff_seconds / 2:
            return policy.max_backoff_seconds
        cap *= 2
    return min(cap, policy.max_backoff_seconds)


def _retry_delay(
    policy: ExternalRetryPolicyLike,
    failed_attempt: int,
    previous_delay: float,
) -> float:
    """Apply the configured jitter strategy to the next retry delay."""
    cap = _backoff_cap(policy, failed_attempt)
    if policy.jitter == "equal":
        return random.uniform(cap / 2, cap)
    if policy.jitter == "decorrelated":
        return min(
            policy.max_backoff_seconds,
            random.uniform(
                policy.initial_backoff_seconds,
                previous_delay * 3,
            ),
        )
    return random.uniform(0, cap)


class ExternalRequestRunner:
    """Apply one retry policy and total deadline to external HTTP operations."""

    def __init__(
        self,
        policy: ExternalRetryPolicyLike,
        *,
        rate_limiter: RateLimiter | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._policy = policy
        self._rate_limiter = rate_limiter
        self._monotonic = monotonic
        self._started_at = 0.0
        self._deadline = 0.0

    def _failure(
        self,
        error: Exception,
        *,
        attempts: int,
        failed_at: float,
    ) -> ExternalRequestFailure:
        error_code, retriable, http_status = classify_external_failure(error)
        return ExternalRequestFailure(
            error,
            error_code=error_code,
            attempts=attempts,
            elapsed_seconds=max(0.0, failed_at - self._started_at),
            retriable=retriable,
            http_status=http_status,
        )

    async def _await_admission(
        self,
        admission_factory: Callable[[], Awaitable[T]],
        *,
        attempts: int,
        cleanup_factory: (
            Callable[[T], Awaitable[None]] | None
        ) = None,
    ) -> T:
        """Bound local admission by the total deadline, not the HTTP timeout."""
        admission_started_at = self._monotonic()
        remaining_seconds = self._deadline - admission_started_at
        if remaining_seconds <= 0:
            error = asyncio.TimeoutError(
                "external request exceeded its total deadline during admission"
            )
            raise self._failure(
                error,
                attempts=attempts,
                failed_at=admission_started_at,
            ) from error

        admission_task = asyncio.ensure_future(admission_factory())

        async def cleanup_untransferred_admission() -> None:
            if not admission_task.done():
                admission_task.cancel()
            try:
                untransferred_admission = await admission_task
            except (asyncio.CancelledError, Exception):
                return
            if cleanup_factory is not None:
                await cleanup_factory(untransferred_admission)

        try:
            return await asyncio.wait_for(
                asyncio.shield(admission_task),
                timeout=remaining_seconds,
            )
        except asyncio.CancelledError:
            await cleanup_untransferred_admission()
            raise
        except asyncio.TimeoutError as error:
            await cleanup_untransferred_admission()
            failed_at = self._monotonic()
            deadline_error = asyncio.TimeoutError(
                "external request exceeded its total deadline during admission"
            )
            raise self._failure(
                deadline_error,
                attempts=attempts,
                failed_at=failed_at,
            ) from error

    async def run(
        self,
        attempt_factory: Callable[[Any], Awaitable[T]],
        *,
        acquire_factory: Callable[[], Awaitable[Any]] | None = None,
        prepare_factory: (
            Callable[[Any], Awaitable[Any]] | None
        ) = None,
        release_factory: (
            Callable[[Any], Awaitable[None]] | None
        ) = None,
        passthrough_error: Callable[[Exception], bool] | None = None,
        started_at: float | None = None,
    ) -> T:
        """Run one logical HTTP operation under distinct admission/HTTP budgets."""
        self._started_at = (
            self._monotonic() if started_at is None else started_at
        )
        self._deadline = (
            self._started_at + self._policy.total_deadline_seconds
        )
        attempts = 0
        previous_retry_delay = self._policy.initial_backoff_seconds
        while attempts < self._policy.max_attempts:
            admission = None
            prepared_admission = None
            admitted = False
            rate_permit_pending = False
            try:
                if acquire_factory is not None:
                    admission = await self._await_admission(
                        acquire_factory,
                        attempts=attempts,
                        cleanup_factory=release_factory,
                    )
                    admitted = True

                try:
                    prepared_admission = admission
                    if prepare_factory is not None:
                        try:
                            prepared_admission = await self._await_admission(
                                lambda: prepare_factory(admission),
                                attempts=attempts,
                            )
                        except ExternalRequestFailure:
                            raise
                        except Exception as error:
                            failed_at = self._monotonic()
                            raise ExternalRequestFailure(
                                error,
                                error_code="client_setup_failed",
                                attempts=attempts,
                                elapsed_seconds=max(
                                    0.0,
                                    failed_at - self._started_at,
                                ),
                                retriable=False,
                                http_status=None,
                            ) from error

                    if self._rate_limiter is not None:
                        rate_limiter = self._rate_limiter
                        await self._await_admission(
                            rate_limiter.acquire,
                            attempts=attempts,
                            cleanup_factory=(
                                lambda _permit: rate_limiter.refund()
                            ),
                        )
                        rate_permit_pending = True

                    attempt_started_at = self._monotonic()
                    remaining_seconds = self._deadline - attempt_started_at
                    if remaining_seconds <= 0:
                        error = asyncio.TimeoutError(
                            "external request exceeded its total deadline"
                        )
                        raise self._failure(
                            error,
                            attempts=attempts,
                            failed_at=attempt_started_at,
                        ) from error

                    attempts += 1
                    rate_permit_pending = False
                    return await asyncio.wait_for(
                        attempt_factory(prepared_admission),
                        timeout=min(
                            self._policy.request_timeout_seconds,
                            remaining_seconds,
                        ),
                    )
                finally:
                    if (
                        rate_permit_pending
                        and self._rate_limiter is not None
                    ):
                        await self._rate_limiter.refund()
                    if (
                        admitted
                        and release_factory is not None
                    ):
                        await release_factory(admission)
            except _EXTERNAL_REQUEST_ERRORS as error:
                if (
                    passthrough_error is not None
                    and passthrough_error(error)
                ):
                    raise

                failed_at = self._monotonic()
                error_code, retriable, http_status = (
                    classify_external_failure(error)
                )
                remaining_seconds = self._deadline - failed_at
                if (
                    retriable
                    and attempts < self._policy.max_attempts
                    and remaining_seconds > 0
                ):
                    retry_delay = _retry_delay(
                        self._policy,
                        attempts,
                        previous_retry_delay,
                    )
                    previous_retry_delay = retry_delay
                    if retry_delay < remaining_seconds:
                        if retry_delay > 0:
                            await asyncio.sleep(retry_delay)
                        continue

                failure = ExternalRequestFailure(
                    error,
                    error_code=error_code,
                    attempts=attempts,
                    elapsed_seconds=max(
                        0.0,
                        failed_at - self._started_at,
                    ),
                    retriable=retriable,
                    http_status=http_status,
                )
                raise failure from error

        raise RuntimeError("unreachable external request retry state")


class _RetryingCreateEndpoint:
    """Apply the external request contract to one SDK ``create`` endpoint."""

    def __init__(
        self,
        endpoint: Any,
        *,
        policy: ExternalRetryPolicyLike,
        rate_limiter: RateLimiter | None,
        response_kind: str,
        acquire_factory: Callable[[], Awaitable[Any]] | None,
        release_factory: (
            Callable[[Any], Awaitable[None]] | None
        ),
    ):
        self._endpoint = endpoint
        self._policy = policy
        self._rate_limiter = rate_limiter
        self._response_kind = response_kind
        self._acquire_factory = acquire_factory
        self._release_factory = release_factory

    def _validate_response(
        self,
        response: Any,
        *,
        expected_choices: int,
    ) -> None:
        """Validate every field judge callers dereference before commit."""
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
                f"external judge returned {actual} choices; "
                f"expected {expected_choices}"
            )

        for choice_index, choice in enumerate(choices):
            if self._response_kind == "chat":
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None)
                if (
                    not isinstance(content, str)
                    or not content.strip()
                ):
                    raise MalformedExternalResponseError(
                        "external judge returned empty or null chat "
                        f"content for choice {choice_index}"
                    )
                continue

            text = getattr(choice, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise MalformedExternalResponseError(
                    "external judge returned empty or null completion "
                    f"text for choice {choice_index}"
                )

    @staticmethod
    def _expected_choices(kwargs: dict[str, Any]) -> int:
        expected_choices = kwargs.get("n", 1)
        if expected_choices is None:
            return 1
        if (
            isinstance(expected_choices, bool)
            or not isinstance(expected_choices, int)
            or expected_choices <= 0
        ):
            raise ValueError(
                "external judge choice count must be a positive integer"
            )
        return expected_choices

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        expected_choices = self._expected_choices(kwargs)
        runner = ExternalRequestRunner(
            self._policy,
            rate_limiter=self._rate_limiter,
        )

        async def attempt(_admission: Any) -> Any:
            response = await self._endpoint.create(*args, **kwargs)
            self._validate_response(
                response,
                expected_choices=expected_choices,
            )
            return response

        return await runner.run(
            attempt,
            acquire_factory=self._acquire_factory,
            release_factory=self._release_factory,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._endpoint, name)


class _RetryingChatEndpoint:
    """Preserve the SDK chat namespace while wrapping completions."""

    def __init__(
        self,
        chat: Any,
        *,
        policy: ExternalRetryPolicyLike,
        rate_limiter: RateLimiter | None,
        acquire_factory: Callable[[], Awaitable[Any]] | None,
        release_factory: (
            Callable[[Any], Awaitable[None]] | None
        ),
    ):
        self._chat = chat
        self.completions = _RetryingCreateEndpoint(
            chat.completions,
            policy=policy,
            rate_limiter=rate_limiter,
            response_kind="chat",
            acquire_factory=acquire_factory,
            release_factory=release_factory,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class RetryingAsyncOpenAI:
    """Narrow AsyncOpenAI proxy for bounded external judge requests."""

    def __init__(
        self,
        client: Any,
        *,
        policy: ExternalRetryPolicyLike,
        rate_limiter: RateLimiter | None,
        acquire_factory: Callable[[], Awaitable[Any]] | None = None,
        release_factory: (
            Callable[[Any], Awaitable[None]] | None
        ) = None,
    ):
        self._client = client
        self.chat = _RetryingChatEndpoint(
            client.chat,
            policy=policy,
            rate_limiter=rate_limiter,
            acquire_factory=acquire_factory,
            release_factory=release_factory,
        )
        self.completions = _RetryingCreateEndpoint(
            client.completions,
            policy=policy,
            rate_limiter=rate_limiter,
            response_kind="completion",
            acquire_factory=acquire_factory,
            release_factory=release_factory,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
