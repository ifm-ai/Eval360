from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from scheduler.model import ExternalRetryPolicy


def _runner_contract():
    import scheduler.external_requests as external_requests

    runner_type = getattr(
        external_requests,
        "ExternalRequestRunner",
        None,
    )
    assert runner_type is not None, "ExternalRequestRunner is not implemented"
    return external_requests, runner_type


def _policy(**overrides: Any) -> ExternalRetryPolicy:
    values = {
        "max_attempts": 2,
        "request_timeout_seconds": 0.1,
        "total_deadline_seconds": 1,
        "initial_backoff_seconds": 0,
        "max_backoff_seconds": 1,
    }
    values.update(overrides)
    return ExternalRetryPolicy(**values)


class _RecordingRateLimiter:
    def __init__(
        self,
        events: list[str],
        acquire: Callable[[], Awaitable[None]] | None = None,
    ):
        self._events = events
        self._acquire = acquire

    async def acquire(self) -> None:
        self._events.append("rate")
        if self._acquire is not None:
            await self._acquire()

    async def refund(self) -> None:
        self._events.append("refund")


@pytest.mark.asyncio
async def test_runner_orders_pool_setup_rate_http_and_release():
    _, runner_type = _runner_contract()
    events: list[str] = []

    async def acquire():
        events.append("pool")
        return "lease"

    async def prepare(admission):
        assert admission == "lease"
        events.append("setup")
        return "client"

    async def attempt(prepared):
        assert prepared == "client"
        events.append("http")
        return "ok"

    async def release(admission):
        assert admission == "lease"
        events.append("release")

    runner = runner_type(
        _policy(),
        rate_limiter=_RecordingRateLimiter(events),
    )

    result = await runner.run(
        attempt,
        acquire_factory=acquire,
        prepare_factory=prepare,
        release_factory=release,
    )

    assert result == "ok"
    assert events == ["pool", "setup", "rate", "http", "release"]


@pytest.mark.asyncio
async def test_client_setup_failure_uses_zero_http_attempts_and_no_permit():
    external_requests, runner_type = _runner_contract()
    events: list[str] = []

    async def acquire():
        events.append("pool")
        return "lease"

    async def prepare(_admission):
        events.append("setup")
        raise RuntimeError("client construction failed")

    async def attempt(_prepared):
        pytest.fail("HTTP must not start after client setup failure")

    async def release(_admission):
        events.append("release")

    runner = runner_type(
        _policy(),
        rate_limiter=_RecordingRateLimiter(events),
    )

    with pytest.raises(
        external_requests.ExternalRequestFailure
    ) as failure:
        await runner.run(
            attempt,
            acquire_factory=acquire,
            prepare_factory=prepare,
            release_factory=release,
        )

    assert failure.value.error_code == "client_setup_failed"
    assert failure.value.attempts == 0
    assert failure.value.retriable is False
    assert events == ["pool", "setup", "release"]


@pytest.mark.asyncio
async def test_rate_admission_timeout_releases_lease_without_http_attempt():
    external_requests, runner_type = _runner_contract()
    events: list[str] = []
    never = asyncio.Event()

    async def blocked_rate():
        await never.wait()

    async def acquire():
        events.append("pool")
        return "lease"

    async def release(_admission):
        events.append("release")

    runner = runner_type(
        _policy(total_deadline_seconds=0.02),
        rate_limiter=_RecordingRateLimiter(events, blocked_rate),
    )

    with pytest.raises(
        external_requests.ExternalRequestFailure
    ) as failure:
        await runner.run(
            lambda _prepared: pytest.fail("HTTP must not start"),
            acquire_factory=acquire,
            release_factory=release,
        )

    assert failure.value.error_code == "request_timeout"
    assert failure.value.attempts == 0
    assert events == ["pool", "rate", "release"]


@pytest.mark.asyncio
async def test_deadline_after_rate_admission_refunds_permit_without_http():
    external_requests, runner_type = _runner_contract()
    events: list[str] = []
    now = 0.0

    async def expire_deadline_during_rate_admission():
        nonlocal now
        now = 2.0

    async def attempt(_prepared):
        events.append("http")
        return "unexpected"

    runner = runner_type(
        _policy(total_deadline_seconds=1),
        rate_limiter=_RecordingRateLimiter(
            events,
            expire_deadline_during_rate_admission,
        ),
        monotonic=lambda: now,
    )

    with pytest.raises(
        external_requests.ExternalRequestFailure
    ) as failure:
        await runner.run(attempt)

    assert failure.value.error_code == "request_timeout"
    assert failure.value.attempts == 0
    assert events == ["rate", "refund"]


@pytest.mark.asyncio
async def test_request_timeout_counts_the_started_http_attempt():
    external_requests, runner_type = _runner_contract()
    never = asyncio.Event()

    async def attempt(_prepared):
        await never.wait()

    runner = runner_type(
        _policy(
            max_attempts=1,
            request_timeout_seconds=0.01,
        )
    )

    with pytest.raises(
        external_requests.ExternalRequestFailure
    ) as failure:
        await runner.run(attempt)

    assert failure.value.error_code == "request_timeout"
    assert failure.value.attempts == 1
    assert failure.value.retriable is True


@pytest.mark.asyncio
async def test_request_timeout_cancels_and_retries_each_http_attempt():
    external_requests, runner_type = _runner_contract()
    started = 0
    cancelled = 0

    async def attempt(_prepared):
        nonlocal started, cancelled
        started += 1
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            cancelled += 1
            raise
        return "late"

    runner = runner_type(
        _policy(
            max_attempts=2,
            request_timeout_seconds=0.01,
            total_deadline_seconds=0.25,
        )
    )

    with pytest.raises(
        external_requests.ExternalRequestFailure
    ) as failure:
        await asyncio.wait_for(runner.run(attempt), timeout=0.2)

    assert failure.value.error_code == "request_timeout"
    assert failure.value.attempts == 2
    assert failure.value.retriable is True
    assert started == 2
    assert cancelled == 2


@pytest.mark.asyncio
async def test_retry_releases_before_the_next_pool_admission():
    _, runner_type = _runner_contract()
    acquires = 0
    releases = 0
    attempts = 0

    async def acquire():
        nonlocal acquires
        if acquires:
            assert releases == 1
        acquires += 1
        return f"lease-{acquires}"

    async def release(_admission):
        nonlocal releases
        releases += 1

    async def attempt(_prepared):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary connection failure")
        return "ok"

    runner = runner_type(_policy())

    assert await runner.run(
        attempt,
        acquire_factory=acquire,
        release_factory=release,
    ) == "ok"
    assert (acquires, attempts, releases) == (2, 2, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("jitter", "expected_uniform_bounds"),
    [
        ("full", (0, 2)),
        ("equal", (1, 2)),
    ],
)
async def test_retry_delay_applies_configured_stateless_jitter(
    monkeypatch,
    jitter,
    expected_uniform_bounds,
):
    external_requests, runner_type = _runner_contract()
    attempts = 0
    uniform_bounds = []
    sleeps = []

    def fake_uniform(lower, upper):
        uniform_bounds.append((lower, upper))
        return 1.5

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def attempt(_prepared):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary connection failure")
        return "ok"

    monkeypatch.setattr(external_requests.random, "uniform", fake_uniform)
    monkeypatch.setattr(external_requests.asyncio, "sleep", fake_sleep)

    runner = runner_type(
        _policy(
            initial_backoff_seconds=2,
            max_backoff_seconds=10,
            total_deadline_seconds=100,
            jitter=jitter,
        ),
        monotonic=lambda: 0,
    )

    assert await runner.run(attempt) == "ok"
    assert uniform_bounds == [expected_uniform_bounds]
    assert sleeps == [1.5]


@pytest.mark.asyncio
async def test_decorrelated_jitter_uses_previous_delay_and_maximum_cap(
    monkeypatch,
):
    external_requests, runner_type = _runner_contract()
    attempts = 0
    uniform_bounds = []
    sampled_delays = iter([4, 7, 12])
    sleeps = []

    def fake_uniform(lower, upper):
        uniform_bounds.append((lower, upper))
        return next(sampled_delays)

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def attempt(_prepared):
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise httpx.ConnectError("temporary connection failure")
        return "ok"

    monkeypatch.setattr(external_requests.random, "uniform", fake_uniform)
    monkeypatch.setattr(external_requests.asyncio, "sleep", fake_sleep)

    runner = runner_type(
        _policy(
            max_attempts=4,
            initial_backoff_seconds=2,
            max_backoff_seconds=10,
            total_deadline_seconds=100,
            jitter="decorrelated",
        ),
        monotonic=lambda: 0,
    )

    assert await runner.run(attempt) == "ok"
    assert uniform_bounds == [(2, 6), (2, 12), (2, 21)]
    assert sleeps == [4, 7, 10]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 429, 500, 503])
async def test_retryable_http_statuses_start_a_second_attempt(status):
    _, runner_type = _runner_contract()
    calls = 0

    async def attempt(_prepared):
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request("POST", "https://api.example.test")
            response = httpx.Response(status, request=request)
            raise httpx.HTTPStatusError(
                "provider failure",
                request=request,
                response=response,
            )
        return "ok"

    assert await runner_type(_policy()).run(attempt) == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_non_retryable_http_4xx_stops_after_one_attempt():
    external_requests, runner_type = _runner_contract()
    calls = 0

    async def attempt(_prepared):
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "https://api.example.test")
        response = httpx.Response(400, request=request)
        raise httpx.HTTPStatusError(
            "bad request",
            request=request,
            response=response,
        )

    with pytest.raises(
        external_requests.ExternalRequestFailure
    ) as failure:
        await runner_type(_policy()).run(attempt)

    assert calls == 1
    assert failure.value.error_code == "backend_4xx"
    assert failure.value.http_status == 400
    assert failure.value.retriable is False


@pytest.mark.asyncio
async def test_malformed_success_is_retryable_inside_runner_boundary():
    external_requests, runner_type = _runner_contract()
    calls = 0

    async def attempt(_prepared):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise external_requests.MalformedExternalResponseError(
                "missing choices"
            )
        return "ok"

    assert await runner_type(_policy()).run(attempt) == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_pool_admission_uses_total_deadline_not_request_timeout():
    _, runner_type = _runner_contract()

    async def acquire():
        await asyncio.sleep(0.03)
        return "lease"

    async def release(_admission):
        return None

    runner = runner_type(
        _policy(
            request_timeout_seconds=0.005,
            total_deadline_seconds=0.2,
        )
    )

    assert await runner.run(
        lambda _prepared: asyncio.sleep(0, result="ok"),
        acquire_factory=acquire,
        release_factory=release,
    ) == "ok"


@pytest.mark.asyncio
async def test_total_deadline_bounds_pool_admission_without_an_attempt():
    external_requests, runner_type = _runner_contract()
    never = asyncio.Event()

    async def acquire():
        await never.wait()

    runner = runner_type(_policy(total_deadline_seconds=0.01))

    with pytest.raises(
        external_requests.ExternalRequestFailure
    ) as failure:
        await runner.run(
            lambda _prepared: pytest.fail("HTTP must not start"),
            acquire_factory=acquire,
        )

    assert failure.value.error_code == "request_timeout"
    assert failure.value.attempts == 0


@pytest.mark.asyncio
async def test_total_deadline_starts_when_logical_operation_runs():
    _, runner_type = _runner_contract()
    now = 0.0

    runner = runner_type(
        _policy(total_deadline_seconds=1),
        monotonic=lambda: now,
    )
    now = 10.0

    assert await runner.run(
        lambda _prepared: asyncio.sleep(0, result="ok")
    ) == "ok"


@pytest.mark.asyncio
async def test_cancellation_during_rate_wait_releases_acquired_lease():
    _, runner_type = _runner_contract()
    rate_started = asyncio.Event()
    never = asyncio.Event()
    released = asyncio.Event()

    async def blocked_rate():
        rate_started.set()
        await never.wait()

    async def acquire():
        return "lease"

    async def release(admission):
        assert admission == "lease"
        released.set()

    runner = runner_type(
        _policy(),
        rate_limiter=_RecordingRateLimiter([], blocked_rate),
    )
    task = asyncio.create_task(
        runner.run(
            lambda _prepared: pytest.fail("HTTP must not start"),
            acquire_factory=acquire,
            release_factory=release,
        )
    )
    await rate_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released.is_set()


@pytest.mark.asyncio
async def test_cancellation_at_pool_acquisition_completion_releases_lease():
    _, runner_type = _runner_contract()
    acquired = asyncio.Event()
    released: list[str] = []
    lease_ready: asyncio.Future[str] = (
        asyncio.get_running_loop().create_future()
    )

    async def acquire():
        acquired.set()
        return await lease_ready

    async def release(admission):
        released.append(admission)

    async def attempt(_prepared):
        pytest.fail("HTTP must not start")

    runner = runner_type(_policy())
    runner_task = asyncio.create_task(
        runner.run(
            attempt,
            acquire_factory=acquire,
            release_factory=release,
        )
    )
    await acquired.wait()

    lease_ready.set_result("lease")
    runner_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await runner_task

    assert released == ["lease"]


@pytest.mark.asyncio
async def test_cancellation_at_rate_acquisition_completion_refunds_permit():
    _, runner_type = _runner_contract()
    events: list[str] = []
    rate_started = asyncio.Event()
    permit_ready: asyncio.Future[None] = (
        asyncio.get_running_loop().create_future()
    )

    async def acquire_rate():
        rate_started.set()
        await permit_ready

    async def attempt(_prepared):
        pytest.fail("HTTP must not start")

    runner = runner_type(
        _policy(),
        rate_limiter=_RecordingRateLimiter(events, acquire_rate),
    )
    runner_task = asyncio.create_task(runner.run(attempt))
    await rate_started.wait()

    permit_ready.set_result(None)
    runner_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await runner_task

    assert events == ["rate", "refund"]
