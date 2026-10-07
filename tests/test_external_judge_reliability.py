"""Focused contracts for bounded external LLM-as-judge requests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError

from scheduler.external_requests import (
    ExternalRequestFailure,
    RATE_LIMITERS,
    canonicalize_external_endpoint,
    get_external_rate_limiter,
)
from scheduler.grader.base import GraderBase, GradingOpenAIConnection
from scheduler.model import ExternalRetryPolicy, ModelInstance
from scheduler.openai_interface import (
    LOCKED_CONNECTIONS,
    get_or_create_connection_pool,
)
from scheduler.utils import ExceptionWrapper


def _chat_response(contents: list[str | None]):
    choices = [
        SimpleNamespace(
            index=index,
            message=SimpleNamespace(content=content),
        )
        for index, content in enumerate(contents)
    ]
    return SimpleNamespace(choices=choices)


def _completion_response(texts: list[str | None]):
    choices = [
        SimpleNamespace(index=index, text=text)
        for index, text in enumerate(texts)
    ]
    return SimpleNamespace(choices=choices)


class _FakeAsyncOpenAI:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        max_retries: int,
        timeout: float | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock())
        )
        self.completions = SimpleNamespace(create=AsyncMock())


@pytest.fixture(autouse=True)
def _clear_external_runtime_registries():
    LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()
    yield
    LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()


def _external_judge(
    *,
    serving_key: str = "shared-service",
    base_url: str = "https://api.example.com/v1",
    api_key: str = "sk-test",
    max_simultaneous_requests: int = 1,
    requests_per_minute: int | None = None,
    retry_policy: ExternalRetryPolicy | None = None,
):
    judge = MagicMock(spec=ModelInstance)
    judge.name = "judge"
    judge.serving_key = serving_key
    judge.is_external = True
    judge.base_url = base_url
    judge.api_key = api_key
    judge.max_simultaneous_requests = max_simultaneous_requests
    judge.requests_per_minute = requests_per_minute
    judge.external_retry_policy = retry_policy or ExternalRetryPolicy(
        max_attempts=1,
        request_timeout_seconds=1,
        total_deadline_seconds=5,
        initial_backoff_seconds=0,
        max_backoff_seconds=1,
    )
    return judge


async def _make_external_judge_client(**judge_kwargs):
    judge = _external_judge(**judge_kwargs)
    canonical_url = canonicalize_external_endpoint(judge.base_url)
    pool = get_or_create_connection_pool(
        judge.serving_key,
        judge.max_simultaneous_requests,
    )
    await pool.add_url(canonical_url)

    task = MagicMock()
    task.grader.llm_as_judge = judge
    connection = GradingOpenAIConnection(
        event_instance=MagicMock(),
        task=task,
        job_manager=MagicMock(),
    )
    created_clients: list[_FakeAsyncOpenAI] = []

    def create_client(**kwargs):
        fake_client = _FakeAsyncOpenAI(**kwargs)
        created_clients.append(fake_client)
        return fake_client

    with patch(
        "scheduler.grader.base.AsyncOpenAI",
        side_effect=create_client,
    ):
        client = await connection.get_client()
    assert len(created_clients) == 1
    fake_client = created_clients[0]
    return judge, connection, client, fake_client, pool


@pytest.mark.asyncio
async def test_judge_uses_one_canonical_endpoint_for_client_and_pool():
    _, connection, _, fake_client, pool = (
        await _make_external_judge_client(
            base_url=" https://API.EXAMPLE.com:443/v1/ ",
        )
    )

    assert connection._url == "https://api.example.com/v1"
    assert fake_client.base_url == "https://api.example.com/v1"
    assert pool.urls == ["https://api.example.com/v1"]


@pytest.mark.asyncio
async def test_generation_and_judge_aliases_share_capacity_and_quota():
    judge = _external_judge(
        base_url="https://API.EXAMPLE.com:443/v1/",
        requests_per_minute=60,
    )
    generation = MagicMock(spec=ModelInstance)
    generation.base_url = "https://api.example.com/v1"
    generation.path = generation.base_url
    generation.api_key = judge.api_key
    generation.requests_per_minute = 60
    generation_limiter = get_external_rate_limiter(generation)

    pool = get_or_create_connection_pool(
        judge.serving_key,
        judge.max_simultaneous_requests,
    )
    await pool.add_url("https://api.example.com/v1")
    held_generation_lease = await pool.acquire(1)

    _, connection, client, fake_client, shared_pool = (
        await _make_external_judge_client(
            base_url=judge.base_url,
            requests_per_minute=60,
        )
    )
    request_started = asyncio.Event()

    async def create(**_kwargs):
        request_started.set()
        return _chat_response(["correct"])

    fake_client.chat.completions.create = AsyncMock(side_effect=create)
    judge_request = asyncio.create_task(
        client.chat.completions.create(
            model="judge",
            messages=[{"role": "user", "content": "grade"}],
        )
    )
    await asyncio.sleep(0.01)

    try:
        assert not request_started.is_set()
        assert (
            client.chat.completions._rate_limiter
            is generation_limiter
        )
        await pool.release(*held_generation_lease)
        response = await asyncio.wait_for(judge_request, timeout=1)
    finally:
        if not judge_request.done():
            judge_request.cancel()
            await asyncio.gather(judge_request, return_exceptions=True)
        if shared_pool._slots["https://api.example.com/v1"][0] == 0:
            await pool.release(*held_generation_lease)

    assert response.choices[0].message.content == "correct"
    assert connection._url == "https://api.example.com/v1"


@pytest.mark.asyncio
async def test_three_judge_calls_observe_capacity_one():
    _, connection, client, fake_client, _ = (
        await _make_external_judge_client(
            max_simultaneous_requests=1,
        )
    )
    active = 0
    peak = 0

    async def create(**_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return _chat_response(["correct"])
        finally:
            active -= 1

    fake_client.chat.completions.create = AsyncMock(side_effect=create)
    await asyncio.gather(
        *[
            client.chat.completions.create(
                model="judge",
                messages=[
                    {"role": "user", "content": f"grade {index}"}
                ],
            )
            for index in range(3)
        ]
    )

    assert peak == 1
    assert connection._url == "https://api.example.com/v1"


@pytest.mark.asyncio
async def test_each_retry_reacquires_quota_and_releases_capacity():
    policy = ExternalRetryPolicy(
        max_attempts=2,
        request_timeout_seconds=1,
        total_deadline_seconds=5,
        initial_backoff_seconds=0,
        max_backoff_seconds=1,
    )
    judge, _, client, fake_client, pool = (
        await _make_external_judge_client(
            retry_policy=policy,
            requests_per_minute=60,
        )
    )
    limiter = get_external_rate_limiter(judge)
    assert limiter is not None
    limiter.acquire = AsyncMock()
    pool.acquire = AsyncMock(wraps=pool.acquire)
    pool.release = AsyncMock(wraps=pool.release)
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            APIConnectionError(
                request=None,
                message="judge unavailable",
            ),
            _chat_response(["correct"]),
        ]
    )

    response = await client.chat.completions.create(
        model="judge",
        messages=[{"role": "user", "content": "grade"}],
    )

    assert response.choices[0].message.content == "correct"
    assert fake_client.chat.completions.create.await_count == 2
    assert limiter.acquire.await_count == 2
    assert pool.acquire.await_count == 2
    assert pool.release.await_count == 2
    assert pool._slots["https://api.example.com/v1"][0] == 1


@pytest.mark.asyncio
async def test_judge_cancellation_returns_capacity():
    _, _, client, fake_client, pool = (
        await _make_external_judge_client()
    )
    pool.acquire = AsyncMock(wraps=pool.acquire)
    pool.release = AsyncMock(wraps=pool.release)
    request_started = asyncio.Event()

    async def never_finishes(**_kwargs):
        request_started.set()
        await asyncio.Future()

    fake_client.chat.completions.create = AsyncMock(
        side_effect=never_finishes
    )
    request = asyncio.create_task(
        client.chat.completions.create(
            model="judge",
            messages=[{"role": "user", "content": "grade"}],
        )
    )
    await request_started.wait()
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request

    assert pool.acquire.await_count == 1
    assert pool.release.await_count == 1
    assert pool._slots["https://api.example.com/v1"][0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed,request_kwargs,expected",
    [
        (
            SimpleNamespace(choices=[]),
            {},
            _chat_response(["correct"]),
        ),
        (_chat_response([None]), {}, _chat_response(["correct"])),
        (_chat_response([""]), {}, _chat_response(["correct"])),
        (_chat_response(["   "]), {}, _chat_response(["correct"])),
        (
            _chat_response(["only-one"]),
            {"n": 2},
            _chat_response(["first", "second"]),
        ),
    ],
    ids=[
        "missing-choice",
        "null-content",
        "empty-content",
        "whitespace-content",
        "wrong-choice-count",
    ],
)
async def test_malformed_chat_success_retries_then_recovers(
    malformed,
    request_kwargs,
    expected,
):
    policy = ExternalRetryPolicy(
        max_attempts=2,
        request_timeout_seconds=1,
        total_deadline_seconds=5,
        initial_backoff_seconds=0,
        max_backoff_seconds=1,
    )
    _, _, client, fake_client, _ = (
        await _make_external_judge_client(retry_policy=policy)
    )
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[malformed, expected]
    )

    response = await client.chat.completions.create(
        model="judge",
        messages=[{"role": "user", "content": "grade"}],
        **request_kwargs,
    )

    assert response is expected
    assert fake_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_json_decode_failure_retries_inside_policy():
    policy = ExternalRetryPolicy(
        max_attempts=2,
        request_timeout_seconds=1,
        total_deadline_seconds=5,
        initial_backoff_seconds=0,
        max_backoff_seconds=1,
    )
    _, _, client, fake_client, _ = (
        await _make_external_judge_client(retry_policy=policy)
    )
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            json.JSONDecodeError("invalid JSON", "x", 0),
            _chat_response(["correct"]),
        ]
    )

    response = await client.chat.completions.create(
        model="judge",
        messages=[{"role": "user", "content": "grade"}],
    )

    assert response.choices[0].message.content == "correct"
    assert fake_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_malformed_completion_success_retries_then_recovers():
    policy = ExternalRetryPolicy(
        max_attempts=2,
        request_timeout_seconds=1,
        total_deadline_seconds=5,
        initial_backoff_seconds=0,
        max_backoff_seconds=1,
    )
    _, _, client, fake_client, _ = (
        await _make_external_judge_client(retry_policy=policy)
    )
    expected = _completion_response(["correct"])
    fake_client.completions.create = AsyncMock(
        side_effect=[
            _completion_response(["   "]),
            expected,
        ]
    )

    response = await client.completions.create(
        model="judge",
        prompt="grade",
    )

    assert response is expected
    assert fake_client.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_malformed_success_exhaustion_is_structured():
    policy = ExternalRetryPolicy(
        max_attempts=2,
        request_timeout_seconds=1,
        total_deadline_seconds=5,
        initial_backoff_seconds=0,
        max_backoff_seconds=1,
    )
    _, _, client, fake_client, _ = (
        await _make_external_judge_client(retry_policy=policy)
    )
    fake_client.chat.completions.create = AsyncMock(
        return_value=_chat_response([None])
    )

    with pytest.raises(ExternalRequestFailure) as exc_info:
        await client.chat.completions.create(
            model="judge",
            messages=[{"role": "user", "content": "grade"}],
        )

    assert exc_info.value.error_code == "malformed_response"
    assert exc_info.value.attempts == 2
    assert exc_info.value.retriable is True
    assert fake_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_judge_uses_configured_timeout_and_attempt_limit():
    policy = ExternalRetryPolicy(
        max_attempts=2,
        request_timeout_seconds=0.01,
        total_deadline_seconds=1,
        initial_backoff_seconds=0,
        max_backoff_seconds=1,
    )
    _, _, client, fake_client, _ = (
        await _make_external_judge_client(retry_policy=policy)
    )
    fake_client.chat.completions.create = AsyncMock(
        side_effect=APIConnectionError(
            request=None,
            message="judge unavailable",
        )
    )

    with pytest.raises(ExternalRequestFailure) as exc_info:
        await client.chat.completions.create(
            model="judge",
            messages=[{"role": "user", "content": "grade"}],
        )

    assert fake_client.timeout == 0.01
    assert fake_client.chat.completions.create.await_count == 2
    assert exc_info.value.attempts == 2


class _ContractGrader(GraderBase):
    async def grade_sample(self, sample):
        return sample

    async def run(self):
        return {}


async def _sample_stream(*samples):
    for sample in samples:
        yield sample


def _make_contract_grader(*samples):
    event = MagicMock()
    event.parser_type = "noop"
    return _ContractGrader(
        _sample_stream(*samples),
        event_manager=MagicMock(),
        job_manager=MagicMock(),
        event=event,
    )


def _external_failure() -> ExternalRequestFailure:
    return ExternalRequestFailure(
        RuntimeError("judge unavailable"),
        error_code="endpoint_unreachable",
        attempts=3,
        elapsed_seconds=1.25,
        retriable=True,
        http_status=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("nonblocking", [False, True])
async def test_grader_preserves_direct_and_grouped_external_evidence(
    grouped,
    nonblocking,
):
    sample = {"row": 0, "generations": ["answer"]}
    grader = _make_contract_grader(sample)
    failure = _external_failure()
    raised = (
        ExceptionGroup("judge failed", [RuntimeError("peer"), failure])
        if grouped
        else failure
    )

    async def fail(_sample):
        raise raised

    method = (
        grader.async_grade_all_samples_nonblocking
        if nonblocking
        else grader.async_grade_all_samples
    )
    results = [item async for item in method(grade_fn=fail)]

    assert len(results) == 1
    assert isinstance(results[0], ExceptionWrapper)
    assert results[0].exception is raised
    assert ("judge failed" if grouped else "judge unavailable") in results[0].trace
    if grouped:
        assert isinstance(results[0].exception, ExceptionGroup)
        assert "peer" in results[0].trace
    assert results[0].error_code == "endpoint_unreachable"
    assert results[0].attempts == 3
    assert results[0].elapsed_seconds == 1.25
    assert results[0].retriable is True


@pytest.mark.asyncio
async def test_closing_nonblocking_grader_cancels_unyielded_tasks():
    grader = _make_contract_grader(
        {"row": 0, "generations": ["first"]},
        {"row": 1, "generations": ["second"]},
    )
    second_started = asyncio.Event()
    second_cancelled = asyncio.Event()
    second_task: asyncio.Task | None = None

    async def grade(sample):
        nonlocal second_task
        if sample["row"] == 0:
            await second_started.wait()
            return sample
        second_task = asyncio.current_task()
        second_started.set()
        try:
            await asyncio.Future()
        finally:
            second_cancelled.set()

    results = grader.async_grade_all_samples_nonblocking(
        grade_fn=grade
    )
    first = await asyncio.wait_for(anext(results), timeout=1)
    assert first["row"] == 0
    assert second_task is not None

    try:
        await results.aclose()
        assert second_cancelled.is_set()
        assert second_task.done()
    finally:
        if second_task is not None and not second_task.done():
            second_task.cancel()
            await asyncio.gather(second_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_nonblocking_grader_propagates_child_cancellation():
    grader = _make_contract_grader(
        {"row": 0, "generations": ["answer"]},
    )

    async def cancel_during_grade(_sample):
        raise asyncio.CancelledError()

    results = grader.async_grade_all_samples_nonblocking(
        grade_fn=cancel_during_grade
    )

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(anext(results), timeout=1)
