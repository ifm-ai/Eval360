"""Focused contracts for external choice-scoring requests.

Ordinary completion/chat generation and external judges have independent
contracts. This suite covers the two-phase choice-scoring protocol only.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from scheduler.cache_salt import CacheSaltConfig
from scheduler.event import EventInstance
from scheduler.external_requests import (
    RATE_LIMITERS,
    canonicalize_external_endpoint,
)
from scheduler.grader.choice_scoring import ChoiceScoring
from scheduler.job import JobManager
from scheduler.model import ExternalRetryPolicy, ModelInstance, ModelType
from scheduler.openai_interface import LOCKED_CONNECTIONS, OpenAIConnection
from scheduler.progress import ProgressManager
from scheduler.task import AsyncGenerationTask
from scheduler.utils import ExceptionWrapper


EXTERNAL_URL = "https://api.openai.com/v1"


@pytest.fixture(autouse=True)
def clear_external_runtime_registries():
    LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()
    yield
    LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()


def make_choice_connection(
    *,
    max_simultaneous: int = 1,
    retry_policy: ExternalRetryPolicy | None = None,
) -> OpenAIConnection:
    model = MagicMock(spec=ModelInstance)
    model.name = "choice-model"
    model.api_model_name = "provider-model"
    model.serving_key = "choice-model"
    model.max_simultaneous_requests = max_simultaneous
    model.model_type = ModelType.BASE
    model.openai_kwargs = {}
    model.cache_salt = CacheSaltConfig()
    model.prompt_prefix_instructions = None
    model.is_external = True
    model.base_url = EXTERNAL_URL
    model.api_key = "sk-test"
    model.requests_per_minute = None
    model.external_retry_policy = retry_policy or ExternalRetryPolicy()

    task = MagicMock(spec=AsyncGenerationTask)
    task.average_over = [1]
    task.pass_at = [1]
    task.openai_settings = None
    task.grader = MagicMock()
    task.grader.type = "choice_scoring"

    connection = OpenAIConnection(
        model=model,
        task=task,
        event_instance=MagicMock(spec=EventInstance),
        job_manager=MagicMock(spec=JobManager),
        progress_manager=MagicMock(spec=ProgressManager),
        new_field_name="generations",
    )
    canonical_url = canonicalize_external_endpoint(EXTERNAL_URL)
    LOCKED_CONNECTIONS[model.serving_key]._slots[canonical_url] = [
        max_simultaneous
    ]
    return connection


def choice_response(logprobs_by_choice):
    choices = []
    generated_choices = 0
    for index, logprobs in enumerate(logprobs_by_choice):
        choice = MagicMock()
        choice.index = index
        choice.finish_reason = "stop"
        choice.stop_reason = None
        if logprobs is None:
            choice.logprobs = None
        else:
            choice.logprobs = MagicMock()
            if isinstance(logprobs, dict):
                for key, value in logprobs.items():
                    setattr(choice.logprobs, key, value)
                choice.logprobs.model_dump.return_value = logprobs
            else:
                choice.logprobs.token_logprobs = logprobs
                choice.logprobs.text_offset = None
                choice.logprobs.model_dump.return_value = {
                    "token_logprobs": logprobs
                }
                choice.text = "x"
                generated_choices += 1
        choices.append(choice)
    response = MagicMock()
    response.choices = choices
    response.usage = (
        {"completion_tokens": generated_choices}
        if generated_choices
        else None
    )
    return response


def with_tail_evidence(response, texts, *, completion_tokens):
    for choice, text in zip(response.choices, texts):
        choice.text = text
    response.usage = {"completion_tokens": completion_tokens}
    return response


def echoed_choice_response(
    logprobs_by_choice,
    prompts,
    *,
    generated_tail: bool,
    completion_tokens: int | None = None,
):
    texts = (
        [f"{prompt} tail" for prompt in prompts]
        if generated_tail
        else list(prompts)
    )
    if completion_tokens is None:
        completion_tokens = len(prompts) if generated_tail else 0
    return with_tail_evidence(
        choice_response(logprobs_by_choice),
        texts,
        completion_tokens=completion_tokens,
    )


def echoed_choice_response_factory(
    *responses,
    generated_tail: bool,
    completion_tokens: int | None = None,
):
    response_batches = iter(responses)

    async def create(*, prompt, **_kwargs):
        return echoed_choice_response(
            next(response_batches),
            prompt,
            generated_tail=generated_tail,
            completion_tokens=completion_tokens,
        )

    return create


def choice_prompt(**updates):
    prompt = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "completion_input": "Question:",
        "scoring_completions": ["A", "B"],
        "scoring_completion_labels": ["A", "B"],
        "scoring_completion_n_tokens": [1, 1],
        "ground_truth": "A",
    }
    prompt.update(updates)
    return prompt


def valid_full_response():
    return choice_response(
        [
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.3],
            },
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.7],
            },
        ]
    )


def valid_completion_response():
    return choice_response(
        [
            {
                "text_offset": [0, 7],
                "token_logprobs": [None, -0.2],
            },
            {
                "text_offset": [0, 7],
                "token_logprobs": [None, -0.4],
            },
        ]
    )


async def requests(*items):
    for item in items:
        yield item


async def collect(generator):
    return [item async for item in generator]


@pytest.mark.asyncio
@pytest.mark.parametrize("token_count", [True, 1.9, 0.5])
async def test_invalid_token_count_is_rejected_before_http(token_count):
    connection = make_choice_connection()
    client = MagicMock()
    client.completions.create = AsyncMock()
    connection._clients[EXTERNAL_URL] = client
    prompt = choice_prompt(
        scoring_completions=["A"],
        scoring_completion_labels=["A"],
        scoring_completion_n_tokens=[token_count],
    )

    results = await collect(
        connection.launch_requests(
            requests(prompt),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert isinstance(results[0], ExceptionWrapper)
    assert "positive integral" in results[0].trace
    client.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_offset_scoring_excludes_generated_token_tail():
    connection = make_choice_connection()
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=echoed_choice_response_factory(
            [
                [None, -0.25, -99.0],
                [None, -0.75, -98.0],
            ],
            [
                [None, -0.20, -97.0],
                [None, -0.40, -96.0],
            ],
            generated_tail=True,
        )
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(choice_prompt()),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )
    scored = ChoiceScoring._compute_choice_metrics(results[0])

    assert client.completions.create.await_count == 2
    assert all(
        call.kwargs["echo"] is True
        and call.kwargs["max_tokens"] == 1
        for call in client.completions.create.await_args_list
    )
    assert scored["choice_nll"] == pytest.approx([0.25, 0.75])
    assert scored["choice_nll_completion"] == pytest.approx([0.20, 0.40])


@pytest.mark.asyncio
async def test_no_offset_scoring_rejects_unproven_text_shape():
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    client = MagicMock()
    client.completions.create = AsyncMock(
        return_value=with_tail_evidence(
            choice_response(
                [
                    [None, -0.25],
                    [None, -0.75],
                ]
            ),
            ["", ""],
            completion_tokens=0,
        )
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(choice_prompt()),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )
    assert client.completions.create.await_count == 1
    assert isinstance(results[0], ExceptionWrapper)
    assert results[0].error_code == "malformed_response"


@pytest.mark.asyncio
async def test_no_offset_scoring_accepts_echoed_prompt_without_tail():
    connection = make_choice_connection()
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=echoed_choice_response_factory(
            [
                [None, -0.25],
                [None, -0.75],
            ],
            [
                [None, -0.25],
                [None, -0.75],
            ],
            generated_tail=False,
        )
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(choice_prompt()),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 2
    assert not isinstance(results[0], ExceptionWrapper)
    scored = ChoiceScoring._compute_choice_metrics(results[0])
    assert scored["choice_nll"] == pytest.approx([0.25, 0.75])
    assert scored["choice_nll_completion"] == pytest.approx([0.25, 0.75])


@pytest.mark.asyncio
async def test_empty_offsets_use_proven_generated_tail_evidence():
    connection = make_choice_connection()
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=echoed_choice_response_factory(
            [
                {
                    "text_offset": [],
                    "token_logprobs": [None, -0.25, -99.0],
                },
                {
                    "text_offset": [],
                    "token_logprobs": [None, -0.75, -98.0],
                },
            ],
            [
                {
                    "text_offset": [],
                    "token_logprobs": [None, -0.20, -97.0],
                },
                {
                    "text_offset": [],
                    "token_logprobs": [None, -0.40, -96.0],
                },
            ],
            generated_tail=True,
        )
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(choice_prompt()),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    scored = ChoiceScoring._compute_choice_metrics(results[0])

    assert scored["choice_nll"] == pytest.approx([0.25, 0.75])
    assert scored["choice_nll_completion"] == pytest.approx([0.20, 0.40])


@pytest.mark.asyncio
async def test_no_offset_scoring_rejects_multiple_generated_tails():
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=echoed_choice_response_factory(
            [
                [None, -0.25, -99.0, -98.0],
                [None, -0.75, -97.0, -96.0],
            ],
            generated_tail=True,
            completion_tokens=4,
        )
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(choice_prompt()),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 1
    assert isinstance(results[0], ExceptionWrapper)
    assert getattr(results[0], "error_code", None) == "malformed_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformation",
    ["duplicate-indices", "missing-logprobs"],
)
async def test_malformed_success_retries_before_phase_commit(malformation):
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    malformed = valid_full_response()
    if malformation == "duplicate-indices":
        malformed.choices[1].index = 0
    else:
        malformed.choices[0].logprobs = None
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=[
            malformed,
            valid_full_response(),
            valid_completion_response(),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(choice_prompt()),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 3
    assert not isinstance(results[0], ExceptionWrapper)
    assert results[0]["choice_scoring_full_logprobs"][0][
        "token_logprobs"
    ] == [None, -0.3]


@pytest.mark.asyncio
async def test_non_json_choice_metadata_retries_inside_phase():
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    malformed = choice_response(
        [
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -9.0],
            },
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -8.0],
            },
        ]
    )
    malformed.choices[0].finish_reason = object()
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=[
            malformed,
            valid_full_response(),
            valid_completion_response(),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(choice_prompt()),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 3
    assert results[0]["choice_scoring_full_logprobs"][0][
        "token_logprobs"
    ] == [None, -0.3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "nested_logprob",
    [
        {"logprob": "-0.3"},
        MagicMock(logprob="-0.3"),
    ],
    ids=["dict-numeric-string", "object-numeric-string"],
)
async def test_live_nested_numeric_string_is_strictly_retried(
    nested_logprob,
):
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    malformed = choice_response(
        [
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, nested_logprob],
            },
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.7],
            },
        ]
    )
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=[
            malformed,
            valid_full_response(),
            valid_completion_response(),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(
                choice_prompt(
                    scoring_completion_n_tokens=None,
                )
            ),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 3
    assert not isinstance(results[0], ExceptionWrapper)
    assert results[0]["choice_scoring_full_logprobs"][0][
        "token_logprobs"
    ] == [None, -0.3]


def test_persisted_numeric_string_logprob_remains_compatible():
    payload = {
        "text_offset": [0, 9],
        "token_logprobs": [None, {"logprob": "-0.3"}],
    }

    total, count = ChoiceScoring._sum_suffix_token_logprobs_with_count(
        payload,
        None,
        suffix_start_chars=9,
        prompt_text="Question: A",
    )

    assert total == pytest.approx(-0.3)
    assert count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_logprobs",
    [
        {
            "text_offset": [9],
            "token_logprobs": [None],
        },
        {
            "text_offset": [9],
            "token_logprobs": ["not-a-number"],
        },
        {
            "text_offset": [9],
            "token_logprobs": [float("nan")],
        },
        {
            "text_offset": [9],
            "token_logprobs": [float("inf")],
        },
        {
            "text_offset": [9.0],
            "token_logprobs": [-0.1],
        },
        {
            "text_offset": [9, 0],
            "token_logprobs": [-0.1, -0.2],
        },
        {
            "text_offset": [0],
            "token_logprobs": [None, -0.1],
        },
    ],
    ids=[
        "null-suffix",
        "non-numeric-suffix",
        "nan-suffix",
        "infinite-suffix",
        "non-integer-offset",
        "unordered-offsets",
        "offset-length-mismatch",
    ],
)
async def test_unusable_suffix_retries_transactionally(
    malformed_logprobs,
):
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    malformed = choice_response(
        [
            malformed_logprobs,
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.7],
            },
        ]
    )
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=[
            malformed,
            valid_full_response(),
            valid_completion_response(),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(
                choice_prompt(
                    scoring_completion_n_tokens=None,
                )
            ),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 3
    assert not isinstance(results[0], ExceptionWrapper)
    assert results[0]["choice_scoring_full_logprobs"][0][
        "token_logprobs"
    ] == [None, -0.3]


@pytest.mark.asyncio
async def test_unusable_suffix_exhausts_as_malformed_response():
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    malformed = choice_response(
        [
            {
                "text_offset": [9],
                "token_logprobs": [None],
            },
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.7],
            },
        ]
    )
    client = MagicMock()
    client.completions.create = AsyncMock(return_value=malformed)
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(
                choice_prompt(
                    scoring_completion_n_tokens=None,
                )
            ),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 2
    assert isinstance(results[0], ExceptionWrapper)
    assert results[0].error_code == "malformed_response"
    assert results[0].attempts == 2


@pytest.mark.asyncio
async def test_single_prompt_fallback_retries_msgpack_failure():
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    msgpack_error = APIConnectionError(
        request=None,
        message="MessagePack data is malformed",
    )
    valid_full_a = choice_response(
        [
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.3],
            }
        ]
    )
    valid_full_b = choice_response(
        [
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.7],
            }
        ]
    )
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=[
            msgpack_error,
            msgpack_error,
            valid_full_a,
            valid_full_b,
            valid_completion_response(),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(
                choice_prompt(
                    scoring_completion_n_tokens=None,
                )
            ),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 5
    assert results[0]["choice_scoring_metadata"]["request_modes"] == {
        "full": "single_prompt_fallback",
        "completion": "batched",
    }


@pytest.mark.asyncio
async def test_terminal_single_prompt_msgpack_is_malformed():
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=APIConnectionError(
            request=None,
            message="MessagePack data is malformed",
        )
    )
    connection._clients[EXTERNAL_URL] = client

    prompt = choice_prompt()
    result = await connection._make_external_request(
        0,
        prompt,
        {**prompt, "generations": []},
        1,
        0,
    )

    assert client.completions.create.await_count == 3
    assert isinstance(result, ExceptionWrapper)
    assert result.error_code == "malformed_response"
    assert result.attempts == 2


def context_length_error() -> BadRequestError:
    request = httpx.Request("POST", f"{EXTERNAL_URL}/completions")
    return BadRequestError(
        "maximum context length exceeded",
        response=httpx.Response(400, request=request),
        body={
            "message": "maximum context length exceeded",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        },
    )


@pytest.mark.asyncio
async def test_context_overflow_is_normal_and_next_row_continues():
    connection = make_choice_connection()

    async def create(**kwargs):
        if any("overflow" in prompt for prompt in kwargs["prompt"]):
            raise context_length_error()
        return echoed_choice_response(
            [
                [None, -0.3, -9.0],
                [None, -0.7, -8.0],
            ],
            kwargs["prompt"],
            generated_tail=True,
        )

    client = MagicMock()
    client.completions.create = AsyncMock(side_effect=create)
    connection._clients[EXTERNAL_URL] = client
    overflow = choice_prompt(row=0, completion_input="overflow")
    valid = choice_prompt(row=1, completion_input="valid")

    results = await collect(
        connection.launch_requests(
            requests(overflow, valid),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )
    results.sort(key=lambda result: result["row"])

    assert client.completions.create.await_count == 3
    assert results[0]["generations"] == [""]
    assert results[0]["eval360_input_too_long"] is True
    assert "choice_scoring_full_logprobs" in results[1]


@pytest.mark.asyncio
async def test_cancellation_between_phases_releases_lease_without_partial_row():
    connection = make_choice_connection()
    completion_started = asyncio.Event()
    completion_cancelled = asyncio.Event()

    async def create(**kwargs):
        if kwargs["prompt"][0].startswith("Answer:"):
            completion_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                completion_cancelled.set()
                raise
        return valid_full_response()

    client = MagicMock()
    client.completions.create = AsyncMock(side_effect=create)
    connection._clients[EXTERNAL_URL] = client
    collect_task = asyncio.create_task(
        collect(
            connection.launch_requests(
                requests(choice_prompt()),
                offset=0,
                completion_hook=AsyncMock(),
            )
        )
    )
    await asyncio.wait_for(completion_started.wait(), timeout=1)

    connection.cancel()
    results = await asyncio.wait_for(collect_task, timeout=1)

    assert results == []
    assert completion_cancelled.is_set()
    assert LOCKED_CONNECTIONS["choice-model"]._slots[EXTERNAL_URL][0] == 1


@pytest.mark.asyncio
async def test_choice_scoring_deadline_is_shared_across_phases():
    connection = make_choice_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=1,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )
    )
    clock = [0.0]
    connection._monotonic = lambda: clock[0]

    async def full_phase_after_deadline(*_args, **_kwargs):
        clock[0] = 1.1
        return valid_full_response()

    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=full_phase_after_deadline
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(choice_prompt()),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 1
    assert isinstance(results[0], ExceptionWrapper)
    assert results[0].error_code == "request_timeout"
    assert results[0].attempts == 0
    assert results[0].elapsed_seconds == pytest.approx(1.1)
