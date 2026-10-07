"""Contracts for ordinary completion/chat requests to external endpoints.

Choice-scoring and judge requests deliberately live in separate test modules so
this file can be used as the focused CI/evaluation gate for the ordinary
generation stack layer.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    BadRequestError,
)
from openai._models import construct_type
from openai.types import Completion
from openai.types.chat import ChatCompletion

from scheduler.cache_salt import CacheSaltConfig
from scheduler.event import EventInstance
from scheduler.external_requests import (
    RATE_LIMITERS,
    canonicalize_external_endpoint,
)
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


def make_external_connection(
    *,
    model_type: ModelType = ModelType.CHAT,
    max_simultaneous: int = 4,
    average_over: int = 1,
    retry_policy: ExternalRetryPolicy | None = None,
    requests_per_minute: int | None = None,
) -> OpenAIConnection:
    model = MagicMock(spec=ModelInstance)
    model.name = "test-model"
    model.api_model_name = "provider-model"
    model.serving_key = "test-model"
    model.max_simultaneous_requests = max_simultaneous
    model.model_type = model_type
    model.openai_kwargs = {}
    model.cache_salt = CacheSaltConfig()
    model.prompt_prefix_instructions = None
    model.is_external = True
    model.base_url = EXTERNAL_URL
    model.api_key = "sk-test"
    model.requests_per_minute = requests_per_minute
    model.external_retry_policy = retry_policy or ExternalRetryPolicy()

    task = MagicMock(spec=AsyncGenerationTask)
    task.average_over = [average_over]
    task.pass_at = [1]
    task.openai_settings = None
    task.grader = MagicMock()
    task.grader.type = "multiple_choice"

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
    connection._job_manager.is_url_live = MagicMock(return_value=True)
    return connection


def completion_response(
    texts: list[object],
    *,
    indices: list[object] | None = None,
    logprobs: list[dict | None] | None = None,
):
    choices = []
    for position, text in enumerate(texts):
        choice = MagicMock()
        choice.index = (
            position if indices is None else indices[position]
        )
        choice.text = text
        choice.finish_reason = "stop"
        choice.stop_reason = None
        if logprobs is None or logprobs[position] is None:
            choice.logprobs = None
        else:
            choice.logprobs = MagicMock()
            choice.logprobs.model_dump.return_value = logprobs[position]
        choices.append(choice)
    response = MagicMock()
    response.choices = choices
    response.usage = None
    return response


def chat_response(
    contents: list[object],
    *,
    indices: list[object] | None = None,
    logprobs: list[dict | None] | None = None,
    finish_reasons: list[str | None] | None = None,
    reasoning_contents: list[str | None] | None = None,
    tool_calls: list[list[dict] | None] | None = None,
):
    choices = []
    for position, content in enumerate(contents):
        choice = MagicMock()
        choice.index = (
            position if indices is None else indices[position]
        )
        choice.finish_reason = (
            "stop"
            if finish_reasons is None
            else finish_reasons[position]
        )
        choice.stop_reason = None
        choice.message.content = content
        choice.message.reasoning_content = (
            None
            if reasoning_contents is None
            else reasoning_contents[position]
        )
        choice.message.model_extra = {}
        raw_tool_calls = (
            None if tool_calls is None else tool_calls[position]
        )
        if raw_tool_calls is None:
            choice.message.tool_calls = []
        else:
            choice.message.tool_calls = []
            for raw_tool_call in raw_tool_calls:
                tool_call = MagicMock()
                tool_call.model_dump.return_value = raw_tool_call
                choice.message.tool_calls.append(tool_call)
        if logprobs is None or logprobs[position] is None:
            choice.logprobs = None
        else:
            choice.logprobs = MagicMock()
            choice.logprobs.model_dump.return_value = logprobs[position]
        choices.append(choice)
    response = MagicMock()
    response.choices = choices
    response.usage = None
    return response


def sdk_completion_response(text):
    return construct_type(
        type_=Completion,
        value={
            "id": "completion-id",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "text": text,
                }
            ],
            "created": 0,
            "model": "provider-model",
            "object": "text_completion",
        },
    )


def sdk_chat_response(content):
    return construct_type(
        type_=ChatCompletion,
        value={
            "id": "chat-id",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "annotations": [],
                        "content": content,
                        "refusal": None,
                        "role": "assistant",
                    },
                }
            ],
            "created": 0,
            "model": "provider-model",
            "object": "chat.completion",
        },
    )


async def requests(*items):
    for item in items:
        yield item


async def collect(generator):
    return [item async for item in generator]


def prompt_for(model_type: ModelType) -> dict:
    return {
        "chat_input": [{"role": "user", "content": "hello"}],
        "completion_input": "hello",
    }


def endpoint_for(client, model_type: ModelType):
    if model_type is ModelType.BASE:
        return client.completions
    return client.chat.completions


def response_for(model_type: ModelType, values, **kwargs):
    if model_type is ModelType.BASE:
        return completion_response(values, **kwargs)
    return chat_response(values, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
async def test_malformed_sdk_success_retries_and_recovers(model_type):
    connection = make_external_connection(
        model_type=model_type,
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    malformed = APIResponseValidationError(
        response=httpx.Response(
            200,
            request=httpx.Request("POST", f"{EXTERNAL_URL}/completions"),
        ),
        body={"choices": "invalid"},
        message="invalid response schema",
    )
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    endpoint.create = AsyncMock(
        side_effect=[
            malformed,
            response_for(model_type, ["answer"]),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 2
    assert results[0]["generations"] == ["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
async def test_duplicate_choice_indices_retry_before_commit(model_type):
    connection = make_external_connection(
        model_type=model_type,
        max_simultaneous=2,
        average_over=2,
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    endpoint.create = AsyncMock(
        side_effect=[
            response_for(
                model_type,
                ["discarded-a", "discarded-b"],
                indices=[0, 0],
            ),
            response_for(model_type, ["answer-a", "answer-b"]),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 2
    assert results[0]["generations"] == ["answer-a", "answer-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
@pytest.mark.parametrize(
    "indices",
    [[0, 2], [0, True]],
    ids=["missing-index", "boolean-index"],
)
async def test_invalid_choice_identity_exhausts_as_malformed(
    model_type,
    indices,
):
    connection = make_external_connection(
        model_type=model_type,
        max_simultaneous=2,
        average_over=2,
        retry_policy=ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    endpoint.create = AsyncMock(
        return_value=response_for(
            model_type,
            ["bad-a", "bad-b"],
            indices=indices,
        )
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 1
    assert isinstance(results[0], ExceptionWrapper)
    assert getattr(results[0], "error_code", None) == "malformed_response"
    assert getattr(results[0], "attempts", None) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
async def test_out_of_order_choices_are_sorted_by_identity(model_type):
    connection = make_external_connection(
        model_type=model_type,
        max_simultaneous=2,
        average_over=2,
    )
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    endpoint.create = AsyncMock(
        return_value=response_for(
            model_type,
            ["second", "first"],
            indices=[1, 0],
        )
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert results[0]["generations"] == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
@pytest.mark.parametrize("logprobs_first", [False, True])
async def test_mixed_parallel_logprobs_retry_without_partial_commit(
    model_type,
    logprobs_first,
):
    connection = make_external_connection(
        model_type=model_type,
        max_simultaneous=2,
        average_over=2,
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    one_logprobs = {"tokens": ["bad"], "token_logprobs": [-0.5]}
    mixed = (
        [one_logprobs, None]
        if logprobs_first
        else [None, one_logprobs]
    )
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    endpoint.create = AsyncMock(
        side_effect=[
            response_for(
                model_type,
                ["discarded-a", "discarded-b"],
                logprobs=mixed,
            ),
            response_for(model_type, ["answer-a", "answer-b"]),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 2
    assert results[0]["generations"] == ["answer-a", "answer-b"]
    assert "logprobs" not in results[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_first", [False, True])
async def test_optional_choice_metadata_preserves_within_batch_identity(
    metadata_first,
):
    connection = make_external_connection(
        max_simultaneous=2,
        average_over=2,
    )
    present = 0 if metadata_first else 1
    reasoning = [None, None]
    reasoning[present] = "reasoning"
    calls = [None, None]
    calls[present] = [{"type": "function", "id": "call-1"}]
    finish_reasons = [None, None]
    finish_reasons[present] = "stop"
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=chat_response(
            ["first", "second"],
            reasoning_contents=reasoning,
            tool_calls=calls,
            finish_reasons=finish_reasons,
        )
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(ModelType.CHAT)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    expected_reasoning = [None, None]
    expected_reasoning[present] = "reasoning"
    expected_calls = [None, None]
    expected_calls[present] = [
        {"type": "function", "id": "call-1"}
    ]
    expected_finishes = [None, None]
    expected_finishes[present] = "stop"
    assert results[0]["reasoning"] == expected_reasoning
    assert results[0]["tool_calls"] == expected_calls
    assert results[0]["finish_reasons"] == expected_finishes


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_first", [False, True])
async def test_optional_choice_metadata_preserves_cross_batch_identity(
    metadata_first,
):
    connection = make_external_connection(
        max_simultaneous=4,
        average_over=5,
    )
    metadata_count = 4 if metadata_first else 1
    metadata_response = chat_response(
        ["with-metadata"]
        + [
            f"metadata-batch-plain-{index}"
            for index in range(1, metadata_count)
        ],
        reasoning_contents=["reasoning"] + [None] * (metadata_count - 1),
        tool_calls=[
            [{"type": "function", "id": "call-1"}]
        ] + [None] * (metadata_count - 1),
        finish_reasons=["stop"] + [None] * (metadata_count - 1),
    )
    plain_count = 1 if metadata_first else 4
    plain_response = chat_response(
        [f"plain-{index}" for index in range(plain_count)],
        finish_reasons=[None] * plain_count,
    )
    responses = (
        [metadata_response, plain_response]
        if metadata_first
        else [plain_response, metadata_response]
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=responses)
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(ModelType.CHAT)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    present = 0 if metadata_first else 4
    expected_reasoning = [None] * 5
    expected_reasoning[present] = "reasoning"
    expected_calls = [None] * 5
    expected_calls[present] = [
        {"type": "function", "id": "call-1"}
    ]
    expected_finishes = [None] * 5
    expected_finishes[present] = "stop"
    assert results[0]["reasoning"] == expected_reasoning
    assert results[0]["tool_calls"] == expected_calls
    assert results[0]["finish_reasons"] == expected_finishes


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
@pytest.mark.parametrize("logprobs_first", [False, True])
async def test_cross_batch_logprobs_mismatch_retries_before_commit(
    model_type,
    logprobs_first,
):
    connection = make_external_connection(
        model_type=model_type,
        max_simultaneous=4,
        average_over=5,
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    logprob = {"tokens": ["answer"], "token_logprobs": [-0.5]}
    first_logprobs = [logprob] * 4 if logprobs_first else None
    mismatched_logprobs = None if logprobs_first else [logprob]
    corrected_logprobs = [logprob] if logprobs_first else None
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    endpoint.create = AsyncMock(
        side_effect=[
            response_for(
                model_type,
                [f"answer-{index}" for index in range(4)],
                logprobs=first_logprobs,
            ),
            response_for(
                model_type,
                ["discarded"],
                logprobs=mismatched_logprobs,
            ),
            response_for(
                model_type,
                ["answer-4"],
                logprobs=corrected_logprobs,
            ),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 3
    assert results[0]["generations"] == [
        "answer-0",
        "answer-1",
        "answer-2",
        "answer-3",
        "answer-4",
    ]
    if logprobs_first:
        assert results[0]["logprobs"] == [logprob] * 5
    else:
        assert "logprobs" not in results[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("response_kind", ["completion", "chat"])
async def test_non_string_sdk_payload_retries_transactionally(response_kind):
    model_type = (
        ModelType.BASE
        if response_kind == "completion"
        else ModelType.CHAT
    )
    connection = make_external_connection(
        model_type=model_type,
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    malformed = (
        sdk_completion_response(123)
        if model_type is ModelType.BASE
        else sdk_chat_response({"bad": "shape"})
    )
    valid = (
        sdk_completion_response("answer")
        if model_type is ModelType.BASE
        else sdk_chat_response("answer")
    )
    endpoint.create = AsyncMock(side_effect=[malformed, valid])
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 2
    assert results[0]["generations"] == ["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "reasoning", "expected_generation"),
    [
        pytest.param(None, None, "", id="null-terminal"),
        pytest.param("", None, "", id="empty"),
        pytest.param("   ", None, "   ", id="whitespace"),
        pytest.param(
            None,
            "usable reasoning",
            "",
            id="null-with-reasoning",
        ),
        pytest.param(
            "",
            "usable reasoning",
            "usable reasoning",
            id="empty-with-reasoning",
        ),
        pytest.param(
            "   ",
            "usable reasoning",
            "   ",
            id="whitespace-with-reasoning",
        ),
    ],
)
async def test_chat_content_preserves_existing_generation_behavior(
    content,
    reasoning,
    expected_generation,
):
    connection = make_external_connection()
    response = chat_response([content])
    response.choices[0].message.reasoning_content = reasoning
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(ModelType.CHAT)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.chat.completions.create.await_count == 1
    assert results[0]["generations"] == [expected_generation]
    if reasoning is None:
        assert "reasoning" not in results[0]
    else:
        assert results[0]["reasoning"] == [reasoning]


@pytest.mark.asyncio
async def test_null_chat_content_with_nonterminal_finish_retries_before_commit():
    connection = make_external_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            chat_response([None], finish_reasons=["error"]),
            chat_response(["answer"]),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(ModelType.CHAT)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.chat.completions.create.await_count == 2
    assert results[0]["generations"] == ["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
@pytest.mark.parametrize(
    "metadata_kind",
    ["finish_reason", "usage", "nonfinite_logprobs"],
)
async def test_non_json_response_metadata_retries_before_commit(
    model_type,
    metadata_kind,
):
    connection = make_external_connection(
        model_type=model_type,
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    malformed = response_for(model_type, ["discarded"])
    if metadata_kind == "finish_reason":
        malformed.choices[0].finish_reason = object()
    elif metadata_kind == "usage":
        malformed.usage = object()
    else:
        malformed.choices[0].logprobs = MagicMock()
        malformed.choices[0].logprobs.model_dump.return_value = {
            "token_logprobs": [float("nan")],
        }

    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    endpoint.create = AsyncMock(
        side_effect=[
            malformed,
            response_for(model_type, ["answer"]),
        ],
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 2
    assert results[0]["generations"] == ["answer"]
    assert results[0]["finish_reasons"] == ["stop"]


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_kind", ["reasoning", "tool_calls"])
async def test_malformed_chat_metadata_retries_before_commit(metadata_kind):
    connection = make_external_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    malformed = chat_response(["discarded"])
    if metadata_kind == "reasoning":
        malformed.choices[0].message.reasoning_content = {
            "not": "text",
        }
    else:
        malformed.choices[0].message.tool_calls = object()

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            malformed,
            chat_response(["answer"]),
        ],
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(ModelType.CHAT)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.chat.completions.create.await_count == 2
    assert results[0]["generations"] == ["answer"]


@pytest.mark.asyncio
async def test_owned_key_collision_cannot_partially_commit_generation():
    connection = make_external_connection(
        retry_policy=ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=chat_response(["must-not-leak"]),
    )
    connection._clients[EXTERNAL_URL] = client
    prompt = {
        **prompt_for(ModelType.CHAT),
        "generation_metadata": {"input_owned": True},
    }

    results = await collect(
        connection.launch_requests(
            requests(prompt),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert isinstance(results[0], ExceptionWrapper)
    assert results[0].instance["generations"] == []
    assert results[0].instance["generation_metadata"] == {
        "input_owned": True,
    }


@pytest.mark.asyncio
async def test_later_batch_failure_preserves_only_completed_batch():
    connection = make_external_connection(
        model_type=ModelType.BASE,
        max_simultaneous=4,
        average_over=5,
        retry_policy=ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    client = MagicMock()
    client.completions.create = AsyncMock(
        side_effect=[
            completion_response(["one", "two", "three", "four"]),
            APIConnectionError(request=None, message="connection refused"),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(ModelType.BASE)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert client.completions.create.await_count == 2
    assert isinstance(results[0], ExceptionWrapper)
    assert results[0].instance["generations"] == [
        "one",
        "two",
        "three",
        "four",
    ]
    assert getattr(results[0], "attempts", None) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
async def test_external_prompt_deadline_is_shared_across_capacity_batches(
    model_type,
):
    connection = make_external_connection(
        model_type=model_type,
        max_simultaneous=1,
        average_over=2,
        retry_policy=ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=1,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    clock = [0.0]
    connection._monotonic = lambda: clock[0]
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)

    async def first_response_after_deadline(*_args, **_kwargs):
        clock[0] = 1.1
        return response_for(model_type, ["first"])

    endpoint.create = AsyncMock(side_effect=first_response_after_deadline)
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 1
    assert isinstance(results[0], ExceptionWrapper)
    assert results[0].instance["generations"] == ["first"]
    assert results[0].error_code == "request_timeout"
    assert results[0].attempts == 0
    assert results[0].elapsed_seconds == pytest.approx(1.1)


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
@pytest.mark.parametrize("model_type", [ModelType.BASE, ModelType.CHAT])
async def test_context_overflow_is_terminal_normal_result(model_type):
    connection = make_external_connection(
        model_type=model_type,
        average_over=2,
        retry_policy=ExternalRetryPolicy(
            max_attempts=3,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    client = MagicMock()
    endpoint = endpoint_for(client, model_type)
    endpoint.create = AsyncMock(side_effect=context_length_error())
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(model_type)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    assert endpoint.create.await_count == 1
    assert results[0]["generations"] == ["", ""]
    assert results[0]["eval360_input_too_long"] is True
    assert not isinstance(results[0], ExceptionWrapper)


@pytest.mark.asyncio
async def test_context_overflow_pads_prior_choice_metadata():
    connection = make_external_connection(
        max_simultaneous=4,
        average_over=5,
        retry_policy=ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    choice_logprob = {
        "tokens": ["answer"],
        "token_logprobs": [-0.5],
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            chat_response(
                ["answer-0", "answer-1", "answer-2", "answer-3"],
                reasoning_contents=["reasoning"] * 4,
                tool_calls=[[{"id": "call-0"}]] * 4,
                finish_reasons=["stop"] * 4,
                logprobs=[choice_logprob] * 4,
            ),
            context_length_error(),
        ]
    )
    connection._clients[EXTERNAL_URL] = client

    results = await collect(
        connection.launch_requests(
            requests(prompt_for(ModelType.CHAT)),
            offset=0,
            completion_hook=AsyncMock(),
        )
    )

    result = results[0]
    assert result["generations"] == [
        "answer-0",
        "answer-1",
        "answer-2",
        "answer-3",
        "",
    ]
    for field in (
        "finish_reasons",
        "reasoning",
        "tool_calls",
        "generation_metadata",
        "logprobs",
    ):
        assert len(result[field]) == len(result["generations"])
        assert result[field][-1] is None
    assert result["eval360_input_too_long"] is True


@pytest.mark.asyncio
async def test_client_setup_failure_does_not_charge_provider_or_attempt():
    connection = make_external_connection(
        max_simultaneous=1,
        requests_per_minute=60,
        retry_policy=ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        ),
    )
    limiter = connection._rate_limiter
    assert limiter is not None
    limiter.acquire = AsyncMock()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=chat_response(["answer"])
    )
    connection._get_client = MagicMock(
        side_effect=[RuntimeError("client setup failed"), client]
    )

    results = await asyncio.wait_for(
        collect(
            connection.launch_requests(
                requests(
                    {**prompt_for(ModelType.CHAT), "row": 0},
                    {**prompt_for(ModelType.CHAT), "row": 1},
                ),
                offset=0,
                completion_hook=AsyncMock(),
            )
        ),
        timeout=1,
    )

    results.sort(key=lambda result: result.instance["row"]
                 if isinstance(result, ExceptionWrapper)
                 else result["row"])
    assert isinstance(results[0], ExceptionWrapper)
    assert getattr(results[0], "error_code", None) == "client_setup_failed"
    assert getattr(results[0], "attempts", None) == 0
    assert results[1]["generations"] == ["answer"]
    limiter.acquire.assert_awaited_once()
    assert (
        LOCKED_CONNECTIONS["test-model"]
        ._slots[EXTERNAL_URL][0]
        == 1
    )


@pytest.mark.asyncio
async def test_fatal_request_task_cancels_and_awaits_siblings():
    connection = make_external_connection()
    pool = LOCKED_CONNECTIONS[connection._model.serving_key]
    all_started = asyncio.Event()
    request_tasks = []
    cancelled_siblings = 0

    async def acquire(_requests_left):
        nonlocal cancelled_siblings
        request_tasks.append(asyncio.current_task())
        call_index = len(request_tasks)
        if call_index == 3:
            all_started.set()
        if call_index == 1:
            await all_started.wait()
            raise RuntimeError("pool failed")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled_siblings += 1
            raise

    pool.acquire = AsyncMock(side_effect=acquire)
    prompts = [
        {**prompt_for(ModelType.CHAT), "row": row}
        for row in range(3)
    ]

    with pytest.raises(RuntimeError, match="pool failed"):
        await asyncio.wait_for(
            collect(
                connection.launch_requests(
                    requests(*prompts),
                    offset=0,
                    completion_hook=AsyncMock(),
                )
            ),
            timeout=1,
        )

    await asyncio.sleep(0)
    assert cancelled_siblings == 2
    assert all(task.done() for task in request_tasks)
