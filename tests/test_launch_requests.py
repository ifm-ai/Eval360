"""
Unit tests for OpenAIConnection.launch_requests.

Verifies:
  - all requests are dispatched (BASE and CHAT model types)
  - results are yielded as they complete (no ordered buffering)
  - completion_hook is called exactly once after all requests finish
  - multiple generations per prompt (n > 1) are batched correctly
  - transient network errors are retried up to 3 times (4 total attempts)
  - errors that exhaust retries produce an ExceptionWrapper in the output
  - connection-level errors (ConnectError, ClientConnectorError, APIConnectionError)
    break to the outer retry loop immediately rather than exhausting 3 retries
  - dead-URL detection breaks the inner loop so a new URL is obtained via get_live_url
  - null content/text from VLLM raises APIConnectionError instead of writing empty generations
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx
import aiohttp
from openai import APIConnectionError

from scheduler.openai_interface import OpenAIConnection, LOCKED_CONNECTIONS
from scheduler.model import CacheSaltConfig, ModelType, ModelInstance
from scheduler.task import AsyncGenerationTask
from scheduler.job import JobManager
from scheduler.event import EventInstance
from scheduler.progress import ProgressManager
from scheduler.utils import ExceptionWrapper, Sentinel

FAKE_URL = "http://fake:8000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completions_response(texts, finish_reasons=None, usage=None):
    finish_reasons = finish_reasons or [None] * len(texts)
    resp = MagicMock()
    choices = []
    for text, finish_reason in zip(texts, finish_reasons):
        choice = MagicMock()
        choice.text = text
        choice.finish_reason = finish_reason
        choice.stop_reason = None
        choice.logprobs = None
        choices.append(choice)
    resp.choices = choices
    resp.usage = usage
    return resp


def _chat_response(contents, finish_reasons=None, usage=None):
    finish_reasons = finish_reasons or [None] * len(contents)
    choices = []
    for content, finish_reason in zip(contents, finish_reasons):
        choice = MagicMock()
        choice.message.content = content
        choice.message.model_extra = {}
        choice.finish_reason = finish_reason
        choice.stop_reason = None
        choice.logprobs = None
        choices.append(choice)
    resp = MagicMock()
    resp.choices = choices
    resp.usage = usage
    return resp


def _chat_response_with_message_payloads(payloads, finish_reasons=None, usage=None):
    finish_reasons = finish_reasons or [None] * len(payloads)
    choices = []
    for payload, finish_reason in zip(payloads, finish_reasons):
        choice = MagicMock()
        choice.message.content = payload.get("content")
        choice.message.model_extra = payload.get("model_extra", {})
        for key, value in choice.message.model_extra.items():
            setattr(choice.message, key, value)
        choice.finish_reason = finish_reason
        choice.stop_reason = None
        choice.logprobs = None
        choices.append(choice)
    resp = MagicMock()
    resp.choices = choices
    resp.usage = usage
    return resp


@pytest.fixture(autouse=True)
def clear_locked_connections():
    """Prevent connection-slot state leaking between tests."""
    LOCKED_CONNECTIONS.clear()
    yield
    LOCKED_CONNECTIONS.clear()


def make_conn(
    model_type=ModelType.BASE,
    max_simultaneous=4,
    average_over=None,
    pass_at=None,
    name="test-model",
):
    model = MagicMock(spec=ModelInstance)
    model.name = name
    model.serving_key = name  # use name as serving key for test simplicity
    model.max_simultaneous_requests = max_simultaneous
    model.model_type = model_type
    model.openai_kwargs = {}
    model.cache_salt = CacheSaltConfig()
    model.prompt_prefix_instructions = None
    model.is_external = False
    model.requests_per_minute = None

    task = MagicMock(spec=AsyncGenerationTask)
    task.average_over = average_over or [1]
    task.pass_at = pass_at or [1]
    task.openai_settings = None

    conn = OpenAIConnection(
        model=model,
        task=task,
        event_instance=MagicMock(spec=EventInstance),
        job_manager=MagicMock(spec=JobManager),
        progress_manager=MagicMock(spec=ProgressManager),
        new_field_name="generations",
    )
    # Seed the pool with FAKE_URL so pool.acquire() returns immediately in tests
    LOCKED_CONNECTIONS[name]._slots[FAKE_URL] = [max_simultaneous]
    conn._clients[FAKE_URL] = MagicMock()
    conn._job_manager.is_url_live = MagicMock(return_value=True)
    return conn


async def _requests(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLaunchRequests:

    @pytest.mark.asyncio
    async def test_base_model_all_requests_made(self):
        conn = make_conn(model_type=ModelType.BASE)
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(3)]
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[_completions_response([f"out{i}"]) for i in range(3)]
        )

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        assert conn._clients[FAKE_URL].completions.create.call_count == 3
        assert [r["generations"] for r in results] == [["out0"], ["out1"], ["out2"]]
        assert all("logprobs" not in r for r in results)

    @pytest.mark.asyncio
    async def test_chat_model_all_requests_made(self):
        conn = make_conn(model_type=ModelType.CHAT)
        prompts = [
            {"chat_input": [{"role": "user", "content": f"q{i}"}], "completion_input": ""}
            for i in range(3)
        ]
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            side_effect=[_chat_response([f"ans{i}"]) for i in range(3)]
        )

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        assert conn._clients[FAKE_URL].chat.completions.create.call_count == 3
        assert [r["generations"] for r in results] == [["ans0"], ["ans1"], ["ans2"]]

    @pytest.mark.asyncio
    async def test_base_model_preserves_generation_metadata_and_usage(self):
        conn = make_conn(model_type=ModelType.BASE, pass_at=[2])
        prompt = {"completion_input": "p0", "chat_input": []}
        usage = MagicMock()
        usage.model_dump.return_value = {"completion_tokens": 1, "prompt_tokens": 7, "total_tokens": 8}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(
                ["", "A"],
                finish_reasons=["stop", "length"],
                usage=usage,
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["", "A"]
        assert results[0]["generation_metadata"] == [
            {"finish_reason": "stop", "stop_reason": None},
            {"finish_reason": "length", "stop_reason": None},
        ]
        assert results[0]["response_usage"] == [
            {"completion_tokens": 1, "prompt_tokens": 7, "total_tokens": 8}
        ]

    @pytest.mark.asyncio
    async def test_chat_model_preserves_generation_metadata_and_usage(self):
        conn = make_conn(model_type=ModelType.CHAT, pass_at=[2])
        prompt = {"chat_input": [{"role": "user", "content": "q0"}], "completion_input": ""}
        usage = MagicMock()
        usage.model_dump.return_value = {"completion_tokens": 4, "prompt_tokens": 9, "total_tokens": 13}
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(
                ["<think>reasoning</think>\nB", ""],
                finish_reasons=["stop", "stop"],
                usage=usage,
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["<think>reasoning</think>\nB", ""]
        assert results[0]["generation_metadata"] == [
            {"finish_reason": "stop", "stop_reason": None},
            {"finish_reason": "stop", "stop_reason": None},
        ]
        assert results[0]["response_usage"] == [
            {"completion_tokens": 4, "prompt_tokens": 9, "total_tokens": 13}
        ]

    @pytest.mark.asyncio
    async def test_chat_model_falls_back_to_reasoning_content_when_content_empty(self):
        conn = make_conn(model_type=ModelType.CHAT)
        prompt = {"chat_input": [{"role": "user", "content": "q0"}], "completion_input": ""}
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response_with_message_payloads(
                [{"content": "", "model_extra": {"reasoning_content": "Answer: B"}}],
                finish_reasons=["stop"],
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["Answer: B"]
        assert results[0]["generation_metadata"] == [
            {"finish_reason": "stop", "stop_reason": None},
        ]

    @pytest.mark.asyncio
    async def test_chat_model_preserves_reasoning_when_content_none(self):
        conn = make_conn(model_type=ModelType.CHAT)
        prompt = {"chat_input": [{"role": "user", "content": "q0"}], "completion_input": ""}
        usage = MagicMock()
        usage.model_dump.return_value = {"completion_tokens": 2, "prompt_tokens": 9, "total_tokens": 11}
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response_with_message_payloads(
                [{"content": None, "model_extra": {"reasoning": "Answer: C"}}],
                finish_reasons=["stop"],
                usage=usage,
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == [""]
        assert results[0]["reasoning"] == ["Answer: C"]
        assert results[0]["generation_metadata"] == [
            {"finish_reason": "stop", "stop_reason": None},
        ]
        assert results[0]["response_usage"] == [
            {"completion_tokens": 2, "prompt_tokens": 9, "total_tokens": 11}
        ]

    @pytest.mark.asyncio
    async def test_results_returned_as_they_complete_not_buffered(self):
        """First request is artificially delayed so later requests finish first.
        Results are yielded in completion order (not input order)."""
        conn = make_conn(model_type=ModelType.BASE)
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(4)]

        async def slow_first(prompt, **kwargs):
            if prompt == "p0":
                await asyncio.sleep(0.02)   # slowest — will finish last
            return _completions_response([f"result_for_{prompt}"])

        conn._clients[FAKE_URL].completions.create = slow_first

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        # All results present, each with correct generation
        assert sorted([r["completion_input"] for r in results]) == ["p0", "p1", "p2", "p3"]
        for r in results:
            prompt = r["completion_input"]
            assert r["generations"] == [f"result_for_{prompt}"]
        # p0 was delayed, so it should NOT be first (it completes last)
        assert results[-1]["completion_input"] == "p0"

    @pytest.mark.asyncio
    async def test_completion_hook_called_exactly_once(self):
        conn = make_conn(model_type=ModelType.BASE)
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(3)]
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[_completions_response(["out"]) for _ in range(3)]
        )
        hook = AsyncMock()

        await _collect(conn.launch_requests(_requests(*prompts), offset=0, completion_hook=hook))

        hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_num_generations_taken_from_average_over_when_larger(self):
        """average_over=[32], pass_at=[1] → num_generations=32 (average_over dominates)."""
        conn = make_conn(model_type=ModelType.BASE, average_over=[32], pass_at=[1])
        prompt = {"completion_input": "p0", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[_completions_response(["x"] * n) for n in [4] * 8]
        )
        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )
        assert len(results[0]["generations"]) == 32

    @pytest.mark.asyncio
    async def test_num_generations_uses_max_across_both_lists(self):
        """average_over=[8, 16], pass_at=[4, 32] → num_generations=32 (max across both)."""
        conn = make_conn(model_type=ModelType.BASE, average_over=[8, 16], pass_at=[4, 32])
        prompt = {"completion_input": "p0", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[_completions_response(["x"] * n) for n in [4] * 8]
        )
        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )
        assert len(results[0]["generations"]) == 32

    @pytest.mark.asyncio
    async def test_multiple_generations_batched_into_single_request(self):
        """average_over=[2], pass_at=[4] → num_generations=4.
        With max_simultaneous=4, all 4 should be in one API call (n=4)."""
        conn = make_conn(model_type=ModelType.BASE, average_over=[2], pass_at=[4])
        prompt = {"completion_input": "p0", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["a", "b", "c", "d"])
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert conn._clients[FAKE_URL].completions.create.call_count == 1
        call_kwargs = conn._clients[FAKE_URL].completions.create.call_args
        assert call_kwargs.kwargs["n"] == 4
        assert results[0]["generations"] == ["a", "b", "c", "d"]

    @pytest.mark.asyncio
    async def test_transient_network_error_retries_three_times_then_succeeds(self):
        """Three failures followed by a success should succeed after 4 total attempts."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            httpx.HTTPError("fail"),
            httpx.HTTPError("fail"),
            httpx.HTTPError("fail"),
            _completions_response(["ok"]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert conn._clients[FAKE_URL].completions.create.call_count == 4
        assert results[0]["generations"] == ["ok"]

    @pytest.mark.asyncio
    async def test_exhausted_retries_yields_exception_wrapper(self):
        """Four consecutive failures exhaust all retries; result is an ExceptionWrapper
        whose instance is the partial result dict (with a 'generations' key)."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=httpx.HTTPError("always fails")
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert conn._clients[FAKE_URL].completions.create.call_count == 4  # 1 + 3 retries
        assert len(results) == 1
        assert isinstance(results[0], ExceptionWrapper)
        # instance should be the partial result dict, not the bare input
        assert "generations" in results[0].instance
        assert results[0].instance["generations"] == []
        assert results[0].instance["completion_input"] == "p"

    @pytest.mark.asyncio
    async def test_error_on_one_prompt_does_not_prevent_others(self):
        """An ExceptionWrapper for one prompt should not stop the other prompts."""
        conn = make_conn(model_type=ModelType.BASE)
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(3)]
        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            httpx.HTTPError("fail"),   # p0 attempt 1
            httpx.HTTPError("fail"),   # p0 attempt 2
            httpx.HTTPError("fail"),   # p0 attempt 3
            httpx.HTTPError("fail"),   # p0 attempt 4 — exhausted
            _completions_response(["out1"]),
            _completions_response(["out2"]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        exception_results = [r for r in results if isinstance(r, ExceptionWrapper)]
        ok_results = [r for r in results if not isinstance(r, ExceptionWrapper)]
        assert len(exception_results) == 1
        assert len(ok_results) == 2
        assert sorted(r["generations"] for r in ok_results) == [["out1"], ["out2"]]


# ---------------------------------------------------------------------------
# force_logprobs tests
# ---------------------------------------------------------------------------

def make_conn_with_logprobs(model_type, force_logprobs):
    model = MagicMock()
    model.name = "test-model"
    model.max_simultaneous_requests = 4
    model.model_type = model_type
    model.openai_kwargs = {}
    model.prompt_prefix_instructions = None
    model.is_external = False
    model.requests_per_minute = None

    task = MagicMock()
    task.average_over = [1]
    task.pass_at = [1]
    task.openai_settings = None

    conn = OpenAIConnection(
        model=model,
        task=task,
        event_instance=MagicMock(),
        job_manager=MagicMock(),
        progress_manager=MagicMock(),
        new_field_name="generations",
        force_logprobs=force_logprobs,
    )
    return conn


class TestForceLogprobs:

    def test_base_model_sets_logprobs_count(self):
        conn = make_conn_with_logprobs(ModelType.BASE, force_logprobs=True)
        assert conn._openai_kwargs["logprobs"] == 5

    def test_base_model_does_not_set_top_logprobs(self):
        conn = make_conn_with_logprobs(ModelType.BASE, force_logprobs=True)
        assert "top_logprobs" not in conn._openai_kwargs

    def test_chat_model_sets_logprobs_true(self):
        conn = make_conn_with_logprobs(ModelType.CHAT, force_logprobs=True)
        assert conn._openai_kwargs["logprobs"] is True

    def test_chat_model_sets_top_logprobs(self):
        conn = make_conn_with_logprobs(ModelType.CHAT, force_logprobs=True)
        assert conn._openai_kwargs["top_logprobs"] == 5

    def test_force_logprobs_false_sets_nothing(self):
        conn = make_conn_with_logprobs(ModelType.CHAT, force_logprobs=False)
        assert "logprobs" not in conn._openai_kwargs
        assert "top_logprobs" not in conn._openai_kwargs

    def test_does_not_override_existing_logprobs(self):
        model = MagicMock()
        model.name = "test-model"
        model.max_simultaneous_requests = 4
        model.model_type = ModelType.CHAT
        model.openai_kwargs = {"logprobs": True, "top_logprobs": 10}
        model.prompt_prefix_instructions = None

        task = MagicMock()
        task.average_over = [1]
        task.pass_at = [1]
        task.openai_settings = None

        conn = OpenAIConnection(
            model=model, task=task,
            event_instance=MagicMock(), job_manager=MagicMock(),
            progress_manager=MagicMock(), new_field_name="generations",
            force_logprobs=True,
        )
        assert conn._openai_kwargs["top_logprobs"] == 10


class TestOpenAIKwargsMerging:

    def test_model_extra_body_merges_with_task_extra_body(self):
        model = MagicMock()
        model.name = "test-model"
        model.max_simultaneous_requests = 4
        model.model_type = ModelType.CHAT
        model.openai_kwargs = {
            "extra_body": {
                "chat_template_kwargs": {"reasoning_effort": "high"},
            },
            "temperature": 0.7,
        }
        model.prompt_prefix_instructions = None

        task = MagicMock()
        task.average_over = [1]
        task.pass_at = [1]
        task.openai_settings = {
            "extra_body": {
                "guided_choice": ["A", "B"],
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "max_tokens": 4096,
        }

        conn = OpenAIConnection(
            model=model,
            task=task,
            event_instance=MagicMock(),
            job_manager=MagicMock(),
            progress_manager=MagicMock(),
            new_field_name="generations",
        )

        assert conn._openai_kwargs["extra_body"]["guided_choice"] == ["A", "B"]
        assert conn._openai_kwargs["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "high",
        }
        assert conn._openai_kwargs["temperature"] == 0.7
        assert conn._openai_kwargs["max_tokens"] == 4096

    def test_model_reasoning_effort_overrides_task_reasoning_effort(self):
        model = MagicMock()
        model.name = "test-model"
        model.max_simultaneous_requests = 4
        model.model_type = ModelType.CHAT
        model.openai_kwargs = {
            "extra_body": {
                "chat_template_kwargs": {"reasoning_effort": "high"},
            },
        }
        model.prompt_prefix_instructions = None

        task = MagicMock()
        task.average_over = [1]
        task.pass_at = [1]
        task.openai_settings = {
            "extra_body": {
                "chat_template_kwargs": {"reasoning_effort": "medium"},
            },
        }

        conn = OpenAIConnection(
            model=model,
            task=task,
            event_instance=MagicMock(),
            job_manager=MagicMock(),
            progress_manager=MagicMock(),
            new_field_name="generations",
        )

        assert conn._openai_kwargs["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Large-scale parametrized tests
# ---------------------------------------------------------------------------

# Powers of 2 from 1 → 16 for num_gen; n_prompts=5000 cases are marked @long
# and skipped in CI (see conftest --long / pyproject.toml addopts).
_LARGE_SCALE_PARAMS = [
    pytest.param(num_gen, n_prompts,
                 marks=pytest.mark.long if n_prompts == 5_000 else [])
    for num_gen in [1, 2, 4, 8, 16]
    for n_prompts in [1_000, 5_000]
]


class TestLaunchRequestsLargeScale:

    @pytest.mark.parametrize("num_gen,n_prompts", _LARGE_SCALE_PARAMS)
    @pytest.mark.asyncio
    async def test_all_requests_made_and_correct_generation_count(self, num_gen, n_prompts):
        """Every prompt produces exactly num_gen generations."""
        # High max_simultaneous so connection-slot contention doesn't dominate runtime
        conn = make_conn(
            model_type=ModelType.BASE,
            max_simultaneous=512,
            average_over=[1],
            pass_at=[num_gen],
            name=f"model-{n_prompts}-{num_gen}",
        )
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(n_prompts)]

        async def mock_create(**kwargs):
            n = kwargs.get("n", 1)
            return _completions_response(["out"] * n)

        conn._clients[FAKE_URL].completions.create = mock_create

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == n_prompts
        for r in results:
            assert len(r["generations"]) == num_gen

    @pytest.mark.parametrize("num_gen,n_prompts", _LARGE_SCALE_PARAMS)
    @pytest.mark.asyncio
    async def test_all_results_returned_large_scale(self, num_gen, n_prompts):
        """All results are returned even when requests complete out of order.

        Every even-indexed prompt yields immediately; every odd-indexed prompt
        yields to the event loop first (asyncio.sleep(0)), causing even prompts
        to finish before the odd prompt that precedes them in the input.
        Results are no longer guaranteed in input order.
        """
        conn = make_conn(
            model_type=ModelType.BASE,
            max_simultaneous=512,
            average_over=[1],
            pass_at=[num_gen],
            name=f"order-model-{n_prompts}-{num_gen}",
        )
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(n_prompts)]

        async def mock_create(**kwargs):
            prompt = kwargs.get("prompt", "")
            idx = int(prompt[1:])
            if idx % 2 == 1:
                await asyncio.sleep(0)  # yield — lets even-indexed tasks overtake
            n = kwargs.get("n", 1)
            return _completions_response([prompt] * n)

        conn._clients[FAKE_URL].completions.create = mock_create

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == n_prompts
        result_prompts = sorted((r["completion_input"] for r in results), key=lambda x: int(x[1:]))
        assert result_prompts == [f"p{i}" for i in range(n_prompts)]


# ---------------------------------------------------------------------------
# prompt_prefix_instructions tests
# ---------------------------------------------------------------------------

def make_conn_with_prefix(model_type, prefix):
    model = MagicMock()
    model.name = "test-model"
    model.serving_key = "test-model"
    model.max_simultaneous_requests = 4
    model.model_type = model_type
    model.openai_kwargs = {}
    model.cache_salt = CacheSaltConfig()
    model.prompt_prefix_instructions = prefix
    model.is_external = False
    model.requests_per_minute = None

    task = MagicMock()
    task.average_over = [1]
    task.pass_at = [1]
    task.openai_settings = None

    conn = OpenAIConnection(
        model=model,
        task=task,
        event_instance=MagicMock(),
        job_manager=MagicMock(),
        progress_manager=MagicMock(),
        new_field_name="generations",
    )
    LOCKED_CONNECTIONS["test-model"]._slots[FAKE_URL] = [4]
    conn._clients[FAKE_URL] = MagicMock()
    conn._job_manager.is_url_live = MagicMock(return_value=True)
    return conn


class TestPromptPrefixInstructions:

    @pytest.mark.asyncio
    async def test_base_model_prefix_prepended_to_completion_input(self):
        conn = make_conn_with_prefix(ModelType.BASE, "Think step by step.")
        prompts = [{"completion_input": "What is 2+2?", "chat_input": []}]
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["4"])
        )

        await _collect(conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock()))

        call_kwargs = conn._clients[FAKE_URL].completions.create.call_args
        assert call_kwargs.kwargs["prompt"] == "Think step by step.\n\nWhat is 2+2?"

    @pytest.mark.asyncio
    async def test_base_model_no_prefix_passes_completion_input_unchanged(self):
        conn = make_conn_with_prefix(ModelType.BASE, None)
        prompts = [{"completion_input": "What is 2+2?", "chat_input": []}]
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["4"])
        )

        await _collect(conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock()))

        call_kwargs = conn._clients[FAKE_URL].completions.create.call_args
        assert call_kwargs.kwargs["prompt"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_chat_model_prefix_inserted_as_system_message_when_none_present(self):
        conn = make_conn_with_prefix(ModelType.CHAT, "Be concise.")
        prompts = [{"chat_input": [{"role": "user", "content": "Hello"}], "completion_input": ""}]
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(["Hi"])
        )

        await _collect(conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock()))

        call_kwargs = conn._clients[FAKE_URL].chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be concise."}
        assert messages[1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_chat_model_prefix_prepended_to_existing_system_message(self):
        conn = make_conn_with_prefix(ModelType.CHAT, "Be concise.")
        prompts = [{
            "chat_input": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "completion_input": "",
        }]
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(["Hi"])
        )

        await _collect(conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock()))

        call_kwargs = conn._clients[FAKE_URL].chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise.\n\nYou are a helpful assistant."
        assert messages[1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_chat_model_no_prefix_passes_messages_unchanged(self):
        conn = make_conn_with_prefix(ModelType.CHAT, None)
        original_messages = [{"role": "user", "content": "Hello"}]
        prompts = [{"chat_input": original_messages, "completion_input": ""}]
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(["Hi"])
        )

        await _collect(conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock()))

        call_kwargs = conn._clients[FAKE_URL].chat.completions.create.call_args
        assert call_kwargs.kwargs["messages"] is original_messages

    @pytest.mark.asyncio
    async def test_prefix_does_not_mutate_original_elem(self):
        """The prefix logic must deepcopy chat_input, not mutate the source element."""
        conn = make_conn_with_prefix(ModelType.CHAT, "Be concise.")
        original_messages = [{"role": "user", "content": "Hello"}]
        prompts = [{"chat_input": original_messages, "completion_input": ""}]
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(["Hi"])
        )

        await _collect(conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock()))

        # original_messages must be unmodified
        assert original_messages == [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Connection-level errors break immediately to the outer loop
# ---------------------------------------------------------------------------
# Bug: when a VLLM node dies, connection errors (APIConnectionError,
# httpx.ConnectError, aiohttp.ClientConnectorError) were exhausting all 3
# retries against the dead server before giving up, causing the progress bar
# to fake-advance as all in-flight requests completed simultaneously with errors.
#
# Fix: these errors now set done=True immediately, skipping all retries and
# returning to the outer `while requests_left` loop which calls get_live_url()
# again (blocking until a healthy URL is available).
# ---------------------------------------------------------------------------

FAKE_URL2 = "http://fake2:8000"


def _make_connector_error():
    from aiohttp.client_reqrep import ConnectionKey
    key = ConnectionKey(host="fake", port=8000, is_ssl=False, ssl=False,
                        proxy=None, proxy_auth=None, proxy_headers_hash=None)
    return aiohttp.ClientConnectorError(connection_key=key, os_error=OSError("refused"))


class TestConnectionErrorBreaksImmediately:
    """Connection-level errors must not exhaust the 3-retry budget.

    The inner retry loop must exit immediately (done=True), and the outer
    per-generation loop must call get_live_url() again to obtain a new URL.
    """

    @pytest.mark.asyncio
    async def test_api_connection_error_does_not_retry_three_times(self):
        """APIConnectionError on a live URL: exits inner loop after 1 attempt,
        not 4 (1 + 3 retries)."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}

        # Second call to get_live_url still returns FAKE_URL (URL not yet evicted)
        conn._job_manager.get_live_url = AsyncMock(return_value=FAKE_URL)
        conn._job_manager.is_url_live = MagicMock(return_value=True)

        # First call raises APIConnectionError; second succeeds
        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            APIConnectionError(request=None, message="connection refused"),
            _completions_response(["ok"]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["ok"]
        # Only 2 create calls: 1 fail (no retries) + 1 succeed
        assert conn._clients[FAKE_URL].completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_httpx_connect_error_does_not_retry_three_times(self):
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}

        conn._job_manager.get_live_url = AsyncMock(return_value=FAKE_URL)
        conn._job_manager.is_url_live = MagicMock(return_value=True)

        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            httpx.ConnectError("connection refused"),
            _completions_response(["ok"]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["ok"]
        assert conn._clients[FAKE_URL].completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_aiohttp_connector_error_does_not_retry_three_times(self):
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}

        conn._job_manager.get_live_url = AsyncMock(return_value=FAKE_URL)
        conn._job_manager.is_url_live = MagicMock(return_value=True)

        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            _make_connector_error(),
            _completions_response(["ok"]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["ok"]
        assert conn._clients[FAKE_URL].completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_connection_error_calls_pool_acquire_again(self):
        """After a connection error, the outer loop re-acquires a URL from the pool."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}

        conn._job_manager.is_url_live = MagicMock(return_value=True)
        conn._clients[FAKE_URL2] = MagicMock()

        pool = LOCKED_CONNECTIONS["test-model"]

        # Simulate scheduler eviction: when FAKE_URL fails, remove it and add FAKE_URL2
        async def create_then_evict(*args, **kwargs):
            pool._slots.pop(FAKE_URL, None)
            pool._slots[FAKE_URL2] = [1]
            raise APIConnectionError(request=None, message="refused")

        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=create_then_evict)
        conn._clients[FAKE_URL2].completions.create = AsyncMock(
            return_value=_completions_response(["ok"])
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["ok"]
        assert conn._clients[FAKE_URL2].completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_dead_url_detected_breaks_inner_loop(self):
        """When is_url_live returns False, the inner loop must break immediately
        and the outer loop must re-acquire a URL from the pool."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}

        conn._clients[FAKE_URL2] = MagicMock()
        pool = LOCKED_CONNECTIONS["test-model"]
        pool._slots[FAKE_URL2] = [1]  # pre-add backup URL

        # Simulate scheduler eviction when dead URL is detected:
        # is_url_live(FAKE_URL) → False and removes it from pool
        def is_url_live_side_effect(url):
            if url == FAKE_URL:
                pool._slots.pop(FAKE_URL, None)
                return False
            return True

        conn._job_manager.is_url_live = MagicMock(side_effect=is_url_live_side_effect)

        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=httpx.HTTPError("server died")
        )
        conn._clients[FAKE_URL2].completions.create = AsyncMock(
            return_value=_completions_response(["ok"])
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["ok"]
        # FAKE_URL got 1 attempt; FAKE_URL2 got 1 attempt — not 4 on dead URL
        assert conn._clients[FAKE_URL].completions.create.call_count == 1
        assert conn._clients[FAKE_URL2].completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_non_connection_httpx_error_still_retries_three_times(self):
        """Non-connection httpx.HTTPError on a live URL must exhaust the 3-retry budget."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}

        conn._job_manager.get_live_url = AsyncMock(return_value=FAKE_URL)
        conn._job_manager.is_url_live = MagicMock(return_value=True)

        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            httpx.HTTPError("transient"),
            httpx.HTTPError("transient"),
            httpx.HTTPError("transient"),
            _completions_response(["ok"]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["ok"]
        assert conn._clients[FAKE_URL].completions.create.call_count == 4  # 1 + 3 retries


# ---------------------------------------------------------------------------
# Null content/text from VLLM raises APIConnectionError
# ---------------------------------------------------------------------------
# Bug: when a VLLM server is dying it can return null text/content in choices.
# These were written as empty generations instead of being treated as errors,
# causing silent data corruption.
#
# Fix: null text (BASE) or null content (CHAT) now raises APIConnectionError
# which then follows the connection-error path (breaks to outer loop).
# ---------------------------------------------------------------------------

class TestNullResponseRaisesError:

    @pytest.mark.asyncio
    async def test_base_model_null_text_raises_api_connection_error(self):
        """Null text in a BASE response must raise APIConnectionError, not write empty str."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}
        conn._job_manager.is_url_live = MagicMock(return_value=True)

        null_response = MagicMock()
        null_response.choices = [MagicMock(text=None, logprobs=None)]

        # After the null response raises an error, succeed on retry
        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            null_response,
            _completions_response(["ok"]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        # Should recover and produce valid output (connection error path → outer loop retry)
        assert results[0]["generations"] == ["ok"]
        # Must not have written an empty string as a generation
        assert "" not in results[0]["generations"]

    @pytest.mark.asyncio
    async def test_chat_model_null_content_stop_reason_gives_empty_string(self):
        """Null content with finish_reason='stop' means a thinking-only response — treat as empty string."""
        conn = make_conn(model_type=ModelType.CHAT)
        prompt = {"chat_input": [{"role": "user", "content": "q"}], "completion_input": ""}
        conn._job_manager.is_url_live = MagicMock(return_value=True)

        null_choice = MagicMock()
        null_choice.message.content = None
        null_choice.message.reasoning_content = None
        null_choice.message.model_extra = {}
        null_choice.message.tool_calls = []
        null_choice.logprobs = None
        null_choice.finish_reason = "stop"

        null_response = MagicMock()
        null_response.choices = [null_choice]

        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(return_value=null_response)

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == [""]
        assert None not in results[0]["generations"]

    @pytest.mark.asyncio
    async def test_chat_model_null_content_bad_finish_reason_gives_exception_wrapper(self):
        """Null content with finish_reason not in (stop, length) suggests a server error — yields ExceptionWrapper."""
        conn = make_conn(model_type=ModelType.CHAT)
        prompt = {"chat_input": [{"role": "user", "content": "q"}], "completion_input": ""}
        conn._job_manager.is_url_live = MagicMock(return_value=True)

        null_choice = MagicMock()
        null_choice.message.content = None
        null_choice.message.reasoning_content = None
        null_choice.message.model_extra = {}
        null_choice.message.tool_calls = []
        null_choice.logprobs = None
        null_choice.finish_reason = "error"  # not stop/length → RuntimeError → ExceptionWrapper

        null_response = MagicMock()
        null_response.choices = [null_choice]

        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(return_value=null_response)

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert isinstance(results[0], ExceptionWrapper)
        assert None not in (results[0].instance.get("generations") or [])

    @pytest.mark.asyncio
    async def test_null_text_does_not_write_none_as_generation(self):
        """Even across multiple null responses from a dying server, None must never
        appear in the generations list — each null triggers a retry (connection-error
        path), and once a healthy URL is available the result is written correctly."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}

        null_response = MagicMock()
        null_response.choices = [MagicMock(text=None, logprobs=None)]

        # Two null responses on FAKE_URL → connection-error path → pool retries.
        # After 2 nulls, simulate scheduler evicting FAKE_URL and adding FAKE_URL2.
        conn._clients[FAKE_URL2] = MagicMock()
        pool = LOCKED_CONNECTIONS["test-model"]
        pool._slots[FAKE_URL2] = [4]

        conn._job_manager.is_url_live = MagicMock(return_value=True)

        null_call_count = [0]

        async def null_with_eviction(*args, **kwargs):
            null_call_count[0] += 1
            if null_call_count[0] >= 2:
                pool._slots.pop(FAKE_URL, None)  # simulate scheduler eviction
            return null_response

        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=null_with_eviction)
        conn._clients[FAKE_URL2].completions.create = AsyncMock(
            return_value=_completions_response(["ok"])
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["ok"]
        assert None not in results[0]["generations"]
        assert "" not in results[0]["generations"]
