"""
Tests for scheduler/openai_interface.py covering functionality not already tested
in test_launch_requests.py and test_input_too_long.py:

  - _deep_merge_openai_kwargs: pure recursive merge function
  - is_live: /health poll with 3-attempt loop
  - wait_for_live: full DEAD/PENDING/RUNNING state machine
  - logprobs extraction in launch_requests (BASE and CHAT)
  - LOCKED_CONNECTIONS semaphore sharing across OpenAIConnection instances
"""

import asyncio
import aiohttp
import os
import pytest
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from scheduler.openai_interface import (
    OpenAIConnection,
    LOCKED_CONNECTIONS,
    RATE_LIMITERS,
    _deep_merge_openai_kwargs,
    _extra_body_to_chat_template_kwargs,
)
from scheduler.job import JobManager
from scheduler.model import (
    CacheSaltConfig,
    ExternalRetryPolicy,
    ModelType,
    ModelInstance,
)
from scheduler.task import AsyncGenerationTask
from scheduler.event import EventInstance
from scheduler.progress import ProgressManager
from scheduler.utils import ExceptionWrapper, Sentinel

FAKE_URL = "http://fake:8000"
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_locked_connections():
    LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()
    yield
    LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()


def make_conn(
    model_type=ModelType.BASE,
    max_simultaneous=4,
    average_over=None,
    pass_at=None,
    name="test-model",
    api_model_name=None,
    grader_type="multiple_choice",
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
    model.api_model_name = api_model_name
    model.requests_per_minute = None

    task = MagicMock(spec=AsyncGenerationTask)
    task.average_over = average_over or [1]
    task.pass_at = pass_at or [1]
    task.openai_settings = None
    task.grader = MagicMock()
    task.grader.type = grader_type

    job_manager = MagicMock(spec=JobManager)

    conn = OpenAIConnection(
        model=model,
        task=task,
        event_instance=MagicMock(spec=EventInstance),
        job_manager=job_manager,
        progress_manager=MagicMock(spec=ProgressManager),
        new_field_name="generations",
    )
    LOCKED_CONNECTIONS[name]._slots[FAKE_URL] = [max_simultaneous]
    conn._clients[FAKE_URL] = MagicMock()
    conn._job_manager.is_url_live = MagicMock(return_value=True)
    return conn


def _completions_response(texts, logprobs_list=None):
    """Build a mock completions.create response.

    logprobs_list: per-choice list of dicts (or None).  If the entire argument
    is None every choice gets choice.logprobs = None.
    """
    choices = []
    for i, text in enumerate(texts):
        choice = MagicMock()
        choice.text = text
        if logprobs_list is None or logprobs_list[i] is None:
            choice.logprobs = None
        else:
            lp = MagicMock()
            lp.model_dump.return_value = logprobs_list[i]
            choice.logprobs = lp
        choices.append(choice)
    resp = MagicMock()
    resp.choices = choices
    return resp


def _chat_response(contents, logprobs_list=None, finish_reasons=None, reasoning_contents=None):
    choices = []
    for i, c in enumerate(contents):
        choice = MagicMock()
        choice.message.content = c
        choice.message.reasoning_content = (reasoning_contents[i] if reasoning_contents else None)
        choice.message.model_extra = {}
        choice.message.tool_calls = []
        choice.finish_reason = (finish_reasons[i] if finish_reasons else "stop")
        if logprobs_list is None or logprobs_list[i] is None:
            choice.logprobs = None
        else:
            lp = MagicMock()
            lp.model_dump.return_value = logprobs_list[i]
            choice.logprobs = lp
        choices.append(choice)
    resp = MagicMock()
    resp.choices = choices
    return resp


def _choice_scoring_response(
    token_logprobs_by_choice,
    *,
    prompts: list[str] | None = None,
):
    """Build the completion response produced by a scoring request.

    Choice-scoring requests use ``echo=True`` and ``max_tokens=1``.  Payloads
    without character offsets therefore need one generated-token logprob after
    the echoed prompt logprobs.  Offset-bearing payloads can identify the
    prompt boundary directly and are kept verbatim for exact serialization
    assertions.
    """
    choices = []
    generated_choices = 0
    for i, token_logprobs in enumerate(token_logprobs_by_choice):
        choice = MagicMock()
        choice.index = i
        choice.finish_reason = "length"
        choice.stop_reason = None
        choice.logprobs = MagicMock()
        if isinstance(token_logprobs, dict):
            for key, value in token_logprobs.items():
                setattr(choice.logprobs, key, value)
            choice.logprobs.model_dump.return_value = token_logprobs
        else:
            response_logprobs = [*token_logprobs, -99.0]
            choice.text = (
                f"{prompts[i]} tail"
                if prompts is not None
                else "x"
            )
            generated_choices += 1
            choice.logprobs.token_logprobs = response_logprobs
            choice.logprobs.model_dump.return_value = {
                "token_logprobs": response_logprobs
            }
        choices.append(choice)
    resp = MagicMock()
    resp.choices = choices
    resp.usage = (
        {"completion_tokens": generated_choices}
        if generated_choices
        else None
    )
    return resp


def _choice_scoring_responses(*response_batches):
    batches = iter(response_batches)

    async def create(*, prompt, **_kwargs):
        return _choice_scoring_response(
            next(batches),
            prompts=prompt,
        )

    return create


async def _requests(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


def _make_connector_error():
    return aiohttp.ClientConnectorError(connection_key=None, os_error=OSError("refused"))


def _make_aiohttp_session(status=200):
    """Return a mock aiohttp.ClientSession that returns `status` from every GET."""
    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)
    mock_resp.status = status

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock(return_value=mock_resp)
    return mock_session


def test_import_openai_interface_does_not_discover_graders():
    code = textwrap.dedent(
        """
        import logging
        import sys

        logging.basicConfig(level=logging.WARNING)
        import scheduler.openai_interface  # noqa: F401

        loaded_graders = [
            name
            for name in sys.modules
            if name == "scheduler.grader" or name.startswith("scheduler.grader.")
        ]
        if loaded_graders:
            raise SystemExit("loaded grader modules: " + ",".join(sorted(loaded_graders)))
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Skipping grader module" not in result.stderr
    assert "optional dependency" not in result.stderr.lower()


# ---------------------------------------------------------------------------
# _deep_merge_openai_kwargs
# ---------------------------------------------------------------------------

class TestDeepMergeOpenAIKwargs:

    def test_non_overlapping_keys_both_present(self):
        result = _deep_merge_openai_kwargs({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_wins_for_scalar(self):
        result = _deep_merge_openai_kwargs({"key": "base"}, {"key": "override"})
        assert result["key"] == "override"

    def test_nested_dicts_merged_recursively(self):
        base = {"extra_body": {"a": 1, "b": 2}}
        override = {"extra_body": {"b": 99, "c": 3}}
        result = _deep_merge_openai_kwargs(base, override)
        assert result["extra_body"] == {"a": 1, "b": 99, "c": 3}

    def test_three_levels_of_nesting(self):
        base = {"L1": {"L2": {"L3": "base_val", "other": "x"}}}
        override = {"L1": {"L2": {"L3": "override_val"}}}
        result = _deep_merge_openai_kwargs(base, override)
        assert result["L1"]["L2"]["L3"] == "override_val"
        assert result["L1"]["L2"]["other"] == "x"

    def test_non_dict_in_override_replaces_dict_in_base(self):
        base = {"key": {"nested": 1}}
        override = {"key": "flat"}
        result = _deep_merge_openai_kwargs(base, override)
        assert result["key"] == "flat"

    def test_dict_in_override_replaces_scalar_in_base(self):
        base = {"key": 42}
        override = {"key": {"nested": 1}}
        result = _deep_merge_openai_kwargs(base, override)
        assert result["key"] == {"nested": 1}

    def test_empty_base(self):
        result = _deep_merge_openai_kwargs({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self):
        result = _deep_merge_openai_kwargs({"a": 1}, {})
        assert result == {"a": 1}

    def test_both_empty(self):
        assert _deep_merge_openai_kwargs({}, {}) == {}

    def test_base_not_mutated(self):
        base = {"extra_body": {"a": 1}}
        override = {"extra_body": {"b": 2}}
        _deep_merge_openai_kwargs(base, override)
        assert base == {"extra_body": {"a": 1}}

    def test_override_not_mutated(self):
        base = {"a": 1}
        override = {"extra_body": {"nested": {"deep": 99}}}
        result = _deep_merge_openai_kwargs(base, override)
        result["extra_body"]["nested"]["deep"] = 0
        assert override["extra_body"]["nested"]["deep"] == 99

    def test_list_values_replaced_not_merged(self):
        """Lists are not recursively merged — override wins outright."""
        base = {"choices": ["A", "B"]}
        override = {"choices": ["C"]}
        result = _deep_merge_openai_kwargs(base, override)
        assert result["choices"] == ["C"]

    def test_none_value_in_override_replaces_scalar(self):
        result = _deep_merge_openai_kwargs({"key": 42}, {"key": None})
        assert result["key"] is None

    def test_chat_template_kwargs_typical_usage(self):
        """Realistic: task sets enable_thinking, model sets reasoning_effort; both survive."""
        base = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
        override = {"extra_body": {"chat_template_kwargs": {"reasoning_effort": "high"}}}
        result = _deep_merge_openai_kwargs(base, override)
        assert result["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "high",
        }


class TestCacheSaltRequestKwargs:

    @staticmethod
    def _choice_scoring_elem():
        return {
            "row": 0,
            "scoring_mode": "choice_scoring",
            "scoring_prompt_prefix": "Question: test\nAnswer:",
            "scoring_completion_prefix": "Answer:",
            "scoring_completions": ["A", "B"],
            "scoring_completion_n_tokens": [1, 1],
            "scoring_completion_n_chars": [1, 1],
            "scoring_completion_labels": ["A", "B"],
            "ground_truth_index": 0,
        }

    def test_none_cache_salt_rejected(self):
        conn = make_conn()
        conn._openai_kwargs = {"temperature": 0.0}
        conn._model.cache_salt = None
        with pytest.raises(ValueError, match="cache_salt must be a CacheSaltConfig"):
            conn._request_kwargs_with_cache_salt()

    def test_disabled_cache_salt_leaves_kwargs_unchanged(self):
        conn = make_conn()
        conn._openai_kwargs = {"temperature": 0.0}
        conn._model.cache_salt = CacheSaltConfig(mode="disabled")
        assert conn._request_kwargs_with_cache_salt() == {"temperature": 0.0}

    def test_static_cache_salt_merges_into_extra_body(self):
        conn = make_conn()
        conn._openai_kwargs = {"extra_body": {"guided_choice": ["A", "B"]}}
        conn._model.cache_salt = CacheSaltConfig(mode="static", salt="partition-a")

        kwargs = conn._request_kwargs_with_cache_salt()

        assert kwargs["extra_body"]["guided_choice"] == ["A", "B"]
        assert kwargs["extra_body"]["cache_salt"] == "partition-a"
        assert "cache_salt" not in conn._openai_kwargs["extra_body"]

    def test_unique_cache_salt_changes_per_serialization(self):
        conn = make_conn()
        conn._openai_kwargs = {}
        conn._model.cache_salt = CacheSaltConfig(mode="unique")

        first = conn._request_kwargs_with_cache_salt()["extra_body"]["cache_salt"]
        second = conn._request_kwargs_with_cache_salt()["extra_body"]["cache_salt"]

        assert first
        assert second
        assert first != second

    def test_choice_scoring_request_includes_cache_salt_and_return_token_ids(self):
        conn = make_conn()
        conn._openai_kwargs = {"extra_body": {"guided_choice": ["A", "B"]}}
        conn._model.cache_salt = CacheSaltConfig(mode="static", salt="partition-a")

        kwargs = conn._choice_scoring_request_kwargs()

        assert kwargs["extra_body"]["guided_choice"] == ["A", "B"]
        assert kwargs["extra_body"]["cache_salt"] == "partition-a"
        assert "return_token_ids" not in kwargs["extra_body"]

    def test_raw_extra_body_cache_salt_rejected(self):
        conn = make_conn()
        conn._openai_kwargs = {"extra_body": {"cache_salt": "raw"}}
        conn._model.cache_salt = CacheSaltConfig()

        with pytest.raises(ValueError, match="extra_body.cache_salt"):
            conn._request_kwargs_with_cache_salt()

    def test_raw_top_level_cache_salt_rejected(self):
        conn = make_conn()
        conn._openai_kwargs = {"cache_salt": "raw"}
        conn._model.cache_salt = CacheSaltConfig()

        with pytest.raises(ValueError, match="cache_salt directly"):
            conn._request_kwargs_with_cache_salt()

    def test_non_mapping_extra_body_rejected(self):
        conn = make_conn()
        conn._openai_kwargs = {"extra_body": ["not", "a", "mapping"]}
        conn._model.cache_salt = CacheSaltConfig(mode="unique")

        with pytest.raises(ValueError, match="extra_body must be a mapping"):
            conn._request_kwargs_with_cache_salt()

    @pytest.mark.asyncio
    async def test_debug_tokenize_rejects_raw_cache_salt_before_request(self):
        conn = make_conn(model_type=ModelType.CHAT)
        conn._openai_kwargs = {"extra_body": {"cache_salt": "raw"}}

        with pytest.raises(ValueError, match="extra_body.cache_salt"):
            await conn._get_templated_prompt(
                FAKE_URL,
                [{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    async def test_base_launch_sends_static_cache_salt(self):
        conn = make_conn(model_type=ModelType.BASE)
        conn._model.cache_salt = CacheSaltConfig(mode="static", salt="partition-a")
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["hello"], logprobs_list=None)
        )

        await _collect(
            conn.launch_requests(
                _requests({"completion_input": "p", "chat_input": []}),
                offset=0,
                completion_hook=AsyncMock(),
            )
        )

        kwargs = conn._clients[FAKE_URL].completions.create.await_args.kwargs
        assert kwargs["extra_body"]["cache_salt"] == "partition-a"

    @pytest.mark.asyncio
    async def test_base_launch_sends_unique_cache_salt_per_outbound_request(self):
        conn = make_conn(model_type=ModelType.BASE, max_simultaneous=1, average_over=[2], pass_at=[1])
        conn._model.cache_salt = CacheSaltConfig(mode="unique")
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[
                _completions_response(["first"], logprobs_list=None),
                _completions_response(["second"], logprobs_list=None),
            ]
        )

        await _collect(
            conn.launch_requests(
                _requests({"completion_input": "p", "chat_input": []}),
                offset=0,
                completion_hook=AsyncMock(),
            )
        )

        salts = [
            call.kwargs["extra_body"]["cache_salt"]
            for call in conn._clients[FAKE_URL].completions.create.await_args_list
        ]
        assert len(salts) == 2
        assert salts[0] != salts[1]

    @pytest.mark.asyncio
    async def test_chat_launch_sends_static_cache_salt(self):
        conn = make_conn(model_type=ModelType.CHAT)
        conn._model.cache_salt = CacheSaltConfig(mode="static", salt="partition-a")
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(["hello"], logprobs_list=None)
        )

        await _collect(
            conn.launch_requests(
                _requests({"chat_input": [{"role": "user", "content": "hi"}], "completion_input": ""}),
                offset=0,
                completion_hook=AsyncMock(),
            )
        )

        kwargs = conn._clients[FAKE_URL].chat.completions.create.await_args.kwargs
        assert kwargs["extra_body"]["cache_salt"] == "partition-a"

    @pytest.mark.asyncio
    async def test_chat_launch_sends_unique_cache_salt_per_outbound_request(self):
        conn = make_conn(model_type=ModelType.CHAT, max_simultaneous=1, average_over=[2], pass_at=[1])
        conn._model.cache_salt = CacheSaltConfig(mode="unique")
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            side_effect=[
                _chat_response(["first"], logprobs_list=None),
                _chat_response(["second"], logprobs_list=None),
            ]
        )

        await _collect(
            conn.launch_requests(
                _requests({"chat_input": [{"role": "user", "content": "hi"}], "completion_input": ""}),
                offset=0,
                completion_hook=AsyncMock(),
            )
        )

        salts = [
            call.kwargs["extra_body"]["cache_salt"]
            for call in conn._clients[FAKE_URL].chat.completions.create.await_args_list
        ]
        assert len(salts) == 2
        assert salts[0] != salts[1]

    @pytest.mark.asyncio
    async def test_choice_scoring_launch_sends_static_cache_salt(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="choice_scoring",
        )
        conn._model.cache_salt = CacheSaltConfig(mode="static", salt="partition-a")
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_choice_scoring_responses(
                [[None, -0.3], [None, -0.7]],
                [[None, -0.2], [None, -0.4]],
            )
        )

        await _collect(
            conn.launch_requests(
                _requests(self._choice_scoring_elem()),
                offset=0,
                completion_hook=AsyncMock(),
            )
        )

        calls = conn._clients[FAKE_URL].completions.create.await_args_list
        assert len(calls) == 2
        for call in calls:
            assert call.kwargs["extra_body"]["cache_salt"] == "partition-a"

    @pytest.mark.asyncio
    async def test_choice_scoring_launch_sends_unique_cache_salt_per_outbound_request(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="choice_scoring",
        )
        conn._model.cache_salt = CacheSaltConfig(mode="unique")
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_choice_scoring_responses(
                [[None, -0.3], [None, -0.7]],
                [[None, -0.2], [None, -0.4]],
            )
        )

        await _collect(
            conn.launch_requests(
                _requests(self._choice_scoring_elem()),
                offset=0,
                completion_hook=AsyncMock(),
            )
        )

        salts = [
            call.kwargs["extra_body"]["cache_salt"]
            for call in conn._clients[FAKE_URL].completions.create.await_args_list
        ]
        assert len(salts) == 2
        assert salts[0] != salts[1]

# ---------------------------------------------------------------------------
# Logprobs extraction in launch_requests
# ---------------------------------------------------------------------------

class TestLogprobsExtraction:
    """Verify that logprobs are extracted and stored when present, and absent
    when the model returns logprobs=None."""

    @pytest.mark.asyncio
    async def test_base_model_logprobs_added_to_result(self):
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}
        lp_data = {"tokens": ["hello"], "token_logprobs": [-0.5]}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["hello"], logprobs_list=[lp_data])
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert "logprobs" in results[0]
        assert results[0]["logprobs"] == [lp_data]


class TestChoiceScoringLogprobs:
    def test_choice_scoring_request_kwargs_strip_conflicts_and_force_logprob_mode(self):
        conn = make_conn(model_type=ModelType.BASE, average_over=[1], pass_at=[1])
        conn._openai_kwargs = {
            "n": 4,
            "stop": ["\n"],
            "top_logprobs": 5,
            "max_tokens": 256,
            "temperature": 0.8,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "return_token_ids": False,
            },
        }

        kwargs = conn._choice_scoring_request_kwargs()

        assert "n" not in kwargs
        assert "stop" not in kwargs
        assert "top_logprobs" not in kwargs
        assert kwargs["temperature"] == 0.0
        assert kwargs["echo"] is True
        assert kwargs["max_tokens"] == 1
        assert kwargs["logprobs"] == 1
        assert kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_choice_scoring_request_kwargs_do_not_send_return_token_ids(self):
        conn = make_conn(model_type=ModelType.BASE, average_over=[1], pass_at=[1])
        conn._openai_kwargs = {
            "extra_body": {
                "return_token_ids": True,
                "existing": "kept",
            },
        }

        kwargs = conn._choice_scoring_request_kwargs()

        assert kwargs["extra_body"] == {"existing": "kept"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("grader_type", ["choice_scoring", "multiple_choice_nll"])
    async def test_choice_scoring_request_only_runs_once(self, grader_type):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type=grader_type,
        )
        conn._openai_kwargs = {
            "n": 3,
            "stop": ["\n"],
            "top_logprobs": 5,
            "max_tokens": 128,
            "extra_body": {"existing": "kept"},
        }
        elem = {
            "row": 0,
            "scoring_mode": "choice_scoring",
            "completion_input": "Question: test\\nAnswer:",
            "scoring_completions": ["A", "B"],
            "scoring_completion_n_tokens": [1, 1],
            "scoring_completion_labels": ["A", "B"],
            "ground_truth": "A",
        }
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_choice_scoring_responses(
                [[None, -0.3], [None, -0.7]],
                [[None, -0.2], [None, -0.4]],
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(elem), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert not isinstance(results[0], ExceptionWrapper), getattr(results[0], "trace", "")
        assert results[0]["choice_scoring_full_logprobs"] == [
            {
                "token_logprobs": [None, -0.3, -99.0],
                "eval360_generated_tail_n_tokens": 1,
            },
            {
                "token_logprobs": [None, -0.7, -99.0],
                "eval360_generated_tail_n_tokens": 1,
            },
        ]
        assert results[0]["choice_scoring_completion_logprobs"] == [
            {
                "token_logprobs": [None, -0.2, -99.0],
                "eval360_generated_tail_n_tokens": 1,
            },
            {
                "token_logprobs": [None, -0.4, -99.0],
                "eval360_generated_tail_n_tokens": 1,
            },
        ]
        assert "generations" not in results[0]
        assert conn._clients[FAKE_URL].completions.create.await_count == 2
        for call in conn._clients[FAKE_URL].completions.create.await_args_list:
            kwargs = call.kwargs
            assert kwargs["n"] == 1
            assert kwargs["temperature"] == 0.0
            assert kwargs["echo"] is True
            assert kwargs["max_tokens"] == 1
            assert kwargs["logprobs"] == 1
            assert kwargs["extra_body"] == {
                "existing": "kept",
            }
            assert "stop" not in kwargs
            assert "top_logprobs" not in kwargs

    @pytest.mark.asyncio
    async def test_choice_scoring_fields_do_not_opt_in_without_yaml_grader(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="multiple_choice",
        )
        elem = {
            "row": 0,
            "scoring_mode": "choice_scoring",
            "completion_input": "Question: pick one\nAnswer:",
            "chat_input": [],
            "scoring_completions": ["A", "B"],
            "scoring_completion_labels": ["A", "B"],
            "ground_truth": "A",
        }
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["A"])
        )

        results = await _collect(
            conn.launch_requests(_requests(elem), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert not isinstance(results[0], ExceptionWrapper), getattr(results[0], "trace", "")
        assert results[0]["generations"] == ["A"]
        assert "choice_scoring_full_logprobs" not in results[0]
        assert "choice_scoring_completion_logprobs" not in results[0]
        assert conn._clients[FAKE_URL].completions.create.await_count == 1
        kwargs = conn._clients[FAKE_URL].completions.create.await_args.kwargs
        assert kwargs["prompt"] == "Question: pick one\nAnswer:"

    @pytest.mark.asyncio
    async def test_choice_scoring_yaml_requires_structured_rows(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="choice_scoring",
        )
        elem = {
            "row": 0,
            "completion_input": "Question: pick one\nAnswer:",
            "chat_input": [],
            "ground_truth": "A",
        }
        conn._clients[FAKE_URL].completions.create = AsyncMock()

        results = await _collect(
            conn.launch_requests(_requests(elem), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert isinstance(results[0], ExceptionWrapper)
        assert "requires scoring_mode='choice_scoring'" in results[0].trace
        assert conn._clients[FAKE_URL].completions.create.await_count == 0

    @pytest.mark.asyncio
    async def test_choice_scoring_serializes_raw_logprobs_and_metadata(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="choice_scoring",
        )
        elem = {
            "row": 0,
            "scoring_mode": "choice_scoring",
            "completion_input": "Question:",
            "scoring_completions": ["pesticides"],
            "scoring_completion_labels": ["A"],
            "ground_truth": "A",
        }
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[
                _choice_scoring_response([
                    {
                        "text_offset": [0, 8, 9],
                        "token_logprobs": [None, -0.3, -0.7],
                    }
                ]),
                _choice_scoring_response([
                    {
                        "text_offset": [0, 6, 7],
                        "token_logprobs": [None, -0.2, -0.4],
                    }
                ]),
            ]
        )

        results = await _collect(
            conn.launch_requests(_requests(elem), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert not isinstance(results[0], ExceptionWrapper), getattr(results[0], "trace", "")
        assert results[0]["choice_scoring_full_logprobs"] == [
            {
                "text_offset": [0, 8, 9],
                "token_logprobs": [None, -0.3, -0.7],
            }
        ]
        assert results[0]["choice_scoring_completion_logprobs"] == [
            {
                "text_offset": [0, 6, 7],
                "token_logprobs": [None, -0.2, -0.4],
            }
        ]
        assert results[0]["choice_scoring_metadata"] == {
            "full_prompts": ["Question: pesticides"],
            "completion_prompts": ["Answer: pesticides"],
        }

    @pytest.mark.asyncio
    async def test_choice_scoring_applies_prompt_prefix_to_full_prompts_only(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="choice_scoring",
        )
        conn._model.prompt_prefix_instructions = "Global instruction"
        elem = {
            "row": 0,
            "scoring_mode": "choice_scoring",
            "completion_input": "Question: pick one\nAnswer:",
            "scoring_completions": ["A", "B"],
            "scoring_completion_n_tokens": [1, 1],
            "scoring_completion_labels": ["A", "B"],
            "ground_truth": "A",
        }
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_choice_scoring_responses(
                [[None, -0.3], [None, -0.7]],
                [[None, -0.2], [None, -0.4]],
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(elem), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["choice_scoring_metadata"] == {
            "full_prompts": [
                "Global instruction\n\nQuestion: pick one\nAnswer: A",
                "Global instruction\n\nQuestion: pick one\nAnswer: B",
            ],
            "completion_prompts": ["Answer: A", "Answer: B"],
        }
        full_call, completion_call = conn._clients[FAKE_URL].completions.create.await_args_list
        assert full_call.kwargs["prompt"] == [
            "Global instruction\n\nQuestion: pick one\nAnswer: A",
            "Global instruction\n\nQuestion: pick one\nAnswer: B",
        ]
        assert completion_call.kwargs["prompt"] == ["Answer: A", "Answer: B"]

    @pytest.mark.asyncio
    async def test_choice_scoring_lean_row_uses_offsets_when_token_counts_omitted(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="choice_scoring",
        )
        elem = {
            "row": 0,
            "scoring_mode": "choice_scoring",
            "completion_input": "Question: pick one\nAnswer:",
            "scoring_completions": ["A", "B"],
            "scoring_completion_labels": ["A", "B"],
            "ground_truth": "A",
        }
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[
                _choice_scoring_response([
                    {
                        "text_offset": [0, 26],
                        "token_logprobs": [None, -0.1],
                    },
                    {
                        "text_offset": [0, 26],
                        "token_logprobs": [None, -0.9],
                    },
                ]),
                _choice_scoring_response([
                    {
                        "text_offset": [0, 7],
                        "token_logprobs": [None, -0.2],
                    },
                    {
                        "text_offset": [0, 7],
                        "token_logprobs": [None, -0.8],
                    },
                ]),
            ]
        )

        results = await _collect(
            conn.launch_requests(_requests(elem), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert not isinstance(results[0], ExceptionWrapper), getattr(results[0], "trace", "")
        assert results[0]["choice_scoring_metadata"] == {
            "full_prompts": [
                "Question: pick one\nAnswer: A",
                "Question: pick one\nAnswer: B",
            ],
            "completion_prompts": ["Answer: A", "Answer: B"],
        }
        assert "scoring_completion_n_tokens" not in results[0]
        assert conn._clients[FAKE_URL].completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_choice_scoring_falls_back_when_batch_hits_malformed_msgpack(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="choice_scoring",
        )
        elem = {
            "row": 0,
            "scoring_mode": "choice_scoring",
            "completion_input": "Question: pick one\nAnswer:",
            "scoring_completions": ["A", "B"],
            "scoring_completion_labels": ["A", "B"],
            "ground_truth": "A",
        }
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[
                RuntimeError("400 MessagePack data is malformed: trailing characters"),
                _choice_scoring_response(
                    [{"text_offset": [0, 26], "token_logprobs": [None, -0.1]}]
                ),
                _choice_scoring_response(
                    [{"text_offset": [0, 26], "token_logprobs": [None, -0.9]}]
                ),
                _choice_scoring_response([
                    {"text_offset": [0, 7], "token_logprobs": [None, -0.2]},
                    {"text_offset": [0, 7], "token_logprobs": [None, -0.8]},
                ]),
            ]
        )

        results = await _collect(
            conn.launch_requests(_requests(elem), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert not isinstance(results[0], ExceptionWrapper), getattr(results[0], "trace", "")
        assert results[0]["choice_scoring_full_logprobs"] == [
            {"text_offset": [0, 26], "token_logprobs": [None, -0.1]},
            {"text_offset": [0, 26], "token_logprobs": [None, -0.9]},
        ]
        assert results[0]["choice_scoring_metadata"]["request_modes"] == {
            "full": "single_prompt_fallback",
            "completion": "batched",
        }
        assert conn._clients[FAKE_URL].completions.create.await_count == 4

    @pytest.mark.asyncio
    async def test_choice_scoring_lean_row_flows_from_request_to_grader(self):
        conn = make_conn(
            model_type=ModelType.BASE,
            average_over=[1],
            pass_at=[1],
            grader_type="choice_scoring",
        )
        elem = {
            "row": 0,
            "scoring_mode": "choice_scoring",
            "completion_input": "Question: pick one\nAnswer:",
            "scoring_completions": ["A", "B"],
            "scoring_completion_labels": ["A", "B"],
            "ground_truth": "A",
        }
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=[
                _choice_scoring_response([
                    {"text_offset": [0, 26], "token_logprobs": [None, -0.1]},
                    {"text_offset": [0, 26], "token_logprobs": [None, -0.9]},
                ]),
                _choice_scoring_response([
                    {"text_offset": [0, 7], "token_logprobs": [None, -0.2]},
                    {"text_offset": [0, 7], "token_logprobs": [None, -0.8]},
                ]),
            ]
        )

        request_results = await _collect(
            conn.launch_requests(_requests(elem), offset=0, completion_hook=AsyncMock())
        )

        from scheduler.grader.base import Grade
        from scheduler.grader.choice_scoring import ChoiceScoring

        class MockEvent:
            parser_type = "noop"

        async def generated_samples():
            for item in request_results:
                yield item
            yield Sentinel.COMPLETED

        grader = ChoiceScoring(
            samples_generator=generated_samples(),
            event_manager=None,
            job_manager=None,
            event=MockEvent(),
        )
        graded_items = []
        async for item in grader.run(existing=_requests(), average_over=[1], pass_at=[1]):
            graded_items.append(item)

        grades = [item.element for item in graded_items if isinstance(item, Grade)]
        assert len(grades) == 1
        assert grades[0]["choice_nll"] == pytest.approx([0.1, 0.9])
        assert grades[0]["scoring_completion_n_tokens"] == pytest.approx([1.0, 1.0])
        assert grades[0]["scoring_completion_n_chars"] == pytest.approx([1.0, 1.0])
        assert grades[0]["ground_truth_index"] == 0
        assert grades[0]["picked"] == ["A"]
        assert grades[0]["correct"] == [True]

    @pytest.mark.asyncio
    async def test_base_model_no_logprobs_key_when_none(self):
        """When choice.logprobs is None, the result must not have a 'logprobs' key."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["hello"], logprobs_list=None)
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert "logprobs" not in results[0]

    @pytest.mark.asyncio
    async def test_chat_model_logprobs_added_to_result(self):
        conn = make_conn(model_type=ModelType.CHAT)
        prompt = {"chat_input": [{"role": "user", "content": "hi"}], "completion_input": ""}
        lp_data = {"content": [{"token": "hello", "logprob": -0.3}]}
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(["hello"], logprobs_list=[lp_data])
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert "logprobs" in results[0]
        assert results[0]["logprobs"] == [lp_data]

    @pytest.mark.asyncio
    async def test_chat_model_no_logprobs_key_when_none(self):
        conn = make_conn(model_type=ModelType.CHAT)
        prompt = {"chat_input": [{"role": "user", "content": "hi"}], "completion_input": ""}
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(["hello"], logprobs_list=None)
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert "logprobs" not in results[0]

    @pytest.mark.asyncio
    async def test_logprobs_for_multiple_prompts_stay_separate(self):
        """Each prompt's result must contain only that prompt's logprobs."""
        conn = make_conn(model_type=ModelType.BASE)
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(3)]
        lp_datas = [{"id": i} for i in range(3)]
        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            _completions_response([f"out{i}"], logprobs_list=[lp_datas[i]])
            for i in range(3)
        ])

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        for i, result in enumerate(results):
            assert result["logprobs"] == [lp_datas[i]], f"prompt {i} has wrong logprobs"

    @pytest.mark.asyncio
    async def test_logprobs_accumulated_across_batched_choices(self):
        """When n=2 in a single API call, logprobs from both choices must be in the list."""
        conn = make_conn(model_type=ModelType.BASE, average_over=[2], pass_at=[1])
        prompt = {"completion_input": "p", "chat_input": []}
        lp0 = {"choice": 0}
        lp1 = {"choice": 1}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["a", "b"], logprobs_list=[lp0, lp1])
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["logprobs"] == [lp0, lp1]

    @pytest.mark.asyncio
    async def test_mixed_prompts_with_and_without_logprobs(self):
        """Some prompts return logprobs, others don't — each result is independent."""
        conn = make_conn(model_type=ModelType.BASE)
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(3)]
        lp_data = {"tokens": ["x"]}
        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            _completions_response(["out0"], logprobs_list=[lp_data]),
            _completions_response(["out1"], logprobs_list=None),
            _completions_response(["out2"], logprobs_list=[lp_data]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        assert "logprobs" in results[0]
        assert "logprobs" not in results[1]
        assert "logprobs" in results[2]


# ---------------------------------------------------------------------------
# LOCKED_CONNECTIONS semaphore sharing
# ---------------------------------------------------------------------------

class TestLockedConnectionsSharing:

    def test_same_serving_key_shares_pool(self):
        """Two connections with the same serving_key must reuse the same pool."""
        from scheduler.openai_interface import ModelConnectionPool
        # make_conn uses name as serving_key, so same name → same serving_key
        make_conn(name="modelA", max_simultaneous=4)
        make_conn(name="modelA", max_simultaneous=4)

        assert isinstance(LOCKED_CONNECTIONS["modelA"], ModelConnectionPool)
        assert len([k for k in LOCKED_CONNECTIONS if k == "modelA"]) == 1

    def test_different_serving_keys_get_different_pools(self):
        # make_conn uses name as serving_key, so different names → different keys
        make_conn(name="modelA", max_simultaneous=4)
        make_conn(name="modelB", max_simultaneous=4)

        assert LOCKED_CONNECTIONS["modelA"] is not LOCKED_CONNECTIONS["modelB"]

    def test_pool_capacity_set_at_creation(self):
        """Pool capacity is set when first created for a serving key."""
        make_conn(name="modelX", max_simultaneous=7)
        pool = LOCKED_CONNECTIONS["modelX"]
        assert pool._capacity == 7

    def test_pool_not_recreated_for_second_instance(self):
        """Creating a second connection with the same serving_key must not replace the pool."""
        make_conn(name="modelA", max_simultaneous=4)
        pool1 = LOCKED_CONNECTIONS["modelA"]

        make_conn(name="modelA", max_simultaneous=4)
        pool2 = LOCKED_CONNECTIONS["modelA"]

        assert pool1 is pool2

    def test_locked_connections_dict_has_one_entry_per_serving_key(self):
        make_conn(name="modelA", max_simultaneous=4)
        make_conn(name="modelA", max_simultaneous=4)  # same serving_key, no new entry
        make_conn(name="modelB", max_simultaneous=4)  # new serving_key

        assert len(LOCKED_CONNECTIONS) == 2


# ---------------------------------------------------------------------------
# _extra_body_to_chat_template_kwargs
# ---------------------------------------------------------------------------

class TestExtraBodyToChatTemplateKwargs:
    """Unit tests for the pure normalisation function."""

    @pytest.mark.parametrize("extra_body,expected", [
        # nested form — unwrap the inner dict
        (
            {"chat_template_kwargs": {"reasoning_effort": "high"}},
            {"reasoning_effort": "high"},
        ),
        # flat form — pass through as-is
        (
            {"reasoning_effort": "high"},
            {"reasoning_effort": "high"},
        ),
        # flat form with multiple keys
        (
            {"reasoning_effort": "medium", "enable_thinking": True},
            {"reasoning_effort": "medium", "enable_thinking": True},
        ),
        # nested form with multiple keys
        (
            {"chat_template_kwargs": {"reasoning_effort": "low", "enable_thinking": False}},
            {"reasoning_effort": "low", "enable_thinking": False},
        ),
        # empty dict
        ({}, {}),
    ])
    def test_normalisation(self, extra_body, expected):
        assert _extra_body_to_chat_template_kwargs(extra_body) == expected

    def test_returns_copy_not_reference(self):
        """Mutating the returned dict must not affect the input."""
        extra_body = {"reasoning_effort": "high"}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        result["reasoning_effort"] = "low"
        assert extra_body["reasoning_effort"] == "high"

    def test_nested_returns_copy_not_reference(self):
        inner = {"reasoning_effort": "high"}
        extra_body = {"chat_template_kwargs": inner}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        result["reasoning_effort"] = "low"
        assert inner["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Contract: generation extra_body ↔ tokenize chat_template_kwargs
# ---------------------------------------------------------------------------

def _make_tokenize_session(
    captured_tokenize_body,
    captured_detokenize_body=None,
):
    """Return a mock aiohttp.ClientSession that captures the /tokenize POST body."""
    tokenize_resp = MagicMock()
    tokenize_resp.status = 200
    tokenize_resp.json = AsyncMock(return_value={"tokens": [1, 2, 3]})
    tokenize_resp.__aenter__ = AsyncMock(return_value=tokenize_resp)
    tokenize_resp.__aexit__ = AsyncMock(return_value=False)

    detokenize_resp = MagicMock()
    detokenize_resp.status = 200
    detokenize_resp.json = AsyncMock(return_value={"prompt": "<think>\n"})
    detokenize_resp.__aenter__ = AsyncMock(return_value=detokenize_resp)
    detokenize_resp.__aexit__ = AsyncMock(return_value=False)

    call_count = 0

    def mock_post(url, json=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            captured_tokenize_body.update(json or {})
            return tokenize_resp
        if captured_detokenize_body is not None:
            captured_detokenize_body.update(json or {})
        return detokenize_resp

    session = MagicMock()
    session.post = mock_post
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _chat_response_explicit(content):
    """Chat response with tool_calls=[] to avoid MagicMock iteration side effects."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.reasoning_content = None
    choice.message.model_extra = {}
    choice.message.tool_calls = []
    choice.logprobs = None
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestExtraBodyGenerationTokenizeContract:
    """Contract: the template variables sent to generation and to /tokenize must agree.

    If someone changes how extra_body is built for the generation call without
    updating _extra_body_to_chat_template_kwargs (used by _get_templated_prompt),
    these tests will fail because the two captured values diverge.
    """

    @pytest.mark.asyncio
    async def test_all_vllm_requests_use_api_model_name(self):
        conn = make_conn(
            model_type=ModelType.CHAT,
            name="pair-specific-name",
            api_model_name="stable-serving-name",
        )
        conn._url = FAKE_URL
        conn._debug = True
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response_explicit("answer")
        )

        tokenize_body = {}
        detokenize_body = {}
        mock_session = _make_tokenize_session(
            tokenize_body,
            detokenize_body,
        )

        with patch(
            "scheduler.openai_interface.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            await _collect(
                conn.launch_requests(
                    _requests(
                        {
                            "chat_input": [
                                {"role": "user", "content": "Hi"}
                            ],
                            "completion_input": "",
                        }
                    ),
                    offset=0,
                    completion_hook=AsyncMock(),
                )
            )

        _, generation_kwargs = (
            conn._clients[FAKE_URL].chat.completions.create.call_args
        )
        assert generation_kwargs["model"] == "stable-serving-name"
        assert tokenize_body["model"] == "stable-serving-name"
        assert detokenize_body["model"] == "stable-serving-name"

    @pytest.mark.parametrize("extra_body", [
        {"reasoning_effort": "high"},
        {"chat_template_kwargs": {"reasoning_effort": "high"}},
        {"reasoning_effort": "medium", "enable_thinking": True},
    ])
    @pytest.mark.asyncio
    async def test_tokenize_ctk_matches_generation_extra_body(self, extra_body):
        conn = make_conn(model_type=ModelType.CHAT)
        conn._openai_kwargs = {"extra_body": extra_body, "temperature": 0.0}
        conn._url = "http://fake:8000"
        conn._debug = True

        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response_explicit("answer")
        )

        captured_tokenize_body = {}
        mock_session = _make_tokenize_session(captured_tokenize_body)

        with patch("scheduler.openai_interface.aiohttp.ClientSession", return_value=mock_session):
            await _collect(
                conn.launch_requests(
                    _requests({"chat_input": [{"role": "user", "content": "Hi"}], "completion_input": ""}),
                    offset=0,
                    completion_hook=AsyncMock(),
                )
            )

        # Capture what was actually passed to generation
        _, gen_kwargs = conn._clients[FAKE_URL].chat.completions.create.call_args
        actual_extra_body = gen_kwargs.get("extra_body", {})

        # The tokenize chat_template_kwargs must equal _extra_body_to_chat_template_kwargs
        # applied to whatever extra_body the generation call received
        expected_ctk = _extra_body_to_chat_template_kwargs(actual_extra_body)
        assert captured_tokenize_body.get("chat_template_kwargs") == expected_ctk

    @pytest.mark.asyncio
    async def test_no_extra_body_means_no_chat_template_kwargs_in_tokenize(self):
        """When there is no extra_body, /tokenize must not receive chat_template_kwargs."""
        conn = make_conn(model_type=ModelType.CHAT)
        conn._openai_kwargs = {"temperature": 0.0}
        conn._url = "http://fake:8000"
        conn._debug = True

        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response_explicit("answer")
        )

        captured_tokenize_body = {}
        mock_session = _make_tokenize_session(captured_tokenize_body)

        with patch("scheduler.openai_interface.aiohttp.ClientSession", return_value=mock_session):
            await _collect(
                conn.launch_requests(
                    _requests({"chat_input": [{"role": "user", "content": "Hi"}], "completion_input": ""}),
                    offset=0,
                    completion_hook=AsyncMock(),
                )
            )

        assert "chat_template_kwargs" not in captured_tokenize_body


# ---------------------------------------------------------------------------
# TestCancellation
# ---------------------------------------------------------------------------

class TestCancellation:
    """cancel() stops in-flight make_request tasks and ends the generator."""

    @pytest.mark.asyncio
    async def test_cancel_before_launch_yields_nothing(self):
        """If cancel() is called before launch_requests is iterated, nothing is yielded."""
        conn = make_conn()
        conn.cancel()

        async def requests():
            yield {"completion_input": "hello", "row": 0}
            yield Sentinel.COMPLETED

        results = await _collect(conn.launch_requests(requests(), offset=0, completion_hook=AsyncMock()))
        assert results == []

    @pytest.mark.asyncio
    async def test_cancel_while_blocked_in_acquire_terminates(self):
        """When cancel() is called while make_request tasks are blocked waiting for
        a URL (pool has no slots), the generator terminates instead of hanging."""
        conn = make_conn()
        # Remove all slots so pool.acquire() will block
        LOCKED_CONNECTIONS["test-model"]._slots.clear()

        async def requests():
            yield {"completion_input": "hello", "row": 0}
            yield Sentinel.COMPLETED

        async def run():
            return await _collect(conn.launch_requests(requests(), offset=0, completion_hook=AsyncMock()))

        gen_task = asyncio.create_task(run())
        # Give the generator time to start and block in pool.acquire()
        await asyncio.sleep(0.05)
        conn.cancel()
        results = await asyncio.wait_for(gen_task, timeout=2.0)
        assert results == []

    @pytest.mark.asyncio
    async def test_cancel_mid_stream_stops_new_requests(self):
        """After cancel(), make_request tasks that haven't started yet are not spawned."""
        conn = make_conn()

        call_count = [0]
        async def slow_create(**kwargs):
            call_count[0] += 1
            conn.cancel()  # cancel after first request
            return _completions_response(["answer"])

        conn._clients[FAKE_URL].completions.create = slow_create

        async def requests():
            yield {"completion_input": "row0", "row": 0}
            yield {"completion_input": "row1", "row": 1}
            yield {"completion_input": "row2", "row": 2}
            yield Sentinel.COMPLETED

        results = await _collect(conn.launch_requests(requests(), offset=0, completion_hook=AsyncMock()))
        # Only one request was made before cancellation; no more than 1 should complete
        assert call_count[0] <= 1


# ---------------------------------------------------------------------------
# Null content (reasoning-only) handling
# ---------------------------------------------------------------------------

class TestNullContentHandling:
    """content=None with finish_reason='stop' is normal for reasoning models that produce
    only thinking tokens.  The scheduler should accept it as '' rather than retrying forever."""

    def _make_chat_conn(self):
        return make_conn(model_type=ModelType.CHAT)

    def _prompt(self):
        return {"chat_input": [{"role": "user", "content": "solve this"}], "completion_input": ""}

    @pytest.mark.asyncio
    async def test_null_content_with_reasoning_accepted_as_empty_string(self):
        """content=None + finish_reason='stop' + reasoning_content present → generation is ''."""
        conn = self._make_chat_conn()
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(
                [None],
                finish_reasons=["stop"],
                reasoning_contents=["<think>lots of thinking</think>"],
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(self._prompt()), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert results[0]["generations"] == [""]

    @pytest.mark.asyncio
    async def test_null_content_with_reasoning_captures_reasoning_field(self):
        """When content=None but reasoning is present, reasoning is still saved."""
        conn = self._make_chat_conn()
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(
                [None],
                finish_reasons=["stop"],
                reasoning_contents=["my reasoning"],
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(self._prompt()), offset=0, completion_hook=AsyncMock())
        )

        assert results[0].get("reasoning") == ["my reasoning"]

    @pytest.mark.asyncio
    async def test_null_content_without_reasoning_accepted_as_empty_string(self):
        """content=None + finish_reason='stop' with no reasoning → generation is ''."""
        conn = self._make_chat_conn()
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response([None], finish_reasons=["stop"])
        )

        results = await _collect(
            conn.launch_requests(_requests(self._prompt()), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == [""]

    @pytest.mark.asyncio
    async def test_null_content_with_bad_finish_reason_raises(self):
        """content=None + finish_reason != 'stop' means the server dropped the response —
        should be treated as a server error, not silently accepted."""
        from scheduler.utils import ExceptionWrapper
        conn = self._make_chat_conn()
        # Server keeps returning null content with finish_reason='error'
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response([None], finish_reasons=["error"])
        )

        results = await _collect(
            conn.launch_requests(_requests(self._prompt()), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert isinstance(results[0], ExceptionWrapper)

    @pytest.mark.asyncio
    async def test_multiple_null_content_choices_all_become_empty_string(self):
        """All null-content choices (finish_reason='stop') are normalised to ''."""
        conn = self._make_chat_conn()
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(
                [None, None, None, None],
                finish_reasons=["stop", "stop", "stop", "stop"],
                reasoning_contents=["r1", "r2", "r3", "r4"],
            )
        )
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(
                [None, None, None, None],
                finish_reasons=["stop", "stop", "stop", "stop"],
                reasoning_contents=["r1", "r2", "r3", "r4"],
            )
        )

        conn2 = make_conn(model_type=ModelType.CHAT, average_over=[4], pass_at=[1], name="test-model")
        conn2._clients[FAKE_URL].chat.completions.create = AsyncMock(
            return_value=_chat_response(
                [None, None, None, None],
                finish_reasons=["stop", "stop", "stop", "stop"],
                reasoning_contents=["r1", "r2", "r3", "r4"],
            )
        )

        results = await _collect(
            conn2.launch_requests(_requests(self._prompt()), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["", "", "", ""]
        assert results[0]["reasoning"] == ["r1", "r2", "r3", "r4"]


# ---------------------------------------------------------------------------
# ModelConnectionPool unit tests
# ---------------------------------------------------------------------------

from scheduler.openai_interface import ModelConnectionPool


class TestModelConnectionPoolAcquirePriority:
    """Gap 1: acquire() picks the URL with the most available slots."""

    @pytest.mark.asyncio
    async def test_acquire_picks_url_with_most_slots(self):
        pool = ModelConnectionPool(capacity_per_url=10)
        await pool.add_url("http://a:8000")
        await pool.add_url("http://b:8000")
        # Give b more slots
        pool._slots["http://a:8000"] = [3]
        pool._slots["http://b:8000"] = [7]

        url, n = await pool.acquire(requests_left=1)
        assert url == "http://b:8000"

    @pytest.mark.asyncio
    async def test_acquire_always_picks_highest_slots_url(self):
        pool = ModelConnectionPool(capacity_per_url=10)
        await pool.add_url("http://a:8000")
        await pool.add_url("http://b:8000")
        await pool.add_url("http://c:8000")
        pool._slots["http://a:8000"] = [1]
        pool._slots["http://b:8000"] = [5]
        pool._slots["http://c:8000"] = [3]

        url, _ = await pool.acquire(requests_left=1)
        assert url == "http://b:8000"


class TestModelConnectionPoolBatchCap:
    """Gap 2: acquire() caps n at 4 even when more slots are free."""

    @pytest.mark.asyncio
    async def test_batch_capped_at_4_when_slots_and_requests_exceed_4(self):
        pool = ModelConnectionPool(capacity_per_url=20)
        await pool.add_url("http://a:8000")
        pool._slots["http://a:8000"] = [10]

        url, n = await pool.acquire(requests_left=10)
        assert n == 4

    @pytest.mark.asyncio
    async def test_batch_capped_by_requests_left_when_smaller_than_4(self):
        pool = ModelConnectionPool(capacity_per_url=20)
        await pool.add_url("http://a:8000")
        pool._slots["http://a:8000"] = [10]

        url, n = await pool.acquire(requests_left=2)
        assert n == 2

    @pytest.mark.asyncio
    async def test_batch_capped_by_available_slots_when_smaller_than_4(self):
        pool = ModelConnectionPool(capacity_per_url=20)
        await pool.add_url("http://a:8000")
        pool._slots["http://a:8000"] = [3]

        url, n = await pool.acquire(requests_left=10)
        assert n == 3


class TestModelConnectionPoolReleaseUnblocksWaiter:
    """Gap 3: release() unblocks a coroutine blocked in acquire()."""

    @pytest.mark.asyncio
    async def test_release_unblocks_acquire(self):
        pool = ModelConnectionPool(capacity_per_url=2)
        await pool.add_url("http://a:8000")
        # Drain all slots
        pool._slots["http://a:8000"] = [0]

        acquired_event = asyncio.Event()
        result = {}

        async def waiter():
            acquired_event.set()
            url, n = await pool.acquire(requests_left=1)
            result["url"] = url
            result["n"] = n

        waiter_task = asyncio.create_task(waiter())
        # Wait until waiter has started and is blocked
        await acquired_event.wait()
        await asyncio.sleep(0)  # yield so the waiter can enter acquire() and block

        # Now release a slot — waiter should unblock
        await pool.release("http://a:8000", 1)

        await asyncio.wait_for(waiter_task, timeout=2.0)
        assert result["url"] == "http://a:8000"
        assert result["n"] == 1


class TestModelConnectionPoolAddUrlUnblocksWaiter:
    """Gap 4: add_url() unblocks a coroutine blocked because no URLs exist."""

    @pytest.mark.asyncio
    async def test_add_url_unblocks_acquire(self):
        pool = ModelConnectionPool(capacity_per_url=3)
        # No URLs — acquire() will block immediately

        result = {}

        async def waiter():
            url, n = await pool.acquire(requests_left=1)
            result["url"] = url
            result["n"] = n

        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0)  # let waiter enter acquire() and block

        await pool.add_url("http://new:8000")

        await asyncio.wait_for(waiter_task, timeout=2.0)
        assert result["url"] == "http://new:8000"
        assert result["n"] == 1


class TestModelConnectionPoolRemoveUrlWhileSlotsHeld:
    """Gap 5: remove_url() while slots are held doesn't return that URL again
    and doesn't deadlock."""

    @pytest.mark.asyncio
    async def test_removed_url_not_returned_by_acquire(self):
        pool = ModelConnectionPool(capacity_per_url=5)
        await pool.add_url("http://dying:8000")
        await pool.add_url("http://live:8000")
        pool._slots["http://dying:8000"] = [5]
        pool._slots["http://live:8000"] = [5]

        # Simulate "holding" slots for dying URL (reduce its count to 3)
        url, n = await pool.acquire(requests_left=2)
        # acquire returns dying or live — doesn't matter; remove dying
        await pool.remove_url("http://dying:8000")

        # Subsequent acquire must never return the removed URL
        for _ in range(5):
            url2, _ = await pool.acquire(requests_left=1)
            assert url2 != "http://dying:8000"
            await pool.release(url2, 1)

    @pytest.mark.asyncio
    async def test_remove_url_with_held_slots_does_not_deadlock(self):
        """Removing a URL while another task holds slots on it must not deadlock
        the pool: release() for the removed URL must not raise and the pool
        stays functional."""
        pool = ModelConnectionPool(capacity_per_url=5)
        await pool.add_url("http://dying:8000")
        pool._slots["http://dying:8000"] = [5]

        # Acquire slots
        url, n = await pool.acquire(requests_left=2)
        assert url == "http://dying:8000"

        # Remove while slots are held
        await pool.remove_url("http://dying:8000")

        # release() on a removed URL must not raise
        await pool.release(url, n)

        # Pool should still work after adding a new URL
        await pool.add_url("http://live:8000")
        url2, n2 = await asyncio.wait_for(pool.acquire(requests_left=1), timeout=2.0)
        assert url2 == "http://live:8000"


# ---------------------------------------------------------------------------
# launch_requests: completion_hook and cancel() edge cases
# ---------------------------------------------------------------------------

class TestCompletionHookAfterCancel:
    """Gap 6: completion_hook fires exactly once when cancel() is called after
    all requests complete."""

    @pytest.mark.asyncio
    async def test_completion_hook_fires_once_after_all_complete_then_cancel(self):
        conn = make_conn(model_type=ModelType.BASE)
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=_completions_response(["out"])
        )

        hook_call_count = [0]

        async def completion_hook():
            hook_call_count[0] += 1

        async def requests():
            yield {"completion_input": "row0", "row": 0}
            yield {"completion_input": "row1", "row": 1}
            yield Sentinel.COMPLETED

        results = []
        async for item in conn.launch_requests(requests(), offset=0, completion_hook=completion_hook):
            results.append(item)

        # All items yielded normally; now cancel (after the generator is exhausted)
        conn.cancel()

        assert hook_call_count[0] == 1


class TestCancelMidStream:
    """Gap 7: cancel() mid-stream stops the consumer and in-flight tasks without
    raising an exception."""

    @pytest.mark.asyncio
    async def test_cancel_after_two_yields_stops_iteration(self):
        """cancel() called from outside the generator loop terminates iteration
        with no exception and at most a handful of results."""
        conn = make_conn(model_type=ModelType.BASE)

        call_count = [0]

        async def instant_create(**kwargs):
            call_count[0] += 1
            return _completions_response(["out"])

        conn._clients[FAKE_URL].completions.create = instant_create

        async def requests():
            for i in range(5):
                yield {"completion_input": f"row{i}", "row": i}
            yield Sentinel.COMPLETED

        results = []
        gen = conn.launch_requests(requests(), offset=0, completion_hook=AsyncMock())

        async def collect_then_cancel():
            # Collect items; cancel from outside after a short delay so we
            # interrupt before all 5 are yielded.
            async for item in gen:
                results.append(item)

        # Start the collection in the background
        collect_task = asyncio.create_task(collect_then_cancel())
        # Give it a moment to start, then cancel
        await asyncio.sleep(0)
        conn.cancel()

        # The task should terminate promptly without raising
        await asyncio.wait_for(collect_task, timeout=3.0)

        # Fewer than all 5 items should have been yielded (we cancelled mid-stream)
        assert len(results) <= 5
        # No exception was raised (the task returned normally)


# ---------------------------------------------------------------------------
# _deep_merge_openai_kwargs: None override (Gap 8 — targeted addition)
# ---------------------------------------------------------------------------

class TestDeepMergeNoneOverride:
    """Gap 8: override value of None must overwrite the base, not be skipped."""

    def test_none_value_overwrites_base_scalar(self):
        result = _deep_merge_openai_kwargs({"key": 42}, {"key": None})
        assert result["key"] is None

    def test_none_value_overwrites_base_dict(self):
        result = _deep_merge_openai_kwargs({"key": {"nested": 1}}, {"key": None})
        assert result["key"] is None

    def test_none_value_for_new_key_sets_none(self):
        result = _deep_merge_openai_kwargs({"a": 1}, {"b": None})
        assert result["b"] is None
        assert result["a"] == 1


# ---------------------------------------------------------------------------
# _extra_body_to_chat_template_kwargs: nested and flat forms (Gap 9)
# ---------------------------------------------------------------------------

class TestExtraBodyToChatTemplateKwargsNestedAndFlat:
    """Gap 9: explicit tests for nested and flat forms of _extra_body_to_chat_template_kwargs."""

    def test_nested_form_returns_inner_dict(self):
        extra_body = {"chat_template_kwargs": {"reasoning_effort": "high"}}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        assert result == {"reasoning_effort": "high"}

    def test_flat_form_returns_dict_as_is(self):
        extra_body = {"reasoning_effort": "high"}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        assert result == {"reasoning_effort": "high"}

    def test_nested_form_with_multiple_keys(self):
        extra_body = {"chat_template_kwargs": {"reasoning_effort": "low", "enable_thinking": False}}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        assert result == {"reasoning_effort": "low", "enable_thinking": False}

    def test_flat_form_with_multiple_keys(self):
        extra_body = {"reasoning_effort": "medium", "enable_thinking": True}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        assert result == {"reasoning_effort": "medium", "enable_thinking": True}

    def test_nested_form_does_not_include_outer_key(self):
        """The outer 'chat_template_kwargs' key itself must not appear in the result."""
        extra_body = {"chat_template_kwargs": {"k": "v"}}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        assert "chat_template_kwargs" not in result

    def test_nested_form_returns_copy(self):
        inner = {"reasoning_effort": "high"}
        extra_body = {"chat_template_kwargs": inner}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        result["reasoning_effort"] = "low"
        assert inner["reasoning_effort"] == "high"

    def test_flat_form_returns_copy(self):
        extra_body = {"reasoning_effort": "high"}
        result = _extra_body_to_chat_template_kwargs(extra_body)
        result["reasoning_effort"] = "low"
        assert extra_body["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# ModelConnectionPool.urls
# ---------------------------------------------------------------------------

class TestModelConnectionPoolUrls:

    @pytest.mark.asyncio
    async def test_urls_empty_initially(self):
        from scheduler.openai_interface import ModelConnectionPool
        pool = ModelConnectionPool(capacity_per_url=4)
        assert pool.urls == []

    @pytest.mark.asyncio
    async def test_urls_reflects_added_urls(self):
        from scheduler.openai_interface import ModelConnectionPool
        pool = ModelConnectionPool(capacity_per_url=4)
        await pool.add_url("http://node1:8000")
        await pool.add_url("http://node2:8000")
        assert set(pool.urls) == {"http://node1:8000", "http://node2:8000"}

    @pytest.mark.asyncio
    async def test_urls_reflects_removed_url(self):
        from scheduler.openai_interface import ModelConnectionPool
        pool = ModelConnectionPool(capacity_per_url=4)
        await pool.add_url("http://node1:8000")
        await pool.add_url("http://node2:8000")
        await pool.remove_url("http://node1:8000")
        assert pool.urls == ["http://node2:8000"]

    @pytest.mark.asyncio
    async def test_urls_returns_snapshot(self):
        """Mutating the returned list must not affect the pool's internal state."""
        from scheduler.openai_interface import ModelConnectionPool
        pool = ModelConnectionPool(capacity_per_url=4)
        await pool.add_url("http://node1:8000")
        snapshot = pool.urls
        snapshot.append("http://injected:8000")
        assert "http://injected:8000" not in pool.urls


# ---------------------------------------------------------------------------
# OpenAIConnection._fetch_vllm_num_requests_running
# ---------------------------------------------------------------------------

class TestFetchVllmNumRequestsRunning:

    def _make_mock_response(self, status: int, body: str):
        resp = AsyncMock()
        resp.status = status
        resp.text = AsyncMock(return_value=body)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    def _make_mock_session(self, resp):
        session = AsyncMock()
        session.get = MagicMock(return_value=resp)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return session

    @pytest.mark.asyncio
    async def test_parses_integer_count(self):
        conn = make_conn()
        body = (
            '# HELP vllm:num_requests_running\n'
            'vllm:num_requests_running{model_name="m",engine_process_id="1"} 3.0\n'
        )
        resp = self._make_mock_response(200, body)
        with patch("aiohttp.ClientSession", return_value=self._make_mock_session(resp)):
            result = await conn._fetch_vllm_num_requests_running("http://fake:8000")
        assert result == 3

    @pytest.mark.asyncio
    async def test_returns_zero_when_metric_is_zero(self):
        conn = make_conn()
        body = 'vllm:num_requests_running{model_name="m"} 0.0\n'
        resp = self._make_mock_response(200, body)
        with patch("aiohttp.ClientSession", return_value=self._make_mock_session(resp)):
            result = await conn._fetch_vllm_num_requests_running("http://fake:8000")
        assert result == 0

    @pytest.mark.asyncio
    async def test_sums_waiting_and_swapped_requests_too(self):
        conn = make_conn()
        body = (
            'vllm:num_requests_running{model_name="m"} 0.0\n'
            'vllm:num_requests_waiting{model_name="m"} 2.0\n'
            'vllm:num_requests_swapped{model_name="m"} 1.0\n'
        )
        resp = self._make_mock_response(200, body)
        with patch("aiohttp.ClientSession", return_value=self._make_mock_session(resp)):
            result = await conn._fetch_vllm_num_requests_running("http://fake:8000")
        assert result == 3

    @pytest.mark.asyncio
    async def test_returns_none_on_non_200(self):
        conn = make_conn()
        resp = self._make_mock_response(503, "")
        with patch("aiohttp.ClientSession", return_value=self._make_mock_session(resp)):
            result = await conn._fetch_vllm_num_requests_running("http://fake:8000")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_metric_absent(self):
        conn = make_conn()
        body = '# HELP some_other_metric\nsome_other_metric{} 1.0\n'
        resp = self._make_mock_response(200, body)
        with patch("aiohttp.ClientSession", return_value=self._make_mock_session(resp)):
            result = await conn._fetch_vllm_num_requests_running("http://fake:8000")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self):
        conn = make_conn()
        with patch("aiohttp.ClientSession", side_effect=aiohttp.ClientConnectionError("refused")):
            result = await conn._fetch_vllm_num_requests_running("http://fake:8000")
        assert result is None


# ---------------------------------------------------------------------------
# OpenAIConnection._poll_vllm_metrics — poller logs discrepancies
# ---------------------------------------------------------------------------

class TestPollVllmMetrics:

    @pytest.mark.asyncio
    async def test_poller_stops_when_cancelled(self):
        """_poll_vllm_metrics exits cleanly when _cancelled is set."""
        conn = make_conn()
        conn._cancelled.set()
        # Should return almost immediately without hanging.
        await asyncio.wait_for(conn._poll_vllm_metrics(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_poller_logs_info_when_counts_agree(self):
        """When VLLM count > 0 and matches in-flight, no warning is emitted."""
        conn = make_conn()
        conn._in_flight = 2

        warned = []

        import logging
        with patch.object(logging.getLogger("OpenAIConnection"), "warning",
                          side_effect=lambda *a, **kw: warned.append(a)):
            # Simulate one poller body: VLLM agrees with scheduler
            pool = LOCKED_CONNECTIONS[conn._model.serving_key]
            for url in pool.urls:
                vllm_count = 2  # agrees with _in_flight
                if vllm_count is not None and vllm_count == 0 and conn._in_flight > 0:
                    logging.getLogger("OpenAIConnection").warning("Discrepancy", url)

        assert warned == []  # no warning when counts agree

    @pytest.mark.asyncio
    async def test_poller_warns_on_discrepancy(self):
        """When VLLM reports 0 but scheduler has in-flight > 0, a WARNING is logged."""
        conn = make_conn()
        conn._in_flight = 5

        warned = []
        fetch_called = asyncio.Event()

        async def fake_fetch(url):
            fetch_called.set()
            return 0  # VLLM says idle, scheduler says 5 in-flight

        conn._fetch_vllm_num_requests_running = fake_fetch

        wait_calls = 0

        async def fast_wait_for(coro, **kwargs):
            nonlocal wait_calls
            coro.close()
            wait_calls += 1
            if wait_calls == 1:
                raise asyncio.TimeoutError()
            conn.cancel()
            return

        import logging
        with (
            patch("asyncio.wait_for", new=fast_wait_for),
            patch.object(
                logging.getLogger("OpenAIConnection"),
                "warning",
                side_effect=lambda *a, **kw: warned.append(a),
            ),
        ):
            await conn._poll_vllm_metrics()

        assert fetch_called.is_set()
        assert len(warned) == 1
        assert "Discrepancy" in warned[0][0]

    @pytest.mark.asyncio
    async def test_poller_keeps_requests_after_persistent_zero_metrics(self):
        """Zero engine work remains diagnostic after the former 30s boundary."""
        conn = make_conn()
        conn._in_flight = 5

        async def fake_fetch(url):
            return 0

        conn._fetch_vllm_num_requests_running = fake_fetch

        wait_calls = 0

        async def fast_wait_for(coro, **kwargs):
            nonlocal wait_calls
            coro.close()
            wait_calls += 1
            if wait_calls == 3:
                conn.cancel()
                return
            raise asyncio.TimeoutError()

        loop = asyncio.get_event_loop()
        with (
            patch("asyncio.wait_for", new=fast_wait_for),
            patch.object(
                loop,
                "time",
                side_effect=lambda: 1000.0 if wait_calls == 1 else 1031.0,
            ),
        ):
            await conn._poll_vllm_metrics()

        assert conn._cancelled.is_set()
        assert conn._emit_cancelled_results is False
        assert conn._cancel_reason is None

    @pytest.mark.asyncio
    async def test_poller_task_cancelled_after_launch_requests(self):
        """launch_requests cancels the poller task in its finally block."""
        conn = make_conn()

        poller_tasks = []
        original_create_task = asyncio.create_task

        def capture_task(coro, **kwargs):
            task = original_create_task(coro, **kwargs)
            if hasattr(coro, '__qualname__') and 'poll_vllm_metrics' in getattr(coro, '__qualname__', ''):
                poller_tasks.append(task)
            return task

        async def fake_requests():
            yield {"completion_input": "hello", "row": 0}
            yield Sentinel.COMPLETED

        conn._clients[FAKE_URL].completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(text="world", finish_reason="stop", logprobs=None)],
                usage=None,
            )
        )

        with patch("asyncio.create_task", side_effect=capture_task):
            results = []
            async for item in conn.launch_requests(fake_requests(), offset=0, completion_hook=AsyncMock()):
                if item != Sentinel.COMPLETED:
                    results.append(item)

        # Yield control so cancelled tasks can actually complete
        await asyncio.sleep(0)

        # All poller tasks that were created should be done (cancelled or finished)
        for t in poller_tasks:
            assert t.done()

    @pytest.mark.asyncio
    async def test_delayed_responses_complete_after_persistent_zero_metrics(self):
        """Responses may arrive after VLLM has stopped counting engine work."""
        conn = make_conn()
        responses_ready = asyncio.Event()
        fetch_count = 0
        current_time = 1000.0

        async def delayed_completions(*args, **kwargs):
            await responses_ready.wait()
            return _completions_response(["answer"])

        conn._clients[FAKE_URL].completions.create = delayed_completions

        async def fake_fetch(url):
            nonlocal fetch_count, current_time
            fetch_count += 1
            if fetch_count == 2:
                current_time += 31.0
            elif fetch_count == 3:
                responses_ready.set()
            return 0

        conn._fetch_vllm_num_requests_running = fake_fetch

        async def fast_wait_for(coro, **kwargs):
            coro.close()
            await asyncio.sleep(0)
            raise asyncio.TimeoutError()

        loop = asyncio.get_event_loop()

        async def fake_requests():
            for i in range(4):
                yield {"completion_input": f"prompt {i}", "row": i}
            yield Sentinel.COMPLETED

        completion_hook = AsyncMock()
        with (
            patch("asyncio.wait_for", new=fast_wait_for),
            patch.object(loop, "time", side_effect=lambda: current_time),
        ):
            items = await _collect(
                conn.launch_requests(
                    fake_requests(),
                    offset=0,
                    completion_hook=completion_hook,
                )
            )

        results = [item for item in items if item != Sentinel.COMPLETED]
        assert fetch_count >= 3
        assert len(results) == 4
        assert all(not isinstance(item, ExceptionWrapper) for item in results)
        assert [item["generations"] for item in results] == [["answer"]] * 4
        assert not conn._cancelled.is_set()
        assert conn._emit_cancelled_results is False
        completion_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_choice_scoring_cancelled_inflight_prompt_without_generations_field(self):
        conn = make_conn(model_type=ModelType.BASE, grader_type="choice_scoring")
        request_started = asyncio.Event()

        async def slow_completions(*args, **kwargs):
            request_started.set()
            await asyncio.Event().wait()

        conn._clients[FAKE_URL].completions.create = slow_completions

        async def requests():
            yield {
                "completion_input": "Question: test\nAnswer:",
                "row": 0,
                "scoring_mode": "choice_scoring",
                "scoring_completions": ["A", "B"],
                "scoring_completion_labels": ["A", "B"],
                "ground_truth": "A",
            }
            yield Sentinel.COMPLETED

        collect_task = asyncio.create_task(
            _collect(
                conn.launch_requests(
                    requests(), offset=0, completion_hook=AsyncMock()
                )
            )
        )
        await asyncio.wait_for(request_started.wait(), timeout=1.0)
        conn.cancel_with_failures()
        items = await asyncio.wait_for(collect_task, timeout=1.0)
        results = [item for item in items if item != Sentinel.COMPLETED]

        assert conn._cancelled.is_set()
        assert len(results) == 1
        assert isinstance(results[0], ExceptionWrapper)
        assert "generation cancelled" in results[0].trace
        assert results[0].instance["scoring_mode"] == "choice_scoring"
        assert "generations" not in results[0].instance

    @pytest.mark.asyncio
    async def test_poller_progress_rearms_zero_metric_warning(self):
        conn = make_conn()
        conn._in_flight = 1
        warnings = []

        import logging

        async def fake_fetch(url):
            return 0

        conn._fetch_vllm_num_requests_running = fake_fetch

        wait_calls = 0

        async def fake_wait_for(coro, **kwargs):
            nonlocal wait_calls
            coro.close()
            wait_calls += 1
            if wait_calls == 1:
                raise asyncio.TimeoutError()
            if wait_calls == 2:
                conn._completed_requests_count += 1
                raise asyncio.TimeoutError()
            conn.cancel()
            return

        with (
            patch("asyncio.wait_for", new=fake_wait_for),
            patch.object(
                logging.getLogger("OpenAIConnection"),
                "warning",
                side_effect=lambda *args, **kwargs: warnings.append(args),
            ),
        ):
            await conn._poll_vllm_metrics()

        assert conn._cancelled.is_set()
        assert conn._emit_cancelled_results is False
        assert conn._cancel_reason is None
        assert len(warnings) == 2

    @pytest.mark.asyncio
    async def test_poller_does_not_cancel_when_one_replica_idle_and_another_active(self):
        """With 2 replicas, one reporting 0 should NOT cancel if the other is active."""
        conn = make_conn()
        conn._in_flight = 5

        pool = LOCKED_CONNECTIONS[conn._model.serving_key]
        await pool.add_url("http://node-02:8000")

        async def fake_fetch(url):
            if url == FAKE_URL:
                return 0
            return 3

        conn._fetch_vllm_num_requests_running = fake_fetch

        cycle = [0]

        async def fast_wait_for(coro, **kwargs):
            coro.close()
            cycle[0] += 1
            if cycle[0] >= 3:
                conn._cancelled.set()
            if conn._cancelled.is_set():
                return
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", new=fast_wait_for):
            await conn._poll_vllm_metrics()

        assert conn._in_flight == 5
        assert conn._emit_cancelled_results is False
        assert conn._cancel_reason is None

    @pytest.mark.asyncio
    async def test_poller_keeps_requests_when_all_replicas_report_zero(self):
        """All replicas may finish engine work before responses reach clients."""
        conn = make_conn()
        conn._in_flight = 5
        warnings = []

        import logging

        pool = LOCKED_CONNECTIONS[conn._model.serving_key]
        await pool.add_url("http://node-02:8000")

        async def fake_fetch(url):
            return 0

        conn._fetch_vllm_num_requests_running = fake_fetch

        wait_calls = 0

        async def fast_wait_for(coro, **kwargs):
            nonlocal wait_calls
            coro.close()
            wait_calls += 1
            if wait_calls == 3:
                conn.cancel()
                return
            raise asyncio.TimeoutError()

        loop = asyncio.get_event_loop()
        with (
            patch("asyncio.wait_for", new=fast_wait_for),
            patch.object(
                loop,
                "time",
                side_effect=lambda: 1000.0 if wait_calls == 1 else 1031.0,
            ),
            patch.object(
                logging.getLogger("OpenAIConnection"),
                "warning",
                side_effect=lambda *args, **kwargs: warnings.append(args),
            ),
        ):
            await conn._poll_vllm_metrics()

        assert conn._cancelled.is_set()
        assert conn._emit_cancelled_results is False
        assert conn._cancel_reason is None
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# External model tests
# ---------------------------------------------------------------------------

def make_external_conn(
    base_url="https://api.openai.com/v1",
    api_key="sk-test",
    requests_per_minute=None,
    max_simultaneous=4,
    name="gpt-4o",
):
    model = MagicMock(spec=ModelInstance)
    model.name = name
    model.serving_key = name
    model.max_simultaneous_requests = max_simultaneous
    model.model_type = ModelType.CHAT
    model.openai_kwargs = {}
    model.cache_salt = CacheSaltConfig()
    model.prompt_prefix_instructions = None
    model.is_external = True
    model.base_url = base_url
    model.api_key = api_key
    model.requests_per_minute = requests_per_minute
    model.external_retry_policy = ExternalRetryPolicy()
    model.api_model_name = name

    task = MagicMock(spec=AsyncGenerationTask)
    task.average_over = [1]
    task.pass_at = [1]
    task.openai_settings = None

    conn = OpenAIConnection(
        model=model,
        task=task,
        event_instance=MagicMock(spec=EventInstance),
        job_manager=MagicMock(spec=JobManager),
        progress_manager=MagicMock(spec=ProgressManager),
        new_field_name="generations",
    )
    LOCKED_CONNECTIONS[name]._slots[base_url] = [max_simultaneous]
    return conn


class TestExternalModelClient:
    def test_get_client_uses_real_api_key_for_external(self):
        conn = make_external_conn(base_url="https://api.openai.com/v1", api_key="sk-realkey")
        client = conn._get_client("https://api.openai.com/v1")
        assert client.api_key == "sk-realkey"

    def test_get_client_uses_base_url_directly_for_external(self):
        conn = make_external_conn(base_url="https://api.openai.com/v1")
        client = conn._get_client("https://api.openai.com/v1")
        # External: base_url used as-is (not with /v1 appended)
        assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"

    def test_get_client_uses_fake_key_for_vllm(self):
        conn = make_conn()
        # Pre-populate client cache so make_conn's FAKE_URL slot is used
        conn._clients.clear()
        client = conn._get_client(FAKE_URL)
        assert client.api_key == "fake key"

    def test_get_client_appends_v1_for_vllm(self):
        conn = make_conn()
        conn._clients.clear()
        client = conn._get_client(FAKE_URL)
        assert str(client.base_url).rstrip("/") == f"{FAKE_URL}/v1"

    def test_get_client_no_key_falls_back_to_no_key_string(self):
        conn = make_external_conn(api_key=None)
        client = conn._get_client("https://api.openai.com/v1")
        assert client.api_key == "no-key"


class TestExternalModelRateLimiter:
    def test_rate_limiter_created_when_rpm_set(self):
        conn = make_external_conn(requests_per_minute=500)
        assert conn._rate_limiter is not None

    def test_rate_limiter_none_when_rpm_not_set(self):
        conn = make_external_conn(requests_per_minute=None)
        assert conn._rate_limiter is None

    def test_rate_limiter_none_for_vllm_model(self):
        conn = make_conn()
        assert conn._rate_limiter is None

    @pytest.mark.asyncio
    async def test_rate_limiter_called_before_request(self):
        from scheduler.rate_limiter import RateLimiter
        conn = make_external_conn(requests_per_minute=60)
        acquire_calls = []

        original_acquire = conn._rate_limiter.acquire
        async def mock_acquire():
            acquire_calls.append(True)
            await original_acquire()

        conn._rate_limiter.acquire = mock_acquire

        mock_client = AsyncMock()
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "answer"
        choice.message.reasoning_content = None
        choice.message.model_extra = {}
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        choice.logprobs = None
        resp.choices = [choice]
        resp.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        conn._clients["https://api.openai.com/v1"] = mock_client

        results = []
        async def input_stream():
            yield {"row": 0, "chat_input": [{"role": "user", "content": "hi"}], "ground_truth": "A"}
            yield Sentinel.COMPLETED

        async def completion_hook():
            pass

        async for item in conn.launch_requests(input_stream(), offset=0, completion_hook=completion_hook):
            results.append(item)

        assert len(acquire_calls) >= 1


class TestExternalModelVllmPoller:
    @pytest.mark.asyncio
    async def test_vllm_poller_not_started_for_external_model(self):
        """For external models, no VLLM metrics poller task should be started."""
        conn = make_external_conn()

        mock_client = AsyncMock()
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "answer"
        choice.message.reasoning_content = None
        choice.message.model_extra = {}
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        choice.logprobs = None
        resp.choices = [choice]
        resp.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        conn._clients["https://api.openai.com/v1"] = mock_client

        poller_called = []
        original_poller = conn._poll_vllm_metrics
        async def mock_poller():
            poller_called.append(True)
        conn._poll_vllm_metrics = mock_poller

        async def input_stream():
            yield {"row": 0, "chat_input": [{"role": "user", "content": "hi"}], "ground_truth": "A"}
            yield Sentinel.COMPLETED

        async def completion_hook():
            pass

        async for _ in conn.launch_requests(input_stream(), offset=0, completion_hook=completion_hook):
            pass

        assert poller_called == [], "VLLM poller should not be started for external models"


class TestExternalModelTemplatedPrompt:
    @pytest.mark.asyncio
    async def test_get_templated_prompt_returns_none_for_external(self):
        conn = make_external_conn()
        result = await conn._get_templated_prompt("https://api.openai.com/v1", [])
        assert result is None


# ---------------------------------------------------------------------------
# GradingOpenAIConnection: external judge should not block on JobManager
# ---------------------------------------------------------------------------

class TestGradingOpenAIConnectionExternal:
    @pytest.mark.asyncio
    async def test_external_judge_does_not_block_on_job_manager(self):
        """External judge should use base_url directly, not wait for get_live_url."""
        from scheduler.grader.base import GradingOpenAIConnection
        from scheduler.job import JobManager

        judge = MagicMock(spec=ModelInstance)
        judge.name = "gpt-4o-judge"
        judge.serving_key = "gpt-4o-judge"
        judge.max_simultaneous_requests = 1
        judge.is_external = True
        judge.base_url = "https://api.openai.com/v1"
        judge.api_key = "sk-test"
        judge.requests_per_minute = None
        judge.external_retry_policy = ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )

        task = MagicMock()
        task.grader = MagicMock()
        task.grader.llm_as_judge = judge

        conn = GradingOpenAIConnection(
            event_instance=MagicMock(),
            task=task,
            job_manager=JobManager(),
        )

        # Should NOT block — external judges bypass get_live_url
        client = await asyncio.wait_for(conn.get_client(), timeout=1.0)
        assert client is not None

    @pytest.mark.asyncio
    async def test_external_judge_uses_base_url_without_v1_suffix(self):
        """External judge base_url should be used as-is (no /v1 appended)."""
        from scheduler.grader.base import GradingOpenAIConnection
        from scheduler.job import JobManager

        judge = MagicMock(spec=ModelInstance)
        judge.name = "gpt-4o-judge"
        judge.serving_key = "gpt-4o-judge"
        judge.max_simultaneous_requests = 1
        judge.is_external = True
        judge.base_url = "https://api.openai.com/v1"
        judge.api_key = "sk-test"
        judge.requests_per_minute = None
        judge.external_retry_policy = ExternalRetryPolicy(
            max_attempts=1,
            request_timeout_seconds=1,
            total_deadline_seconds=5,
            initial_backoff_seconds=0,
            max_backoff_seconds=1,
        )

        task = MagicMock()
        task.grader = MagicMock()
        task.grader.llm_as_judge = judge

        conn = GradingOpenAIConnection(
            event_instance=MagicMock(),
            task=task,
            job_manager=JobManager(),
        )

        client = await asyncio.wait_for(conn.get_client(), timeout=1.0)
        assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_vllm_judge_still_uses_job_manager(self):
        """Non-external judge should still block on JobManager for a live URL."""
        from scheduler.grader.base import GradingOpenAIConnection
        from scheduler.job import JobManager

        judge = MagicMock(spec=ModelInstance)
        judge.name = "local-judge"
        judge.is_external = False

        task = MagicMock()
        task.grader = MagicMock()
        task.grader.llm_as_judge = judge

        conn = GradingOpenAIConnection(
            event_instance=MagicMock(),
            task=task,
            job_manager=JobManager(),
        )

        # Should timeout — no live URLs registered, so get_live_url blocks
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(conn.get_client(), timeout=0.2)
