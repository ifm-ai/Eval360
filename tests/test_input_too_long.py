"""
Tests for the "input too long" handling in the generation + grading pipeline.

When VLLM rejects a prompt due to context-length overflow it returns HTTP 400
(BadRequestError).  The pipeline must:
  1. Detect the error in make_request and produce a normal result dict with
     eval360_input_too_long=True and generations=["", ...] — NOT an ExceptionWrapper.
  2. Short-circuit grading in async_grade_all_samples so grade_fn is never
     called and the sample receives correct=[0, ...].
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from openai import BadRequestError

from scheduler.openai_interface import OpenAIConnection, LOCKED_CONNECTIONS, _is_input_too_long
from scheduler.grader.base import GraderBase
from scheduler.model import CacheSaltConfig, ModelType, ModelInstance
from scheduler.task import AsyncGenerationTask
from scheduler.job import JobManager
from scheduler.event import EventInstance
from scheduler.progress import ProgressManager
from scheduler.utils import Sentinel, ExceptionWrapper

FAKE_URL = "http://fake:8000"


# ---------------------------------------------------------------------------
# Helpers shared with testlaunch_requests
# ---------------------------------------------------------------------------

def _completions_response(texts):
    resp = MagicMock()
    resp.choices = [MagicMock(text=t, logprobs=None) for t in texts]
    return resp


@pytest.fixture(autouse=True)
def clear_locked_connections():
    LOCKED_CONNECTIONS.clear()
    yield
    LOCKED_CONNECTIONS.clear()


def make_conn(model_type=ModelType.BASE, average_over=None, pass_at=None, name="test-model"):
    model = MagicMock(spec=ModelInstance)
    model.name = name
    model.serving_key = name  # use name as serving key for test simplicity
    model.max_simultaneous_requests = 4
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
    LOCKED_CONNECTIONS[name]._slots[FAKE_URL] = [4]
    conn._clients[FAKE_URL] = MagicMock()
    conn._job_manager.is_url_live = MagicMock(return_value=True)
    return conn


def _make_bad_request_error(message: str, code: str | None = None) -> BadRequestError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    # The OpenAI SDK reads e.code as body.get("code"), so code must be top-level.
    body: dict = {"message": message, "type": "invalid_request_error"}
    if code is not None:
        body["code"] = code
    return BadRequestError(message=message, response=response, body=body)


async def _requests(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


# ---------------------------------------------------------------------------
# Tests for _is_input_too_long
# ---------------------------------------------------------------------------

class TestIsInputTooLong:
    def test_code_context_length_exceeded(self):
        e = _make_bad_request_error("too long", code="context_length_exceeded")
        assert _is_input_too_long(e) is True

    def test_message_context_length(self):
        e = _make_bad_request_error(
            "This model's maximum context length is 4096 tokens. However, your request has 5000 tokens."
        )
        assert _is_input_too_long(e) is True

    def test_message_maximum_context(self):
        e = _make_bad_request_error("Exceeds maximum context window of 8192 tokens.")
        assert _is_input_too_long(e) is True

    def test_unrelated_400_error(self):
        e = _make_bad_request_error("Invalid temperature value: 3.0")
        assert _is_input_too_long(e) is False

    def test_non_bad_request_error(self):
        assert _is_input_too_long(ValueError("context length exceeded")) is False
        assert _is_input_too_long(RuntimeError("maximum context")) is False


# ---------------------------------------------------------------------------
# Tests for make_request / launch_requests behaviour
# ---------------------------------------------------------------------------

class TestInputTooLongInMakeRequest:

    @pytest.mark.asyncio
    async def test_context_length_error_produces_normal_dict_not_exception_wrapper(self):
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_make_bad_request_error(
                "This model's maximum context length is 4096 tokens.",
                code="context_length_exceeded",
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert len(results) == 1
        assert not isinstance(results[0], ExceptionWrapper)
        assert results[0]["eval360_input_too_long"] is True

    @pytest.mark.asyncio
    async def test_generations_filled_with_empty_strings(self):
        conn = make_conn(model_type=ModelType.BASE, average_over=[1], pass_at=[1])
        prompt = {"completion_input": "p", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_make_bad_request_error(
                "This model's maximum context length is 4096 tokens.",
                code="context_length_exceeded",
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == [""]

    @pytest.mark.asyncio
    async def test_generations_length_matches_num_generations(self):
        """With average_over=[4], pass_at=[1] → num_generations=4; expect 4 empty strings."""
        conn = make_conn(model_type=ModelType.BASE, average_over=[4], pass_at=[1])
        prompt = {"completion_input": "p", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_make_bad_request_error(
                "This model's maximum context length is 4096 tokens.",
                code="context_length_exceeded",
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["generations"] == ["", "", "", ""]

    @pytest.mark.asyncio
    async def test_input_fields_preserved_on_result(self):
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "my-prompt", "chat_input": [], "row": 42}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_make_bad_request_error(
                "This model's maximum context length is 4096 tokens.",
                code="context_length_exceeded",
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["completion_input"] == "my-prompt"
        assert results[0]["row"] == 42

    @pytest.mark.asyncio
    async def test_no_retry_on_context_length_error(self):
        """Context-length errors are permanent; make_request must NOT retry."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_make_bad_request_error(
                "This model's maximum context length is 4096 tokens.",
                code="context_length_exceeded",
            )
        )

        await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert conn._clients[FAKE_URL].completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_other_prompts_unaffected(self):
        """input_too_long on one prompt must not block other prompts from completing."""
        conn = make_conn(model_type=ModelType.BASE)
        prompts = [{"completion_input": f"p{i}", "chat_input": []} for i in range(3)]
        conn._clients[FAKE_URL].completions.create = AsyncMock(side_effect=[
            _make_bad_request_error(
                "This model's maximum context length is 4096 tokens.",
                code="context_length_exceeded",
            ),
            _completions_response(["out1"]),
            _completions_response(["out2"]),
        ])

        results = await _collect(
            conn.launch_requests(_requests(*prompts), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["eval360_input_too_long"] is True
        assert results[1]["generations"] == ["out1"]
        assert results[2]["generations"] == ["out2"]

    @pytest.mark.asyncio
    async def test_unrelated_400_error_becomes_exception_wrapper(self):
        """A BadRequestError that is NOT context-length must still become ExceptionWrapper."""
        conn = make_conn(model_type=ModelType.BASE)
        prompt = {"completion_input": "p", "chat_input": []}
        conn._clients[FAKE_URL].completions.create = AsyncMock(
            side_effect=_make_bad_request_error("Invalid temperature: 99.0")
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert isinstance(results[0], ExceptionWrapper)

    @pytest.mark.asyncio
    async def test_chat_model_context_length_error(self):
        """Same behaviour for CHAT model type."""
        conn = make_conn(model_type=ModelType.CHAT)
        prompt = {"completion_input": "", "chat_input": [{"role": "user", "content": "q"}]}
        conn._clients[FAKE_URL].chat.completions.create = AsyncMock(
            side_effect=_make_bad_request_error(
                "This model's maximum context length is 4096 tokens.",
                code="context_length_exceeded",
            )
        )

        results = await _collect(
            conn.launch_requests(_requests(prompt), offset=0, completion_hook=AsyncMock())
        )

        assert results[0]["eval360_input_too_long"] is True
        assert results[0]["generations"] == [""]


# ---------------------------------------------------------------------------
# Tests for async_grade_all_samples short-circuit
# ---------------------------------------------------------------------------

class _ConcreteGrader(GraderBase):
    """Minimal concrete subclass for testing GraderBase.async_grade_all_samples."""

    def grade_sample(self, sample):
        raise AssertionError("grade_sample must not be called for input_too_long samples")

    async def run(self):
        pass


async def _make_generator(*items):
    for item in items:
        yield item


def make_grader(samples):
    event = MagicMock()
    event.parser_type = "noop"
    grader = _ConcreteGrader(
        samples_generator=_make_generator(*samples),
        event_manager=MagicMock(),
        job_manager=MagicMock(),
        event=event,
    )
    return grader


async def collect_grades(grader):
    results = []
    async for item in grader.async_grade_all_samples(start=0, grade_fn=grader.grade_sample):
        results.append(item)
    return results


class TestAsyncGradeAllSamplesInputTooLong:

    @pytest.mark.asyncio
    async def test_input_too_long_sets_correct_to_zeros(self):
        sample = {
            "row": 0,
            "generations": ["", ""],
            "eval360_input_too_long": True,
            "ground_truth": "A",
        }
        grader = make_grader([sample, Sentinel.COMPLETED])
        results = await collect_grades(grader)
        grades = [r for r in results if r != Sentinel.COMPLETED]
        assert grades[0]["correct"] == [0, 0]

    @pytest.mark.asyncio
    async def test_grade_fn_not_called(self):
        """grade_sample raises AssertionError if called — the test passes only if it isn't."""
        sample = {
            "row": 0,
            "generations": [""],
            "eval360_input_too_long": True,
            "ground_truth": "A",
        }
        grader = make_grader([sample, Sentinel.COMPLETED])
        # If grade_sample is called, AssertionError bubbles up as ExceptionWrapper.
        results = await collect_grades(grader)
        assert not any(isinstance(r, ExceptionWrapper) for r in results)

    @pytest.mark.asyncio
    async def test_input_too_long_preserves_other_fields(self):
        sample = {
            "row": 7,
            "generations": [""],
            "eval360_input_too_long": True,
            "ground_truth": "B",
            "completion_input": "some long prompt",
        }
        grader = make_grader([sample, Sentinel.COMPLETED])
        results = await collect_grades(grader)
        grades = [r for r in results if r != Sentinel.COMPLETED]
        assert grades[0]["row"] == 7
        assert grades[0]["completion_input"] == "some long prompt"
        assert grades[0]["eval360_input_too_long"] is True

    @pytest.mark.asyncio
    async def test_correct_length_matches_generations_length(self):
        for n in [1, 2, 4, 8]:
            sample = {
                "row": 0,
                "generations": [""] * n,
                "eval360_input_too_long": True,
                "ground_truth": "A",
            }
            grader = make_grader([sample, Sentinel.COMPLETED])
            results = await collect_grades(grader)
            grades = [r for r in results if r != Sentinel.COMPLETED]
            assert grades[0]["correct"] == [0] * n

    @pytest.mark.asyncio
    async def test_normal_sample_still_reaches_grade_fn(self):
        """Samples without input_too_long must still be passed to grade_fn."""
        graded = []

        async def grade_fn(sample):
            graded.append(sample["row"])
            sample["correct"] = [1]
            return sample

        event = MagicMock()
        event.parser_type = "noop"
        samples = [
            {"row": 0, "generations": ["ans"], "ground_truth": "ans"},
            Sentinel.COMPLETED,
        ]
        grader = _ConcreteGrader(
            samples_generator=_make_generator(*samples),
            event_manager=MagicMock(),
            job_manager=MagicMock(),
            event=event,
        )
        results = [r async for r in grader.async_grade_all_samples(start=0, grade_fn=grade_fn)]
        assert graded == [0]

    @pytest.mark.asyncio
    async def test_mixed_samples(self):
        """input_too_long and normal samples can coexist in the same stream."""
        graded_rows = []

        async def grade_fn(sample):
            graded_rows.append(sample["row"])
            sample["correct"] = [1]
            return sample

        event = MagicMock()
        event.parser_type = "noop"
        samples = [
            {"row": 0, "generations": [""], "eval360_input_too_long": True, "ground_truth": "A"},
            {"row": 1, "generations": ["ans"], "ground_truth": "ans"},
            {"row": 2, "generations": [""], "eval360_input_too_long": True, "ground_truth": "B"},
            Sentinel.COMPLETED,
        ]
        grader = _ConcreteGrader(
            samples_generator=_make_generator(*samples),
            event_manager=MagicMock(),
            job_manager=MagicMock(),
            event=event,
        )
        results = [r async for r in grader.async_grade_all_samples(start=0, grade_fn=grade_fn)]
        grades = [r for r in results if r != Sentinel.COMPLETED]

        assert grades[0]["correct"] == [0]   # input_too_long
        assert grades[1]["correct"] == [1]   # normal
        assert grades[2]["correct"] == [0]   # input_too_long
        assert graded_rows == [1]            # only row 1 reached grade_fn


# ---------------------------------------------------------------------------
# Tests for async_grade_all_samples_nonblocking
# ---------------------------------------------------------------------------

def make_grader_nonblocking(samples):
    event = MagicMock()
    event.parser_type = "noop"
    grader = _ConcreteGrader(
        samples_generator=_make_generator(*samples),
        event_manager=MagicMock(),
        job_manager=MagicMock(),
        event=event,
    )
    return grader


async def collect_nonblocking(grader, grade_fn, start=0):
    results = []
    async for item in grader.async_grade_all_samples_nonblocking(start=start, grade_fn=grade_fn):
        results.append(item)
    return results


class TestAsyncGradeAllSamplesNonblocking:

    @pytest.mark.asyncio
    async def test_results_yielded_in_original_order(self):
        """Even if tasks finish out of order, results must come out in submission order."""
        order = []

        async def grade_fn(sample):
            # row 0 takes longer — if results weren't reordered, row 1 would come first
            await asyncio.sleep(0.05 if sample["row"] == 0 else 0.0)
            order.append(sample["row"])
            sample["correct"] = [1]
            return sample

        grader = make_grader_nonblocking([
            {"row": 0, "generations": ["a"], "ground_truth": "a"},
            {"row": 1, "generations": ["b"], "ground_truth": "b"},
            Sentinel.COMPLETED,
        ])
        results = await collect_nonblocking(grader, grade_fn)
        grades = [r for r in results if r != Sentinel.COMPLETED]
        assert [g["row"] for g in grades] == [0, 1]

    @pytest.mark.asyncio
    async def test_tasks_run_concurrently(self):
        """All N tasks should start before the first one finishes (true concurrency)."""
        started = []

        async def grade_fn(sample):
            started.append(sample["row"])
            await asyncio.sleep(0.05)
            sample["correct"] = [1]
            return sample

        grader = make_grader_nonblocking([
            {"row": i, "generations": ["a"], "ground_truth": "a"}
            for i in range(4)
        ] + [Sentinel.COMPLETED])
        await collect_nonblocking(grader, grade_fn)
        # All 4 tasks should have started (not sequentially awaited)
        assert sorted(started) == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_start_skips_already_processed(self):
        """Samples with index < start must be skipped."""
        graded = []

        async def grade_fn(sample):
            graded.append(sample["row"])
            sample["correct"] = [1]
            return sample

        grader = make_grader_nonblocking([
            {"row": 0, "generations": ["a"], "ground_truth": "a"},
            {"row": 1, "generations": ["b"], "ground_truth": "b"},
            {"row": 2, "generations": ["c"], "ground_truth": "c"},
            Sentinel.COMPLETED,
        ])
        await collect_nonblocking(grader, grade_fn, start=2)
        assert graded == [2]

    @pytest.mark.asyncio
    async def test_sentinel_passed_through(self):
        """Sentinel.COMPLETED must appear in the output."""
        async def grade_fn(sample):
            sample["correct"] = [1]
            return sample

        grader = make_grader_nonblocking([Sentinel.COMPLETED])
        results = await collect_nonblocking(grader, grade_fn)
        assert Sentinel.COMPLETED in results

    @pytest.mark.asyncio
    async def test_input_too_long_zeroed_without_calling_grade_fn(self):
        """eval360_input_too_long samples get correct=[0,...] and grade_fn is not called."""
        graded = []

        async def grade_fn(sample):
            graded.append(sample["row"])
            sample["correct"] = [1]
            return sample

        grader = make_grader_nonblocking([
            {"row": 0, "generations": ["", ""], "eval360_input_too_long": True, "ground_truth": "A"},
            Sentinel.COMPLETED,
        ])
        results = await collect_nonblocking(grader, grade_fn)
        grades = [r for r in results if r != Sentinel.COMPLETED]
        assert grades[0]["correct"] == [0, 0]
        assert graded == []

    @pytest.mark.asyncio
    async def test_exception_in_grade_fn_becomes_exception_wrapper(self):
        """If grade_fn raises, the result must be an ExceptionWrapper, not a crash."""
        async def grade_fn(sample):
            raise ValueError("grading failed")

        grader = make_grader_nonblocking([
            {"row": 0, "generations": ["a"], "ground_truth": "a"},
            Sentinel.COMPLETED,
        ])
        results = await collect_nonblocking(grader, grade_fn)
        errors = [r for r in results if isinstance(r, ExceptionWrapper)]
        assert len(errors) == 1
        assert isinstance(errors[0].exception, ValueError)

    @pytest.mark.asyncio
    async def test_exception_wrapper_in_stream_passed_through(self):
        """An ExceptionWrapper already in the sample stream must be yielded as-is."""
        async def grade_fn(sample):
            sample["correct"] = [1]
            return sample

        err = ExceptionWrapper(exception=RuntimeError("upstream"), trace="", instance={})
        grader = make_grader_nonblocking([err, Sentinel.COMPLETED])
        results = await collect_nonblocking(grader, grade_fn)
        assert err in results
