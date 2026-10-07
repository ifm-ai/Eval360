"""Tests for error record handling in the grader pipeline.

Covers two bugs fixed together:
  1. Generation error records (exception/trace in generations file, no `generations`
     key) must count as incorrect (correct=[0]) rather than being silently excluded
     from the accuracy denominator.
  2. finish_reason='length' with content=None must be treated as an empty-string
     generation rather than raising a RuntimeError.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from scheduler.grader.base import Grade, Score
from scheduler.grader.match import Match
from scheduler.utils import Sentinel
from scheduler.openai_interface import LOCKED_CONNECTIONS


# ---------------------------------------------------------------------------
# Helpers shared with test_grader_metadata_integration.py
# ---------------------------------------------------------------------------

class MockEvent:
    def __init__(self, parser_type="noop"):
        self.parser_type = parser_type


async def make_samples_generator(*samples):
    for s in samples:
        yield s
    yield Sentinel.COMPLETED


async def empty_existing():
    if False:
        yield


async def collect_run(grader, average_over=None, pass_at=None):
    grades = []
    scores = {}
    async for item in grader.run(
        existing=empty_existing(),
        average_over=average_over or [1],
        pass_at=pass_at or [1],
    ):
        if isinstance(item, Grade):
            grades.append(item.element)
        elif isinstance(item, Score):
            scores[item.name] = item.value
    return grades, scores


def _make_error_record(ground_truth="A"):
    """Simulate a generations-file record where generation failed."""
    return {
        "row": 0,
        "completion_input": "Q",
        "chat_input": [{"role": "user", "content": "Q"}],
        "ground_truth": ground_truth,
        "exception": "VLLM returned null content for 1/1 choices with finish_reason != stop/length",
        "trace": "RuntimeError: ...",
    }


def _make_good_record(generation, ground_truth):
    return {
        "row": 1,
        "completion_input": "Q",
        "generations": [generation],
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# Error records count as incorrect
# ---------------------------------------------------------------------------

class TestErrorRecordCountsAsIncorrect:

    @pytest.mark.asyncio
    async def test_error_record_has_correct_zero(self):
        """An error record must come out of the grader with correct=[0]."""
        grader = Match(
            samples_generator=make_samples_generator(_make_error_record()),
            event_manager=None,
            job_manager=None,
            event=MockEvent("noop"),
        )
        grades, _ = await collect_run(grader)
        assert len(grades) == 1
        assert grades[0]["correct"] == [0]

    @pytest.mark.asyncio
    async def test_error_record_reduces_accuracy(self):
        """One correct + one error record → accuracy = 0.5, not 1.0."""
        samples = [
            _make_good_record("A", "A"),   # correct
            _make_error_record("A"),        # error → incorrect
        ]
        grader = Match(
            samples_generator=make_samples_generator(*samples),
            event_manager=None,
            job_manager=None,
            event=MockEvent("noop"),
        )
        _, scores = await collect_run(grader)
        assert scores["accuracy (avg over 1)"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_all_error_records_gives_zero_accuracy(self):
        """All error records → accuracy = 0.0, not NaN."""
        samples = [_make_error_record() for _ in range(3)]
        grader = Match(
            samples_generator=make_samples_generator(*samples),
            event_manager=None,
            job_manager=None,
            event=MockEvent("noop"),
        )
        _, scores = await collect_run(grader)
        assert scores["accuracy (avg over 1)"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_error_record_mixed_with_correct_and_incorrect(self):
        """2 correct + 1 incorrect + 1 error → accuracy = 2/4 = 0.5."""
        samples = [
            _make_good_record("A", "A"),   # correct
            _make_good_record("B", "A"),   # incorrect
            _make_good_record("A", "A"),   # correct
            _make_error_record("A"),        # error → incorrect
        ]
        grader = Match(
            samples_generator=make_samples_generator(*samples),
            event_manager=None,
            job_manager=None,
            event=MockEvent("noop"),
        )
        _, scores = await collect_run(grader)
        assert scores["accuracy (avg over 1)"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# finish_reason='length' with null content is accepted as empty string
# ---------------------------------------------------------------------------

class TestNullContentFinishReasonLength:

    @pytest.fixture(autouse=True)
    def clear_connections(self):
        LOCKED_CONNECTIONS.clear()
        yield
        LOCKED_CONNECTIONS.clear()

    def _make_chat_conn(self):
        from scheduler.openai_interface import OpenAIConnection, LOCKED_CONNECTIONS
        from scheduler.model import CacheSaltConfig, ModelType, ModelInstance
        from scheduler.task import AsyncGenerationTask
        from scheduler.event import EventInstance
        from scheduler.job import JobManager
        from scheduler.progress import ProgressManager

        FAKE_URL = "http://fake:8000"
        model = MagicMock(spec=ModelInstance)
        model.name = "test-model"
        model.serving_key = "test-model"
        model.max_simultaneous_requests = 4
        model.model_type = ModelType.CHAT
        model.openai_kwargs = {}
        model.cache_salt = CacheSaltConfig()
        model.prompt_prefix_instructions = None

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
        LOCKED_CONNECTIONS["test-model"]._slots[FAKE_URL] = [4]
        conn._clients[FAKE_URL] = MagicMock()
        conn._job_manager.is_url_live = MagicMock(return_value=True)
        return conn, FAKE_URL

    def _chat_response(self, content, finish_reason, reasoning=None):
        choice = MagicMock()
        choice.message.content = content
        choice.message.reasoning_content = reasoning
        choice.message.model_extra = {}
        choice.message.tool_calls = []
        choice.finish_reason = finish_reason
        choice.logprobs = None
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    async def _collect(self, gen):
        return [item async for item in gen]

    async def _requests(self):
        yield {"chat_input": [{"role": "user", "content": "Q"}], "completion_input": ""}

    @pytest.mark.asyncio
    async def test_null_content_length_finish_reason_accepted_as_empty(self):
        """content=None + finish_reason='length' must be accepted as '' (not an error)."""
        conn, url = self._make_chat_conn()
        conn._clients[url].chat.completions.create = AsyncMock(
            return_value=self._chat_response(None, "length")
        )
        results = await self._collect(
            conn.launch_requests(self._requests(), offset=0, completion_hook=AsyncMock())
        )
        assert len(results) == 1
        assert results[0]["generations"] == [""]

    @pytest.mark.asyncio
    async def test_null_content_length_finish_reason_with_reasoning(self):
        """content=None + finish_reason='length' + reasoning → '' generation, reasoning saved."""
        conn, url = self._make_chat_conn()
        conn._clients[url].chat.completions.create = AsyncMock(
            return_value=self._chat_response(None, "length", reasoning="my thinking")
        )
        results = await self._collect(
            conn.launch_requests(self._requests(), offset=0, completion_hook=AsyncMock())
        )
        assert results[0]["generations"] == [""]
        assert results[0].get("reasoning") == ["my thinking"]


# ---------------------------------------------------------------------------
# Gap 1: async_grade_all_samples_nonblocking yields in order with mid-stream exception
# ---------------------------------------------------------------------------

def _make_good_sample(row: int, generation: str = "A", ground_truth: str = "A") -> dict:
    return {
        "row": row,
        "completion_input": "Q",
        "generations": [generation],
        "ground_truth": ground_truth,
    }


class TestNonblockingOrderAndException:
    """
    async_grade_all_samples_nonblocking must:
    - surface exceptions (not silently swallow them)
    - yield results in the original input order even when a middle sample raises
    """

    @pytest.mark.asyncio
    async def test_exception_mid_stream_is_surfaced_as_exception_wrapper(self):
        """If sample index 2 raises, an ExceptionWrapper is yielded (not silently dropped)."""
        from scheduler.utils import ExceptionWrapper
        from scheduler.grader.base import AccuracyGraderBase
        from scheduler.grader.registry import register

        call_count = 0

        @register("_test_nonblocking_raises")
        class _RaisingGrader(AccuracyGraderBase):
            async def grade_sample(self, sample):
                nonlocal call_count
                call_count += 1
                if sample["row"] == 2:
                    raise ValueError("deliberate error on row 2")
                sample["correct"] = [1]
                return sample

        samples = [_make_good_sample(i) for i in range(5)]

        async def _gen():
            for s in samples:
                yield s
            yield Sentinel.COMPLETED

        grader = _RaisingGrader(
            samples_generator=_gen(),
            event_manager=None,
            job_manager=None,
            event=MockEvent("noop"),
        )

        results = []
        async for item in grader.async_grade_all_samples_nonblocking(
            start=0,
            grade_fn=grader.grade_sample,
        ):
            results.append(item)

        # The last item should be Sentinel.COMPLETED
        assert results[-1] == Sentinel.COMPLETED
        non_sentinel = results[:-1]
        assert len(non_sentinel) == 5

        # Row 2 must produce an ExceptionWrapper
        error_item = non_sentinel[2]
        assert isinstance(error_item, ExceptionWrapper), (
            f"expected ExceptionWrapper at index 2, got {type(error_item)}"
        )
        assert "deliberate error on row 2" in error_item.trace

    @pytest.mark.asyncio
    async def test_exception_mid_stream_results_in_input_order(self):
        """Results are yielded in input order: 0, 1, exception-for-2, 3, 4."""
        from scheduler.utils import ExceptionWrapper
        from scheduler.grader.base import AccuracyGraderBase
        from scheduler.grader.registry import register

        @register("_test_nonblocking_order")
        class _OrderCheckGrader(AccuracyGraderBase):
            async def grade_sample(self, sample):
                if sample["row"] == 2:
                    raise ValueError("error on 2")
                sample["correct"] = [1]
                return sample

        samples = [_make_good_sample(i) for i in range(5)]

        async def _gen():
            for s in samples:
                yield s
            yield Sentinel.COMPLETED

        grader = _OrderCheckGrader(
            samples_generator=_gen(),
            event_manager=None,
            job_manager=None,
            event=MockEvent("noop"),
        )

        results = []
        async for item in grader.async_grade_all_samples_nonblocking(
            start=0,
            grade_fn=grader.grade_sample,
        ):
            results.append(item)

        non_sentinel = [r for r in results if r != Sentinel.COMPLETED]
        assert len(non_sentinel) == 5

        # Indices 0, 1, 3, 4 must be normal graded dicts with correct row number
        for expected_row, idx in [(0, 0), (1, 1), (3, 3), (4, 4)]:
            item = non_sentinel[idx]
            assert isinstance(item, dict), f"index {idx} should be a dict, got {type(item)}"
            assert item["row"] == expected_row, f"index {idx} has row={item['row']}, expected {expected_row}"

        # Index 2 must be an ExceptionWrapper
        assert isinstance(non_sentinel[2], ExceptionWrapper), (
            f"index 2 should be ExceptionWrapper, got {type(non_sentinel[2])}"
        )


# ---------------------------------------------------------------------------
# Gap 2: initialize() is safe to call once per stream (concurrent _grade_tasks)
# ---------------------------------------------------------------------------

class TestInitializeCalledOnce:
    """
    initialize() must be called exactly once even when multiple _grade_task
    coroutines start concurrently (as happens in async_grade_all_samples_nonblocking).
    """

    @pytest.mark.asyncio
    async def test_initialize_called_exactly_once_with_concurrent_grading(self):
        from scheduler.grader.base import AccuracyGraderBase
        from scheduler.grader.registry import register

        init_call_count = 0

        @register("_test_initialize_once")
        class _CountingInitGrader(AccuracyGraderBase):
            async def initialize(self):
                nonlocal init_call_count
                # Small sleep to maximise overlap between concurrent _grade_task coroutines
                import asyncio as _asyncio
                await _asyncio.sleep(0)
                init_call_count += 1

            async def grade_sample(self, sample):
                sample["correct"] = [1]
                return sample

        # Use enough samples that multiple _grade_task coroutines will be in flight
        samples = [_make_good_sample(i) for i in range(8)]

        async def _gen():
            for s in samples:
                yield s
            yield Sentinel.COMPLETED

        grader = _CountingInitGrader(
            samples_generator=_gen(),
            event_manager=None,
            job_manager=None,
            event=MockEvent("noop"),
        )

        results = []
        async for item in grader.async_grade_all_samples_nonblocking(
            start=0,
            grade_fn=grader.grade_sample,
        ):
            results.append(item)

        assert init_call_count == 1, (
            f"initialize() should be called exactly once, was called {init_call_count} times"
        )
