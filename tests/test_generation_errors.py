"""
Tests for generation error handling:
  - ExceptionWrapper.instance carries the partial result dict (has 'generations' key)
  - async_grade_all_samples passes through ExceptionWrapper items from generation
    without trying to grade them
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from scheduler.utils import ExceptionWrapper, Sentinel
from scheduler.grader.base import GraderBase
from scheduler.event import GradingEventInstance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_exception_wrapper(input_item=None):
    instance = input_item or {"completion_input": "q", "chat_input": [], "generations": []}
    return ExceptionWrapper(
        exception=ValueError("context length exceeded"),
        trace="Traceback...",
        instance=instance,
    )


async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


class _ConcreteGrader(GraderBase):
    """Minimal concrete grader for testing async_grade_all_samples."""

    def grade_sample(self, sample):
        return {"correct": [True]}

    async def run(self):
        pass


def make_grader(samples):
    db = MagicMock()
    event = MagicMock(spec=GradingEventInstance)
    event.parser_type = "noop"
    return _ConcreteGrader(
        samples_generator=_aiter(*samples),
        event_manager=MagicMock(),
        job_manager=MagicMock(),
        event=event,
    )


# ---------------------------------------------------------------------------
# async_grade_all_samples: ExceptionWrapper passthrough
# ---------------------------------------------------------------------------

class TestAsyncGradeAllSamplesExceptionPassthrough:

    @pytest.mark.asyncio
    async def test_exception_wrapper_passed_through_without_grading(self):
        """ExceptionWrapper items from generation must be yielded as-is."""
        wrapper = make_exception_wrapper()
        grader = make_grader([wrapper, Sentinel.COMPLETED])

        results = await _collect(
            grader.async_grade_all_samples(start=0, grade_fn=AsyncMock())
        )

        assert results[0] is wrapper
        assert results[1] is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_grade_fn_not_called_for_exception_wrapper(self):
        """grade_fn should never be called for ExceptionWrapper items."""
        wrapper = make_exception_wrapper()
        grade_fn = AsyncMock(return_value={"correct": [True]})
        grader = make_grader([wrapper, Sentinel.COMPLETED])

        await _collect(grader.async_grade_all_samples(start=0, grade_fn=grade_fn))

        grade_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_wrapper_mixed_with_normal_items(self):
        """Normal items are graded; ExceptionWrapper items are passed through."""
        normal = {"completion_input": "q", "chat_input": [], "generations": ["ans"],
                  "ground_truth": "ans", "parsed_generations": ["ans"]}
        wrapper = make_exception_wrapper()
        grade_fn = AsyncMock(return_value={"correct": [True]})
        grader = make_grader([normal, wrapper, Sentinel.COMPLETED])

        results = await _collect(
            grader.async_grade_all_samples(start=0, grade_fn=grade_fn)
        )

        assert grade_fn.call_count == 1
        assert results[1] is wrapper
        assert results[2] is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_exception_wrapper_before_start_is_skipped(self):
        """ExceptionWrapper items before 'start' are skipped like normal items."""
        wrapper = make_exception_wrapper()
        normal = {"completion_input": "q", "chat_input": [], "generations": ["ans"]}
        grade_fn = AsyncMock(return_value={"correct": [True]})
        grader = make_grader([wrapper, normal, Sentinel.COMPLETED])

        results = await _collect(
            grader.async_grade_all_samples(start=1, grade_fn=grade_fn)
        )

        # wrapper is at index 0 which is < start=1, so skipped; normal is graded
        assert grade_fn.call_count == 1
        assert results[-1] is Sentinel.COMPLETED
