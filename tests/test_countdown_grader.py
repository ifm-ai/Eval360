"""Tests for the Countdown lightweight grader."""

import math

import pytest

from scheduler.grader.base import Grade, Score
from scheduler.grader.countdown import Countdown, evaluate_equation, score_equation, validate_equation
from scheduler.utils import ExceptionWrapper, Sentinel


class MockEvent:
    """Minimal event carrying a parser type for grader construction."""

    parser_type = "noop"


async def make_samples(*samples):
    """Yield samples followed by the completion sentinel."""
    for sample in samples:
        yield sample
    yield Sentinel.COMPLETED


async def empty_existing():
    """Yield no existing graded rows."""
    if False:
        yield


async def existing_row():
    """Yield one existing graded row."""
    yield {"row": 0, "correct": [1]}


def make_grader(*samples):
    """Build a Countdown grader with a noop parser."""
    return Countdown(
        samples_generator=make_samples(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


class TestCountdownHelpers:
    """What: validates Countdown helpers that reject unsafe or invalid equations.
    Executes: `validate_equation`, `evaluate_equation`, and `score_equation` directly.
    Why: these helpers decide whether Countdown answers are usable before scoring aggregates.
    """

    def test_validate_equation_requires_exact_number_multiset(self):
        """What: verifies validate_equation accepts only the exact available numbers.
        Executes: `validate_equation` regex number extraction and sorted multiset comparison.
        Why: the Countdown puzzle requires every provided number exactly once.
        """
        assert validate_equation("1 * 2 * 3 * 4", [1, 2, 3, 4])
        assert not validate_equation("1 * 2 * 3", [1, 2, 3, 4])
        assert not validate_equation("1 * 2 * 3 * 3", [1, 2, 3, 4])
        assert not validate_equation(None, [1, 2, 3, 4])

    def test_evaluate_equation_rejects_non_arithmetic_and_errors(self):
        """What: verifies evaluate_equation returns None for unsafe text and runtime errors.
        Executes: `evaluate_equation` allow-list filtering and exception handling around eval.
        Why: malformed or non-arithmetic model text must not crash grading or execute code.
        """
        assert evaluate_equation("(1 + 2) * 3") == 9
        assert evaluate_equation("__import__('os').system('true')") is None
        assert evaluate_equation("1 / 0") is None

    def test_score_equation_requires_valid_numbers_and_target(self):
        """What: verifies score_equation returns one only for a valid equation hitting target.
        Executes: `score_equation` through validation, evaluation, and target comparison.
        Why: Countdown correctness depends on both legal number use and the numeric target.
        """
        assert score_equation("1 * 2 * 3 * 4", 24, [1, 2, 3, 4]) == 1
        assert score_equation("1 + 2 + 3 + 4", 24, [1, 2, 3, 4]) == 0
        assert score_equation("1 * 2 * 3", 6, [1, 2, 3, 4]) == 0
        assert score_equation("(1 + 2", 3, [1, 2]) == 0
        assert score_equation(None, 24, [1, 2, 3, 4]) == 0


class TestCountdownGradeSample:
    """What: validates Countdown sample grading for sentinels and mixed generations.
    Executes: `Countdown.grade_sample` sentinel handling and per-generation scoring.
    Why: sample-level grading is the core contract consumed by the shared run loop.
    """

    @pytest.mark.asyncio
    async def test_grade_sample_passes_completion_sentinel_through(self):
        """What: verifies the completion sentinel is returned unchanged.
        Executes: `Countdown.grade_sample` before dict assertions or answer scoring.
        Why: stream completion must pass through without being treated as a sample.
        """
        grader = make_grader()

        result = await grader.grade_sample(Sentinel.COMPLETED)

        assert result is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_grade_sample_scores_mixed_generation_list(self):
        """What: verifies a sample can contain correct, wrong, invalid, and missing equations.
        Executes: `Countdown.grade_sample` iterating parsed generations via `score_equation`.
        Why: mixed model outputs should produce independent correctness flags without mutation.
        """
        grader = make_grader()
        sample = {
            "parsed_generations": [
                "1 * 2 * 3 * 4",
                "1 + 2 + 3 + 4",
                "1 * 2 * 3",
                None,
            ],
            "ground_truth": {"target": 24, "numbers": [1, 2, 3, 4]},
        }

        result = await grader.grade_sample(sample)

        assert result["correct"] == [1, 0, 0, 0]
        assert "correct" not in sample


class TestCountdownRun:
    """What: validates Countdown run defaults, existing rows, and exception passthrough.
    Executes: `AccuracyGraderBase.run` through the concrete `Countdown` grader.
    Why: Countdown relies on the shared resume, aggregation, and error-yielding behavior.
    """

    @pytest.mark.asyncio
    async def test_run_defaults_skip_existing_rows_and_score_new_rows(self):
        """What: verifies default average/pass settings combine existing rows with new grades.
        Executes: `AccuracyGraderBase.run` skip-row resume and default avg/pass metrics.
        Why: resuming should avoid regrading completed rows while still counting them.
        """
        skipped = {
            "row": 0,
            "generations": ["1 + 1"],
            "ground_truth": {"target": 2, "numbers": [1, 1]},
        }
        graded = {
            "row": 1,
            "generations": ["1 + 2"],
            "ground_truth": {"target": 4, "numbers": [1, 2]},
        }
        grader = make_grader(skipped, graded)
        grades = []
        scores = {}

        async for item in grader.run(existing=existing_row(), average_over=[], pass_at=[]):
            if isinstance(item, Grade):
                grades.append(item.element)
            elif isinstance(item, Score):
                scores[item.name] = item.value

        assert [grade["row"] for grade in grades] == [1]
        assert scores["accuracy (avg over 1)"] == pytest.approx(0.5)
        assert scores["accuracy (pass@1)"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_run_passes_exception_wrapper_through(self):
        """What: verifies ExceptionWrapper items from generation are yielded unchanged.
        Executes: `async_grade_all_samples` exception passthrough inside `Countdown.run`.
        Why: generation failures must remain visible while aggregate scores become NaN.
        """
        wrapper = ExceptionWrapper(
            exception=RuntimeError("generation failed"),
            trace="RuntimeError: generation failed",
            instance={"row": 0},
        )
        grader = make_grader(wrapper)
        items = []

        async for item in grader.run(existing=empty_existing(), average_over=[1], pass_at=[1]):
            items.append(item)

        assert items[0] is wrapper
        assert any(item is Sentinel.COMPLETED for item in items)
        assert any(isinstance(item, Score) and math.isnan(item.value) for item in items)
