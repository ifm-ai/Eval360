"""Tests for the Sum Puzzle lightweight grader."""

import pytest

from scheduler.grader.sum_puzzle import SumPuzzle, parse_variables, score_answer
from scheduler.utils import Sentinel


class MockEvent:
    """Minimal event carrying a parser type for grader construction."""

    parser_type = "noop"


async def make_samples(*samples):
    """Yield samples followed by the completion sentinel."""
    for sample in samples:
        yield sample
    yield Sentinel.COMPLETED


def make_grader(*samples):
    """Build a SumPuzzle grader with a noop parser."""
    return SumPuzzle(
        samples_generator=make_samples(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


class TestSumPuzzleHelpers:
    """What: validates Sum Puzzle helpers that parse and score exact assignments.
    Executes: `parse_variables` and `score_answer` directly.
    Why: Sum Puzzle grading depends on extracting both variables before exact comparison.
    """

    def test_parse_variables_requires_x_and_y_assignments(self):
        """What: verifies parse_variables returns None unless both x and y are present.
        Executes: `parse_variables` regex extraction, numeric coercion, and missing-key guard.
        Why: malformed answers should not be scored as legitimate assignments.
        """
        assert parse_variables("x = 2, y = 3") == {"x": 2, "y": 3}
        assert parse_variables("x=2.0 y=3") == {"x": 2.0, "y": 3}
        assert parse_variables("x = 2 only") is None
        assert parse_variables(None) is None

    def test_score_answer_handles_missing_and_mismatched_values(self):
        """What: verifies score_answer returns zero for missing keys, non-mappings, or wrong values.
        Executes: `score_answer` exact x/y comparison and KeyError/TypeError fallback.
        Why: wrong or incomplete assignments should score zero without raising.
        """
        ground_truth = {"x": 2, "y": 3}

        assert score_answer({"x": 2, "y": 3}, ground_truth) == 1
        assert score_answer({"x": 3, "y": 2}, ground_truth) == 0
        assert score_answer({"x": 2}, ground_truth) == 0
        assert score_answer(None, ground_truth) == 0


class TestSumPuzzleGradeSample:
    """What: validates SumPuzzle sample grading for sentinels and mixed parsed answers.
    Executes: `SumPuzzle.grade_sample` sentinel handling and parsed-answer scoring.
    Why: sample grading is where parsed model text becomes per-generation correctness.
    """

    @pytest.mark.asyncio
    async def test_grade_sample_passes_completion_sentinel_through(self):
        """What: verifies the completion sentinel is returned unchanged.
        Executes: `SumPuzzle.grade_sample` before dict assertions or variable parsing.
        Why: stream completion must pass through without being graded as an answer.
        """
        grader = make_grader()

        result = await grader.grade_sample(Sentinel.COMPLETED)

        assert result is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_grade_sample_scores_mixed_generation_list(self):
        """What: verifies a sample can contain exact, swapped, incomplete, and unparsable answers.
        Executes: `SumPuzzle.grade_sample` through `parse_variables` and `score_answer`.
        Why: each generation should receive an independent score without mutating input.
        """
        grader = make_grader()
        sample = {
            "parsed_generations": [
                "x=2 y=3",
                "x=3 y=2",
                "x=2",
                None,
            ],
            "ground_truth": {"x": 2, "y": 3},
        }

        result = await grader.grade_sample(sample)

        assert result["correct"] == [1, 0, 0, 0]
        assert "correct" not in sample
