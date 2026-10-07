"""Tests for the ORDER Puzzle lightweight grader."""

import math

import pytest

from scheduler.grader.base import Grade, Score
from scheduler.grader.order_puzzle import OrderPuzzle, parse_order_answer, verify_order
from scheduler.utils import ExceptionWrapper, Sentinel


class MockEvent:
    """Minimal event carrying a parser type for grader construction."""

    parser_type = "noop"


VALID_CONSTRAINTS = {
    "n": 3,
    "not_sold": [{"0": 2}],
    "sold_before": [{"pivot": 2, "sold_before_pivot": [0]}],
    "sold_after": [{"pivot": 0, "sold_after_pivot": [1]}],
    "sold_between": [{"pivot": 1, "sold_before_pivot": 0, "sold_after_pivot": 2}],
}


async def make_samples(*samples):
    """Yield samples followed by the completion sentinel."""
    for sample in samples:
        yield sample
    yield Sentinel.COMPLETED


async def make_incomplete_samples(*samples):
    """Yield samples without a completion sentinel."""
    for sample in samples:
        yield sample


async def empty_existing():
    """Yield no existing graded rows."""
    if False:
        yield


async def existing_order_row():
    """Yield one existing graded ORDER row."""
    yield {"row": 0, "correct": [1, 0], "difficulty": "easy"}


async def existing_empty_order_row():
    """Yield an existing row with no correctness values."""
    yield {"row": 0, "correct": [], "difficulty": "empty"}


def make_sample(row=0, answer="0, 1, 2", difficulty="easy"):
    """Build an ORDER puzzle sample."""
    return {
        "row": row,
        "generations": [answer],
        "ground_truth": {"constraints": VALID_CONSTRAINTS},
        "difficulty": difficulty,
    }


def make_grader(*samples):
    """Build an OrderPuzzle grader with a noop parser."""
    return OrderPuzzle(
        samples_generator=make_samples(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


def make_incomplete_grader(*samples):
    """Build an OrderPuzzle grader whose stream never completes."""
    return OrderPuzzle(
        samples_generator=make_incomplete_samples(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


class TestOrderPuzzleHelpers:
    """What: validates ORDER helpers that parse answers and check constraints.
    Executes: `parse_order_answer` and `verify_order` across each constraint family.
    Why: ORDER puzzle correctness rests on both output parsing and constraint validation.
    """

    def test_parse_order_answer_accepts_commas_and_whitespace(self):
        """What: verifies parse_order_answer accepts comma-separated and whitespace-separated integers.
        Executes: `parse_order_answer` splitting comma/whitespace text and parse failures.
        Why: model outputs may use either separator and malformed text must score cleanly.
        """
        assert parse_order_answer("0, 1, 2") == [0, 1, 2]
        assert parse_order_answer("0 1 2") == [0, 1, 2]
        assert parse_order_answer("not a list") is None

    def test_verify_order_accepts_solution_satisfying_all_constraints(self):
        """What: verifies a valid permutation satisfying every constraint returns True.
        Executes: `verify_order` length, permutation, not-sold, before, after, and between checks.
        Why: the happy path proves the fixture models a satisfiable ORDER puzzle.
        """
        assert verify_order([0, 1, 2], VALID_CONSTRAINTS)

    @pytest.mark.parametrize(
        "solution",
        [
            [0, 1],
            [0, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
        ],
    )
    def test_verify_order_rejects_invalid_solutions(self, solution):
        """What: verifies invalid length, duplicate items, and violated constraints return False.
        Executes: `verify_order` rejection paths for malformed permutations and constraints.
        Why: invalid orders should fail before they can be counted as solved puzzles.
        """
        assert not verify_order(solution, VALID_CONSTRAINTS)

    def test_verify_order_rejects_failed_between_constraint(self):
        """What: verifies a sold_between constraint fails when neither side condition is true.
        Executes: `verify_order` sold-between logic where both permitted relationships fail.
        Why: the between constraint has custom boolean logic that needs direct coverage.
        """
        constraints = {
            "n": 3,
            "not_sold": [],
            "sold_before": [],
            "sold_after": [],
            "sold_between": [{"pivot": 1, "sold_before_pivot": 2, "sold_after_pivot": 0}],
        }

        assert not verify_order([0, 1, 2], constraints)

    def test_verify_order_rejects_failed_sold_after_constraint(self):
        """What: verifies a sold_after constraint fails when the pivot is not before the item.
        Executes: `verify_order` sold-after comparison for pivot/item ordering.
        Why: reversed order should invalidate a solution even when the list is a permutation.
        """
        constraints = {
            "n": 3,
            "not_sold": [],
            "sold_before": [],
            "sold_after": [{"pivot": 0, "sold_after_pivot": [1]}],
            "sold_between": [],
        }

        assert not verify_order([1, 0, 2], constraints)


class TestOrderPuzzleGradeSample:
    """What: validates OrderPuzzle sample grading for sentinels and mixed answers.
    Executes: `OrderPuzzle.grade_sample` sentinel handling and parsed-answer scoring.
    Why: sample grading ties parser output to ORDER constraint verification.
    """

    @pytest.mark.asyncio
    async def test_grade_sample_passes_completion_sentinel_through(self):
        """What: verifies the completion sentinel is returned unchanged.
        Executes: `OrderPuzzle.grade_sample` before dict assertions or constraint checks.
        Why: stream completion must pass through without being graded as a puzzle row.
        """
        grader = make_grader()

        result = await grader.grade_sample(Sentinel.COMPLETED)

        assert result is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_grade_sample_scores_mixed_generation_list(self):
        """What: verifies correct, invalid, unparsable, and missing answers receive separate scores.
        Executes: `OrderPuzzle.grade_sample` through parse, `verify_order`, and missing-answer guards.
        Why: every generated ordering should produce its own correctness flag.
        """
        grader = make_grader()
        sample = {
            "parsed_generations": ["0, 1, 2", "2, 0, 1", "not a list", None],
            "ground_truth": {"constraints": VALID_CONSTRAINTS},
        }

        result = await grader.grade_sample(sample)

        assert result["correct"] == [1, 0, 0, 0]
        assert "correct" not in sample


class TestOrderPuzzleRun:
    """What: validates OrderPuzzle run aggregation, existing rows, and difficulties.
    Executes: the custom `OrderPuzzle.run` aggregation and difficulty scoring.
    Why: ORDER adds per-difficulty metrics and completion handling on top of grading.
    """

    @pytest.mark.asyncio
    async def test_run_defaults_include_existing_rows_and_per_difficulty_scores(self):
        """What: verifies existing rows are counted and the first generated sample is skipped.
        Executes: `OrderPuzzle.run` existing-row count skip and difficulty buckets.
        Why: resume mode should not duplicate grades but must include prior rows in scores.
        """
        skipped = make_sample(row=0, answer="2, 0, 1", difficulty="easy")
        graded = make_sample(row=1, answer="2, 0, 1", difficulty="hard")
        grader = make_grader(skipped, graded)
        grades = []
        scores = {}

        async for item in grader.run(existing=existing_order_row(), average_over=[], pass_at=[]):
            if isinstance(item, Grade):
                grades.append(item.element)
            elif isinstance(item, Score):
                scores[item.name] = item.value

        assert [grade["row"] for grade in grades] == [1]
        assert scores["accuracy (avg over 1)"] == pytest.approx(0.5)
        assert scores["easy accuracy (avg over 1)"] == pytest.approx(1.0)
        assert scores["hard accuracy (avg over 1)"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_run_reports_nan_for_empty_overall_and_difficulty_buckets(self):
        """What: verifies empty correctness lists report NaN for overall and difficulty scores.
        Executes: `OrderPuzzle.run` no-correctness branches for overall and difficulty metrics.
        Why: empty score buckets need explicit NaN output instead of misleading zeros.
        """
        grader = make_grader(make_sample())
        scores = {}

        async for item in grader.run(existing=existing_empty_order_row(), average_over=[1], pass_at=[1]):
            if isinstance(item, Score):
                scores[item.name] = item.value

        assert math.isnan(scores["accuracy (avg over 1)"])
        assert math.isnan(scores["accuracy (pass@1)"])
        assert math.isnan(scores["empty accuracy (avg over 1)"])
        assert math.isnan(scores["empty accuracy (pass@1)"])

    @pytest.mark.asyncio
    async def test_run_passes_exception_wrapper_through(self):
        """What: verifies ExceptionWrapper items from generation are yielded unchanged.
        Executes: `OrderPuzzle.run` handling of `ExceptionWrapper` from grading.
        Why: generation failures should remain visible to callers of the run stream.
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

    @pytest.mark.asyncio
    async def test_run_emits_bootstrap_when_average_over_exceeds_one(self):
        """What: verifies average_over greater than one emits the custom bootstrap score.
        Executes: `OrderPuzzle.run` overall `average_over > 1` bootstrap calculation.
        Why: repeated ORDER attempts need uncertainty reporting alongside average accuracy.
        """
        sample = make_sample(row=0, answer="0, 1, 2", difficulty="easy")
        sample["generations"] = ["0, 1, 2", "2, 0, 1"]
        grader = make_grader(sample)
        scores = {}

        async for item in grader.run(existing=empty_existing(), average_over=[2], pass_at=[1]):
            if isinstance(item, Score):
                scores[item.name] = item.value

        assert scores["bootstrap_std (avg over 2)"] == 0.0

    @pytest.mark.asyncio
    async def test_run_returns_without_scores_when_stream_never_completes(self):
        """What: verifies a stream without Sentinel.COMPLETED returns before emitting aggregate scores.
        Executes: `OrderPuzzle.run` early return when the generator never completes.
        Why: aggregate metrics should only be emitted after a complete sample stream.
        """
        grader = make_incomplete_grader(make_sample())
        items = []

        async for item in grader.run(existing=empty_existing(), average_over=[1], pass_at=[1]):
            items.append(item)

        assert any(isinstance(item, Grade) for item in items)
        assert not any(isinstance(item, Score) for item in items)
