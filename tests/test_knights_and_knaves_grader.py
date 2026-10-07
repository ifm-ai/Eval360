"""Tests for the Knights and Knaves lightweight grader."""

import math

import pytest

from scheduler.grader.base import Grade, Score
from scheduler.grader.knights_and_knaves import (
    KnightsAndKnaves,
    check_answer,
    parse_model_answer,
    parse_solution_text_format,
)
from scheduler.utils import ExceptionWrapper, Sentinel


class MockEvent:
    """Minimal event carrying a parser type for grader construction."""

    parser_type = "noop"


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


async def existing_kk_row():
    """Yield one existing graded Knights and Knaves row."""
    yield {"row": 0, "correct": [1, 0], "difficulty": "easy"}


async def existing_empty_kk_row():
    """Yield an existing row with no correctness values."""
    yield {"row": 0, "correct": [], "difficulty": "empty"}


def make_sample(row=0, answer="Alice is a knight. Bob is a knave.", difficulty="easy"):
    """Build a Knights and Knaves sample."""
    return {
        "row": row,
        "generations": [answer],
        "ground_truth": "Alice is a knight.\nBob is a knave.",
        "names": ["Alice", "Bob"],
        "difficulty": difficulty,
    }


def make_grader(*samples):
    """Build a KnightsAndKnaves grader with a noop parser."""
    return KnightsAndKnaves(
        samples_generator=make_samples(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


def make_incomplete_grader(*samples):
    """Build a KnightsAndKnaves grader whose stream never completes."""
    return KnightsAndKnaves(
        samples_generator=make_incomplete_samples(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


class TestKnightsAndKnavesHelpers:
    """What: validates Knights and Knaves helpers that parse and compare role assignments.
    Executes: `parse_solution_text_format`, `parse_model_answer`, and `check_answer`.
    Why: Knights and Knaves grading depends on complete, name-specific role extraction.
    """

    def test_parse_solution_text_format_extracts_roles_by_name(self):
        """What: verifies ground-truth solution text is parsed into lowercase roles.
        Executes: `parse_solution_text_format` over blank-prefixed multiline solution text.
        Why: dataset solutions are line-oriented and become the canonical role map.
        """
        parsed = parse_solution_text_format("\nAlice is a knight.\nBob is a knave.")

        assert parsed == {"Alice": "knight", "Bob": "knave"}

    def test_parse_model_answer_returns_none_when_incomplete(self):
        """What: verifies model answers must mention every expected name.
        Executes: `parse_model_answer` per-name regex extraction and incomplete-answer guard.
        Why: missing one character's role should make the whole logical answer unscoreable.
        """
        assert parse_model_answer("Alice is a knight. Bob is a knave.", ["Alice", "Bob"]) == {
            "Alice": "knight",
            "Bob": "knave",
        }
        assert parse_model_answer("Alice is a knight.", ["Alice", "Bob"]) is None

    def test_check_answer_requires_all_roles_to_match(self):
        """What: verifies a solution is correct only when every role matches ground truth.
        Executes: `check_answer` for exact matches, mismatched roles, and `None` solutions.
        Why: the puzzle is scored as a single all-or-nothing role assignment.
        """
        gt = {"Alice": "knight", "Bob": "knave"}

        assert check_answer(gt, {"Alice": "knight", "Bob": "knave"}) == 1
        assert check_answer(gt, {"Alice": "knave", "Bob": "knave"}) == 0
        assert check_answer(gt, None) == 0


class TestKnightsAndKnavesGradeSample:
    """What: validates KnightsAndKnaves sample grading for sentinels and mixed answers.
    Executes: `KnightsAndKnaves.grade_sample` sentinel and parsed-answer scoring routes.
    Why: sample grading connects parsed model text to the helper-level role checks.
    """

    @pytest.mark.asyncio
    async def test_grade_sample_passes_completion_sentinel_through(self):
        """What: verifies the completion sentinel is returned unchanged.
        Executes: `KnightsAndKnaves.grade_sample` before dict assertions or parsing.
        Why: stream completion must pass through without being graded as a puzzle.
        """
        grader = make_grader()

        result = await grader.grade_sample(Sentinel.COMPLETED)

        assert result is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_grade_sample_scores_mixed_generation_list(self):
        """What: verifies correct, wrong, and missing parsed answers receive separate scores.
        Executes: `KnightsAndKnaves.grade_sample` through solution parsing and `check_answer`.
        Why: each generation should get an independent score, including missing answers.
        """
        grader = make_grader()
        sample = {
            "parsed_generations": [
                "Alice is a knight. Bob is a knave.",
                "Alice is a knave. Bob is a knave.",
                None,
            ],
            "ground_truth": "Alice is a knight.\nBob is a knave.",
            "names": ["Alice", "Bob"],
        }

        result = await grader.grade_sample(sample)

        assert result["correct"] == [1, 0, 0]
        assert "correct" not in sample


class TestKnightsAndKnavesRun:
    """What: validates KnightsAndKnaves run aggregation, existing rows, and difficulties.
    Executes: the custom `KnightsAndKnaves.run` aggregation and difficulty scoring.
    Why: this grader extends the shared loop with per-difficulty metrics and completion checks.
    """

    @pytest.mark.asyncio
    async def test_run_defaults_include_existing_rows_and_per_difficulty_scores(self):
        """What: verifies existing rows are counted and the first generated sample is skipped.
        Executes: `KnightsAndKnaves.run` existing-row count skip and difficulty buckets.
        Why: resume mode must avoid duplicate grading while preserving per-difficulty scores.
        """
        skipped = make_sample(row=0, answer="Alice is a knave. Bob is a knave.", difficulty="easy")
        graded = make_sample(row=1, answer="Alice is a knave. Bob is a knave.", difficulty="hard")
        grader = make_grader(skipped, graded)
        grades = []
        scores = {}

        async for item in grader.run(existing=existing_kk_row(), average_over=[], pass_at=[]):
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
        Executes: `KnightsAndKnaves.run` no-correctness branches for overall and difficulty.
        Why: empty buckets have no denominator and should not report a numeric accuracy.
        """
        grader = make_grader(make_sample())
        scores = {}

        async for item in grader.run(existing=existing_empty_kk_row(), average_over=[1], pass_at=[1]):
            if isinstance(item, Score):
                scores[item.name] = item.value

        assert math.isnan(scores["accuracy (avg over 1)"])
        assert math.isnan(scores["accuracy (pass@1)"])
        assert math.isnan(scores["empty accuracy (avg over 1)"])
        assert math.isnan(scores["empty accuracy (pass@1)"])

    @pytest.mark.asyncio
    async def test_run_passes_exception_wrapper_through(self):
        """What: verifies ExceptionWrapper items from generation are yielded unchanged.
        Executes: `KnightsAndKnaves.run` handling of `ExceptionWrapper` from grading.
        Why: generation failures should be yielded to callers instead of swallowed.
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
        Executes: `KnightsAndKnaves.run` overall `average_over > 1` bootstrap calculation.
        Why: multi-generation averages should expose uncertainty alongside accuracy.
        """
        sample = make_sample(row=0, answer="Alice is a knight. Bob is a knave.", difficulty="easy")
        sample["generations"] = [
            "Alice is a knight. Bob is a knave.",
            "Alice is a knave. Bob is a knave.",
        ]
        grader = make_grader(sample)
        scores = {}

        async for item in grader.run(existing=empty_existing(), average_over=[2], pass_at=[1]):
            if isinstance(item, Score):
                scores[item.name] = item.value

        assert scores["bootstrap_std (avg over 2)"] == 0.0

    @pytest.mark.asyncio
    async def test_run_returns_without_scores_when_stream_never_completes(self):
        """What: verifies a stream without Sentinel.COMPLETED returns before emitting aggregate scores.
        Executes: `KnightsAndKnaves.run` early return when the generator never completes.
        Why: aggregate scores should only be emitted after a completed grading stream.
        """
        grader = make_incomplete_grader(make_sample())
        items = []

        async for item in grader.run(existing=empty_existing(), average_over=[1], pass_at=[1]):
            items.append(item)

        assert any(isinstance(item, Grade) for item in items)
        assert not any(isinstance(item, Score) for item in items)
