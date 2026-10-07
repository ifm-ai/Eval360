"""Tests for the RULER lightweight grader."""

import math

import pytest

from scheduler.grader.ruler import RulerGrader, check_ruler
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
    """Build a RulerGrader with a noop parser."""
    return RulerGrader(
        samples_generator=make_samples(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


class TestRulerHelpers:
    """What: validates RULER helper checks for task-specific matching.
    Executes: `check_ruler` across NIAH, QA, list-task, empty, and unknown-task routes.
    Why: RULER uses task names to select distinct scoring logic inside one helper.
    """

    def test_check_ruler_single_value_uses_word_boundaries(self):
        """What: verifies NIAH single-value tasks reject substring-only matches.
        Executes: `check_ruler` via the `_match_single_value` word-boundary regex.
        Why: needle answers should match whole tokens without accepting larger words.
        """
        result = check_ruler("niah_single_1", ["needle found", "needless"], ["needle"])

        assert result["correct"] == [True, False]
        assert result["accuracy"] == pytest.approx(0.5)

    def test_check_ruler_multi_value_expands_expected_values(self):
        """What: verifies NIAH multi-value tasks score each expected value per generation.
        Executes: `check_ruler` NIAH multi-value route that extends one score per expected value.
        Why: partial retrieval should lower accuracy instead of collapsing to one boolean.
        """
        result = check_ruler("niah_multivalue", ["alpha appears"], ["alpha", "beta"])

        assert result["correct"] == [True, False]
        assert result["accuracy"] == pytest.approx(0.5)

    def test_check_ruler_question_answer_accepts_any_expected_answer(self):
        """What: verifies QA tasks score a generation correct when any expected answer appears.
        Executes: `check_ruler` QA route and single-result bootstrap default.
        Why: QA fixtures may provide alternate acceptable answers for the same question.
        """
        result = check_ruler("ruler_qa_squad", ["The answer is Paris."], ["Lyon", "Paris"])

        assert result["correct"] == [True]
        assert result["bootstrap_std"] == 0.0

    @pytest.mark.parametrize(
        ("task", "generation", "expected"),
        [
            ("ruler_fwe", "alpha and beta", ["alpha", "gamma"]),
            ("ruler_cwe", "alpha and beta", ["alpha", "gamma"]),
            ("ruler_vt", "abc XYZ", ["xyz", "nope"]),
        ],
    )
    def test_check_ruler_list_tasks_score_each_expected_value(self, task, generation, expected):
        """What: verifies FWE, CWE, and VT tasks append one score for each expected value.
        Executes: `check_ruler` list-task routes, including VT uppercase normalization.
        Why: these RULER tasks grade expected values independently within each generation.
        """
        result = check_ruler(task, [generation], expected)

        assert result["correct"] == [True, False]

    def test_check_ruler_empty_inputs_report_nan_accuracy(self):
        """What: verifies an empty generation list has no denominator and reports NaN accuracy.
        Executes: `check_ruler` with no generations and the empty-correctness metrics path.
        Why: empty input should expose an undefined accuracy rather than a passing score.
        """
        result = check_ruler("ruler_qa_hotpot", [], ["answer"])

        assert result["correct"] == []
        assert math.isnan(result["accuracy"])
        assert result["bootstrap_std"] == 0.0

    def test_check_ruler_rejects_unknown_task(self):
        """What: verifies unknown ruler tasks raise ValueError.
        Executes: `check_ruler` unsupported-task error branch.
        Why: misspelled or unsupported task names should fail fast during grading.
        """
        with pytest.raises(ValueError, match="Unknown ruler_task"):
            check_ruler("unknown", ["answer"], ["answer"])


class TestRulerGradeSample:
    """What: validates RulerGrader sample grading for sentinels and parser fallback.
    Executes: `RulerGrader.grade_sample` sentinel handling and generation selection.
    Why: sample grading must choose parsed text or raw fallback before task scoring.
    """

    @pytest.mark.asyncio
    async def test_grade_sample_passes_completion_sentinel_through(self):
        """What: verifies the completion sentinel is returned unchanged.
        Executes: `RulerGrader.grade_sample` before sample assertions or `check_ruler`.
        Why: stream completion must pass through without being treated as RULER input.
        """
        grader = make_grader()

        result = await grader.grade_sample(Sentinel.COMPLETED)

        assert result is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_grade_sample_uses_raw_generation_when_parsed_is_none(self):
        """What: verifies raw generations are used when parser output is None.
        Executes: `RulerGrader.grade_sample` raw-generation fallback into `check_ruler`.
        Why: parser misses should still grade usable raw model text.
        """
        grader = make_grader()
        sample = {
            "completion_input": "Find the key.",
            "generations": ["The secret needle is here."],
            "parsed_generations": [None],
            "ground_truth": "needle",
            "ruler_task": "niah_single_1",
        }

        result = await grader.grade_sample(sample)

        assert result["correct"] == [True]
        assert result["accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_grade_sample_prefers_parsed_generation_when_present(self):
        """What: verifies parsed generations are used ahead of raw text when available.
        Executes: `RulerGrader.grade_sample` parsed-generation precedence into `check_ruler`.
        Why: parser-normalized output should override raw text when both are present.
        """
        grader = make_grader()
        sample = {
            "completion_input": "Find the key.",
            "generations": ["raw text without the key"],
            "parsed_generations": ["needle"],
            "ground_truth": "needle",
            "ruler_task": "niah_single_1",
        }

        result = await grader.grade_sample(sample)

        assert result["correct"] == [True]
