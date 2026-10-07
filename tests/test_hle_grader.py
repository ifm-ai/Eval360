"""Tests for HLE grader — _extract_mc_answer, _check_multiple_choice, parse_judge_response, grade_sample."""

import asyncio
import pytest
from unittest.mock import MagicMock

from scheduler.grader.hle import _extract_mc_answer, _check_multiple_choice, HLEJudge, HLEGrader
from scheduler.utils import Sentinel


def make_judge():
    """Return an HLEJudge with external dependencies mocked out."""
    judge = HLEJudge.__new__(HLEJudge)
    judge.task = MagicMock()
    judge.task.grader.llm_as_judge = MagicMock()
    judge.model = MagicMock()
    judge.openai_connection = MagicMock()
    return judge


def make_grader(model=MagicMock()):
    """Return an HLEGrader with external dependencies mocked out."""
    grader = HLEGrader.__new__(HLEGrader)
    grader.task = MagicMock()
    grader.task.grader.llm_as_judge = model
    grader.model = model
    grader.openai_connection = MagicMock()
    return grader


def make_sample(answer_type, ground_truth="A", generations=None):
    gens = generations or ["Answer: A"]
    return {
        "row": 0,
        "completion_input": "question",
        "ground_truth": ground_truth,
        "answer_type": answer_type,
        "generations": gens,
        "parsed_generations": gens,
    }


# ---------------------------------------------------------------------------
# tests for function `_extract_mc_answer``
# ---------------------------------------------------------------------------

class TestExtractMcAnswer:

    # Model output format — "Answer: X" should be extracted
    def test_standard_answer_prefix(self):
        assert _extract_mc_answer("Answer: A") == "A"

    # No "Answer:" prefix — falls back to first letter in text
    def test_fallback_first_letter_in_sentence(self):
        assert _extract_mc_answer("The correct option is D.") == "T"

    # None generation means the model produced no output
    def test_none_input(self):
        assert _extract_mc_answer(None) is None

    # Empty string is also a missing generation
    def test_empty_string(self):
        assert _extract_mc_answer("") is None

    # Extracted multiple choice answer must always be uppercased for consistent comparison
    def test_answer_uppercases_result(self):
        assert _extract_mc_answer("Answer: b") == "B"


# ---------------------------------------------------------------------------
# tests for function `_check_multiple_choice`
# ---------------------------------------------------------------------------

class TestCheckMultipleChoice:

    # Each generation is graded independently and accuracy reflects the fraction correct
    def test_multiple_generations_mixed(self):
        result = _check_multiple_choice(["Answer: A", "Answer: B", "Answer: A"], "A")
        assert result["correct"] == [True, False, True]
        assert round(result["accuracy"], 4) == round(2 / 3, 4)


# ---------------------------------------------------------------------------
# tests for `HLEJudge.parse_judge_response`
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:

    # Test JSON response from judge model -- correct answer
    def test_json_correct_yes(self):
        judge = make_judge()
        assert judge.parse_judge_response('{"correct": "yes"}') is True

    # Test JSON response from judge model -- incorrect answer
    def test_json_correct_no(self):
        judge = make_judge()
        assert judge.parse_judge_response('{"correct": "no"}') is False

    # Regex match when no JSON is present in the judge response
    def test_line_format_yes(self):
        judge = make_judge()
        assert judge.parse_judge_response("correct: yes") is True

    # Reasoning models wrap output in <think> tags — reasoning content must be ignored
    def test_think_tag_stripped(self):
        judge = make_judge()
        response = "<think>some reasoning</think>\ncorrect: yes"
        assert judge.parse_judge_response(response) is True

    # Judge returning None means no verdict — should be treated as incorrect
    def test_none_response(self):
        judge = make_judge()
        assert judge.parse_judge_response(None) is False

    # Malformed JSON should not crash — falls back to regex matching
    def test_malformed_json_falls_back_to_line_match(self):
        judge = make_judge()
        assert judge.parse_judge_response("{invalid json}\ncorrect: yes") is True


# ---------------------------------------------------------------------------
# tests for `HLEGrader.grade_sample`
# ---------------------------------------------------------------------------

class TestGradeSample:

    # multiple choice samples are graded locally — no LLM judge call needed
    def test_multiple_choice_dispatches_without_llm(self):
        grader = make_grader()
        sample = make_sample("multipleChoice", ground_truth="A", generations=["Answer: A"])
        result = asyncio.run(grader.grade_sample(sample))
        assert result["correct"] == [True]

    # Without a judge model, exactMatch falls back to strict string comparison
    def test_exact_match_no_model_falls_back_to_string_match(self):
        grader = make_grader(model=None)
        sample = make_sample("exactMatch", ground_truth="Paris", generations=["Paris"])
        result = asyncio.run(grader.grade_sample(sample))
        assert result["correct"] == [True]

    # Unrecognised answer_type should raise an error
    def test_unknown_answer_type_raises(self):
        grader = make_grader()
        sample = make_sample("unknown_type")
        with pytest.raises(ValueError, match="Unknown HLE answer_type"):
            asyncio.run(grader.grade_sample(sample))

    # Each record must have an answer_type field — missing it should raise an error
    def test_missing_answer_type_raises(self):
        grader = make_grader()
        sample = make_sample("multipleChoice")
        del sample["answer_type"]
        with pytest.raises(AssertionError):
            asyncio.run(grader.grade_sample(sample))

    # Sentinel signals end-of-stream and must pass through without processing
    def test_sentinel_passes_through(self):
        grader = make_grader()
        result = asyncio.run(grader.grade_sample(Sentinel.COMPLETED))
        assert result == Sentinel.COMPLETED
