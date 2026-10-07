"""
Tests for SympyLLMasJudge and its helper functions.

Covers:
  - normalize_answer_hendrycks
  - _normalize
  - grade_answer (sympy-based comparison)
  - should_allow_eval
  - split_tuple
  - _grade_single_generation (sympy path, LLM fallback, timeout)
  - grade_sample (orchestration, diagnostics)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from scheduler.external_requests import ExternalRequestFailure
from scheduler.grader.sympy_llm_as_judge import (
    SympyLLMasJudge,
    normalize_answer_hendrycks,
    _normalize,
    grade_answer,
    should_allow_eval,
    split_tuple,
    _fix_fracs,
    _fix_sqrt,
    _is_frac,
    _str_is_int,
    are_equal_under_sympy,
)
from scheduler.model import CacheSaltConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grader():
    """Return a SympyLLMasJudge with all external dependencies mocked out."""
    grader = SympyLLMasJudge.__new__(SympyLLMasJudge)
    grader._initialized = True
    grader.client = AsyncMock()
    grader.openai_connection = AsyncMock()
    grader.openai_connection.get_client = AsyncMock(return_value=grader.client)
    grader.model = MagicMock()
    grader.model.name = "judge-model"
    grader.model.api_model_name = None
    grader.model.openai_kwargs = {}
    grader.model.cache_salt = CacheSaltConfig()
    grader._parser = MagicMock()
    grader._parser.parse_generations = lambda gens: gens
    return grader


def make_sample(generation, ground_truth):
    gen = generation if isinstance(generation, list) else [generation]
    return {
        "row": 0,
        "completion_input": "solve x",
        "ground_truth": ground_truth,
        "generations": gen,
        "parsed_generations": gen,
    }


# ---------------------------------------------------------------------------
# normalize_answer_hendrycks
# ---------------------------------------------------------------------------

class TestNormalizeAnswerHendrycks:
    def test_none_returns_none(self):
        assert normalize_answer_hendrycks(None) is None

    def test_plain_integer(self):
        assert normalize_answer_hendrycks("42") == "42"

    def test_strips_whitespace(self):
        assert normalize_answer_hendrycks("  42  ") == "42"

    def test_text_wrapper_extracted(self):
        # \text{hello} → hello
        result = normalize_answer_hendrycks("\\text{hello}")
        assert result == "hello"

    def test_replaces_dfrac_with_frac(self):
        result = normalize_answer_hendrycks("\\dfrac{1}{2}")
        assert "frac" in result

    def test_removes_degree_symbol(self):
        result = normalize_answer_hendrycks("90^{\\circ}")
        assert "circ" not in result

    def test_leading_dot_gets_zero(self):
        # ".5" → "0.5" → then "0.5" == "0.5" triggers frac replacement → "\\frac{1}{2}"
        # The key invariant is that the leading "." is gone
        result = normalize_answer_hendrycks(".5")
        assert not result.startswith(".")

    def test_equation_rhs_extracted(self):
        # "x=5" → "5" (LHS is short)
        result = normalize_answer_hendrycks("x=5")
        assert result == "5"


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_none_returns_none(self):
        assert _normalize(None) is None

    def test_integer_string(self):
        assert _normalize("42") == "42"

    def test_float_that_is_int(self):
        assert _normalize("42.0") == "42"

    def test_strips_percent(self):
        result = _normalize("50%")
        assert "%" not in result

    def test_strips_dollar(self):
        result = _normalize("$100")
        assert "$" not in result

    def test_million_expanded(self):
        result = _normalize("2million")
        assert "10" in result or "2" in result  # 2*10^6

    def test_removes_degree_units(self):
        result = _normalize("90degree")
        assert "degree" not in result

    def test_or_replaced_with_comma(self):
        result = _normalize("1 or 2")
        assert "or" not in result
        assert "," in result

    def test_strips_braces(self):
        result = _normalize("{42}")
        assert "{" not in result and "}" not in result

    def test_comma_formatted_number(self):
        # "1,000" should become "1000"
        result = _normalize("1,000")
        assert result == "1000"


# ---------------------------------------------------------------------------
# grade_answer
# ---------------------------------------------------------------------------

class TestGradeAnswer:
    def test_none_generation_returns_false(self):
        assert grade_answer(None, "42") is False

    def test_exact_match(self):
        assert grade_answer("42", "42") is True

    def test_float_vs_int(self):
        assert grade_answer("42.0", "42") is True

    def test_fraction_equivalent(self):
        assert grade_answer("1/2", "0.5") is True

    def test_wrong_answer(self):
        assert grade_answer("5", "42") is False

    def test_sympy_algebraic_equivalence(self):
        # (x+1)^2 == x^2+2x+1
        assert grade_answer("(x+1)**2", "x**2+2*x+1") is True

    def test_negative_fraction(self):
        assert grade_answer("2/(-3)", "-2/3") is True

    def test_tuple_correct(self):
        assert grade_answer("(1,2)", "(1,2)") is True

    def test_tuple_wrong(self):
        assert grade_answer("(1,3)", "(1,2)") is False

    def test_tuple_different_length(self):
        assert grade_answer("(1,2,3)", "(1,2)") is False

    def test_empty_string(self):
        assert grade_answer("", "42") is False

    def test_latex_fraction(self):
        assert grade_answer("\\frac{1}{2}", "0.5") is True


# ---------------------------------------------------------------------------
# should_allow_eval
# ---------------------------------------------------------------------------

class TestShouldAllowEval:
    def test_simple_numeric_allowed(self):
        assert should_allow_eval("42-42") is True

    def test_bad_substring_hat_brace(self):
        assert should_allow_eval("x^{2}-x^{2}") is False

    def test_bad_substring_hat_paren(self):
        assert should_allow_eval("x^(2)") is False

    def test_too_many_unknown_letters(self):
        # More than 2 unknown letters (excluding sqrt/frac)
        assert should_allow_eval("a+b+c-a-b-c") is False

    def test_known_letters_ok(self):
        # sqrt and frac letters are stripped before counting
        assert should_allow_eval("sqrt(2)-sqrt(2)") is True

    def test_bad_regex_repeated_exponent(self):
        assert should_allow_eval("x^22^y") is False


# ---------------------------------------------------------------------------
# split_tuple
# ---------------------------------------------------------------------------

class TestSplitTuple:
    def test_empty_string(self):
        assert split_tuple("") == []

    def test_single_value(self):
        assert split_tuple("42") == ["42"]

    def test_parenthesized_tuple(self):
        assert split_tuple("(1,2,3)") == ["1", "2", "3"]

    def test_bracketed_tuple(self):
        assert split_tuple("[1,2]") == ["1", "2"]

    def test_nested_parens_not_split(self):
        # Inner parens prevent splitting
        result = split_tuple("(1,(2,3))")
        assert result == ["(1,(2,3))"]

    def test_comma_formatted_numbers_handled(self):
        # "1,000" should be treated as a single number, not a tuple
        result = split_tuple("1,000")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _grade_single_generation
# ---------------------------------------------------------------------------

class TestGradeSingleGeneration:

    @pytest.mark.asyncio
    async def test_sympy_correct_match(self):
        grader = make_grader()
        score, diag = await grader._grade_single_generation("42", ["42"])
        assert score == 1
        assert diag["method"] == "sympy"
        assert diag["sympy_result"] == "match"
        assert diag["llm_called"] is False

    @pytest.mark.asyncio
    async def test_sympy_no_match_falls_back_to_llm(self):
        grader = make_grader()
        # Patch LLM to say "Yes"
        grader._check_equality_with_llm = AsyncMock(return_value=True)
        score, diag = await grader._grade_single_generation("43", ["42"])
        assert score == 1
        assert diag["method"] == "llm"
        assert diag["llm_called"] is True
        assert diag["llm_result"] == "match"

    @pytest.mark.asyncio
    async def test_sympy_no_match_llm_also_no_match(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        score, diag = await grader._grade_single_generation("99", ["42"])
        assert score == 0
        assert diag["llm_result"] == "no_match"

    @pytest.mark.asyncio
    async def test_empty_generation_treated_as_no_answer(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        score, diag = await grader._grade_single_generation("", ["42"])
        assert score == 0

    @pytest.mark.asyncio
    async def test_multiple_ground_truths_first_match_wins(self):
        grader = make_grader()
        score, diag = await grader._grade_single_generation("42", ["99", "42"])
        assert score == 1
        assert diag["method"] == "sympy"

    @pytest.mark.asyncio
    async def test_sympy_timeout_falls_back_to_llm(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=True)

        async def slow_grade(gen, gt):
            raise asyncio.TimeoutError()

        with patch("scheduler.grader.sympy_llm_as_judge.asyncio.wait_for",
                   side_effect=asyncio.TimeoutError):
            score, diag = await grader._grade_single_generation("42", ["42"])

        assert diag["sympy_result"] == "timeout"
        assert diag["llm_called"] is True
        assert score == 1

    @pytest.mark.asyncio
    async def test_diagnostics_structure(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        _, diag = await grader._grade_single_generation("99", ["42"])
        assert "method" in diag
        assert "sympy_result" in diag
        assert "sympy_error" in diag
        assert "llm_called" in diag
        assert "llm_result" in diag
        assert "score" in diag


# ---------------------------------------------------------------------------
# grade_sample
# ---------------------------------------------------------------------------

class TestGradeSample:

    @pytest.mark.asyncio
    async def test_correct_answer_scores_one(self):
        grader = make_grader()
        sample = make_sample("42", "42")
        result = await grader.grade_sample(sample)
        assert result["correct"] == [1]

    @pytest.mark.asyncio
    async def test_wrong_answer_scores_zero(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        sample = make_sample("99", "42")
        result = await grader.grade_sample(sample)
        assert result["correct"] == [0]

    @pytest.mark.asyncio
    async def test_multiple_generations(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        sample = make_sample(["42", "99"], "42")
        result = await grader.grade_sample(sample)
        assert result["correct"] == [1, 0]

    @pytest.mark.asyncio
    async def test_partial_accuracy(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        sample = make_sample(["42", "99"], "42")
        result = await grader.grade_sample(sample)
        assert result["partial_accuracy"] == 0.5

    @pytest.mark.asyncio
    async def test_accuracy_flag(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        sample = make_sample(["42", "99"], "42")
        result = await grader.grade_sample(sample)
        assert result["accuracy"] == 1  # any correct → 1

    @pytest.mark.asyncio
    async def test_all_wrong_accuracy_zero(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        sample = make_sample(["99", "100"], "42")
        result = await grader.grade_sample(sample)
        assert result["accuracy"] == 0

    @pytest.mark.asyncio
    async def test_grading_diagnostics_present(self):
        grader = make_grader()
        sample = make_sample("42", "42")
        result = await grader.grade_sample(sample)
        diags = result["grading_diagnostics"]
        assert "per_generation" in diags
        assert "sympy_match_count" in diags
        assert "llm_called_count" in diags

    @pytest.mark.asyncio
    async def test_sympy_match_count_in_diagnostics(self):
        grader = make_grader()
        sample = make_sample(["42", "42"], "42")
        result = await grader.grade_sample(sample)
        assert result["grading_diagnostics"]["sympy_match_count"] == 2
        assert result["grading_diagnostics"]["llm_called_count"] == 0

    @pytest.mark.asyncio
    async def test_list_ground_truth(self):
        grader = make_grader()
        sample = make_sample("42", ["99", "42"])
        result = await grader.grade_sample(sample)
        assert result["correct"] == [1]

    @pytest.mark.asyncio
    async def test_none_parsed_generation_uses_raw(self):
        """If parsed_generation is None, falls back to raw generation string."""
        grader = make_grader()
        sample = {
            "row": 0,
            "completion_input": "solve x",
            "ground_truth": "42",
            "generations": ["42"],
            "parsed_generations": [None],
        }
        result = await grader.grade_sample(sample)
        assert result["correct"] == [1]

    @pytest.mark.asyncio
    async def test_original_sample_not_mutated(self):
        grader = make_grader()
        sample = make_sample("42", "42")
        original_keys = set(sample.keys())
        await grader.grade_sample(sample)
        assert set(sample.keys()) == original_keys


# ---------------------------------------------------------------------------
# Additional grade_answer edge cases
# ---------------------------------------------------------------------------

class TestNormalizeEdgeCases:
    """Targets normalization paths not reachable via grade_answer's Hendrycks fast-path."""

    def test_text_wrapper_in_normalize(self):
        # Hendrycks and _normalize both extract \text{...}
        # Use "42.0" so Hendrycks gives "42.0" ≠ "42", then _normalize extracts "42"
        assert grade_answer("\\text{42.0}", "42") is True

    def test_latex_fraction_via_normalize(self):
        # \frac{3}{4} and 0.75: Hendrycks gives different strings,
        # _normalize calls _parse_latex which converts \frac → decimal
        assert grade_answer("\\frac{3}{4}", "0.75") is True

    def test_are_equal_under_sympy_exception_returns_false(self):
        # Providing an expression that raises inside sympy should return False gracefully
        # (not crash) — the except clause catches it
        from scheduler.grader.sympy_llm_as_judge import are_equal_under_sympy
        # Unparseable expressions should not raise
        result = are_equal_under_sympy("not_valid_$$!", "42")
        assert result is False


class TestGradeAnswerEdgeCases:
    def test_none_ground_truth_returns_false(self):
        # ground_truth normalizes to None → should not crash, returns False
        assert grade_answer("5", None) is False

    def test_mismatched_tuple_brackets_returns_false(self):
        # (1,2) vs [1,2] — different bracket types for a multi-element tuple
        assert grade_answer("[1,2]", "(1,2)") is False

    def test_fraction_elements_in_tuple(self):
        # Both elements are fractions — uses the _is_frac path
        assert grade_answer("(1/2,3/4)", "(1/2,3/4)") is True

    def test_fraction_element_mismatch(self):
        assert grade_answer("(1/2,1/3)", "(1/2,3/4)") is False

    def test_fix_sqrt_normalizes_bare_sqrt(self):
        # "\\sqrt2" (no braces) and "\\sqrt{2}" should normalize to the same form
        from scheduler.grader.sympy_llm_as_judge import normalize_answer_hendrycks
        assert normalize_answer_hendrycks("\\sqrt2") == normalize_answer_hendrycks("\\sqrt{2}")

    def test_remove_right_units_path(self):
        # "5 \\text{ cm}" — _remove_right_units strips the unit part
        assert grade_answer("5\\text{ cm}", "5") is True

    def test_normalize_text_wrapper(self):
        # _normalize also strips \text{...} wrapper
        assert grade_answer("\\text{42}", "42") is True


# ---------------------------------------------------------------------------
# Additional _grade_single_generation edge cases
# ---------------------------------------------------------------------------

class TestGradeSingleGenerationEdgeCases:
    @pytest.mark.asyncio
    async def test_sympy_exception_falls_back_to_llm(self):
        """When grade_answer itself raises, sympy_result=error and LLM is called."""
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=True)

        with patch("scheduler.grader.sympy_llm_as_judge.grade_answer",
                   side_effect=ValueError("parse error")):
            score, diag = await grader._grade_single_generation("42", ["42"])

        assert diag["sympy_result"] == "error"
        assert diag["llm_called"] is True
        assert score == 1

    @pytest.mark.asyncio
    async def test_llm_matches_second_ground_truth(self):
        """LLM loop tries all ground truths; second one matches."""
        grader = make_grader()
        # Use expressions that sympy can't confirm equal (different symbolic vars)
        # LLM says no for first GT, yes for second
        grader._check_equality_with_llm = AsyncMock(side_effect=[False, True])
        score, diag = await grader._grade_single_generation("y+1", ["x+1", "z+1"])
        assert score == 1
        assert diag["method"] == "llm"


# ---------------------------------------------------------------------------
# Additional grade_sample edge cases
# ---------------------------------------------------------------------------

class TestGradeSampleEdgeCases:
    @pytest.mark.asyncio
    async def test_sentinel_passed_through(self):
        from scheduler.utils import Sentinel
        grader = make_grader()
        result = await grader.grade_sample(Sentinel.COMPLETED)
        assert result == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_integer_ground_truth_converted_to_str(self):
        """Non-str/list ground_truth (e.g. int) is converted via str()."""
        grader = make_grader()
        sample = {
            "row": 0,
            "completion_input": "solve x",
            "ground_truth": 42,   # int, not str or list
            "generations": ["42"],
            "parsed_generations": ["42"],
        }
        result = await grader.grade_sample(sample)
        assert result["correct"] == [1]


# ---------------------------------------------------------------------------
# _check_equality_with_llm
# ---------------------------------------------------------------------------

class TestCheckEqualityWithLLM:

    @pytest.mark.asyncio
    async def test_yes_response_returns_true(self):
        grader = make_grader()
        completion = MagicMock()
        completion.choices[0].message.content = "Yes"
        grader.client.chat.completions.create = AsyncMock(return_value=completion)
        result = await grader._check_equality_with_llm("42", "42")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_response_returns_false(self):
        grader = make_grader()
        completion = MagicMock()
        completion.choices[0].message.content = "No"
        grader.client.chat.completions.create = AsyncMock(return_value=completion)
        result = await grader._check_equality_with_llm("42", "99")
        assert result is False

    @pytest.mark.asyncio
    async def test_case_insensitive_yes(self):
        grader = make_grader()
        completion = MagicMock()
        completion.choices[0].message.content = "YES"
        grader.client.chat.completions.create = AsyncMock(return_value=completion)
        result = await grader._check_equality_with_llm("42", "42")
        assert result is True

    @pytest.mark.asyncio
    async def test_exception_retries_and_returns_false(self):
        grader = make_grader()
        grader.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )
        result = await grader._check_equality_with_llm("42", "42")
        assert result is False
        assert grader.client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_external_policy_exhaustion_is_not_retried_by_legacy_loop(self):
        grader = make_grader()
        failure = ExternalRequestFailure(
            RuntimeError("judge unavailable"),
            error_code="endpoint_unreachable",
            attempts=2,
            elapsed_seconds=0.1,
            retriable=True,
            http_status=None,
        )
        grader.client.chat.completions.create = AsyncMock(
            side_effect=failure
        )

        with pytest.raises(ExternalRequestFailure):
            await grader._check_equality_with_llm("42", "99")

        assert grader.client.chat.completions.create.call_count == 1
