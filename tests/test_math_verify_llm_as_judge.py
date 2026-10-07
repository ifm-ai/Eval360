"""
Tests for MathVerifyLLMasJudge and its helper functions.

Covers:
  - _grade_single_generation: math_verify path, LLM fallback, diagnostics
  - grade_sample: orchestration, diagnostics, edge cases
  - _check_equality_with_llm: response parsing, think-tag stripping, retries
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from scheduler.external_requests import ExternalRequestFailure
from scheduler.grader.base import Grade
from scheduler.grader.math_verify_llm_as_judge import MathVerifyLLMasJudge
from scheduler.model import CacheSaltConfig
from scheduler.utils import Sentinel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grader():
    """Return a MathVerifyLLMasJudge with all external dependencies mocked out."""
    grader = MathVerifyLLMasJudge.__new__(MathVerifyLLMasJudge)
    grader._initialized = True
    grader.client = AsyncMock()
    grader.openai_connection = AsyncMock()
    grader.openai_connection.get_client = AsyncMock(return_value=grader.client)
    grader.model = MagicMock()
    grader.model.name = "judge-model"
    grader.model.api_model_name = None
    grader.model.openai_kwargs = {}
    grader.model.cache_salt = CacheSaltConfig()
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


async def _stream(*items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_run_accepts_base_skip_rows_contract():
    event = MagicMock()
    event.parser_type = "noop"
    task = MagicMock()
    task.grader.llm_as_judge = MagicMock()
    persisted_sample = make_sample("42", "42")
    fresh_sample = make_sample("42", "42")
    fresh_sample["row"] = 1
    grader = MathVerifyLLMasJudge(
        samples_generator=_stream(
            persisted_sample,
            fresh_sample,
            Sentinel.COMPLETED,
        ),
        event_manager=MagicMock(),
        job_manager=MagicMock(),
        event=event,
        task=task,
    )

    results = [
        item
        async for item in grader.run(
            existing=_stream({"row": 0, "correct": [1]}),
            average_over=[1],
            pass_at=[1],
        )
    ]

    grades = [item for item in results if isinstance(item, Grade)]
    assert len(grades) == 1
    assert grades[0].element["row"] == 1
    assert grades[0].element["correct"] == [1]
    assert results[-1] == Sentinel.COMPLETED


# ---------------------------------------------------------------------------
# _grade_single_generation — math_verify path
# ---------------------------------------------------------------------------

class TestGradeSingleGenerationMathVerify:

    @pytest.mark.asyncio
    async def test_exact_integer_match(self):
        grader = make_grader()
        score, diag = await grader._grade_single_generation("72", ["72"])
        assert score == 1
        assert diag["method"] == "math_verify"
        assert diag["math_verify_result"] == "match"
        assert diag["llm_called"] is False

    @pytest.mark.asyncio
    async def test_latex_fraction_match(self):
        grader = make_grader()
        score, diag = await grader._grade_single_generation(r"\frac{3}{56}", [r"\frac{3}{56}"])
        assert score == 1
        assert diag["method"] == "math_verify"

    @pytest.mark.asyncio
    async def test_sqrt_match(self):
        grader = make_grader()
        score, diag = await grader._grade_single_generation(r"\sqrt{51}", [r"\sqrt{51}"])
        assert score == 1
        assert diag["method"] == "math_verify"

    @pytest.mark.asyncio
    async def test_wrong_answer_no_match(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        score, diag = await grader._grade_single_generation("99", ["72"])
        assert score == 0
        assert diag["math_verify_result"] == "no_match"
        assert diag["llm_called"] is True

    @pytest.mark.asyncio
    async def test_multiple_ground_truths_first_match_wins(self):
        grader = make_grader()
        score, diag = await grader._grade_single_generation("72", ["99", "72"])
        assert score == 1
        assert diag["method"] == "math_verify"

    @pytest.mark.asyncio
    async def test_empty_generation_becomes_no_answer(self):
        """Empty string is replaced with 'No answer' and graded accordingly."""
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        score, diag = await grader._grade_single_generation("", ["72"])
        assert score == 0


# ---------------------------------------------------------------------------
# _grade_single_generation — LLM fallback path
# ---------------------------------------------------------------------------

class TestGradeSingleGenerationLLMFallback:

    @pytest.mark.asyncio
    async def test_llm_matches_when_math_verify_fails(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=True)
        # "seventy-two" can't be parsed by math_verify
        score, diag = await grader._grade_single_generation("seventy-two", ["72"])
        assert score == 1
        assert diag["method"] == "llm"
        assert diag["llm_result"] == "match"
        assert diag["llm_called"] is True

    @pytest.mark.asyncio
    async def test_llm_no_match(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        score, diag = await grader._grade_single_generation("seventy-two", ["42"])
        assert score == 0
        assert diag["llm_result"] == "no_match"

    @pytest.mark.asyncio
    async def test_llm_tries_all_ground_truths(self):
        grader = make_grader()
        # First GT: no, second GT: yes
        grader._check_equality_with_llm = AsyncMock(side_effect=[False, True])
        score, diag = await grader._grade_single_generation("answer", ["wrong", "correct"])
        assert score == 1
        assert diag["method"] == "llm"

    @pytest.mark.asyncio
    async def test_raw_generation_fallback_when_parsed_fails(self):
        """When parsed expression fails LLM judge, raw generation is tried next."""
        grader = make_grader()
        calls = []
        async def mock_check(gt, expr):
            calls.append(expr)
            # parsed "10" fails; full prose succeeds
            return expr != "10"
        grader._check_equality_with_llm = mock_check
        score, diag = await grader._grade_single_generation(
            "10", ["42"], raw_generation="The answer is 42. For example, <answer> \\boxed{10}"
        )
        assert score == 1
        assert diag["method"] == "llm"
        assert "10" in calls
        assert "The answer is 42. For example, <answer> \\boxed{10}" in calls

    @pytest.mark.asyncio
    async def test_raw_generation_not_tried_when_same_as_parsed(self):
        """When raw_generation == generation (noop parser), LLM is called only once."""
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        full_gen = "some prose answer"
        await grader._grade_single_generation(full_gen, ["42"], raw_generation=full_gen)
        assert grader._check_equality_with_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_raw_generation_not_tried_when_none(self):
        """When raw_generation is None, LLM is called once (original behavior)."""
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        await grader._grade_single_generation("10", ["42"], raw_generation=None)
        assert grader._check_equality_with_llm.call_count == 1



# ---------------------------------------------------------------------------
# _grade_single_generation — diagnostics
# ---------------------------------------------------------------------------

class TestGradeSingleGenerationDiagnostics:

    @pytest.mark.asyncio
    async def test_diagnostics_keys_present(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        _, diag = await grader._grade_single_generation("99", ["42"])
        assert "method" in diag
        assert "math_verify_result" in diag
        assert "llm_called" in diag
        assert "llm_result" in diag
        assert "score" in diag

    @pytest.mark.asyncio
    async def test_math_verify_match_diag(self):
        grader = make_grader()
        _, diag = await grader._grade_single_generation("42", ["42"])
        assert diag["math_verify_result"] == "match"
        assert diag["llm_called"] is False
        assert diag["llm_result"] is None

    @pytest.mark.asyncio
    async def test_llm_fallback_diag(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=True)
        _, diag = await grader._grade_single_generation("prose answer", ["42"])
        assert diag["math_verify_result"] == "no_match"
        assert diag["llm_called"] is True
        assert diag["llm_result"] == "match"


# ---------------------------------------------------------------------------
# _check_equality_with_llm
# ---------------------------------------------------------------------------

class TestCheckEqualityWithLLM:

    @pytest.mark.asyncio
    async def test_yes_returns_true(self):
        grader = make_grader()
        completion = MagicMock()
        completion.choices[0].message.content = "Yes"
        grader.client.chat.completions.create = AsyncMock(return_value=completion)
        result = await grader._check_equality_with_llm("42", "42")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_returns_false(self):
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
    async def test_think_tag_stripped_before_parsing(self):
        """</think> tokens in response are stripped; only text after them is used."""
        grader = make_grader()
        completion = MagicMock()
        completion.choices[0].message.content = "<think>Let me compare</think>Yes"
        grader.client.chat.completions.create = AsyncMock(return_value=completion)
        result = await grader._check_equality_with_llm("42", "42")
        assert result is True

    @pytest.mark.asyncio
    async def test_think_tag_no_answer_after(self):
        """If </think> is followed by whitespace only, still evaluates correctly."""
        grader = make_grader()
        completion = MagicMock()
        completion.choices[0].message.content = "<think>reasoning</think>No"
        grader.client.chat.completions.create = AsyncMock(return_value=completion)
        result = await grader._check_equality_with_llm("42", "99")
        assert result is False

    @pytest.mark.asyncio
    async def test_long_solution_uses_solution_grading_template(self):
        """When generation exceeds threshold, SOLUTION_GRADING_TEMPLATE is used."""
        from scheduler.grader.math_verify_llm_as_judge import SOLUTION_GRADING_TEMPLATE, _SOLUTION_LENGTH_THRESHOLD
        grader = make_grader()
        long_solution = "x " * (_SOLUTION_LENGTH_THRESHOLD + 1)
        completion = MagicMock()
        completion.choices[0].message.content = "Yes"
        grader.client.chat.completions.create = AsyncMock(return_value=completion)
        result = await grader._check_equality_with_llm("42", long_solution)
        assert result is True
        call_args = grader.client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "student" in prompt.lower()
        assert "42" in prompt

    @pytest.mark.asyncio
    async def test_short_expression_uses_equality_template(self):
        """When generation is short, EQUALITY_TEMPLATE is used."""
        grader = make_grader()
        completion = MagicMock()
        completion.choices[0].message.content = "Yes"
        grader.client.chat.completions.create = AsyncMock(return_value=completion)
        result = await grader._check_equality_with_llm("42", "42")
        assert result is True
        call_args = grader.client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Expression 1" in prompt

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

    @pytest.mark.asyncio
    async def test_exception_then_success(self):
        """Succeeds on second retry after one failure."""
        grader = make_grader()
        completion = MagicMock()
        completion.choices[0].message.content = "Yes"
        grader.client.chat.completions.create = AsyncMock(
            side_effect=[Exception("fail"), completion]
        )
        result = await grader._check_equality_with_llm("42", "42")
        assert result is True


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
        assert result["accuracy"] == 1

    @pytest.mark.asyncio
    async def test_wrong_answer_scores_zero(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        sample = make_sample("99", "42")
        result = await grader.grade_sample(sample)
        assert result["correct"] == [0]
        assert result["accuracy"] == 0

    @pytest.mark.asyncio
    async def test_multiple_generations_partial(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        sample = make_sample(["42", "99"], "42")
        result = await grader.grade_sample(sample)
        assert result["correct"] == [1, 0]
        assert result["partial_accuracy"] == 0.5
        assert result["accuracy"] == 1

    @pytest.mark.asyncio
    async def test_all_wrong_accuracy_zero(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=False)
        sample = make_sample(["99", "100"], "42")
        result = await grader.grade_sample(sample)
        assert result["accuracy"] == 0

    @pytest.mark.asyncio
    async def test_list_ground_truth(self):
        grader = make_grader()
        sample = make_sample("42", ["99", "42"])
        result = await grader.grade_sample(sample)
        assert result["correct"] == [1]

    @pytest.mark.asyncio
    async def test_integer_ground_truth_converted(self):
        grader = make_grader()
        sample = {
            "row": 0,
            "completion_input": "solve",
            "ground_truth": 42,
            "generations": ["42"],
            "parsed_generations": ["42"],
        }
        result = await grader.grade_sample(sample)
        assert result["correct"] == [1]

    @pytest.mark.asyncio
    async def test_none_parsed_generation_falls_back_to_raw(self):
        grader = make_grader()
        sample = {
            "row": 0,
            "completion_input": "solve",
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

    @pytest.mark.asyncio
    async def test_sentinel_passed_through(self):
        grader = make_grader()
        result = await grader.grade_sample(Sentinel.COMPLETED)
        assert result == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_grading_diagnostics_structure(self):
        grader = make_grader()
        sample = make_sample("42", "42")
        result = await grader.grade_sample(sample)
        diags = result["grading_diagnostics"]
        assert "per_generation" in diags
        assert "math_verify_match_count" in diags
        assert "math_verify_no_match_count" in diags
        assert "llm_called_count" in diags
        assert "llm_match_count" in diags
        assert "llm_no_match_count" in diags

    @pytest.mark.asyncio
    async def test_math_verify_match_count(self):
        grader = make_grader()
        sample = make_sample(["42", "72"], ["42", "72"])
        result = await grader.grade_sample(sample)
        assert result["grading_diagnostics"]["math_verify_match_count"] == 2
        assert result["grading_diagnostics"]["llm_called_count"] == 0

    @pytest.mark.asyncio
    async def test_llm_called_count_when_math_verify_fails(self):
        grader = make_grader()
        grader._check_equality_with_llm = AsyncMock(return_value=True)
        sample = make_sample(["prose answer", "another prose"], "42")
        result = await grader.grade_sample(sample)
        assert result["grading_diagnostics"]["llm_called_count"] == 2
        assert result["grading_diagnostics"]["llm_match_count"] == 2


# ---------------------------------------------------------------------------
# Nonblocking grading generator
# ---------------------------------------------------------------------------

class TestNonblockingGradingGenerator:

    @pytest.mark.asyncio
    async def test_nonblocking_preserves_order(self):
        """Results are yielded in the original sample order, not completion order."""
        from scheduler.utils import Sentinel

        samples = [
            {**make_sample(str(i * 10), str(i * 10)), "row": i}
            for i in range(5)
        ]

        async def samples_gen():
            for s in samples:
                yield s
            yield Sentinel.COMPLETED

        grader = MathVerifyLLMasJudge.__new__(MathVerifyLLMasJudge)
        grader._initialized = True
        grader.client = AsyncMock()
        grader.model = MagicMock()
        grader.model.name = "judge-model"
        grader.model.openai_kwargs = {}
        grader.model.cache_salt = CacheSaltConfig()
        from scheduler.grader.parser_registry import get_parser
        grader._parser = get_parser("noop")
        grader.samples_generator = samples_gen()

        results = []
        async for result in grader._grading_generator(start=0):
            if result == Sentinel.COMPLETED:
                break
            results.append(result)

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r["row"] == i

    @pytest.mark.asyncio
    async def test_nonblocking_skips_already_graded(self):
        """start=N skips the first N samples."""
        from scheduler.utils import Sentinel

        samples = [make_sample(str(i * 10), str(i * 10)) for i in range(3)]
        for i, s in enumerate(samples):
            s["row"] = i

        async def samples_gen():
            for s in samples:
                yield s
            yield Sentinel.COMPLETED

        grader = MathVerifyLLMasJudge.__new__(MathVerifyLLMasJudge)
        grader._initialized = True
        grader.client = AsyncMock()
        grader.model = MagicMock()
        grader.model.name = "judge-model"
        grader.model.openai_kwargs = {}
        grader.model.cache_salt = CacheSaltConfig()
        from scheduler.grader.parser_registry import get_parser
        grader._parser = get_parser("noop")
        grader.samples_generator = samples_gen()

        results = []
        async for result in grader._grading_generator(start=2):
            if result == Sentinel.COMPLETED:
                break
            results.append(result)

        assert len(results) == 1
        assert results[0]["row"] == 2
