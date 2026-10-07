import copy
import asyncio
import logging
from typing import Any

from ..utils import Sentinel
from .base import AccuracyGraderBase
from ..cache_salt import request_kwargs_with_cache_salt
from ..external_requests import ExternalRequestFailure
from ..metrics import get_accuracy, get_bootstrap_accuracy_std
from .registry import register
from .math import parse_answer_with_verify, compare_answers

logger = logging.getLogger(__name__)

# Used when both ground truth and extracted answer are short expressions.
# EQUALITY_TEMPLATE is copied from openai/simple-evals (common.py; MIT,
# Copyright (c) 2024 OpenAI). See THIRD_PARTY_NOTICES.md.
EQUALITY_TEMPLATE = r"""Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s"""

# Used when the student response is a full solution (prose), not a short expression.
SOLUTION_GRADING_TEMPLATE = r"""You are grading a math problem. The correct answer is:
%(ground_truth)s

The final portion of the student's solution is:
%(solution)s

Determine whether the student arrived at the correct answer. Guidelines:
- Focus on the student's FINAL stated answer (the last thing they conclude)
- Accept equivalent forms: e.g., 9,901 = 9901, CDXL = 440 (Roman numerals), different spacing/punctuation
- Accept decimal approximations that equal the exact answer (e.g., 0.8667 ≈ 13/15)
- Give benefit of the doubt on units (e.g., "30 meters" matches 30)

Did the student arrive at the correct answer? Respond with only "Yes" or "No" (without quotes). Do not include a rationale."""

# If generation exceeds this length (chars), treat it as a full solution and use
# SOLUTION_GRADING_TEMPLATE instead of EQUALITY_TEMPLATE.
_SOLUTION_LENGTH_THRESHOLD = 200

# When passing a long solution to the judge, only use the final portion.
# This reduces noise from problem restatement and focuses on the conclusion.
_SOLUTION_TAIL_CHARS = 800


@register("math-verify-llm-as-judge", "math_verify_llm_as_judge")
class MathVerifyLLMasJudge(AccuracyGraderBase):
    """
    Hybrid math grader: tries math_verify first, falls back to LLM equality
    check when math_verify cannot confirm correctness.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.task.grader.llm_as_judge
        self.openai_connection = self.request_openai_connection(self.model)

    def _grading_generator(
        self,
        start: int = 0,
        skip_rows: set[int] | None = None,
    ):
        return self.async_grade_all_samples_nonblocking(
            start=start,
            grade_fn=self.grade_sample,
            skip_rows=skip_rows,
        )

    async def _check_equality_with_llm(self, expr1: str, expr2: str) -> bool:
        if len(expr2) > _SOLUTION_LENGTH_THRESHOLD:
            tail = expr2[-_SOLUTION_TAIL_CHARS:]
            prompt = SOLUTION_GRADING_TEMPLATE % {"ground_truth": expr1, "solution": tail}
        else:
            prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
        messages = [{"role": "user", "content": prompt}]
        client = await self.openai_connection.get_client()
        for _ in range(3):
            try:
                request_kwargs = request_kwargs_with_cache_salt(
                    self.model.openai_kwargs,
                    self.model.cache_salt,
                )
                completion = await client.chat.completions.create(
                    model=self.model.api_model_name or self.model.name,
                    messages=messages,
                    **request_kwargs,
                )
                response = completion.choices[0].message.content
                # Strip thinking tokens if present
                if "</think>" in response:
                    response = response[response.rfind("</think>") + len("</think>"):]
                if response and len(response.strip()) > 0:
                    return "yes" in response.lower().strip()
            except ValueError:
                raise
            except ExternalRequestFailure:
                # The shared external policy has already exhausted its full
                # attempt budget; the legacy loop must not multiply it.
                raise
            except Exception as e:
                logger.warning("LLM equality check failed: %s", e)
                continue
        return False

    async def _grade_single_generation(
        self, generation: str, ground_truths: list[str], raw_generation: str | None = None
    ) -> tuple[int, dict]:
        diag = {
            "method": "none",
            "math_verify_result": None,
            "llm_called": False,
            "llm_result": None,
            "score": 0,
        }

        if not generation:
            generation = "No answer"

        # Step 1: Try math_verify
        for gt in ground_truths:
            try:
                pred_parsed = parse_answer_with_verify(generation)
                if pred_parsed and compare_answers(pred_parsed, gt):
                    diag.update(method="math_verify", math_verify_result="match", score=1)
                    return 1, diag
            except Exception as e:
                logger.warning("math_verify error: %s", e)

        diag["math_verify_result"] = "no_match"

        # Step 2: LLM fallback.
        # When a parser extracted a short expression (e.g. boxed parser), try that
        # first.  If it fails and the raw generation differs (i.e. a parser was
        # active), also try the full raw generation: the real answer may be in the
        # prose but not in the extracted expression.
        llm_candidates: list[str] = [generation]
        if raw_generation is not None and raw_generation != generation:
            llm_candidates.append(raw_generation)

        diag["llm_called"] = True
        for llm_expr in llm_candidates:
            for gt in ground_truths:
                if await self._check_equality_with_llm(gt, llm_expr):
                    diag.update(method="llm", llm_result="match", score=1)
                    return 1, diag

        diag["llm_result"] = "no_match"
        return 0, diag

    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict)
        assert "parsed_generations" in sample
        assert "ground_truth" in sample

        ground_truth = sample["ground_truth"]
        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        elif isinstance(ground_truth, (list, tuple)):
            ground_truths = list(ground_truth)
        else:
            ground_truths = [str(ground_truth)]

        generations = []
        for i, gen in enumerate(sample["parsed_generations"]):
            generations.append(gen if gen is not None else sample["generations"][i])

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for i, gen in enumerate(generations):
                tasks.append(
                    tg.create_task(self._grade_single_generation(
                        gen, ground_truths, raw_generation=sample["generations"][i]
                    ))
                )

        result = copy.deepcopy(sample)
        scores = [t.result()[0] for t in tasks]
        per_gen_diags = [t.result()[1] for t in tasks]

        result["correct"] = scores
        result["partial_accuracy"] = sum(scores) / len(scores)
        result["accuracy"] = 1 if any(scores) else 0
        result["grading_diagnostics"] = {
            "per_generation": per_gen_diags,
            "math_verify_match_count": sum(1 for d in per_gen_diags if d["math_verify_result"] == "match"),
            "math_verify_no_match_count": sum(1 for d in per_gen_diags if d["math_verify_result"] == "no_match"),
            "llm_called_count": sum(1 for d in per_gen_diags if d["llm_called"]),
            "llm_match_count": sum(1 for d in per_gen_diags if d["llm_result"] == "match"),
            "llm_no_match_count": sum(1 for d in per_gen_diags if d["llm_result"] == "no_match"),
        }
        return result
