"""HLE grader — dispatches to multiple-choice or LLM-judge exact-match scoring based on answer_type."""

import copy
import json
import logging
import re
from typing import Any

from ..metrics import get_accuracy
from ..utils import Sentinel
from .llm_as_judge import LLMasJudgeBoxedMatch
from .registry import register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt — adapted from the official HLE evaluation (centerforaisafety/hle,
# hle_eval/run_judge_results.py; MIT). Lightly normalized: upstream's "0|\%|" and
# "100|\%|" are written "0%" and "100%", and one doubled blank line is single.
# See THIRD_PARTY_NOTICES.md.
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


# ---------------------------------------------------------------------------
# Multiple-choice helpers
# ---------------------------------------------------------------------------

def _extract_mc_answer(generation: str | None) -> str | None:
    if not generation:
        return None
    text = generation.strip()
    # Prefer the letter immediately following "Answer:" (we prompt the model this way, see build_hle.py)
    answer_match = re.search(r"Answer:\s*([A-Za-z])", text, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    # Fall back to the first letter in the text
    match = re.search(r"[A-Za-z]", text)
    return match.group(0).upper() if match else None


def _check_multiple_choice(generations: list[str], expected: str) -> dict:
    assert isinstance(generations, list) and len(generations) > 0, "generations must be a non-empty list"
    assert isinstance(expected, str) and expected.strip(), "expected must be a non-empty string"
    expected_norm = expected.strip().upper()
    correct = []
    for gen in generations:
        choice = _extract_mc_answer(gen)
        correct.append(choice == expected_norm)
    return {
        "correct": correct,
        "accuracy": get_accuracy(correct),
    }


# ---------------------------------------------------------------------------
# Judge — overrides prompt and response parser from LLMasJudgeBoxedMatch
# ---------------------------------------------------------------------------

class HLEJudge(LLMasJudgeBoxedMatch):
    """LLM judge for HLE exactMatch questions."""

    # see the `grade_sample` function in `LLMasJudgeBoxedMatch` class
    async def grade_sample(self, sample: Any, *_):
        return await super().grade_sample(sample)

    def create_judge_messages(self, sample: dict, generation: str) -> list:
        prompt = JUDGE_PROMPT.format(
            question=sample["completion_input"],
            response=generation or "No answer",
            correct_answer=sample["ground_truth"],
        )
        return [{"role": "user", "content": prompt}]

    def parse_judge_response(self, response: str | None) -> bool:
        # match behavior of the base class; if the judge model does not return a verdict then assume the verdict is No
        # To Do: add a retry logic later in the `grade_sample` function of the base class
        if not response:
            logger.warning("Judge returned empty/None response")
            return False
        if "</think>" in response:
            response = response[response.rfind("</think>") + len("</think>"):]
        response = response.strip()

        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return str(parsed.get("correct", "")).strip().lower() == "yes"
            except json.JSONDecodeError:
                pass

        line_match = re.search(r"correct:\s*(yes|no)", response, flags=re.IGNORECASE)
        if line_match:
            return line_match.group(1).lower() == "yes"

        logger.warning("Could not parse judge response: %s", response[:200])
        return False


# ---------------------------------------------------------------------------
# Grader — dispatches to MC scoring or the inherited LLM judge
# ---------------------------------------------------------------------------

@register("hle")
class HLEGrader(HLEJudge):
    """Dispatches to _check_multiple_choice or the LLM judge based on answer_type."""

    def _grading_generator(self, start: int = 0, skip_rows: set[int] | None = None):
        # Use concurrent grading since exactMatch samples make async LLM calls
        return self.async_grade_all_samples_nonblocking(
            start=start, grade_fn=self.grade_sample, skip_rows=skip_rows
        )

    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict), "sample must be a dict"
        assert "parsed_generations" in sample
        assert "ground_truth" in sample
        assert "answer_type" in sample, "HLE records must have an 'answer_type' field"

        answer_type = sample["answer_type"]

        if answer_type == "multipleChoice":
            generations = [
                g if g is not None else sample["generations"][i]
                for i, g in enumerate(sample["parsed_generations"])
            ]
            new_fields = _check_multiple_choice(
                generations=generations,
                expected=sample["ground_truth"],
            )
            result = copy.deepcopy(sample)
            result.update(new_fields)
            return result

        elif answer_type == "exactMatch":
            if self.model is None:
                logger.warning("No llm_as_judge configured for HLE exactMatch — falling back to strict string match")
                generations = [
                    g if g is not None else sample["generations"][i]
                    for i, g in enumerate(sample["parsed_generations"])
                ]
                expected_norm = sample["ground_truth"].strip().lower()
                correct = [g.strip().lower() == expected_norm for g in generations]
                result = copy.deepcopy(sample)
                result.update({
                    "correct": correct,
                    "accuracy": get_accuracy(correct),
                })
                return result
            return await super().grade_sample(sample)

        else:
            raise ValueError(f"Unknown HLE answer_type: {answer_type!r}")
