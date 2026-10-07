"""Artificial Analysis LCR grader with the official binary judge protocol."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from ..cache_salt import request_kwargs_with_cache_salt
from .base import AccuracyGraderBase
from .registry import register


OFFICIAL_JUDGE_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT.

"""


@register("lcr_aa")
class LCRAAJudge(AccuracyGraderBase):
    """Grade LCR-AA final answers with strict CORRECT/INCORRECT parsing.

    The shared AA runner treats an empty final answer as incorrect without
    querying the judge. Eval360 currently falls back to separately returned
    reasoning when ``message.content`` is empty, so this grader also detects
    that substituted value and applies the same deterministic policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "lcr_aa"
        self.model = self.task.grader.llm_as_judge
        self.openai_connection = self.request_openai_connection(self.model)

    @staticmethod
    def create_judge_messages(
        question: str,
        official_answer: list[str],
        candidate_answer: str,
    ) -> list[dict[str, str]]:
        prompt = OFFICIAL_JUDGE_TEMPLATE.format(
            question=question,
            official_answer=str(official_answer),
            candidate_answer=candidate_answer,
        )
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def parse_judge_response(response: Any) -> tuple[bool, str | None]:
        if not isinstance(response, str):
            return False, None
        normalized = response.strip().upper()
        if normalized == "CORRECT":
            return True, "CORRECT"
        if normalized == "INCORRECT":
            return False, "INCORRECT"
        return False, None

    @staticmethod
    def _parallel_flag(sample: dict[str, Any], key: str, index: int) -> bool:
        values = sample.get(key)
        return (
            isinstance(values, list)
            and index < len(values)
            and values[index] is True
        )

    @classmethod
    def deterministic_incorrect_reason(
        cls,
        sample: dict[str, Any],
        index: int,
        candidate_answer: Any,
    ) -> str | None:
        if not isinstance(candidate_answer, str) or not candidate_answer.strip():
            return "empty_final"

        if cls._parallel_flag(sample, "parsed_unclosed_think_tag", index):
            return "unclosed_think_tag"

        raw_generations = sample.get("generations")
        if isinstance(raw_generations, list) and index < len(raw_generations):
            raw_generation = raw_generations[index]
            reasoning_values = sample.get("reasoning")
            if (
                isinstance(raw_generation, str)
                and isinstance(reasoning_values, list)
                and raw_generation in reasoning_values
            ):
                return "reasoning_substituted_for_empty_final"

        return None

    @staticmethod
    def _official_answer(sample: dict[str, Any]) -> list[str]:
        value = sample.get("ground_truth")
        if not isinstance(value, list) or not all(
            isinstance(answer, str) for answer in value
        ):
            raise ValueError("LCR-AA ground_truth must be a list of strings")
        return value

    async def grade_sample(self, sample: Any, *_):
        result = copy.deepcopy(sample)
        result["correct"] = [False] * len(sample["parsed_generations"])
        result["lcr_aa_grading_source"] = [None] * len(sample["parsed_generations"])
        result["lcr_aa_judge_label"] = [None] * len(sample["parsed_generations"])
        result["lcr_aa_judge_response"] = [None] * len(sample["parsed_generations"])

        official_answer = self._official_answer(sample)
        question = sample.get("question")
        if not isinstance(question, str) or not question:
            raise ValueError("LCR-AA row is missing a non-empty question")

        pending: list[tuple[int, asyncio.Task]] = []
        client = None
        async with asyncio.TaskGroup() as task_group:
            for index, candidate_answer in enumerate(sample["parsed_generations"]):
                deterministic_reason = self.deterministic_incorrect_reason(
                    sample, index, candidate_answer
                )
                if deterministic_reason is not None:
                    result["lcr_aa_grading_source"][index] = deterministic_reason
                    result["lcr_aa_judge_label"][index] = "INCORRECT"
                    continue

                if client is None:
                    client = await self.openai_connection.get_client()
                messages = self.create_judge_messages(
                    question, official_answer, candidate_answer
                )
                request_kwargs = request_kwargs_with_cache_salt(
                    self.model.openai_kwargs,
                    self.model.cache_salt,
                )
                task = task_group.create_task(
                    client.chat.completions.create(
                        model=self.model.api_model_name or self.model.name,
                        messages=messages,
                        **request_kwargs,
                    )
                )
                pending.append((index, task))

        for index, task in pending:
            response = task.result().choices[0].message.content
            correct, label = self.parse_judge_response(response)
            result["correct"][index] = correct
            result["lcr_aa_grading_source"][index] = "judge"
            result["lcr_aa_judge_label"][index] = label
            result["lcr_aa_judge_response"][index] = response

        result["accuracy"] = sum(result["correct"]) / len(result["correct"])
        return result
