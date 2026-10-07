import copy
from typing import Any, Union, Callable, Optional
import logging
from ..utils import Sentinel
from ..cache_salt import request_kwargs_with_cache_salt
from .base import AccuracyGraderBase
from ..metrics import get_accuracy, get_bootstrap_accuracy_std
import asyncio
import yaml

from .registry import register

logger = logging.getLogger(__name__)


# The opening sentences follow LangChain's QA eval prompt (langchain-ai/langchain,
# MIT, Copyright (c) LangChain, Inc.). See THIRD_PARTY_NOTICES.md.
GRADER_TEMPLATE = """
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either       CORRECT or INCORRECT.

Example Format:
QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

Output:
{{"GRADE": "CORRECT" or "INCORRECT"}}
"""

@register("boxed-llm-as-judge", "llm_as_judge")
class LLMasJudgeBoxedMatch(AccuracyGraderBase):
    # Simple example, but less efficient
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llm_as_judge"
        self.model = self.task.grader.llm_as_judge
        self.openai_connection = self.request_openai_connection(self.model)

    def create_judge_messages(self, sample, generation):
        message = GRADER_TEMPLATE.format(query=sample["completion_input"],
                                         result=generation,
                                         answer=sample["ground_truth"])
        return [
            {
                "role": "system",
                "content": "You are a teacher grading a quiz, and output JSON.",
            },
            {
                "role": "user",
                "content": message
            }
        ]

    def parse_judge_response(self, response):
        try:
            think_close = response.lower().rfind("</think>")
            if think_close != -1:
                response = response[think_close + len("</think>"):]
            response = response.strip()
            grader_response = yaml.safe_load(response)
            if not isinstance(grader_response, dict):
                return 0
            grade = None
            if "GRADE" in grader_response:
                grade = grader_response["GRADE"]
            elif "grade" in grader_response:
                grade = grader_response["grade"]
            elif "answer" in grader_response:
                answer = grader_response["answer"]
                if isinstance(answer, dict):
                    grade = answer.get("GRADE", answer.get("grade"))
            if not isinstance(grade, str):
                return 0
            return 1 if grade.lower() == "correct" else 0
        except (AttributeError, yaml.YAMLError, TypeError):
            return 0

    async def grade_sample(self, sample: Any, *_):
        # if the sample is the sentinel, we have reached the end of the data
        result = copy.deepcopy(sample)
        result["correct"] = []

        client = await self.openai_connection.get_client()
        judge_response_tasks = []
        request_kwargs_with_cache_salt(self.model.openai_kwargs, self.model.cache_salt)
        async with asyncio.TaskGroup() as tg:
            for generation in sample["parsed_generations"]:
                if not generation:
                    generation = "No answer"
                judge_messages = self.create_judge_messages(sample, generation)
                request_kwargs = request_kwargs_with_cache_salt(
                    self.model.openai_kwargs,
                    self.model.cache_salt,
                )
                judge_response_tasks.append(tg.create_task(client.chat.completions.create(
                    model=self.model.api_model_name or self.model.name,
                    messages=judge_messages,
                    **request_kwargs)))
        for task in judge_response_tasks:
            completion = task.result()
            result["correct"].append(self.parse_judge_response(completion.choices[0].message.content))


        result["accuracy"] = sum(result["correct"])/len(result["generations"])
        return result


@register("llm_as_judge_lcr")
class LLMasJudgeLCRMatch(LLMasJudgeBoxedMatch):

    def create_judge_messages(self, sample, generation):
        query: str = sample["completion_input"]
        if "\n=== QUESTION ===\n" in query:
            query = query.split("\n=== QUESTION ===\n")[-1]
        if query.endswith("\n\nAnswer:"):
            query = query[:-len("\n\nAnswer:")]
        message = GRADER_TEMPLATE.format(query=query.strip(" \n"),
                                         result=generation,
                                         answer=sample["ground_truth"])
        return [
            {
                "role": "system",
                "content": "You are a teacher grading a quiz, and output JSON.",
            },
            {
                "role": "user",
                "content": message
            }
        ]
