import copy
import logging
from typing import Any, Dict, AsyncIterator

from ..utils import Sentinel, ExceptionWrapper
from .base import GraderBase, Grade, Score
from .registry import register
from .ifeval_lib.evaluation_lib import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
)

logger = logging.getLogger(__name__)


@register("ifeval", "if_eval")
class IFEval(GraderBase):
    async def run(
        self,
        existing: AsyncIterator[Dict[str, Any]],
        average_over: list[int],
        pass_at: list[int],
    ):
        all_strict_prompt = []
        all_strict_inst = []
        all_loose_prompt = []
        all_loose_inst = []
        count = 0

        async for result in existing:
            if "correct" in result:
                all_strict_prompt.append(result["strict_prompt_follow"])
                all_strict_inst.extend(result["strict_inst_follow"])
                all_loose_prompt.append(result["loose_prompt_follow"])
                all_loose_inst.extend(result["loose_inst_follow"])
            count += 1

        completed = False
        async for result in self.async_grade_all_samples(
            start=count, grade_fn=self.grade_sample
        ):
            if result == Sentinel.COMPLETED:
                completed = True
                break
            elif isinstance(result, ExceptionWrapper):
                yield result
            elif isinstance(result, dict) and "correct" in result:
                all_strict_prompt.append(result["strict_prompt_follow"])
                all_strict_inst.extend(result["strict_inst_follow"])
                all_loose_prompt.append(result["loose_prompt_follow"])
                all_loose_inst.extend(result["loose_inst_follow"])
                yield Grade(element=result)

        if not completed:
            return

        n = len(all_strict_prompt)
        if n == 0:
            yield Score(name="strict_prompt_accuracy", value=float("nan"))
            yield Score(name="strict_instruction_accuracy", value=float("nan"))
            yield Score(name="loose_prompt_accuracy", value=float("nan"))
            yield Score(name="loose_instruction_accuracy", value=float("nan"))
        else:
            yield Score(
                name="strict_prompt_accuracy",
                value=sum(all_strict_prompt) / n,
            )
            yield Score(
                name="strict_instruction_accuracy",
                value=sum(all_strict_inst) / len(all_strict_inst) if all_strict_inst else float("nan"),
            )
            yield Score(
                name="loose_prompt_accuracy",
                value=sum(all_loose_prompt) / n,
            )
            yield Score(
                name="loose_instruction_accuracy",
                value=sum(all_loose_inst) / len(all_loose_inst) if all_loose_inst else float("nan"),
            )
        yield Sentinel.COMPLETED

    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        gt = sample["ground_truth"]
        instruction_id_list = gt["instruction_id_list"]
        kwargs = gt["kwargs"]

        generations = []
        for idx, generation in enumerate(sample["parsed_generations"]):
            if generation is None:
                generations.append(sample["generations"][idx])
            else:
                generations.append(generation)

        response = generations[0]

        inp = InputExample(
            key=sample["row"],
            instruction_id_list=instruction_id_list,
            prompt=sample["completion_input"],
            kwargs=kwargs,
        )
        prompt_to_response = {inp.prompt: response}

        strict_out = test_instruction_following_strict(inp, prompt_to_response)
        loose_out = test_instruction_following_loose(inp, prompt_to_response)

        result = copy.deepcopy(sample)
        result["strict_prompt_follow"] = strict_out.follow_all_instructions
        result["strict_inst_follow"] = strict_out.follow_instruction_list
        result["loose_prompt_follow"] = loose_out.follow_all_instructions
        result["loose_inst_follow"] = loose_out.follow_instruction_list
        # "correct" field required by Grade model
        result["correct"] = [1 if strict_out.follow_all_instructions else 0]
        return result
