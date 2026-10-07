"""Grader for ORDER logic puzzle benchmark."""
import copy
import re
import logging
from collections import defaultdict
from typing import Any, AsyncIterator, Dict

from ..utils import Sentinel, ExceptionWrapper
from ..metrics import get_accuracy, get_bootstrap_accuracy_std, mean_pass_at_k
from .base import AccuracyGraderBase, Grade, Score
from .registry import register

logger = logging.getLogger(__name__)


def verify_order(solution_list: list, constraints: dict) -> bool:
    """Verify that a solution satisfies all ordering constraints."""
    n = constraints["n"]
    if len(solution_list) != n:
        return False
    if set(solution_list) != set(range(n)):
        return False

    order_dict = {item: day for day, item in enumerate(solution_list)}

    for con in constraints["not_sold"]:
        for item_str, day in con.items():
            item = int(item_str)
            if order_dict.get(item) == day:
                return False

    for con in constraints["sold_before"]:
        pivot = con["pivot"]
        items_before = con["sold_before_pivot"]
        for item in items_before:
            if not order_dict.get(item, n) < order_dict.get(pivot, -1):
                return False

    for con in constraints["sold_after"]:
        pivot = con["pivot"]
        items_after = con["sold_after_pivot"]
        for item in items_after:
            if not order_dict.get(pivot, n) < order_dict.get(item, -1):
                return False

    for con in constraints["sold_between"]:
        pivot = con["pivot"]
        before_item = con["sold_before_pivot"]
        after_item = con["sold_after_pivot"]
        cond1 = order_dict.get(before_item, n) < order_dict.get(pivot, -1)
        cond2 = order_dict.get(pivot, n) < order_dict.get(after_item, -1)
        if not (cond1 or cond2):
            return False

    return True


def parse_order_answer(text: str) -> list | None:
    """Parse comma-separated integer list from model output."""
    text = text.strip()
    try:
        return list(map(int, re.split(r'[,\s]+', text.strip())))
    except (ValueError, AttributeError):
        return None


@register("order-puzzle", "order")
class OrderPuzzle(AccuracyGraderBase):
    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict)
        assert "parsed_generations" in sample
        assert "ground_truth" in sample

        gt = sample["ground_truth"]
        constraints = gt["constraints"]

        correct = []
        for gen in sample["parsed_generations"]:
            if gen is None:
                correct.append(0)
            else:
                parsed = parse_order_answer(gen)
                if parsed is None:
                    correct.append(0)
                else:
                    correct.append(1 if verify_order(parsed, constraints) else 0)

        result = copy.deepcopy(sample)
        result["correct"] = correct
        return result

    async def run(self, existing: AsyncIterator[Dict[str, Any]],
                  average_over: list[int], pass_at: list[int]):
        if not pass_at:
            pass_at = [1]
        if not average_over:
            average_over = [1]

        all_correct = []
        all_difficulty = []
        completed = False
        count = 0

        async for result in existing:
            if "correct" in result:
                all_correct.append(result["correct"])
                all_difficulty.append(result.get("difficulty", "unknown"))
            count += 1

        async for result in self.async_grade_all_samples(start=count, grade_fn=self.grade_sample):
            if result == Sentinel.COMPLETED:
                completed = True
                break
            elif isinstance(result, ExceptionWrapper):
                yield result
            elif isinstance(result, dict) and "correct" in result:
                all_correct.append(result["correct"])
                all_difficulty.append(result.get("difficulty", "unknown"))
                yield Grade(element=result)

        if not completed:
            return

        if not all_correct or not any(all_correct):
            for n in average_over:
                yield Score(name=f"accuracy (avg over {n})", value=float("nan"))
            for n in pass_at:
                yield Score(name=f"accuracy (pass@{n})", value=float("nan"))
        else:
            for n in average_over:
                yield Score(name=f"accuracy (avg over {n})", value=get_accuracy([s[:n] for s in all_correct]))
                if n > 1:
                    yield Score(name=f"bootstrap_std (avg over {n})", value=get_bootstrap_accuracy_std([s[:n] for s in all_correct]))
            for n in pass_at:
                per_problem = [(len(s), sum(s)) for s in all_correct]
                yield Score(name=f"accuracy (pass@{n})", value=mean_pass_at_k(per_problem, k=n))

        by_diff = defaultdict(list)
        for corr, diff in zip(all_correct, all_difficulty):
            by_diff[diff].append(corr)

        for diff in sorted(by_diff.keys()):
            diff_correct = by_diff[diff]
            if not diff_correct or not any(diff_correct):
                for n in average_over:
                    yield Score(name=f"{diff} accuracy (avg over {n})", value=float("nan"))
                for n in pass_at:
                    yield Score(name=f"{diff} accuracy (pass@{n})", value=float("nan"))
            else:
                for n in average_over:
                    yield Score(name=f"{diff} accuracy (avg over {n})", value=get_accuracy([s[:n] for s in diff_correct]))
                for n in pass_at:
                    per_problem = [(len(s), sum(s)) for s in diff_correct]
                    yield Score(name=f"{diff} accuracy (pass@{n})", value=mean_pass_at_k(per_problem, k=n))

        yield Sentinel.COMPLETED
