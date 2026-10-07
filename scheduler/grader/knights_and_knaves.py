"""Grader for Knights and Knaves (KK) logical reasoning benchmark.

parse_solution_text_format is copied, and parse_model_answer adapted, from
Unakar/Logic-RL verl/utils/reward_score/kk.py (Apache-2.0), whose NOTICE
reads: "Copyright 2023-2024 Bytedance Ltd. and/or its affiliates".
Modified for Eval360 (debug prints removed). See THIRD_PARTY_NOTICES.md.
"""
import copy
import re
import logging
from collections import defaultdict
from typing import Any, AsyncIterator, Dict, Optional

from ..utils import Sentinel, ExceptionWrapper
from ..metrics import get_accuracy, get_bootstrap_accuracy_std, mean_pass_at_k
from .base import AccuracyGraderBase, Grade, Score
from .registry import register

logger = logging.getLogger(__name__)


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parse ground truth solution text into {name: role} dict."""
    status_dict = {}
    for line in solution_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.search(r"\b([A-Za-z]+)\b.*?\b(knight|knave)\b", line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parse model's answer text into {name: role} dict. Returns None if incomplete."""
    status_dict = {}
    for name in expected_names:
        pattern = re.compile(rf"\b{re.escape(name)}\b.*?\b(knight|knave)\b", re.IGNORECASE)
        match = pattern.search(answer_text)
        if match:
            status_dict[name] = match.group(1).lower()
        else:
            return None
    return status_dict


def check_answer(gt: Dict[str, str], sol: Optional[Dict[str, str]]) -> int:
    """Check if model solution matches ground truth. All must be correct."""
    if sol is None:
        return 0
    for name in gt:
        if gt[name] != sol.get(name):
            return 0
    return 1


@register("knights-and-knaves", "kk")
class KnightsAndKnaves(AccuracyGraderBase):
    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict)
        assert "parsed_generations" in sample
        assert "ground_truth" in sample
        assert "names" in sample

        gt = parse_solution_text_format(sample["ground_truth"])
        names = sample["names"]

        correct = []
        for gen in sample["parsed_generations"]:
            if gen is None:
                correct.append(0)
            else:
                parsed = parse_model_answer(gen, names)
                correct.append(check_answer(gt, parsed))

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

        # Overall scores (same as base class)
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

        # Per-difficulty scores
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
