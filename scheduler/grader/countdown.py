"""Grader for Countdown arithmetic benchmark.

The model must construct an arithmetic equation using given numbers
(each used exactly once) that evaluates to a target value.

The verification functions (validate_equation, evaluate_equation and the
tolerance check in score_equation) are adapted from Jiayi-Pan/TinyZero
verl/utils/reward_score/countdown.py (Apache-2.0), whose NOTICE reads:
"Copyright 2023-2024 Bytedance Ltd. and/or its affiliates".
Modified for Eval360. See THIRD_PARTY_NOTICES.md.
"""
import copy
import re
import logging
from typing import Any

from ..utils import Sentinel
from .base import AccuracyGraderBase
from .registry import register

logger = logging.getLogger(__name__)


def validate_equation(equation_str: str, available_numbers: list[int]) -> bool:
    """Check that the equation uses exactly the available numbers, each once."""
    try:
        numbers_in_eq = sorted(int(n) for n in re.findall(r"\d+", equation_str))
        return numbers_in_eq == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str: str):
    """Safely evaluate an arithmetic expression (digits, +, -, *, /, parens only)."""
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
            return None
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def score_equation(equation: str | None, target: int, numbers: list[int]) -> int:
    """Return 1 if equation is valid and equals target, else 0."""
    if equation is None:
        return 0
    if not validate_equation(equation, numbers):
        return 0
    result = evaluate_equation(equation)
    if result is None:
        return 0
    if abs(result - target) < 1e-5:
        return 1
    return 0


@register("countdown", "cd")
class Countdown(AccuracyGraderBase):
    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict)
        assert "parsed_generations" in sample
        assert "ground_truth" in sample

        gt = sample["ground_truth"]
        target = gt["target"]
        numbers = gt["numbers"]

        correct = []
        for gen in sample["parsed_generations"]:
            correct.append(score_equation(gen, target, numbers))

        result = copy.deepcopy(sample)
        result["correct"] = correct
        return result
