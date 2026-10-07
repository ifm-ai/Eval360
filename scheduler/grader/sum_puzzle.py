"""Grader for Sum puzzle benchmark.

The model must find two positive integers x and y given their sum
and the sum of their squares.
"""
import copy
import re
import logging
from typing import Any, Dict, Optional

from ..utils import Sentinel
from .base import AccuracyGraderBase
from .registry import register

logger = logging.getLogger(__name__)


def parse_variables(answer_text: str) -> Optional[Dict[str, int]]:
    """Parse x=? y=? from model answer text."""
    if answer_text is None:
        return None
    pairs = re.findall(r"(\w+)\s*=\s*([-+]?\d*\.?\d+)", answer_text)
    parsed = {}
    for var, val_str in pairs:
        parsed[var] = int(val_str) if "." not in val_str else float(val_str)
    if "x" not in parsed or "y" not in parsed:
        return None
    return parsed


def score_answer(predicted: Optional[Dict], ground_truth: Dict) -> int:
    """Return 1 if both x and y match, else 0."""
    if predicted is None:
        return 0
    try:
        return int(predicted["x"] == ground_truth["x"] and predicted["y"] == ground_truth["y"])
    except (KeyError, TypeError):
        return 0


@register("sum-puzzle", "sum")
class SumPuzzle(AccuracyGraderBase):
    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict)
        assert "parsed_generations" in sample
        assert "ground_truth" in sample

        gt = sample["ground_truth"]

        correct = []
        for gen in sample["parsed_generations"]:
            predicted = parse_variables(gen)
            correct.append(score_answer(predicted, gt))

        result = copy.deepcopy(sample)
        result["correct"] = correct
        return result
