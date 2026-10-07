import copy
import re
from typing import Any, Union, Callable, Optional
import logging
from ..utils import Sentinel
from ..metrics import get_accuracy, get_bootstrap_accuracy_std
from .base import AccuracyGraderBase, Grade, Score
from .registry import register
from enum import Enum


logger = logging.getLogger(__name__)

_MC_DIGIT_TO_LETTER = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
}


def _normalize_choice_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip().upper()
    if not candidate:
        return None
    return _MC_DIGIT_TO_LETTER.get(candidate, candidate)


# Answer-extraction patterns for multiple choice, tried in priority order.
# Higher-priority entries are explicit structural markers the model is
# prompted to emit ("the answer is (X)", "**X**", "\\boxed{X}"); lower ones
# are looser fallbacks. A-J covers 4-choice MMLU and 10-choice MMLU-Pro.
_LETTER_PATTERNS = [
    re.compile(r"(?:the\s+)?(?:final\s+)?answer\s+is[\s:]*\*{0,2}\(?([A-J])\)?\*{0,2}", re.I),
    re.compile(r"(?:final\s+)?answer[\s:]*[:\-]\s*\*{0,2}\(?([A-J])\)?\*{0,2}", re.I),
    re.compile(r"\\boxed\s*\{\s*\(?([A-J])\)?\s*\}"),
    re.compile(r"\*\*\(?([A-J])\)?\*\*"),
    re.compile(r"\(\s*([A-J])\s*\)"),
]


def _extract_choice_symbol(sampled: str | None) -> str | None:
    """Extract the chosen multiple-choice letter from a generation.

    Reasoning models write prose before the final answer, so the old
    "first letter in the string" heuristic almost always picked a letter
    from a leading word like "The" or "Let". Instead, scan the tail of the
    generation for explicit answer markers in priority order, then fall
    back to an isolated A-J letter near the very end. A bare single-letter
    generation ("A", "(A)") is still handled by the isolated-letter rule.
    """
    if sampled is None:
        return None

    stripped = sampled.strip()
    if not stripped:
        return None

    # The final choice of a reasoning/CoT answer sits at the end; scan the tail.
    tail = stripped[-600:] if len(stripped) > 600 else stripped

    for pat in _LETTER_PATTERNS:
        matches = pat.findall(tail)
        if matches:
            return matches[-1].upper()

    # Fallback: an isolated A-J letter in the last 50 chars.
    isolated = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", tail[-50:])
    if isolated:
        return isolated[-1].upper()

    # Digit fallback (1-5 -> A-E) for numeric-labelled choices.
    digit = re.search(r"(?<!\d)([1-5])(?!\d)", tail[-50:])
    if digit:
        return _MC_DIGIT_TO_LETTER[digit.group(1)]

    # Symbol fallback: a single non-alphanumeric character (e.g. "?") is
    # returned as-is so the picked column still records what the model
    # produced when the response is neither a known letter nor a digit.
    if len(stripped) == 1 and not stripped.isalnum():
        return stripped

    return None


def check_multiple_choice(
    prompt: Any,
    generations: list[str],
    expected: Union[str, list[str]],
):
    """
    Records and checks if generated responses match the expected result.

    Args:
        prompt: The input prompt.
        generations: The generated responses from the model.
        expected: The expected response

    Returns:
        The matched option or None if no match found.
    """

    if isinstance(expected, str):
        expected = [expected] * len(generations)
    picked, match = [None] * len(generations), [None] * len(generations)

    assert (len(generations) == len(expected)), (len(generations), len(expected))
    for i, (sampled, exp) in enumerate(zip(generations, expected)):
        picked[i] = _extract_choice_symbol(sampled)
        expected_symbol = _normalize_choice_symbol(exp[0] if exp else None)
        match[i] = picked[i] is not None and expected_symbol == picked[i]
    result = {
        "picked": picked,
        "correct": match,
        "accuracy": get_accuracy(match),
        "bootstrap_std": get_bootstrap_accuracy_std(match),
    }
    return result


@register("multiple-choice", "multiple_choice")
class MultipleChoice(AccuracyGraderBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    async def grade_sample(self, sample: Any, *_):
        # if the sample is the sentinel, we have reached the end of the data
        if sample == Sentinel.COMPLETED:
            return sample
        # TODO: move this typechecking to pydantic
        assert isinstance(sample, dict), "sample must be a dict"
        assert "completion_input" in sample, "sample must have an 'completion_input' key"
        assert "parsed_generations" in sample, "sample must have an 'parsed_generations' key"
        assert "ground_truth" in sample, "sample must have an 'ground_truth' key"
        assert isinstance(sample["ground_truth"], str) or isinstance(
            sample["ground_truth"], list
        ), "sample['ground_truth'] must be a string or list of strings"

        generations = []
        for index, generation in enumerate(sample["parsed_generations"]):
            if generation is None:
                generations.append(sample["generations"][index])
            else:
                generations.append(generation)

        new_fields = check_multiple_choice(
            prompt=sample["completion_input"],
            generations=generations,
            expected=sample["ground_truth"],
        )

        result = copy.deepcopy(sample)
        result.update(new_fields)
        return result
