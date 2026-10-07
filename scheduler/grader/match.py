import copy
from typing import Any, Union, Callable, Optional
import logging
from ..metrics import get_accuracy, get_bootstrap_accuracy_std
from .base import AccuracyGraderBase
from .registry import register


logger = logging.getLogger(__name__)


def check_match(
    prompt: Any,
    generations: list[str],
    expected: Union[str, list[str], tuple[str]],
    separator: Callable[[str], bool] = None,
    options: Optional[list[str]] = None,
):
    """
    Records and checks if generated responses match the expected result.

    Args:
        prompt: The input prompt.
        generations: The generated responses from the model.
        expected: The expected response or list of responses.
        separator: Optional function to check if a character is a separator.
        options: Optional list of options to match against the sampled response.

    Returns:
        The matched option or None if no match found.
    """
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]
    if options is None:
        options = expected

    picked, match = [None] * len(generations), [None] * len(generations)

    for i, sampled in enumerate(generations):
        for option in options:
            if not sampled.startswith(option):
                continue
            if (
                separator is not None
                and len(sampled) > len(option)
                and not separator(sampled[len(option)])
            ):
                continue
            picked[i] = option
            break

        match[i] = bool(picked[i] in expected)
    result = {
        "picked": picked,
        "correct": match,
        "accuracy": get_accuracy(match),
        "bootstrap_std": get_bootstrap_accuracy_std(match),
    }
    return result


@register("exact-match", "exact_match")
class Match(AccuracyGraderBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    async def grade_sample(self, sample: Any, *_):
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

        new_fields = check_match(
            prompt=sample["completion_input"],
            generations=generations,
            expected=sample["ground_truth"],
        )
        result = copy.deepcopy(sample)
        result.update(new_fields)
        return result
