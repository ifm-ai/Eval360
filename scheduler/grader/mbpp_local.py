"""
Local MBPP grader — no Daytona, no external services.

Pipeline per sample:
  1. Pull the model's parsed_generation.
  2. Extract the code out of the first ```...``` markdown block the model
     emitted (reimplemented here, inspired by extract_code_blocks in
     lm-eval-harness lm_eval/tasks/mbpp/utils.py).
  3. Concatenate the extracted function with the three `assert` lines from
     ground_truth.
  4. Run the combined script in a short-lived python3 subprocess with a
     10 s wall clock, a stripped environment, and a scratch cwd. Exit
     code 0 means all asserts passed.

This is safe enough for MBPP (benign algorithm problems, non-adversarial
model) without needing full container isolation.
"""

import copy
import os
import re
import subprocess
import tempfile
from typing import Any

from ..utils import Sentinel
from ..metrics import get_accuracy, get_bootstrap_accuracy_std
from .base import AccuracyGraderBase
from .registry import register


# Match the body of the first fenced code block.
# Pattern: ``` then optional language tag (python, py, etc.) then newline,
# then capture the body non-greedy, then \n```.
#
# We deliberately do NOT use lm-eval's "prepend ```" trick here: our model
# emits both an opening AND closing fence on its own (reasoning models with
# instruct chat templates do), so prepending ``` causes the regex to match
# an empty body between our prepended fence and the model's opening fence.
_CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\s*\n(.*?)\n?```", re.DOTALL)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[0].strip()
    # Fallback: the model wrote code but never closed the fence (hit max tokens
    # mid-block or forgot). Try to take everything after the opening fence.
    open_match = re.search(r"```(?:[A-Za-z0-9_+-]*)\s*\n", text)
    if open_match:
        return text[open_match.end():].strip()
    return ""


def _run_test(code: str, test_code: str, timeout: float = 10.0) -> bool:
    """Run `code\n<test_code>\n` in a stripped-env subprocess. Return True on exit 0."""
    program = code + "\n" + test_code + "\n"
    with tempfile.TemporaryDirectory(prefix="mbpp_local_") as tmp:
        try:
            result = subprocess.run(
                ["python3", "-c", program],
                capture_output=True,
                timeout=timeout,
                cwd=tmp,
                env={
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "PYTHONUNBUFFERED": "1",
                    "LANG": "C.UTF-8",
                },
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


@register("mbpp-local", "mbpp_local")
class MBPPLocalGrader(AccuracyGraderBase):
    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample
        assert isinstance(sample, dict), "sample must be a dict"
        assert "parsed_generations" in sample, "sample must have parsed_generations"
        assert "ground_truth" in sample, "sample must have ground_truth"

        # Fall back to raw generations if parsed_generations is None.
        generations = []
        raw = sample.get("generations") or []
        for i, g in enumerate(sample["parsed_generations"]):
            if g is None and i < len(raw):
                generations.append(raw[i])
            else:
                generations.append(g or "")

        test_code = sample["ground_truth"]
        if isinstance(test_code, list):
            # Some upstream formats stash tests in a list; join for `exec`.
            test_code = "\n".join(test_code)

        correct = []
        picked_codes = []
        for g in generations:
            code = _extract_code(g)
            # Truncate for storage — we don't want to blow up grades.jsonl
            picked_codes.append(code[:400])
            if not code:
                correct.append(False)
                continue
            correct.append(_run_test(code, test_code))

        result = copy.deepcopy(sample)
        result["picked"] = picked_codes
        result["correct"] = correct
        result["accuracy"] = get_accuracy(correct)
        result["bootstrap_std"] = get_bootstrap_accuracy_std(correct)
        return result
