"""
HumanEval+ grader using Daytona sandboxes for isolated code execution.

Each generated solution is run inside a fresh Daytona sandbox against the
problem's test cases. Subclasses DaytonaGraderBase and overrides only the
methods needed to handle HumanEval+'s test format.

Ground-truth format (stored as a JSON string in the JSONL):
    {
        "test": "<full Python test block string>",
        "entry_point": "<function name>",
        "prompt": "<function signature + docstring>"
    }

Harness construction:
    {prompt}
    {generated_code}

    {test_code}      <- defines check(candidate), assertion(), is_floats(), imports numpy

    [assertion override]  <- wraps original assertion() to record pass/fail per call
    check({entry_point})
    print(json.dumps(_results))
    sys.exit(0 if all passed else 1)

The prompt is always prepended so that:
- For completion models: prompt + generated_body = complete function definition.
- For chat models: the model generates a full function definition; prepending
  the prompt (which ends with a docstring-only stub) is valid Python — the
  second full definition overwrites the first.

The test field (from evalplus/humanevalplus) defines check(candidate) with
inputs/results arrays containing the combined base + plus test cases, but does
NOT call check() itself — the harness appends that call explicitly.

The test field also defines assertion() and is_floats() helpers and imports
numpy — all available in the ganler/evalplus sandbox image.

Sandbox image: ganler/evalplus:v0.2.1 (~153 MB compressed)
  - Python 3.11-slim base
  - numpy pre-installed (used by assertion() for np.allclose float comparisons)
  - No need to pip install anything in the harness script
"""

import json
import logging
from typing import Any

from daytona_sdk import CreateSandboxFromImageParams

from .registry import register
from .daytona_base import DaytonaGraderBase

logger = logging.getLogger(__name__)

# The official EvalPlus Docker image. v0.2.1 is ~153 MB compressed (vs 1.07 GB
# for latest which pulls in transformers, ML deps we don't need). The image has
# Python 3.11, numpy, and all EvalPlus evaluation dependencies pre-installed.
_EVALPLUS_IMAGE = "ganler/evalplus:v0.2.1"

# Appended after the test code to:
# 1. Override assertion() to record pass/fail per call (no details on pass).
# 2. Override check() to wrap candidate with an input tracker — this is the
#    only way to capture inputs, since assertion() only receives (out, exp, atol).
# 3. Run check(), emit JSON results to stdout, exit 0/1.
#
# On failure each entry is: {"passed": false, "input": repr(args), "actual": repr(out), "expected": repr(exp)}
# On success each entry is: {"passed": true}
_HARNESS_SUFFIX = """\

# --- per-assertion result capture ---
import json as _json, sys as _sys

_results = []
_current_input = [None]
_orig_assertion = assertion

def assertion(out, exp, atol):
    try:
        _orig_assertion(out, exp, atol)
        _results.append({{"passed": True}})
    except AssertionError:
        _results.append({{
            "passed": False,
            "input": repr(_current_input[0]),
            "actual": repr(out),
            "expected": repr(exp),
        }})

_orig_check = check

def check(candidate):
    def _tracked(*args, **kwargs):
        _current_input[0] = args
        return candidate(*args, **kwargs)
    _orig_check(_tracked)

check({entry_point})
print(_json.dumps(_results))
_sys.exit(0 if all(r["passed"] for r in _results) else 1)
"""


@register("humaneval-plus-daytona")
class HumanEvalPlusDaytonaGrader(DaytonaGraderBase):
    """Grade HumanEval+ code-generation problems by executing solutions in Daytona sandboxes."""

    def sandbox_params(self, name: str | None = None) -> CreateSandboxFromImageParams:
        """Use the official EvalPlus Docker image which has numpy pre-installed."""
        return CreateSandboxFromImageParams(
            image=_EVALPLUS_IMAGE,
            ephemeral=True,
            name=name,
            labels={"app": "eval360"},
        )

    @staticmethod
    def ground_truth_to_test_list(ground_truth: str) -> list[dict]:
        """Parse the ground_truth JSON string into a single-element test list."""
        return [json.loads(ground_truth)]

    def build_test_harness(self, code: str, test_cases: list[dict]) -> str:
        """
        Build a self-contained Python script that runs the generated code
        against the HumanEval+ test suite and emits per-assertion results as JSON.

        Structure:
            {prompt} + {code}         — defines the candidate function
            {test_code}               — defines check(), assertion(), is_floats(); imports numpy
            [assertion override]      — wraps assertion() to record pass/fail per call
            check({entry_point})      — runs all test cases
            print(json.dumps(...))    — emits per-assertion results to stdout
            sys.exit(0 or 1)          — 0 = all passed, 1 = any failed

        The EvalPlus test field does NOT call check() itself — that's appended here.
        """
        test_case = test_cases[0]
        prompt = test_case["prompt"]
        test_code = test_case["test"]
        entry_point = test_case["entry_point"]

        suffix = _HARNESS_SUFFIX.format(entry_point=entry_point)
        return f"{prompt}\n{code}\n\n{test_code}{suffix}"

    def _grade_sandbox_result(self, response: Any, test_cases: list[dict]) -> tuple[str, list | None]:
        """
        Parse the per-assertion JSON emitted to stdout and compute the grade.

        Returns (grade, details) where details is the list of per-assertion dicts
        (each has "passed": bool, plus "actual"/"expected" on failure).
        On parse failure (crash before print, or timeout), returns wrong_answer
        with an error descriptor.
        """
        # The harness prints our JSON on the last line. Any earlier lines are
        # incidental stdout from the model's code or test helpers — ignore them.
        last_line = (response.result or "").strip().rsplit("\n", 1)[-1]
        try:
            results = json.loads(last_line)
        except (json.JSONDecodeError, ValueError, TypeError):
            return "wrong_answer", [{
                "parse_error": True,
                "raw_output": response.result,
                "exit_code": getattr(response, "exit_code", None),
            }]

        if not results:
            return "wrong_answer", [{"empty_output": True}]

        grade = "pass" if all(r.get("passed") for r in results) else "wrong_answer"
        return grade, results
