"""
LCBv6 grader using Daytona sandboxes for isolated code execution.

Each generated solution is run inside a fresh Daytona sandbox against the
problem's test cases.  The grader implements GraderBase directly (rather
than AccuracyGraderBase) so it can own the full streaming/checkpointing/
scoring pipeline — mirroring the flexibility used by the generation phase.

Daytona quickstart
------------------
Install the SDK:
    pip install daytona-sdk

Authentication — set one of:
    DAYTONA_API_KEY   (preferred)
    DAYTONA_SERVER_URL + DAYTONA_API_KEY

Basic usage (async):
    from daytona_sdk import AsyncDaytona

    async with AsyncDaytona() as daytona:   # reads env vars; closes session on exit
        sandbox = await daytona.create(CreateSandboxFromSnapshotParams(language="python",
                                       ephemeral=True))
        response = await sandbox.process.code_run('print("hello")')
        print(response.result)              # "hello"
        response = await sandbox.process.exec("python solution.py")
        print(response.exit_code)           # 0
        await daytona.stop(sandbox)         # non-blocking async stop
"""

import logging
import json
from typing import Any

from daytona_sdk import CreateSandboxFromSnapshotParams

from .registry import register
from .daytona_base import DaytonaGraderBase
logger = logging.getLogger(__name__)


@register("lcbv6-daytona")
class LCBv6DaytonaGrader(DaytonaGraderBase):
    """Grade LCBv6 code-generation problems by executing solutions in Daytona sandboxes."""
    @staticmethod
    def ground_truth_to_test_list(ground_truth: str) -> list[dict]:
        return json.loads(ground_truth)

    def _build_sandbox_script(self, code: str, test_cases: list[dict]) -> str:
        testtype = test_cases[0].get("testtype", "stdin") if test_cases else "stdin"
        if testtype == "functional":
            return self._build_functional_harness(code, test_cases)
        return self._build_stdin_harness(code, test_cases)

    def _build_stdin_harness(self, code: str, test_cases: list[dict]) -> str:
        """Run each test case as a subprocess with stdin, collect stdout."""
        return f"""\
import subprocess
import json
import sys

solution = {repr(code)}
test_cases = {repr(test_cases)}
results = []
for test in test_cases:
    try:
        proc = subprocess.run(
            ["python", "-c", solution],
            input=test["input"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        results.append({{"stdout": proc.stdout, "exit_code": proc.returncode}})
    except subprocess.TimeoutExpired:
        results.append({{"timeout": True}})
    except Exception as e:
        results.append({{"error": str(e), "exit_code": -1}})
print(json.dumps(results))
"""

    def _build_functional_harness(self, code: str, test_cases: list[dict]) -> str:
        """Exec the solution, call Solution.<method_name> with JSON args per test case.

        method_name is taken from test_cases[0]["method_name"] (set by the converter)
        with a reflection fallback for datasets that don't include it.
        """
        method_name = test_cases[0].get("method_name") if test_cases else None
        if method_name:
            method_lookup = f'method_name = {repr(method_name)}'
        else:
            method_lookup = """\
methods = [m for m in dir(SolutionClass) if not m.startswith("_") and callable(getattr(SolutionClass, m))]
if not methods:
    print(json.dumps([{"error": "No public methods in Solution", "exit_code": -1}] * len(test_cases)))
    sys.exit(0)
method_name = methods[0]"""

        return f"""\
import json
import sys

solution = {repr(code)}
test_cases = {repr(test_cases)}

namespace = {{"__name__": "__solution__"}}
try:
    exec(solution, namespace)
except Exception as e:
    print(json.dumps([{{"error": f"exec failed: {{e}}", "exit_code": -1}}] * len(test_cases)))
    sys.exit(0)

SolutionClass = namespace.get("Solution")
if SolutionClass is None:
    print(json.dumps([{{"error": "No Solution class found", "exit_code": -1}}] * len(test_cases)))
    sys.exit(0)

{method_lookup}
results = []
for test in test_cases:
    try:
        args = [json.loads(line) for line in test["input"].strip().splitlines()]
        actual = getattr(SolutionClass(), method_name)(*args)
        results.append({{"stdout": json.dumps(actual), "exit_code": 0}})
    except Exception as e:
        results.append({{"error": str(e), "exit_code": -1}})
print(json.dumps(results))
"""

    def _grade_sandbox_result(self, response: Any, test_cases: list[dict]) -> tuple[str, list | None]:
        """Compare actual outputs against expected using numeric-aware local comparison.

        Returns ``(grade, details)`` where *details* is the JSON results array
        enriched with an ``"expected"`` field on each element (the raw expected
        output from the test case).
        """
        try:
            results = json.loads(response.result)
        except (json.JSONDecodeError, ValueError, TypeError):
            return "wrong_answer", [{"raw_output": response.result,
                                     "exit_code": getattr(response, "exit_code", None),
                                     "stderr": getattr(response, "stderr", None),
                                     "parse_error": True}]

        if len(results) != len(test_cases):
            return "wrong_answer", [{"raw_output": response.result, "count_mismatch": True,
                                     "got": len(results), "expected_count": len(test_cases)}]

        for result, test in zip(results, test_cases):
            result["expected"] = test["output"]

        testtype = test_cases[0].get("testtype", "stdin") if test_cases else "stdin"
        grade = "pass"
        for result, test in zip(results, test_cases):
            if result.get("timeout"):
                grade = "timeout"
                break
            if result.get("exit_code", 0) != 0:
                grade = "wrong_answer"
                break

            if testtype == "functional":
                grade = self._compare_functional(result["stdout"], test["output"])
            else:
                grade = self._compare_stdin(result["stdout"], test["output"])

            if grade != "pass":
                break

        return grade, results

    def _compare_functional(self, actual_json: str, expected_json: str) -> str:
        """Compare by parsing both sides as JSON values."""
        try:
            if json.loads(actual_json) != json.loads(expected_json):
                return "wrong_answer"
        except (json.JSONDecodeError, ValueError, TypeError):
            return "wrong_answer"
        return "pass"

    def _compare_stdin(self, actual_stdout: str, expected_output: str) -> str:
        """Compare line-by-line with numeric coercion fallback."""
        actual_lines = [line.strip() for line in actual_stdout.rstrip().splitlines()]
        expected_lines = [line.strip() for line in expected_output.rstrip().splitlines()]
        if len(actual_lines) != len(expected_lines):
            return "wrong_answer"
        for actual, expected in zip(actual_lines, expected_lines):
            try:
                if float(actual) != float(expected):
                    return "wrong_answer"
            except (ValueError, TypeError):
                if actual != expected:
                    return "wrong_answer"
        return "pass"

    def build_test_harness(self, code: str, test_cases: list[dict]) -> str:
        """
        Combine the generated *code* with all *test_cases* into a self-contained
        Python script that exits 0 if all tests pass, non-zero otherwise.
        """
        return f"""\
import subprocess
import sys

import os
import tempfile

_log_dir = "/home/daytona" if os.path.isdir("/home/daytona") else tempfile.gettempdir()
solution = {repr(code)}
test_cases = {repr(test_cases)}
with open(os.path.join(_log_dir, "logs.txt"), "w") as _f:
    _f.write(f"{{solution!r}}\\n{{test_cases!r}}\\n")

for i, test in enumerate(test_cases):
    result = subprocess.run(
        ["python", "-c", solution],
        input=test["input"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual_lines = [l.strip() for l in result.stdout.rstrip().splitlines()]
    expected_lines = [l.strip() for l in test["output"].rstrip().splitlines()]
    if actual_lines != expected_lines:
        with open(os.path.join(_log_dir, f"failure_{{i}}.txt"), "w") as _f:
            _f.write(f"expected {{expected_lines!r}}, got {{actual_lines!r}}\\n")
        sys.exit(1)
sys.exit(0)
"""
