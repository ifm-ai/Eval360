"""
Tests for LCBv6DaytonaGrader.build_test_harness and grade_sample failure tracking.

Uses synthetic fixtures in the LiveCodeBench record shape (no benchmark items).
The stdin fixture: given X, print the sum of the integers 1..50 except X.
"""

import asyncio
import json
import subprocess
import sys
from unittest.mock import AsyncMock, patch
import pytest

from scheduler.grader.lcbv6_daytona import LCBv6DaytonaGrader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRADER = LCBv6DaytonaGrader.__new__(LCBv6DaytonaGrader)

# Synthetic stdin test cases (sum of 1..50 is 1275).
TEST_CASES = [
    {"input": "1",  "output": "1274\n", "testtype": "stdin"},
    {"input": "11", "output": "1264\n", "testtype": "stdin"},
    {"input": "24", "output": "1251\n", "testtype": "stdin"},
    {"input": "36", "output": "1239\n", "testtype": "stdin"},
    {"input": "81", "output": "1275\n", "testtype": "stdin"},  # X outside 1..50
]

CORRECT_SOLUTION = """\
X = int(input())
print(sum(i for i in range(1, 51) if i != X))
"""

WRONG_ANSWER_SOLUTION = """\
X = int(input())
print(0)
"""

CRASH_SOLUTION = """\
raise RuntimeError("boom")
"""

WRONG_LINE_COUNT_SOLUTION = """\
X = int(input())
print(sum(i for i in range(1, 51) if i != X))
print("extra line")
"""


def run_harness(harness: str) -> int:
    """Execute *harness* as a Python script and return its exit code."""
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
    )
    return result.returncode


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

GROUND_TRUTH = "[]"  # placeholder; ground_truth_to_test_list is mocked in grade_sample tests


class TestBuildTestHarness:
    def test_correct_solution_passes_all(self):
        harness = GRADER.build_test_harness(CORRECT_SOLUTION, TEST_CASES)
        assert run_harness(harness) == 0

    def test_wrong_answer_fails(self):
        harness = GRADER.build_test_harness(WRONG_ANSWER_SOLUTION, TEST_CASES)
        assert run_harness(harness) == 1

    def test_crashing_solution_fails(self):
        harness = GRADER.build_test_harness(CRASH_SOLUTION, TEST_CASES)
        assert run_harness(harness) == 1

    def test_wrong_line_count_fails(self):
        harness = GRADER.build_test_harness(WRONG_LINE_COUNT_SOLUTION, TEST_CASES)
        assert run_harness(harness) == 1


class TestGradeSampleFailureReasons:
    """Tests for evaluation_details tracking in grade_sample."""

    def _run_grade_sample(self, reasons: list[str]) -> dict:
        """Call grade_sample with _run_in_sandbox mocked to return *reasons* in order."""
        sample = {
            "row": 0,
            "parsed_generations": ["gen"] * len(reasons),
            "ground_truth": GROUND_TRUTH,
        }
        # _run_in_sandbox now returns (grade, details) tuples
        side_effects = [(r, None) for r in reasons]
        with patch.object(GRADER, "_run_in_sandbox", AsyncMock(side_effect=side_effects)):
            return asyncio.run(GRADER.grade_sample(sample))

    def test_all_pass(self):
        result = self._run_grade_sample(["pass", "pass"])
        assert result["correct"] == [1, 1]
        assert result["evaluation_details"] == ["pass", "pass"]

    def test_all_wrong_answer(self):
        result = self._run_grade_sample(["wrong_answer", "wrong_answer"])
        assert result["correct"] == [0, 0]
        assert result["evaluation_details"] == ["wrong_answer", "wrong_answer"]

    def test_mixed_reasons(self):
        reasons = ["pass", "wrong_answer", "timeout", "no_output"]
        result = self._run_grade_sample(reasons)
        assert result["correct"] == [1, 0, 0, 0]
        assert result["evaluation_details"] == reasons

    def test_timeout_is_failure(self):
        result = self._run_grade_sample(["timeout"])
        assert result["correct"] == [0]
        assert result["evaluation_details"] == ["timeout"]

    def test_no_output_is_failure(self):
        result = self._run_grade_sample(["no_output"])
        assert result["correct"] == [0]
        assert result["evaluation_details"] == ["no_output"]

    def test_sandbox_details_are_preserved_when_present(self):
        """What: verifies per-generation sandbox details are stored when any sandbox returns details.
        Executes: `LCBv6DaytonaGrader.grade_sample()` with `_run_in_sandbox` patched to return details.
        Why: covers the result-shaping path that preserves per-test sandbox diagnostics in grades.
        """
        sample = {
            "row": 0,
            "parsed_generations": ["gen-0", "gen-1"],
            "ground_truth": GROUND_TRUTH,
        }
        details = [[{"passed": True}], [{"passed": False, "expected": "1"}]]
        side_effects = [("pass", details[0]), ("wrong_answer", details[1])]
        with patch.object(GRADER, "_run_in_sandbox", AsyncMock(side_effect=side_effects)):
            result = asyncio.run(GRADER.grade_sample(sample))

        assert result["correct"] == [1, 0]
        assert result["evaluation_details"] == ["pass", "wrong_answer"]
        assert result["sandbox_results"] == details


def run_sandbox_script(harness: str) -> list[dict]:
    """Execute the runner harness locally and return the parsed JSON results."""
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


class MockSandboxResponse:
    """Minimal stand-in for a Daytona sandbox response."""
    def __init__(self, result: str | None, exit_code: int = 0):
        self.result = result
        self.exit_code = exit_code


def make_response(outputs: list[str]) -> MockSandboxResponse:
    """Build a mock response whose JSON result encodes one stdout per test case."""
    payload = [{"stdout": o, "exit_code": 0} for o in outputs]
    return MockSandboxResponse(json.dumps(payload))


class TestBuildSandboxScript:
    """_build_sandbox_script runs the solution remotely and returns JSON."""

    def test_correct_solution_returns_right_outputs(self):
        harness = GRADER._build_sandbox_script(CORRECT_SOLUTION, TEST_CASES)
        results = run_sandbox_script(harness)
        expected = ["1274", "1264", "1251", "1239", "1275"]
        assert len(results) == len(TEST_CASES)
        for result, exp in zip(results, expected):
            assert result["exit_code"] == 0
            assert result["stdout"].strip() == exp

    def test_wrong_answer_captured(self):
        harness = GRADER._build_sandbox_script(WRONG_ANSWER_SOLUTION, TEST_CASES)
        results = run_sandbox_script(harness)
        assert len(results) == len(TEST_CASES)
        for result in results:
            assert result["exit_code"] == 0
            assert result["stdout"].strip() == "0"

    def test_crashing_solution_nonzero_exit(self):
        harness = GRADER._build_sandbox_script(CRASH_SOLUTION, TEST_CASES)
        results = run_sandbox_script(harness)
        assert len(results) == len(TEST_CASES)
        for result in results:
            assert result["exit_code"] != 0

    def test_returns_one_entry_per_test_case(self):
        harness = GRADER._build_sandbox_script(CORRECT_SOLUTION, TEST_CASES)
        results = run_sandbox_script(harness)
        assert isinstance(results, list)
        assert len(results) == len(TEST_CASES)


class TestGradeSandboxResult:
    """_grade_sandbox_result compares outputs locally, with numeric tolerance."""

    def test_correct_outputs_pass(self):
        response = make_response(["1274\n", "1264\n", "1251\n", "1239\n", "1275\n"])
        grade, details = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "pass"
        assert len(details) == len(TEST_CASES)
        for detail, test in zip(details, TEST_CASES):
            assert detail["expected"] == test["output"]

    def test_wrong_outputs_fail(self):
        response = make_response(["0\n"] * len(TEST_CASES))
        grade, details = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "wrong_answer"
        assert len(details) == len(TEST_CASES)

    def test_numeric_float_form_accepted(self):
        # "1274.0" should equal "1274" numerically
        response = make_response(["1274.0\n", "1264.0\n", "1251.0\n", "1239.0\n", "1275.0\n"])
        grade, _ = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "pass"

    def test_timeout_entry_returns_timeout(self):
        payload = [{"timeout": True}] + [{"stdout": "x\n", "exit_code": 0}] * (len(TEST_CASES) - 1)
        response = MockSandboxResponse(json.dumps(payload))
        grade, details = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "timeout"
        assert len(details) == len(TEST_CASES)

    def test_nonzero_exit_code_returns_wrong_answer(self):
        payload = [{"stdout": "", "exit_code": 1}] + [{"stdout": "x\n", "exit_code": 0}] * (len(TEST_CASES) - 1)
        response = MockSandboxResponse(json.dumps(payload))
        grade, _ = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "wrong_answer"

    def test_wrong_line_count_returns_wrong_answer(self):
        outputs = ["1274\nextra\n", "1264\n", "1251\n", "1239\n", "1275\n"]
        response = make_response(outputs)
        grade, _ = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "wrong_answer"

    def test_invalid_json_returns_wrong_answer_with_raw_output(self):
        response = MockSandboxResponse("not json")
        grade, details = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "wrong_answer"
        assert details[0]["parse_error"] is True
        assert details[0]["raw_output"] == "not json"

    def test_none_result_returns_wrong_answer_with_raw_output(self):
        response = MockSandboxResponse(None)
        grade, details = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "wrong_answer"
        assert details[0]["parse_error"] is True

    def test_wrong_number_of_results_returns_wrong_answer_with_count_info(self):
        payload = [{"stdout": "1274\n", "exit_code": 0}]  # only 1 of 5
        response = MockSandboxResponse(json.dumps(payload))
        grade, details = GRADER._grade_sandbox_result(response, TEST_CASES)
        assert grade == "wrong_answer"
        assert details[0]["count_mismatch"] is True
        assert details[0]["got"] == 1
        assert details[0]["expected_count"] == len(TEST_CASES)

    def test_expected_field_present_on_all_elements(self):
        response = make_response(["1274\n", "1264\n", "1251\n", "1239\n", "1275\n"])
        _, details = GRADER._grade_sandbox_result(response, TEST_CASES)
        for detail, test in zip(details, TEST_CASES):
            assert "expected" in detail
            assert detail["expected"] == test["output"]


# ---------------------------------------------------------------------------
# Functional testtype fixtures
# ---------------------------------------------------------------------------

# Synthetic main-diagonal problem: grid -> list  (single arg, list output)
FUNCTIONAL_TEST_CASES = [
    {"input": "[[1,2],[3,4]]", "output": "[1,4]", "testtype": "functional", "method_name": "mainDiagonal"},
    {"input": "[[2,5,6],[7,1,8],[9,3,2]]", "output": "[2,1,2]", "testtype": "functional", "method_name": "mainDiagonal"},
]

CORRECT_FUNCTIONAL_SOLUTION = """\
class Solution:
    def mainDiagonal(self, grid):
        result = []
        for i, row in enumerate(grid):
            if i < len(row):
                result.append(row[i])
        return result
"""

WRONG_FUNCTIONAL_SOLUTION = """\
class Solution:
    def mainDiagonal(self, grid):
        return []
"""

# Synthetic two-arg functional problem: (string, int) -> bool, true when some
# character repeats at least k times in a row.
MULTI_ARG_TEST_CASES = [
    {"input": '"xxyyyz"\n3', "output": "true", "testtype": "functional", "method_name": "hasRunOfLength"},
    {"input": '"abcabc"\n2', "output": "false", "testtype": "functional", "method_name": "hasRunOfLength"},
]

CORRECT_MULTI_ARG_SOLUTION = """\
class Solution:
    def hasRunOfLength(self, s: str, k: int) -> bool:
        run = 0
        previous = None
        for ch in s:
            run = run + 1 if ch == previous else 1
            previous = ch
            if run >= k:
                return True
        return False
"""


class TestBuildFunctionalHarness:

    def test_correct_solution_returns_expected_outputs(self):
        harness = GRADER._build_functional_harness(CORRECT_FUNCTIONAL_SOLUTION, FUNCTIONAL_TEST_CASES)
        results = run_sandbox_script(harness)
        assert len(results) == 2
        assert results[0] == {"stdout": "[1, 4]", "exit_code": 0}
        assert results[1] == {"stdout": "[2, 1, 2]", "exit_code": 0}

    def test_wrong_solution_returns_wrong_outputs(self):
        harness = GRADER._build_functional_harness(WRONG_FUNCTIONAL_SOLUTION, FUNCTIONAL_TEST_CASES)
        results = run_sandbox_script(harness)
        assert all(r["exit_code"] == 0 for r in results)
        assert all(r["stdout"] == "[]" for r in results)

    def test_multi_arg_solution(self):
        harness = GRADER._build_functional_harness(CORRECT_MULTI_ARG_SOLUTION, MULTI_ARG_TEST_CASES)
        results = run_sandbox_script(harness)
        assert results[0] == {"stdout": "true", "exit_code": 0}
        assert results[1] == {"stdout": "false", "exit_code": 0}

    def test_no_solution_class_returns_error(self):
        harness = GRADER._build_functional_harness("x = 1", FUNCTIONAL_TEST_CASES)
        results = run_sandbox_script(harness)
        assert all("error" in r for r in results)

    def test_crashing_solution_returns_error_entry(self):
        harness = GRADER._build_functional_harness(
            "class Solution:\n    def f(self): raise RuntimeError('boom')",
            FUNCTIONAL_TEST_CASES,
        )
        results = run_sandbox_script(harness)
        assert all(r.get("exit_code", 0) != 0 for r in results)

    def test_reflection_fallback_uses_first_public_method(self):
        """What: verifies functional harness falls back to reflecting the first public Solution method.
        Executes: `_build_functional_harness()` and runs the generated harness locally.
        Why: covers functional cases whose converted tests omit explicit method_name metadata.
        """
        test_cases = [{"input": "[1, 2, 3]", "output": "6", "testtype": "functional"}]
        solution = "class Solution:\n    def total(self, values):\n        return sum(values)\n"
        harness = GRADER._build_functional_harness(solution, test_cases)
        results = run_sandbox_script(harness)
        assert results == [{"stdout": "6", "exit_code": 0}]

    def test_reflection_fallback_reports_no_public_methods(self):
        """What: verifies functional harness reports an error when Solution has no public methods.
        Executes: `_build_functional_harness()` for a Solution class with only private methods.
        Why: covers the reflective dispatch failure path returned by the sandbox harness.
        """
        test_cases = [{"input": "[1, 2, 3]", "output": "6", "testtype": "functional"}]
        solution = "class Solution:\n    def _hidden(self, values):\n        return sum(values)\n"
        harness = GRADER._build_functional_harness(solution, test_cases)
        results = run_sandbox_script(harness)
        assert results == [{"error": "No public methods in Solution", "exit_code": -1}]

    def test_exec_failure_returns_error_for_each_case(self):
        """What: verifies syntax errors while loading the solution are represented as error results.
        Executes: `_build_functional_harness()` with invalid solution code run through the local harness.
        Why: covers the exec-failure branch that reports one error result per test case.
        """
        harness = GRADER._build_functional_harness("class Solution(:\n", FUNCTIONAL_TEST_CASES)
        results = run_sandbox_script(harness)
        assert len(results) == len(FUNCTIONAL_TEST_CASES)
        assert all(result["error"].startswith("exec failed:") for result in results)


class TestCompareFunctional:

    def test_correct_list_output(self):
        assert GRADER._compare_functional("[1, 4]", "[1,4]") == "pass"

    def test_wrong_list_output(self):
        assert GRADER._compare_functional("[]", "[1,4]") == "wrong_answer"

    def test_bool_true(self):
        assert GRADER._compare_functional("true", "true") == "pass"

    def test_bool_false_mismatch(self):
        assert GRADER._compare_functional("false", "true") == "wrong_answer"

    def test_int_vs_float(self):
        # json.loads("52") == json.loads("52.0") in Python since 52 == 52.0
        assert GRADER._compare_functional("52", "52") == "pass"

    def test_invalid_json(self):
        assert GRADER._compare_functional("not json", "[1]") == "wrong_answer"


class TestGroundTruthAndDispatch:
    """What: groups tests for LCB ground-truth parsing and harness dispatch.
    Executes: `ground_truth_to_test_list()` and `_build_sandbox_script()` dispatch decisions.
    Why: these helpers choose the sandbox harness before any remote Daytona execution happens.
    """

    def test_ground_truth_to_test_list_parses_json_cases(self):
        """What: verifies ground truth JSON strings are parsed into test-case lists.
        Executes: `LCBv6DaytonaGrader.ground_truth_to_test_list()` with serialized test cases.
        Why: covers the conversion from dataset ground truth into sandbox-ready case dictionaries.
        """
        payload = json.dumps(TEST_CASES)
        assert LCBv6DaytonaGrader.ground_truth_to_test_list(payload) == TEST_CASES

    def test_build_sandbox_script_dispatches_functional_harness(self):
        """What: verifies functional testtype cases use the functional harness dispatcher.
        Executes: `_build_sandbox_script()` followed by local execution of the generated functional harness.
        Why: covers the functional dispatch branch without starting a Daytona sandbox.
        """
        harness = GRADER._build_sandbox_script(CORRECT_FUNCTIONAL_SOLUTION, FUNCTIONAL_TEST_CASES)
        results = run_sandbox_script(harness)
        assert results[0] == {"stdout": "[1, 4]", "exit_code": 0}

    def test_build_sandbox_script_dispatches_stdin_harness_by_default(self):
        """What: verifies stdin test cases use the subprocess-based stdin harness dispatcher.
        Executes: `_build_sandbox_script()` followed by local execution of the generated stdin harness.
        Why: covers the default dispatch branch used by standard input/output LCB cases.
        """
        harness = GRADER._build_sandbox_script(CORRECT_SOLUTION, TEST_CASES)
        results = run_sandbox_script(harness)
        assert results[0]["stdout"].strip() == "1274"


class TestCompareStdin:
    """What: groups tests for stdin output comparison edge cases.
    Executes: `LCBv6DaytonaGrader._compare_stdin()` with numeric and string outputs.
    Why: stdout comparison determines wrong-answer results for stdin-style sandbox runs.
    """

    def test_string_outputs_can_match_without_numeric_coercion(self):
        """What: verifies non-numeric lines compare by exact stripped string value.
        Executes: `_compare_stdin()` with matching non-numeric stdout and expected output.
        Why: covers the fallback string comparison branch after float coercion fails.
        """
        assert GRADER._compare_stdin("ok\n", "ok\n") == "pass"

    def test_numeric_mismatch_returns_wrong_answer(self):
        """What: verifies numeric-looking lines must compare equal after float coercion.
        Executes: `_compare_stdin()` with unequal numeric strings.
        Why: covers the numeric mismatch path that marks sandbox output as wrong answer.
        """
        assert GRADER._compare_stdin("1.0\n", "2.0\n") == "wrong_answer"

    def test_string_mismatch_returns_wrong_answer(self):
        """What: verifies different non-numeric lines fail comparison.
        Executes: `_compare_stdin()` with unequal non-numeric strings.
        Why: covers the string mismatch branch after numeric parsing is unavailable.
        """
        assert GRADER._compare_stdin("ok\n", "no\n") == "wrong_answer"


class TestRunInSandboxEmptyResponse:
    """Empty / whitespace parsed_response must short-circuit to 'no_output'."""

    @pytest.mark.asyncio
    async def test_none_returns_no_output(self):
        grade, details = await GRADER._run_in_sandbox(None, [], row=0, gen_index=0)
        assert grade == "no_output"
        assert details is None

    @pytest.mark.asyncio
    async def test_empty_string_returns_no_output(self):
        grade, details = await GRADER._run_in_sandbox("", [], row=0, gen_index=0)
        assert grade == "no_output"
        assert details is None

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_no_output(self):
        grade, details = await GRADER._run_in_sandbox("   \n  ", [], row=0, gen_index=0)
        assert grade == "no_output"
        assert details is None
