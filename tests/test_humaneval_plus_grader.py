"""
Tests for HumanEvalPlusDaytonaGrader.

Covers:
- ground_truth_to_test_list: correct parsing, field validation
- build_test_harness: structure, check() call, prompt prepend, suffix format
- sandbox_params: uses CreateSandboxFromImageParams with evalplus image
- _grade_sandbox_result: pass/fail/empty/parse-error cases
- Harness execution correctness: harness runs as valid Python locally
"""
import json
import textwrap
from unittest.mock import MagicMock

import pytest

from daytona_sdk import CreateSandboxFromImageParams

from scheduler.grader.humaneval_plus_daytona import (
    HumanEvalPlusDaytonaGrader,
    _EVALPLUS_IMAGE,
    _HARNESS_SUFFIX,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_PROMPT = textwrap.dedent("""\
    from typing import List


    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        \"\"\"Check if two numbers in the list are closer than threshold.
        >>> has_close_elements([1.0, 2.0, 3.9], 0.3)
        False
        \"\"\"
""")

# Minimal EvalPlus-style test block: defines assertion() + check(), does NOT call check().
SAMPLE_TEST = textwrap.dedent("""\
    def assertion(out, exp, atol):
        assert out == exp

    def check(candidate):
        assertion(candidate([1.0, 1.1, 2.0], 0.2), True, 0)
        assertion(candidate([1.0, 2.0, 3.0], 0.5), False, 0)
""")

SAMPLE_ENTRY_POINT = "has_close_elements"

SAMPLE_GROUND_TRUTH = json.dumps({
    "test": SAMPLE_TEST,
    "entry_point": SAMPLE_ENTRY_POINT,
    "prompt": SAMPLE_PROMPT,
})


def make_grader():
    return HumanEvalPlusDaytonaGrader.__new__(HumanEvalPlusDaytonaGrader)


def mock_response(result: str, exit_code: int = 0):
    r = MagicMock()
    r.result = result
    r.exit_code = exit_code
    return r


# ---------------------------------------------------------------------------
# ground_truth_to_test_list
# ---------------------------------------------------------------------------

class TestGroundTruthToTestList:
    def test_returns_single_element_list(self):
        result = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        assert isinstance(result, list) and len(result) == 1

    def test_contains_test_field(self):
        result = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        assert result[0]["test"] == SAMPLE_TEST

    def test_contains_entry_point_field(self):
        result = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        assert result[0]["entry_point"] == SAMPLE_ENTRY_POINT

    def test_contains_prompt_field(self):
        result = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        assert result[0]["prompt"] == SAMPLE_PROMPT

    def test_is_static_method(self):
        result = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        assert len(result) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            HumanEvalPlusDaytonaGrader.ground_truth_to_test_list("not valid json")


# ---------------------------------------------------------------------------
# sandbox_params
# ---------------------------------------------------------------------------

class TestSandboxParams:
    def test_returns_from_image_params(self):
        grader = make_grader()
        params = grader.sandbox_params()
        assert isinstance(params, CreateSandboxFromImageParams)

    def test_uses_evalplus_image(self):
        grader = make_grader()
        params = grader.sandbox_params()
        assert params.image == _EVALPLUS_IMAGE

    def test_evalplus_image_is_v021(self):
        # Pin to lightweight v0.2.1 (~153 MB), not latest (~1.07 GB)
        assert "v0.2.1" in _EVALPLUS_IMAGE

    def test_is_ephemeral(self):
        grader = make_grader()
        params = grader.sandbox_params()
        assert params.ephemeral is True

    def test_name_is_passed_through(self):
        grader = make_grader()
        params = grader.sandbox_params(name="my-sandbox")
        assert params.name == "my-sandbox"

    def test_labels_include_eval360_tag(self):
        grader = make_grader()
        params = grader.sandbox_params()
        assert params.labels == {"app": "eval360"}


# ---------------------------------------------------------------------------
# build_test_harness
# ---------------------------------------------------------------------------

class TestBuildTestHarness:
    def test_prompt_is_present(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        harness = grader.build_test_harness("    return True", test_cases)
        assert SAMPLE_PROMPT in harness

    def test_generated_code_is_present(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        code = "    return any(abs(a-b) < threshold for i,a in enumerate(numbers) for j,b in enumerate(numbers) if i!=j)"
        harness = grader.build_test_harness(code, test_cases)
        assert code in harness

    def test_test_code_is_present(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        harness = grader.build_test_harness("    return True", test_cases)
        assert SAMPLE_TEST in harness

    def test_check_call_is_appended(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        harness = grader.build_test_harness("    return True", test_cases)
        assert f"check({SAMPLE_ENTRY_POINT})" in harness

    def test_assertion_and_check_override_present(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        harness = grader.build_test_harness("    return True", test_cases)
        assert "_orig_assertion = assertion" in harness
        assert "_orig_check = check" in harness
        assert "_current_input" in harness
        assert "_results.append" in harness

    def test_json_print_is_present(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        harness = grader.build_test_harness("    return True", test_cases)
        assert "print(_json.dumps(_results))" in harness

    def test_sys_exit_is_present(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        harness = grader.build_test_harness("    return True", test_cases)
        assert "_sys.exit(" in harness

    def test_check_call_comes_after_test_code(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        harness = grader.build_test_harness("    return True", test_cases)
        assert harness.index(SAMPLE_TEST) < harness.index(f"check({SAMPLE_ENTRY_POINT})")

    def test_prompt_comes_before_code(self):
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        code = "    return True"
        harness = grader.build_test_harness(code, test_cases)
        assert harness.index(SAMPLE_PROMPT) < harness.index(code)

    def test_different_entry_points(self):
        grader = make_grader()
        gt = json.dumps({
            "test": "def assertion(out, exp, atol):\n    assert out == exp\ndef check(candidate):\n    assertion(candidate(2), 4, 0)\n",
            "entry_point": "square",
            "prompt": "def square(x):\n",
        })
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(gt)
        harness = grader.build_test_harness("    return x * x", test_cases)
        assert "check(square)" in harness
        assert "check(has_close_elements)" not in harness

    def test_no_numpy_pip_install_in_harness(self):
        """numpy comes from the Docker image, not a pip install in the script."""
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        harness = grader.build_test_harness("    return True", test_cases)
        assert "pip install" not in harness


# ---------------------------------------------------------------------------
# _grade_sandbox_result
# ---------------------------------------------------------------------------

class TestGradeSandboxResult:
    def test_all_passed_returns_pass(self):
        grader = make_grader()
        results_json = json.dumps([{"passed": True}, {"passed": True}])
        grade, details = grader._grade_sandbox_result(mock_response(results_json), [{}])
        assert grade == "pass"

    def test_any_failed_returns_wrong_answer(self):
        grader = make_grader()
        results_json = json.dumps([{"passed": True}, {"passed": False, "actual": "1", "expected": "2"}])
        grade, details = grader._grade_sandbox_result(mock_response(results_json, exit_code=1), [{}])
        assert grade == "wrong_answer"

    def test_all_failed_returns_wrong_answer(self):
        grader = make_grader()
        results_json = json.dumps([{"passed": False}, {"passed": False}])
        grade, details = grader._grade_sandbox_result(mock_response(results_json, exit_code=1), [{}])
        assert grade == "wrong_answer"

    def test_details_contain_per_assertion_entries(self):
        grader = make_grader()
        results_json = json.dumps([
            {"passed": True},
            {"passed": False, "input": "(1, 2)", "actual": "0", "expected": "1"},
        ])
        _, details = grader._grade_sandbox_result(mock_response(results_json), [{}])
        assert len(details) == 2
        assert details[0] == {"passed": True}
        assert details[1]["passed"] is False
        assert "input" in details[1]
        assert "actual" in details[1]
        assert "expected" in details[1]

    def test_invalid_json_returns_wrong_answer_with_parse_error(self):
        grader = make_grader()
        grade, details = grader._grade_sandbox_result(mock_response("not json"), [{}])
        assert grade == "wrong_answer"
        assert details[0]["parse_error"] is True
        assert details[0]["raw_output"] == "not json"

    def test_extra_stdout_before_json_is_ignored(self):
        """Lines printed before the JSON (e.g. from model code) don't break grading."""
        grader = make_grader()
        results_json = json.dumps([{"passed": True}, {"passed": True}])
        output = f"False\nTrue\n{results_json}"
        grade, details = grader._grade_sandbox_result(mock_response(output), [{}])
        assert grade == "pass"
        assert len(details) == 2

    def test_empty_output_returns_wrong_answer(self):
        grader = make_grader()
        grade, details = grader._grade_sandbox_result(mock_response(""), [{}])
        assert grade == "wrong_answer"

    def test_empty_json_array_returns_wrong_answer(self):
        grader = make_grader()
        grade, details = grader._grade_sandbox_result(mock_response("[]"), [{}])
        assert grade == "wrong_answer"
        assert details[0]["empty_output"] is True


class TestGradeSandboxResultEdgeCases:
    """What: groups additional edge cases for HumanEval+ sandbox result parsing.
    Executes: `HumanEvalPlusDaytonaGrader._grade_sandbox_result()` with crafted JSON stdout.
    Why: sandbox result parsing determines the public pass/wrong-answer grade for every generation.
    """

    def test_missing_passed_key_is_wrong_answer(self):
        """What: verifies result entries without a truthy passed field are treated as failures.
        Executes: `_grade_sandbox_result()` with one passing record and one record missing `passed`.
        Why: covers malformed sandbox output that should fail closed as a wrong answer.
        """
        grader = make_grader()
        results_json = json.dumps([{"passed": True}, {"actual": "1", "expected": "2"}])
        grade, details = grader._grade_sandbox_result(mock_response(results_json), [{}])
        assert grade == "wrong_answer"
        assert details[1]["actual"] == "1"

    def test_last_line_json_is_used_after_trailing_whitespace(self):
        """What: verifies the parser accepts incidental stdout before the final JSON line.
        Executes: `_grade_sandbox_result()` with debug output before the final JSON line.
        Why: covers the noisy-stdout sandbox corner case while preserving successful grading.
        """
        grader = make_grader()
        results_json = json.dumps([{"passed": True}])
        grade, details = grader._grade_sandbox_result(mock_response(f"debug\n{results_json}\n"), [{}])
        assert grade == "pass"
        assert details == [{"passed": True}]


# ---------------------------------------------------------------------------
# Harness execution correctness (exec locally, no Daytona)
# ---------------------------------------------------------------------------

class TestHarnessExecution:
    """Run the harness locally via exec() to verify structural and semantic correctness.
    Captures stdout to read the JSON results instead of checking sys.exit().
    """

    @staticmethod
    def _exec_harness(harness: str) -> list[dict]:
        """Exec the harness, capturing stdout. Returns the parsed JSON results list."""
        import io
        import sys

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        raised = None
        try:
            exec(harness, {})
        except SystemExit:
            pass  # sys.exit() is expected
        except Exception as e:
            raised = e
        finally:
            sys.stdout = old_stdout

        if raised:
            raise raised

        output = stdout_capture.getvalue().strip()
        return json.loads(output) if output else []

    def test_correct_completion_model_output_all_pass(self):
        """A correct function body (completion model) passes all assertions."""
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        correct_body = (
            "    from itertools import combinations\n"
            "    return any(abs(a - b) < threshold for a, b in combinations(numbers, 2))\n"
        )
        harness = grader.build_test_harness(correct_body, test_cases)
        results = self._exec_harness(harness)
        assert all(r["passed"] for r in results)
        assert len(results) == 2  # two assertions in SAMPLE_TEST

    def test_correct_chat_model_output_all_pass(self):
        """A correct full function (chat model output) passes all assertions."""
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        full_function = (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    from itertools import combinations\n"
            "    return any(abs(a - b) < threshold for a, b in combinations(numbers, 2))\n"
        )
        harness = grader.build_test_harness(full_function, test_cases)
        results = self._exec_harness(harness)
        assert all(r["passed"] for r in results)

    def test_wrong_answer_records_failures_with_input(self):
        """Failed assertions include input, actual, and expected."""
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        # Always returns True — fails the second assertion (expects False)
        wrong_body = "    return True\n"
        harness = grader.build_test_harness(wrong_body, test_cases)
        results = self._exec_harness(harness)
        failed = [r for r in results if not r["passed"]]
        assert len(failed) >= 1
        assert "input" in failed[0]
        assert "actual" in failed[0]
        assert "expected" in failed[0]

    def test_failure_input_contains_args(self):
        """The input field on a failure is the repr of the args tuple passed to the candidate."""
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        wrong_body = "    return True\n"
        harness = grader.build_test_harness(wrong_body, test_cases)
        results = self._exec_harness(harness)
        failed = [r for r in results if not r["passed"]]
        # The second assertion calls candidate([1.0, 2.0, 3.0], 0.5) — input should reflect this
        assert "[1.0, 2.0, 3.0]" in failed[0]["input"]
        assert "0.5" in failed[0]["input"]

    def test_passed_assertions_have_no_input_or_details(self):
        """Passed assertions are compact — just {"passed": True}, no noise."""
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        correct_body = (
            "    from itertools import combinations\n"
            "    return any(abs(a - b) < threshold for a, b in combinations(numbers, 2))\n"
        )
        harness = grader.build_test_harness(correct_body, test_cases)
        results = self._exec_harness(harness)
        for r in results:
            assert r == {"passed": True}

    def test_syntax_error_propagates(self):
        """Syntactically invalid code raises before any results are printed."""
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        bad_code = "    this is not valid python !!!\n"
        harness = grader.build_test_harness(bad_code, test_cases)
        with pytest.raises(SyntaxError):
            self._exec_harness(harness)

    def test_partial_failure_records_all_assertions(self):
        """All assertions are recorded even when one fails mid-way."""
        grader = make_grader()
        test_cases = HumanEvalPlusDaytonaGrader.ground_truth_to_test_list(SAMPLE_GROUND_TRUTH)
        # Returns True always: first assertion passes, second fails
        body = "    return True\n"
        harness = grader.build_test_harness(body, test_cases)
        results = self._exec_harness(harness)
        assert results[0] == {"passed": True}   # candidate([1.0, 1.1, 2.0], 0.2) == True ✓
        assert results[1]["passed"] is False     # candidate([1.0, 2.0, 3.0], 0.5) == False ✗ (got True)
        assert "input" in results[1]
