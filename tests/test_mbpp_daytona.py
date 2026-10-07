"""
Tests for MBPPDaytonaGrader.build_test_harness and ground_truth_to_test_list.

Uses a simple "is_even" problem as a fixture, with ground_truth formatted
as newline-joined assert statements, optionally preceded by setup code.
"""

import subprocess
import sys
import pytest

from scheduler.grader.mbpp_daytona import MBPPDaytonaGrader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRADER = MBPPDaytonaGrader.__new__(MBPPDaytonaGrader)

# Ground truth mirrors what persist_mbpp.py produces:
# newline-joined assert statements.
GROUND_TRUTH = """\
assert is_even(2) == True
assert is_even(3) == False
assert is_even(0) == True"""

# Ground truth with setup code prepended (some MBPP problems need imports).
GROUND_TRUTH_WITH_SETUP = """\
import math
assert square_root(4) == 2.0
assert square_root(9) == 3.0
assert square_root(1) == 1.0"""

CORRECT_SOLUTION = """\
def is_even(n):
    return n % 2 == 0
"""

WRONG_SOLUTION = """\
def is_even(n):
    return True
"""

CRASH_SOLUTION = """\
def is_even(n):
    raise RuntimeError("boom")
"""

CORRECT_SQRT_SOLUTION = """\
import math

def square_root(n):
    return math.sqrt(n)
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

class TestGroundTruthToTestList:
    def test_returns_single_element_list(self):
        result = GRADER.ground_truth_to_test_list(GROUND_TRUTH)
        assert result == [GROUND_TRUTH]

    def test_element_is_original_string(self):
        result = GRADER.ground_truth_to_test_list(GROUND_TRUTH)
        assert result[0] is GROUND_TRUTH


class TestBuildTestHarness:
    def test_correct_solution_passes(self):
        harness = GRADER.build_test_harness(CORRECT_SOLUTION, GROUND_TRUTH)
        assert run_harness(harness) == 0

    def test_wrong_solution_fails(self):
        harness = GRADER.build_test_harness(WRONG_SOLUTION, GROUND_TRUTH)
        assert run_harness(harness) != 0

    def test_crashing_solution_fails(self):
        harness = GRADER.build_test_harness(CRASH_SOLUTION, GROUND_TRUTH)
        assert run_harness(harness) != 0

    def test_harness_is_concatenation(self):
        harness = GRADER.build_test_harness(CORRECT_SOLUTION, GROUND_TRUTH)
        assert harness == f"{CORRECT_SOLUTION}\n{GROUND_TRUTH}"

    def test_setup_code_in_ground_truth(self):
        """When ground_truth includes setup code (e.g. imports), it should work."""
        harness = GRADER.build_test_harness(CORRECT_SQRT_SOLUTION, GROUND_TRUTH_WITH_SETUP)
        assert run_harness(harness) == 0


# ---------------------------------------------------------------------------
# MBPP+ specific tests (SET_EQ, atol ground_truth, end-to-end harness)
# ---------------------------------------------------------------------------

# SET_EQ ground_truth: uses set() comparison (from persist_mbpp_plus.py)
MBPP_PLUS_SET_EQ_GROUND_TRUTH = """\
assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set([4, 5])"""

CORRECT_SET_EQ_SOLUTION = """\
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
"""

WRONG_SET_EQ_SOLUTION = """\
def similar_elements(test_tup1, test_tup2):
    return test_tup1
"""

# Float tolerance ground_truth: uses abs() comparison (from persist_mbpp_plus.py)
MBPP_PLUS_ATOL_GROUND_TRUTH = """\
assert abs(volume_sphere(3) - 113.09733552923254) <= 0.0001 + 1e-07 * abs(113.09733552923254)"""

CORRECT_ATOL_SOLUTION = """\
import math
def volume_sphere(r):
    return (4.0 / 3) * math.pi * r ** 3
"""

WRONG_ATOL_SOLUTION = """\
def volume_sphere(r):
    return float(r)
"""


class TestMbppPlusSetEqHarness:
    def test_correct_set_eq_passes(self):
        harness = GRADER.build_test_harness(
            CORRECT_SET_EQ_SOLUTION, MBPP_PLUS_SET_EQ_GROUND_TRUTH,
        )
        assert run_harness(harness) == 0

    def test_wrong_set_eq_fails(self):
        harness = GRADER.build_test_harness(
            WRONG_SET_EQ_SOLUTION, MBPP_PLUS_SET_EQ_GROUND_TRUTH,
        )
        assert run_harness(harness) != 0


class TestMbppPlusAtolHarness:
    def test_correct_atol_passes(self):
        harness = GRADER.build_test_harness(
            CORRECT_ATOL_SOLUTION, MBPP_PLUS_ATOL_GROUND_TRUTH,
        )
        assert run_harness(harness) == 0

    def test_wrong_atol_fails(self):
        harness = GRADER.build_test_harness(
            WRONG_ATOL_SOLUTION, MBPP_PLUS_ATOL_GROUND_TRUTH,
        )
        assert run_harness(harness) != 0
