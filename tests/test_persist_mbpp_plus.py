"""Tests for data_layer/persist_mbpp_plus.py assert generation and prompt building."""

import subprocess
import sys

import pytest

from data_layer.persist_mbpp_plus import (
    build_chat_prompt,
    build_completion_prompt,
    build_ground_truth,
    extract_description,
    generate_assert,
    is_float_output,
    run_canonical,
)


# ---------------------------------------------------------------------------
# is_float_output
# ---------------------------------------------------------------------------

class TestIsFloatOutput:
    def test_scalar_float(self):
        assert is_float_output(3.14) is True

    def test_scalar_int(self):
        assert is_float_output(42) is False

    def test_list_of_floats(self):
        assert is_float_output([1.0, 2.0, 3.0]) is True

    def test_tuple_of_floats(self):
        assert is_float_output((1.0, 2.0)) is True

    def test_mixed_list(self):
        assert is_float_output([1.0, 2]) is False

    def test_empty_list(self):
        assert is_float_output([]) is False

    def test_string(self):
        assert is_float_output("hello") is False

    def test_none(self):
        assert is_float_output(None) is False


# ---------------------------------------------------------------------------
# generate_assert — default exact match
# ---------------------------------------------------------------------------

class TestGenerateAssertDefault:
    def test_int_output(self):
        result = generate_assert("add", [1, 2], 3, atol=0)
        assert result == "assert add(1, 2) == 3"

    def test_string_output(self):
        result = generate_assert("greet", ["world"], "hello world", atol=0)
        assert result == "assert greet('world') == 'hello world'"

    def test_list_output(self):
        result = generate_assert("sort_it", [[3, 1, 2]], [1, 2, 3], atol=0)
        assert result == "assert sort_it([3, 1, 2]) == [1, 2, 3]"

    def test_bool_output(self):
        result = generate_assert("is_even", [4], True, atol=0)
        assert result == "assert is_even(4) == True"

    def test_tuple_input(self):
        result = generate_assert("func", [(1, 2), (3, 4)], (1, 3), atol=0)
        assert result == "assert func((1, 2), (3, 4)) == (1, 3)"


# ---------------------------------------------------------------------------
# generate_assert — SET_EQ
# ---------------------------------------------------------------------------

class TestGenerateAssertSetEq:
    def test_set_eq_task(self):
        result = generate_assert(
            "similar_elements", [(3, 4, 5), (5, 7, 4)], (4, 5), atol=0,
        )
        assert "set(similar_elements(" in result
        assert "set(" in result

    def test_set_eq_runs_correctly(self):
        """The generated assert should pass when executed with a correct implementation."""
        code = "def similar_elements(a, b): return tuple(set(a) & set(b))\n"
        assert_str = generate_assert(
            "similar_elements", [(3, 4, 5, 6), (5, 7, 4, 10)], (4, 5), atol=0,
        )
        harness = f"{code}{assert_str}"
        result = subprocess.run(
            [sys.executable, "-c", harness], capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_set_eq_deterministic_repr(self):
        """SET_EQ asserts should be deterministic via sorted()."""
        r1 = generate_assert("Diff", [[1, 2, 3], [1]], [2, 3], atol=0)
        r2 = generate_assert("Diff", [[1, 2, 3], [1]], [3, 2], atol=0)
        # Both should produce the same sorted representation
        assert r1 == r2


# ---------------------------------------------------------------------------
# generate_assert — NOT_NONE
# ---------------------------------------------------------------------------

class TestGenerateAssertNotNone:
    def test_not_none_true(self):
        """When canonical returns something truthy, assert (fn() is not None) == True."""
        # Simulate a regex match (truthy object)
        result = generate_assert("check_str", ["hello"], "match_obj", atol=0)
        assert "is not None" in result
        assert "True" in result

    def test_not_none_false(self):
        """When canonical returns None, assert (fn() is not None) == False."""
        result = generate_assert("check_str", ["xyz"], None, atol=0)
        assert "is not None" in result
        assert "False" in result


# ---------------------------------------------------------------------------
# generate_assert — float tolerance
# ---------------------------------------------------------------------------

class TestGenerateAssertFloat:
    def test_scalar_float_explicit_atol(self):
        result = generate_assert("volume_sphere", [3], 113.097, atol=0.0001)
        assert "abs(" in result
        assert "0.0001" in result

    def test_scalar_float_auto_promote(self):
        """atol=0 with float output should auto-promote to 1e-6."""
        result = generate_assert("find_Volume", [1, 2, 3], 6.0, atol=0)
        assert "abs(" in result
        assert "1e-06" in result

    def test_int_output_no_promote(self):
        """atol=0 with int output should use exact match, not tolerance."""
        result = generate_assert("add", [1, 2], 3, atol=0)
        assert "abs(" not in result
        assert "==" in result

    def test_list_float_elementwise(self):
        result = generate_assert("div_list", [[1, 2], [3, 4]], [0.5, 0.5], atol=0)
        assert "_r = " in result
        assert "len(_r)" in result
        assert "_r[0]" in result
        assert "_r[1]" in result

    def test_float_tolerance_runs_correctly(self):
        """Tolerance assert passes for values within tolerance."""
        code = "def vol(r): import math; return (4/3) * math.pi * r**3\n"
        # Canonical output for r=1: ~4.1887902047863905
        assert_str = generate_assert("vol", [1], 4.1887902047863905, atol=0)
        harness = f"{code}{assert_str}"
        result = subprocess.run(
            [sys.executable, "-c", harness], capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"


# ---------------------------------------------------------------------------
# generate_assert — surface_Area oracle
# ---------------------------------------------------------------------------

class TestGenerateAssertSurfaceArea:
    def test_includes_oracle_alternative(self):
        """surface_Area asserts should accept oracle output when different from canonical."""
        from evalplus.eval._special_oracle import _surface_Area

        # Use an input where canonical and oracle might differ
        result = generate_assert("surface_Area", [3, 4], 42, atol=0)
        oracle = _surface_Area(3, 4)
        if oracle != 42:
            assert "_r" in result
            assert str(oracle) in result
        else:
            assert "==" in result


# ---------------------------------------------------------------------------
# generate_assert — digit_distance_nums oracle
# ---------------------------------------------------------------------------

class TestGenerateAssertDigitDistance:
    def test_includes_oracle_alternative(self):
        from evalplus.eval._special_oracle import _digit_distance_nums

        result = generate_assert("digit_distance_nums", [12, 123], 99, atol=0)
        oracle = _digit_distance_nums(12, 123)
        if oracle != 99:
            assert "_r" in result
            assert str(oracle) in result


# ---------------------------------------------------------------------------
# run_canonical
# ---------------------------------------------------------------------------

class TestRunCanonical:
    def test_simple_function(self):
        code = "\ndef add(a, b):\n    return a + b\n"
        outputs = run_canonical(code, "add", [[1, 2], [3, 4]])
        assert outputs == [3, 7]

    def test_function_with_import(self):
        code = "\nimport math\ndef area(r):\n    return math.pi * r * r\n"
        outputs = run_canonical(code, "area", [[1]])
        assert abs(outputs[0] - 3.141592653589793) < 1e-10


# ---------------------------------------------------------------------------
# extract_description
# ---------------------------------------------------------------------------

class TestExtractDescription:
    def test_standard_prompt(self):
        prompt = '"""\nWrite a function to find shared elements.\nassert similar_elements((3,4),(5,4)) == (4,)\n"""\n'
        assert extract_description(prompt) == "Write a function to find shared elements."

    def test_no_assert(self):
        prompt = '"""\nWrite a function to add two numbers.\n"""\n'
        assert extract_description(prompt) == "Write a function to add two numbers."

    def test_multiline_description(self):
        prompt = '"""\nWrite a function to do X.\nIt should handle Y too.\nassert fn(1) == 2\n"""\n'
        result = extract_description(prompt)
        assert "Write a function to do X." in result
        assert "It should handle Y too." in result


# ---------------------------------------------------------------------------
# build_completion_prompt
# ---------------------------------------------------------------------------

class TestBuildCompletionPrompt:
    def test_basic(self):
        prompt = '"""\nWrite a function.\nassert fn(1) == 2\n"""\n'
        completion_input, prefix = build_completion_prompt(prompt, "my_func")
        assert completion_input.endswith("def my_func(")
        assert prefix == "def my_func("

    def test_prompt_preserved(self):
        prompt = '"""\nDescription.\n"""\n'
        completion_input, _ = build_completion_prompt(prompt, "fn")
        assert '"""' in completion_input


# ---------------------------------------------------------------------------
# build_chat_prompt
# ---------------------------------------------------------------------------

class TestBuildChatPrompt:
    def test_includes_function_name(self):
        prompt = '"""\nDo something.\nassert fn(1) == 2\n"""\n'
        result = build_chat_prompt(prompt, "my_func", None)
        assert "`my_func`" in result

    def test_includes_assert_examples(self):
        prompt = '"""\nDo something.\nassert fn(1) == 2\n"""\n'
        result = build_chat_prompt(prompt, "fn", None)
        assert "assert fn(1) == 2" in result

    def test_includes_suffix(self):
        prompt = '"""\nDo something.\n"""\n'
        result = build_chat_prompt(prompt, "fn", "Use markdown.")
        assert "Use markdown." in result

    def test_no_suffix(self):
        prompt = '"""\nDo something.\n"""\n'
        result = build_chat_prompt(prompt, "fn", None)
        assert "Use markdown" not in result


# ---------------------------------------------------------------------------
# build_ground_truth — integration
# ---------------------------------------------------------------------------

class TestBuildGroundTruth:
    def test_multiple_asserts(self):
        code = "\ndef add(a, b):\n    return a + b\n"
        gt = build_ground_truth("add", code, [[1, 2], [3, 4]], atol=0)
        lines = gt.strip().split("\n")
        assert len(lines) == 2
        assert "add(1, 2) == 3" in lines[0]
        assert "add(3, 4) == 7" in lines[1]

    def test_harness_executes_correctly(self):
        """Ground truth + correct code should execute successfully."""
        code = "\ndef is_even(n):\n    return n % 2 == 0\n"
        gt = build_ground_truth("is_even", code, [[2], [3], [0]], atol=0)
        harness = f"def is_even(n):\n    return n % 2 == 0\n{gt}"
        result = subprocess.run(
            [sys.executable, "-c", harness], capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
