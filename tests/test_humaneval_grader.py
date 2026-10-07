"""Tests for the HumanEval grader and code_completion parser."""
import asyncio
from unittest.mock import patch

import pytest

from scheduler.grader import get_grader
from scheduler.grader.humaneval import DEFAULT_TIMEOUT
from scheduler.grader.parser_registry import get_parser
from scheduler.utils import Sentinel


# ---------------------------------------------------------------------------
# Fixtures: a synthetic problem in the HumanEval record shape (not a benchmark item)
# ---------------------------------------------------------------------------

PROMPT = (
    "from typing import List\n\n\n"
    "def has_negative(values: List[int]) -> bool:\n"
    '    """ Return True if any number in the list is negative.\n'
    "    >>> has_negative([1, 2, 3])\n"
    "    False\n"
    "    >>> has_negative([4, -1, 0])\n"
    "    True\n"
    '    """\n'
)

TEST_CODE = (
    "\n\nMETADATA = {\n    'author': 'synthetic',\n    'dataset': 'test'\n}\n\n\n"
    "def check(candidate):\n"
    "    assert candidate([5, -2, 7]) == True\n"
    "    assert candidate([0, 3, 9]) == False\n"
    "    assert candidate([-8]) == True\n"
    "    assert candidate([]) == False\n\n"
)

CORRECT_SOLUTION = (
    "    for value in values:\n"
    "        if value < 0:\n"
    "            return True\n"
    "\n"
    "    return False\n"
)

WRONG_SOLUTION = "    return False\n"

TIMEOUT_SOLUTION = "    while True: pass\n"


class HumanEvalTestBase:
    """Shared HumanEval sample construction for grader and parser tests.

    What: centralizes a synthetic problem fixture in the HumanEval record shape.
    Executes: grade_sample inputs that mirror the benchmark record shape.
    Why: HumanEval grading depends on benchmark-shaped samples with aligned raw and parsed generations.
    """

    @staticmethod
    def make_sample(generations):
        """Return a HumanEval sample with matching raw and parsed generations.

        What: builds the benchmark fields required by HumanEval grading.
        Executes: the code-completion grading path with a stable task fixture.
        Why: avoids brittle duplication of prompt, tests, and entry-point metadata.
        """
        return {
            "row": 0,
            "completion_input": PROMPT,
            "chat_input": [{"role": "user", "content": PROMPT}],
            "ground_truth": {"test": TEST_CODE, "entry_point": "has_negative"},
            "task_id": "Synthetic/0",
            "generations": generations,
            "parsed_generations": generations,
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestHumanEvalRegistration(HumanEvalTestBase):
    def test_get_grader_humaneval(self):
        from scheduler.grader.humaneval import HumanEval
        assert get_grader("humaneval") is HumanEval

    def test_get_grader_aliases(self):
        cls = get_grader("humaneval")
        assert get_grader("human_eval") is cls
        assert get_grader("human-eval") is cls


# ---------------------------------------------------------------------------
# Grader (grade_sample)
# ---------------------------------------------------------------------------

class TestHumanEvalGrading(HumanEvalTestBase):

    @pytest.fixture
    def grader(self):
        """Create a minimal HumanEval grader instance (no real scheduler wiring)."""
        from scheduler.grader.humaneval import HumanEval

        async def _empty_gen():
            return
            yield  # make it an async generator

        return HumanEval(
            samples_generator=_empty_gen(),
            event_manager=None,
            job_manager=None,
            event=type("E", (), {"parser_type": "passthrough"})(),
            task=None,
        )

    def test_grade_sample_raises_not_implemented(self, grader):
        """Subprocess execution is disabled until robust sandbox is integrated."""
        sample = self.make_sample([CORRECT_SOLUTION])
        with pytest.raises(NotImplementedError, match="Subprocess-based code execution"):
            asyncio.run(grader.grade_sample(sample))

    def test_sentinel_passthrough(self, grader):
        result = asyncio.run(grader.grade_sample(Sentinel.COMPLETED))
        assert result == Sentinel.COMPLETED

    def test_check_correctness_raises_not_implemented(self):
        """check_correctness directly raises NotImplementedError."""
        from scheduler.grader.humaneval import check_correctness
        problem = {
            "task_id": "Synthetic/0",
            "prompt": PROMPT,
            "test": TEST_CODE,
            "entry_point": "has_negative",
        }
        with pytest.raises(NotImplementedError, match="Subprocess-based code execution"):
            check_correctness(problem, CORRECT_SOLUTION, 10.0)

    @pytest.mark.asyncio
    async def test_initialize_uses_task_timeout_metadata(self, grader):
        """What: verifies task metadata overrides the default HumanEval timeout.
        Executes: `HumanEval.initialize()` with a fake task carrying string timeout metadata.
        Why: covers the configuration path that controls executor time limits for code grading.
        """
        grader.task = type("Task", (), {"meta": {"timeout": "2.5"}})()

        await grader.initialize()

        assert grader._timeout == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_grade_sample_with_mocked_check_correctness_strips_completion_variants(self, grader):
        """What: verifies mocked check_correctness receives prompt-stripped and signature-stripped completions.
        Executes: `HumanEval.grade_sample()` with `check_correctness` patched to record executor calls.
        Why: covers prompt fallback, raw-generation fallback, and signature stripping without running untrusted code.
        """
        full_function = (
            "def has_negative(values: List[int]) -> bool:\n"
            '    """Duplicate docstring."""\n'
            "    return True\n"
        )
        sample = self.make_sample([PROMPT + CORRECT_SOLUTION, None, full_function])
        sample["generations"] = ["unused", WRONG_SOLUTION, "unused"]
        calls = []
        outcomes = iter([True, False, True])

        def fake_check_correctness(problem, completion, timeout):
            """Record executor calls and return deterministic pass/fail results."""
            calls.append((problem, completion, timeout))
            return {"passed": next(outcomes)}

        with patch("scheduler.grader.humaneval.check_correctness", side_effect=fake_check_correctness):
            result = await grader.grade_sample(sample)

        assert result["correct"] == [1, 0, 1]
        assert calls[0][1] == CORRECT_SOLUTION
        assert calls[1][1] == WRONG_SOLUTION
        assert calls[2][1] == "    return True\n"
        assert calls[0][0]["task_id"] == "Synthetic/0"
        assert calls[0][2] == pytest.approx(DEFAULT_TIMEOUT)


# ---------------------------------------------------------------------------
# Parser (code_completion)
# ---------------------------------------------------------------------------

class TestCodeCompletionParser(HumanEvalTestBase):
    @pytest.fixture
    def parser(self):
        return get_parser("code_completion")

    def test_bare_code(self, parser):
        code = "    return x + 1\n"
        result = parser.parse_generations([code])
        assert result == [code]

    def test_markdown_fence(self, parser):
        gen = "Here is the solution:\n```python\n    return x + 1\n```\n"
        result = parser.parse_generations([gen])
        assert result == ["    return x + 1\n"]

    def test_fence_no_language(self, parser):
        gen = "```\n    return x + 1\n```"
        result = parser.parse_generations([gen])
        assert result == ["    return x + 1\n"]

    def test_empty_generation(self, parser):
        result = parser.parse_generations(["   "])
        assert result == [None]

    def test_none_generation(self, parser):
        result = parser.parse_generations([None])
        assert result == [None]

    def test_think_tag_stripping(self, parser):
        gen = "<think>\nLet me think about this...\n</think>\n\n    return len(string)"
        result = parser.parse_generations([gen])
        assert result == ["    return len(string)"]

    def test_think_tag_with_fence(self, parser):
        gen = "<think>\nThinking...\n</think>\n\n```python\n    return len(string)\n```"
        result = parser.parse_generations([gen])
        assert result == ["    return len(string)\n"]

    def test_fence_py_shorthand(self, parser):
        gen = "```py\n    return x + 1\n```"
        result = parser.parse_generations([gen])
        assert result == ["    return x + 1\n"]

    def test_truncated_think_with_fence(self, parser):
        """Unclosed <think> (truncated by max_tokens) with a code fence inside."""
        gen = "<think>\nLet me reason...\n```python\n    return x + 1\n```\nMore thinking"
        result = parser.parse_generations([gen])
        assert result == ["    return x + 1\n"]

    def test_truncated_think_no_fence(self, parser):
        """Unclosed <think> with no fence — falls back to text after last blank line."""
        gen = "<think>\nLet me think step by step.\n\n    return x + 1"
        result = parser.parse_generations([gen])
        assert result == ["    return x + 1"]

    def test_truncated_think_all_reasoning(self, parser):
        """Unclosed <think> with only reasoning text and no code — returns last chunk."""
        gen = "<think>\nStep 1: analyze the input\n\nStep 2: process the data"
        result = parser.parse_generations([gen])
        assert result == ["Step 2: process the data"]


# ---------------------------------------------------------------------------
# Function signature stripping
# ---------------------------------------------------------------------------

class TestFunctionSignatureStripping:
    def test_strips_reemitted_def(self):
        from scheduler.grader.humaneval import _strip_function_signature
        gen = "def strlen(string: str) -> int:\n    return len(string)\n"
        result = _strip_function_signature(gen, "strlen")
        assert result == "    return len(string)\n"

    def test_preserves_bare_body(self):
        from scheduler.grader.humaneval import _strip_function_signature
        gen = "    return len(string)\n"
        result = _strip_function_signature(gen, "strlen")
        assert result == "    return len(string)\n"

    def test_strips_def_with_docstring(self):
        from scheduler.grader.humaneval import _strip_function_signature
        gen = 'def strlen(string: str) -> int:\n    """Return length."""\n    return len(string)\n'
        result = _strip_function_signature(gen, "strlen")
        assert result == "    return len(string)\n"
