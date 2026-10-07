"""Tests for the local MBPP grader helpers and run flow."""

from __future__ import annotations

import os
import subprocess
from typing import Any, AsyncIterator

import pytest

from scheduler.grader import mbpp_local as mbpp_local_module
from scheduler.grader.base import Grade, Score
from scheduler.grader.mbpp_local import MBPPLocalGrader
from scheduler.utils import Sentinel


class _MockEvent:
    """Minimal event object that selects passthrough parsing."""

    parser_type = "passthrough"


class MBPPLocalTestBase:
    """Shared async grader setup for local MBPP tests.

    What: centralizes fake sample streams and run collection helpers.
    Executes: the MBPPLocalGrader constructor and inherited `run` stream.
    Why: MBPP local grading must cover subprocess boundaries while preserving the async stream contract.
    """

    @staticmethod
    async def async_iter(*items: Any) -> AsyncIterator[Any]:
        """Yield items as an async iterator for grader tests.

        What: adapts in-memory MBPP samples into the async stream expected by graders.
        Executes: the same async iteration protocol used by scheduler pipelines.
        Why: avoids scheduler dependencies while preserving streaming semantics.
        """
        for item in items:
            yield item

    @classmethod
    def make_grader(cls, *samples: Any) -> MBPPLocalGrader:
        """Create an MBPPLocalGrader with fake scheduler dependencies.

        What: builds the grader with a passthrough parser event and sample stream.
        Executes: MBPPLocalGrader initialization without job-manager state.
        Why: isolates local MBPP grading logic from unrelated scheduler wiring.
        """
        return MBPPLocalGrader(
            samples_generator=cls.async_iter(*samples),
            event_manager=None,
            job_manager=None,
            event=_MockEvent(),
            task=None,
        )

    @classmethod
    async def collect_run(
        cls,
        grader: MBPPLocalGrader,
        average_over: list[int] | None = None,
        pass_at: list[int] | None = None,
    ) -> list[Any]:
        """Collect all objects yielded by MBPPLocalGrader.run().

        What: materializes the async run stream for assertions.
        Executes: inherited `run` with empty existing records and explicit score settings.
        Why: lets tests verify emitted grades, scores, and sentinels deterministically.
        """
        results = []
        async for item in grader.run(
            existing=cls.async_iter(),
            average_over=average_over or [],
            pass_at=pass_at or [],
        ):
            results.append(item)
        return results


class TestExtractCode(MBPPLocalTestBase):
    """What: groups tests for markdown code extraction.
    Executes: `mbpp_local._extract_code()` with plain text and fenced Python blocks.
    Why: extracted code is the only program text passed into the local MBPP subprocess runner.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("", ""),
            ("plain prose", ""),
            ("```python\ndef add(a, b):\n    return a + b\n```", "def add(a, b):\n    return a + b"),
            ("```py\nx = 1\n```", "x = 1"),
            ("```\ny = 2\n```", "y = 2"),
            ("prefix\n```python\nz = 3", "z = 3"),
        ],
    )
    def test_extract_code_handles_fences_and_empty_text(self, text: str, expected: str) -> None:
        """What: verifies code extraction supports fenced, unclosed, and empty generations.
        Executes: `_extract_code()` across empty, prose-only, closed-fence, and unclosed-fence inputs.
        Why: covers common model-output formats before local execution is attempted.
        """
        assert mbpp_local_module._extract_code(text) == expected

    def test_extract_code_uses_first_fenced_block(self) -> None:
        """What: verifies only the first fenced code block is used for MBPP execution.
        Executes: `_extract_code()` on text containing multiple fenced code blocks.
        Why: protects the deterministic first-block selection used for grading.
        """
        text = "```python\nfirst = True\n```\n```python\nsecond = True\n```"
        assert mbpp_local_module._extract_code(text) == "first = True"


class TestRunTest(MBPPLocalTestBase):
    """What: groups tests for the subprocess wrapper used by local MBPP grading.
    Executes: `mbpp_local._run_test()` with monkeypatched subprocess behavior.
    Why: the subprocess boundary is the core local execution path and should remain deterministic.
    """

    def test_run_test_success_invokes_python_with_stripped_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """What: verifies successful subprocess return code 0 maps to True and uses safe defaults.
        Executes: `_run_test()` with `subprocess.run` patched to capture command, cwd, env, and timeout.
        Why: covers the safe local execution contract without launching a live interpreter.
        """
        captured: dict[str, Any] = {}

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            """Record subprocess parameters and return success."""
            captured["cmd"] = cmd
            captured.update(kwargs)
            captured["cwd_exists"] = os.path.isdir(kwargs["cwd"])
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(mbpp_local_module.subprocess, "run", fake_run)

        assert mbpp_local_module._run_test("def add(a, b): return a + b", "assert add(1, 2) == 3") is True
        assert captured["cmd"][0:2] == ["python3", "-c"]
        assert "def add(a, b): return a + b\nassert add(1, 2) == 3\n" in captured["cmd"][2]
        assert captured["capture_output"] is True
        assert captured["timeout"] == 10.0
        assert captured["cwd_exists"] is True
        assert captured["env"] == {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONUNBUFFERED": "1",
            "LANG": "C.UTF-8",
        }

    def test_run_test_nonzero_exit_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What: verifies nonzero subprocess exit codes are graded as failures.
        Executes: `_run_test()` with a fake `CompletedProcess` return code of 1.
        Why: covers failed assertion commands mapping to an incorrect answer.
        """

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            """Return a failing subprocess result."""
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"AssertionError")

        monkeypatch.setattr(mbpp_local_module.subprocess, "run", fake_run)

        assert mbpp_local_module._run_test("def f(): pass", "assert False") is False

    def test_run_test_timeout_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What: verifies timeoutExpired is caught and graded as a failed test.
        Executes: `_run_test()` with `subprocess.run` patched to raise `TimeoutExpired`.
        Why: covers the local infinite-loop timeout path without sleeping.
        """

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            """Raise a timeout using the Python 3.14-compatible constructor."""
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        monkeypatch.setattr(mbpp_local_module.subprocess, "run", fake_run)

        assert mbpp_local_module._run_test("while True: pass", "assert True", timeout=0.1) is False

    def test_run_test_unexpected_exception_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What: verifies unexpected subprocess errors are contained and graded as failures.
        Executes: `_run_test()` with `subprocess.run` patched to raise `OSError`.
        Why: covers interpreter-launch failures while preserving the public boolean contract.
        """

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            """Raise an arbitrary OS error from subprocess.run."""
            raise OSError("python missing")

        monkeypatch.setattr(mbpp_local_module.subprocess, "run", fake_run)

        assert mbpp_local_module._run_test("def f(): pass", "assert True") is False


class TestGradeSample(MBPPLocalTestBase):
    """What: groups tests for MBPPLocalGrader.grade_sample().
    Executes: `MBPPLocalGrader.grade_sample()` with sentinels, malformed samples, and mocked tests.
    Why: grade_sample connects parsed generations to extracted code and correctness arrays.
    """

    @pytest.mark.asyncio
    async def test_sentinel_is_returned_unchanged(self) -> None:
        """What: verifies sentinel.COMPLETED passes through without local execution.
        Executes: `MBPPLocalGrader.grade_sample()` with `Sentinel.COMPLETED`.
        Why: preserves the stream-termination contract used by inherited run loops.
        """
        grader = self.make_grader()
        assert await grader.grade_sample(Sentinel.COMPLETED) == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_grade_sample_uses_raw_fallback_and_list_tests(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """What: verifies none parsed generations fall back to raw generations and list tests are joined.
        Executes: `grade_sample()` with `_run_test` patched after code extraction succeeds.
        Why: covers raw-generation fallback, list ground-truth joining, and no-code handling deterministically.
        """
        calls: list[tuple[str, str]] = []

        def fake_run_test(code: str, test_code: str) -> bool:
            """Record code/test pairs and return deterministic outcomes."""
            calls.append((code, test_code))
            return len(calls) == 1

        monkeypatch.setattr(mbpp_local_module, "_run_test", fake_run_test)
        grader = self.make_grader()
        sample = {
            "parsed_generations": [
                None,
                "```python\ndef is_even(n):\n    return True\n```",
                "no code here",
            ],
            "generations": [
                "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
            ],
            "ground_truth": [
                "assert is_even(2) is True",
                "assert is_even(3) is False",
            ],
        }

        result = await grader.grade_sample(sample)

        joined_tests = "assert is_even(2) is True\nassert is_even(3) is False"
        assert result["correct"] == [True, False, False]
        assert result["accuracy"] == pytest.approx(1 / 3)
        assert result["picked"][0] == "def is_even(n):\n    return n % 2 == 0"
        assert calls == [
            ("def is_even(n):\n    return n % 2 == 0", joined_tests),
            ("def is_even(n):\n    return True", joined_tests),
        ]
        assert "correct" not in sample

    @pytest.mark.asyncio
    async def test_grade_sample_requires_dict_shape(self) -> None:
        """What: verifies invalid samples fail fast with an assertion instead of subprocess execution.
        Executes: `MBPPLocalGrader.grade_sample()` with a non-dict sample.
        Why: covers input validation before any local subprocess code path can run.
        """
        grader = self.make_grader()
        with pytest.raises(AssertionError, match="sample must be a dict"):
            await grader.grade_sample("not a dict")


class TestRunAggregation(MBPPLocalTestBase):
    """What: groups tests for inherited run() defaults and error records.
    Executes: `MBPPLocalGrader.run()` over mocked async sample streams.
    Why: inherited aggregation emits the public MBPP grades, scores, and completion sentinel.
    """

    @pytest.mark.asyncio
    async def test_run_defaults_emit_accuracy_scores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What: verifies empty average/pass lists default to avg@1/pass@1 for local MBPP.
        Executes: `MBPPLocalGrader.run()` with `_run_test` patched to deterministic outcomes.
        Why: covers default scoring aggregation for correct and no-code generations.
        """
        monkeypatch.setattr(mbpp_local_module, "_run_test", lambda code, test_code: "correct" in code)
        samples = [
            {
                "row": 0,
                "generations": ["```python\ncorrect = True\n```"],
                "ground_truth": "assert True",
            },
            {
                "row": 1,
                "generations": ["no fenced code"],
                "ground_truth": "assert True",
            },
            Sentinel.COMPLETED,
        ]
        grader = self.make_grader(*samples)

        results = await self.collect_run(grader)

        grades = [item for item in results if isinstance(item, Grade)]
        scores = {item.name: item.value for item in results if isinstance(item, Score)}
        assert [grade.element["correct"] for grade in grades] == [[True], [False]]
        assert scores["accuracy (avg over 1)"] == pytest.approx(0.5)
        assert scores["accuracy (pass@1)"] == pytest.approx(0.5)
        assert results[-1] == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_run_marks_exception_records_incorrect(self) -> None:
        """What: verifies upstream exception records count as incorrect grade elements.
        Executes: `MBPPLocalGrader.run()` with an input sample carrying an exception marker.
        Why: covers the scheduler failure-record path as an incorrect MBPP grade.
        """
        samples = [{"row": 0, "exception": "generation failed", "generations": ["ignored"]}, Sentinel.COMPLETED]
        grader = self.make_grader(*samples)

        results = await self.collect_run(grader)

        grades = [item for item in results if isinstance(item, Grade)]
        scores = {item.name: item.value for item in results if isinstance(item, Score)}
        assert grades[0].element["correct"] == [0]
        assert scores["accuracy (avg over 1)"] == pytest.approx(0.0)
