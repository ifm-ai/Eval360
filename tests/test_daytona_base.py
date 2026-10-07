"""
Tests for DaytonaGraderBase._async_grade_all_samples_nonblocking.

The grade_fn implementations use asyncio.sleep to simulate out-of-order
completion: later samples finish before earlier ones.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from daytona_sdk.common.errors import DaytonaError, DaytonaRateLimitError

from scheduler.grader.daytona_base import DaytonaGraderBase
from scheduler.utils import Sentinel, ExceptionWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockEvent:
    """Minimal event object that selects the noop parser."""

    model = "mock-model"
    parser_type = "noop"


def make_sample(i: int, delay: float = 0.0) -> dict:
    """Return a sample with a generation and optional artificial delay."""
    return {
        "index": i,
        "delay": delay,
        "generations": [f"gen-{i}"],
        "ground_truth": i,
    }


class DaytonaBaseTestBase:
    """Shared async setup and sandbox fakes for Daytona base tests.

    What: centralizes async generators and mocked Daytona sandbox construction.
    Executes: the same async sample streams and sandbox runner hooks used by the grader.
    Why: Daytona grading must verify sandbox lifecycle behavior without provisioning live remote sandboxes.
    """

    @staticmethod
    async def make_generator(*items):
        """Yield items as an async generator.

        What: adapts in-memory samples into the async stream expected by Daytona graders.
        Executes: the scheduler-style async iteration protocol.
        Why: avoids real scheduler setup while preserving stream ordering behavior.
        """
        for item in items:
            yield item

    @staticmethod
    async def make_async_iter(*items):
        """Yield existing records as an async iterator.

        What: adapts checkpoint records into the async stream expected by `run`.
        Executes: the existing-results iteration path used for resume behavior.
        Why: lets denominator tests cover resume records without filesystem checkpoints.
        """
        for item in items:
            yield item

    @staticmethod
    def make_sandbox_runner(response=None, exec_side_effect=None):
        """Return a grader and fake sandbox ready for `_run_in_sandbox` tests.

        What: wires mocked Daytona create/stop calls around a fake sandbox process.
        Executes: the sandbox execution path without creating a real remote sandbox.
        Why: tests retry, cleanup, and result handling deterministically.
        """
        grader = make_grader([])
        grader.ground_truth_to_test_list = MagicMock(return_value=[{"input": "1"}])
        grader.build_test_harness = MagicMock(return_value="print('ok')")
        sandbox = _FakeSandbox(response=response, exec_side_effect=exec_side_effect)
        grader._daytona.create = AsyncMock(return_value=sandbox)
        grader._daytona.stop = AsyncMock()
        return grader, sandbox


def make_grader(samples):
    """Return a DaytonaGraderBase instance with a fake samples_generator."""
    grader = DaytonaGraderBase.__new__(DaytonaGraderBase)
    # Minimal GraderBase state
    grader.samples_generator = DaytonaBaseTestBase.make_generator(*samples)
    grader.task = None
    grader._event_instance = MockEvent()
    grader._event_manager = MagicMock()
    grader._job_manager = MagicMock()
    grader.openai_connection = None
    grader._initialized = True  # skip initialize()
    grader._run_id = "test1234"
    grader._daytona = MagicMock()
    grader._semaphore = asyncio.Semaphore(600)
    from scheduler.grader.parser_registry import get_parser
    grader._parser = get_parser("noop")
    return grader


async def delay_grade_fn(sample: dict) -> dict:
    """Grade fn that sleeps for sample['delay'] seconds, then marks correct."""
    await asyncio.sleep(sample["delay"])
    sample["correct"] = [1]
    return sample


async def instant_grade_fn(sample: dict) -> dict:
    """Grade fn that returns immediately with a correct result."""
    sample["correct"] = [1]
    return sample


async def failing_grade_fn(sample: dict) -> dict:
    """Grade fn that raises to exercise ExceptionWrapper handling."""
    raise ValueError(f"grade failed for sample {sample['index']}")


async def collect(grader, grade_fn, start=0):
    """Collect all results from the nonblocking Daytona sample grader."""
    results = []
    async for item in grader._async_grade_all_samples_nonblocking(
        start=start, grade_fn=grade_fn
    ):
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNonblockingOrder(DaytonaBaseTestBase):
    @pytest.mark.asyncio
    async def test_results_yielded_in_input_order(self):
        """Even when sample 0 is slow and sample 1 is fast, output is in order."""
        samples = [
            make_sample(0, delay=0.1),
            make_sample(1, delay=0.0),
            make_sample(2, delay=0.0),
            Sentinel.COMPLETED,
        ]
        grader = make_grader(samples)
        results = await collect(grader, delay_grade_fn)

        # Sentinel is last
        assert results[-1] == Sentinel.COMPLETED
        grades = results[:-1]
        assert len(grades) == 3
        assert [r["index"] for r in grades] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_reverse_order_delays(self):
        """Slowest sample first, fastest last — output still ordered."""
        samples = [
            make_sample(0, delay=0.09),
            make_sample(1, delay=0.06),
            make_sample(2, delay=0.03),
            make_sample(3, delay=0.0),
            Sentinel.COMPLETED,
        ]
        grader = make_grader(samples)
        results = await collect(grader, delay_grade_fn)

        assert results[-1] == Sentinel.COMPLETED
        grades = results[:-1]
        assert [r["index"] for r in grades] == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_random_order_delays(self):
        """Mixed delays — output stays in input order regardless of completion order."""
        delays = [0.05, 0.01, 0.03, 0.02, 0.05, 0.01,
                  0.04, 0.02, 0.05, 0.01, 0.03, 0.02,
                  0.04, 0.01, 0.05, 0.02, 0.03, 0.01,
                  0.04, 0.02]
        samples = [make_sample(i, delay=d) for i, d in enumerate(delays)]
        samples.append(Sentinel.COMPLETED)
        grader = make_grader(samples)

        results = await collect(grader, delay_grade_fn)

        assert results[-1] == Sentinel.COMPLETED
        grades = results[:-1]
        assert len(grades) == 20
        assert [r["index"] for r in grades] == list(range(20))

    @pytest.mark.asyncio
    async def test_nonblocking_concurrent_execution(self):
        """Grading tasks run concurrently: total time should be ~max(delay), not sum(delays)."""
        delays = [0.05, 0.01, 0.03, 0.02, 0.05, 0.01,
                  0.04, 0.02, 0.05, 0.01, 0.03, 0.02,
                  0.04, 0.01, 0.05, 0.02, 0.03, 0.01,
                  0.04, 0.02]
        samples = [make_sample(i, delay=d) for i, d in enumerate(delays)]
        samples.append(Sentinel.COMPLETED)
        grader = make_grader(samples)

        import time
        t0 = time.monotonic()
        await collect(grader, delay_grade_fn)
        elapsed = time.monotonic() - t0

        sequential_time = sum(delays)  # ~0.56s if run one-by-one
        max_delay = max(delays)        # ~0.05s if truly concurrent
        # Allow 10x max_delay as a generous upper bound for overhead,
        # still well under sequential time.
        assert elapsed < sequential_time / 2, (
            f"Elapsed {elapsed:.3f}s is too close to sequential {sequential_time:.3f}s — "
            f"grading may not be running concurrently"
        )

    @pytest.mark.asyncio
    async def test_start_skips_samples(self):
        """start=2 skips first two samples."""
        samples = [
            make_sample(0),
            make_sample(1),
            make_sample(2),
            make_sample(3),
            Sentinel.COMPLETED,
        ]
        grader = make_grader(samples)
        results = await collect(grader, instant_grade_fn, start=2)

        assert results[-1] == Sentinel.COMPLETED
        grades = results[:-1]
        assert len(grades) == 2
        assert [r["index"] for r in grades] == [2, 3]

    @pytest.mark.asyncio
    async def test_empty_generator(self):
        """Only a Sentinel — no grades yielded."""
        grader = make_grader([Sentinel.COMPLETED])
        results = await collect(grader, instant_grade_fn)
        assert results == [Sentinel.COMPLETED]

    @pytest.mark.asyncio
    async def test_exception_in_grade_fn_yields_exception_wrapper(self):
        """A grade_fn that raises should yield an ExceptionWrapper to the caller."""
        samples = [make_sample(0), Sentinel.COMPLETED]
        grader = make_grader(samples)
        results = await collect(grader, failing_grade_fn)
        wrappers = [r for r in results if isinstance(r, ExceptionWrapper)]
        assert len(wrappers) == 1
        assert isinstance(wrappers[0].exception, ValueError)

    @pytest.mark.asyncio
    async def test_correct_field_preserved(self):
        """grade_fn result fields are passed through intact."""
        samples = [make_sample(0), make_sample(1), Sentinel.COMPLETED]
        grader = make_grader(samples)
        results = await collect(grader, instant_grade_fn)

        grades = [r for r in results if r != Sentinel.COMPLETED]
        for grade in grades:
            assert grade["correct"] == [1]

    @pytest.mark.asyncio
    async def test_all_samples_concurrent(self):
        """With large delays, all samples should still complete (not deadlock)."""
        samples = [make_sample(i, delay=0.05) for i in range(10)]
        samples.append(Sentinel.COMPLETED)
        grader = make_grader(samples)

        results = await asyncio.wait_for(
            collect(grader, delay_grade_fn), timeout=1.0
        )
        assert results[-1] == Sentinel.COMPLETED
        assert len(results) == 11  # 10 grades + sentinel


# ---------------------------------------------------------------------------
# sandbox_params
# ---------------------------------------------------------------------------

class TestSandboxParams(DaytonaBaseTestBase):
    def test_labels_include_eval360_tag(self):
        grader = make_grader([])
        params = grader.sandbox_params()
        assert params.labels == {"app": "eval360"}

    def test_sandbox_name_format(self):
        import getpass
        grader = make_grader([])
        grader._event_instance = MagicMock()
        grader._event_instance.model = "k2-v2-instruct"
        grader.task = MagicMock()
        grader.task.dataset_name = "lcbv6"
        name = grader._sandbox_name(row=42, gen_index=1)
        user = getpass.getuser()
        assert name.startswith(f"{user}-eval360-k2-v2-instruct-lcbv6-row-42-gen-1-")
        assert name.endswith(grader._run_id)
        assert len(grader._run_id) == 8

    def test_sandbox_params_name_passed_through(self):
        grader = make_grader([])
        params = grader.sandbox_params(name="my-sandbox")
        assert params.name == "my-sandbox"


# ---------------------------------------------------------------------------
# _parse_retry_after / _run_in_sandbox / notifications
# ---------------------------------------------------------------------------

class _FakeSandbox:
    """Minimal async Daytona sandbox test double."""

    def __init__(self, response=None, exec_side_effect=None):
        """Create fake fs/process APIs for DaytonaGraderBase._run_in_sandbox()."""
        self.id = "sandbox-1"
        self.fs = MagicMock()
        self.fs.upload_file = AsyncMock()
        self.process = MagicMock()
        self.process.exec = AsyncMock(return_value=response, side_effect=exec_side_effect)


class TestRetryAfterParsing(DaytonaBaseTestBase):
    """What: groups tests for Daytona retry-after header parsing.
    Executes: `DaytonaGraderBase._parse_retry_after()` with mixed retry header dictionaries.
    Why: parsed retry delays control deterministic rate-limit backoff behavior.
    """

    def test_parse_retry_after_uses_smallest_numeric_header(self):
        """What: verifies multiple retry headers resolve to the smallest numeric value.
        Executes: `_parse_retry_after()` with numeric and non-numeric Retry-After-* headers.
        Why: covers the backoff selection logic used after Daytona rate-limit errors.
        """
        grader = make_grader([])
        headers = {
            "Retry-After-Requests": "2.5",
            "retry-after-tokens": "7",
            "Retry-After-Bad": "not-a-number",
        }
        assert grader._parse_retry_after(headers, default=60.0) == pytest.approx(2.5)

    def test_parse_retry_after_returns_default_for_missing_values(self):
        """What: verifies missing retry-after headers fall back to the supplied default.
        Executes: `_parse_retry_after()` with unrelated headers only.
        Why: covers the default backoff path when Daytona supplies no retry metadata.
        """
        grader = make_grader([])
        assert grader._parse_retry_after({"x": "1"}, default=12.0) == pytest.approx(12.0)


class TestRunInSandbox(DaytonaBaseTestBase):
    """What: groups tests for Daytona sandbox execution using mocked SDK objects.
    Executes: `DaytonaGraderBase._run_in_sandbox()` with fake Daytona create/exec/stop APIs.
    Why: sandbox orchestration is core to code grading and must be covered without live Daytona calls.
    """

    @pytest.mark.asyncio
    async def test_success_uploads_harness_executes_and_stops_sandbox(self):
        """What: verifies successful sandbox execution uploads the harness and returns pass.
        Executes: `_run_in_sandbox()` through mocked create, upload_file, process.exec, and stop calls.
        Why: covers the primary sandbox success path without provisioning a remote sandbox.
        """
        response = MagicMock(exit_code=0)
        grader, sandbox = self.make_sandbox_runner(response=response)

        with patch("scheduler.grader.daytona_base.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            grade, details = await grader._run_in_sandbox("solution", "ground truth", row=2, gen_index=3)

        assert grade == "pass"
        assert details is None
        mock_sleep.assert_awaited_once_with(5)
        grader._daytona.create.assert_awaited_once()
        sandbox.fs.upload_file.assert_awaited_once_with(b"print('ok')", "/tmp/harness.py")
        sandbox.process.exec.assert_awaited_once_with("python /tmp/harness.py", timeout=30)
        grader._daytona.stop.assert_awaited_once_with(sandbox)

    @pytest.mark.asyncio
    async def test_daytona_408_exec_error_returns_timeout_and_stops(self):
        """What: verifies a Daytona 408 during exec is graded as timeout and still cleans up.
        Executes: `_run_in_sandbox()` with `sandbox.process.exec` raising `DaytonaError(408)`.
        Why: covers the remote execution timeout branch while asserting sandbox stop is awaited.
        """
        error = DaytonaError("timed out", status_code=408)
        grader, sandbox = self.make_sandbox_runner(exec_side_effect=error)

        with patch("scheduler.grader.daytona_base.asyncio.sleep", new=AsyncMock()):
            grade, details = await grader._run_in_sandbox("solution", "ground truth", row=0, gen_index=0)

        assert grade == "timeout"
        assert details is None
        grader._daytona.stop.assert_awaited_once_with(sandbox)

    @pytest.mark.asyncio
    async def test_rate_limit_error_sleeps_then_retries(self):
        """What: verifies daytona rate limits are retried using parsed Retry-After headers.
        Executes: `_run_in_sandbox()` with `DaytonaRateLimitError` followed by a successful sandbox.
        Why: covers retry scheduling and proves the parsed retry delay reaches asyncio.sleep.
        """
        response = MagicMock(exit_code=0)
        sandbox = _FakeSandbox(response=response)
        grader = make_grader([])
        grader.ground_truth_to_test_list = MagicMock(return_value=[{"input": "1"}])
        grader.build_test_harness = MagicMock(return_value="print('ok')")
        grader._daytona.create = AsyncMock(side_effect=[
            DaytonaRateLimitError("limited", headers={"Retry-After-Requests": "2.5"}),
            sandbox,
        ])
        grader._daytona.stop = AsyncMock()

        with patch("scheduler.grader.daytona_base.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            grade, _ = await grader._run_in_sandbox("solution", "ground truth", row=0, gen_index=0)

        assert grade == "pass"
        assert grader._daytona.create.await_count == 2
        assert [call.args[0] for call in mock_sleep.await_args_list] == [5, 2.5]

    @pytest.mark.asyncio
    async def test_stop_failure_notifies_leaked_sandbox(self):
        """What: verifies sandbox stop failures log the leaked-sandbox error.
        Executes: `_run_in_sandbox()` with a successful exec and failing `daytona.stop`.
        Why: covers the orphaned-sandbox operator signal corner case.
        """
        response = MagicMock(exit_code=0)
        grader, sandbox = self.make_sandbox_runner(response=response)
        grader._daytona.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

        with (
            patch("scheduler.grader.daytona_base.asyncio.sleep", new=AsyncMock()),
            patch("scheduler.grader.daytona_base.logger") as mock_logger,
        ):
            grade, _ = await grader._run_in_sandbox("solution", "ground truth", row=0, gen_index=0)

        assert grade == "pass"
        mock_logger.error.assert_called_once_with(
            "LEAKED SANDBOX: %s could not be stopped. "
            "Please log in to Daytona and stop/delete it manually.",
            sandbox.id,
        )


# ---------------------------------------------------------------------------
# run() denominator tests — ExceptionWrapper must count toward denominator
# ---------------------------------------------------------------------------

def make_run_grader(grade_results):
    """Return a grader whose _async_grade_all_samples_nonblocking yields grade_results."""
    grader = DaytonaGraderBase.__new__(DaytonaGraderBase)
    grader.task = None
    grader._event_instance = MockEvent()
    grader._event_manager = MagicMock()
    grader._job_manager = MagicMock()
    grader.openai_connection = None
    grader._initialized = True
    grader._run_id = "test1234"
    grader._daytona = MagicMock()
    grader._semaphore = asyncio.Semaphore(600)
    from scheduler.grader.parser_registry import get_parser
    grader._parser = get_parser("noop")
    grader.samples_generator = DaytonaBaseTestBase.make_generator()

    async def fake_grade_all(start, grade_fn, **kwargs):
        for item in grade_results:
            yield item

    grader._async_grade_all_samples_nonblocking = fake_grade_all

    async def fake_cleanup():
        pass

    grader.cleanup = fake_cleanup
    return grader


async def collect_run(grader, existing_records=None):
    existing = DaytonaBaseTestBase.make_async_iter(*(existing_records or []))
    results = []
    async for item in grader.run(existing=existing, average_over=[1], pass_at=[1]):
        results.append(item)
    return results


class TestRunDenominator(DaytonaBaseTestBase):
    @pytest.mark.asyncio
    async def test_exception_wrapper_counts_toward_denominator(self):
        """ExceptionWrapper results must be counted as incorrect (not excluded from denominator)."""
        from scheduler.grader.base import Score
        # 1 correct, 1 ExceptionWrapper → accuracy = 1/2, not 1/1
        grade_results = [
            {"correct": [1]},
            ExceptionWrapper(exception=RuntimeError("fail"), trace="fail", instance={}),
            Sentinel.COMPLETED,
        ]
        grader = make_run_grader(grade_results)
        results = await collect_run(grader)
        scores = [r for r in results if isinstance(r, Score)]
        avg_score = next(s for s in scores if "avg over" in s.name)
        assert avg_score.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_checkpoint_exception_counts_toward_denominator(self):
        """Checkpoint records with 'exception' (no 'correct') must count as incorrect."""
        from scheduler.grader.base import Score
        # Checkpoint has 1 correct + 1 exception → all_correct = [[1], [0]], accuracy = 0.5
        existing_records = [
            {"correct": [1], "row": 0},
            {"exception": "some error", "row": 1},
        ]
        grade_results = [Sentinel.COMPLETED]
        grader = make_run_grader(grade_results)
        results = await collect_run(grader, existing_records=existing_records)
        scores = [r for r in results if isinstance(r, Score)]
        avg_score = next(s for s in scores if "avg over" in s.name)
        assert avg_score.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_mixed_correct_and_exceptions_denominator(self):
        """Mix of correct results and exceptions — denominator is the total count."""
        from scheduler.grader.base import Score
        # 3 correct, 2 exceptions → accuracy = 3/5 = 0.6
        grade_results = [
            {"correct": [1]},
            {"correct": [1]},
            ExceptionWrapper(exception=RuntimeError("e1"), trace="e1", instance={}),
            {"correct": [1]},
            ExceptionWrapper(exception=RuntimeError("e2"), trace="e2", instance={}),
            Sentinel.COMPLETED,
        ]
        grader = make_run_grader(grade_results)
        results = await collect_run(grader)
        scores = [r for r in results if isinstance(r, Score)]
        avg_score = next(s for s in scores if "avg over" in s.name)
        assert avg_score.value == pytest.approx(0.6)
