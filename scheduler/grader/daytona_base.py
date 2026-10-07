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

import asyncio
from collections import Counter
import getpass
import logging
import traceback
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Tuple

from daytona_sdk import AsyncDaytona, CreateSandboxFromSnapshotParams
from daytona_sdk.common.errors import DaytonaError, DaytonaRateLimitError

from ..utils import Sentinel, ExceptionWrapper
from ..metrics import get_accuracy, get_bootstrap_accuracy_std, mean_pass_at_k
from .base import GraderBase, Grade, Score

logger = logging.getLogger(__name__)

# Maximum number of sandboxes running concurrently.
_DEFAULT_CONCURRENCY = 20


class DaytonaGraderBase(GraderBase):
    """Base class for graders that use Daytona sandboxes for isolated code execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._daytona: AsyncDaytona | None = None
        self._semaphore = asyncio.Semaphore(_DEFAULT_CONCURRENCY)
        self._run_id = uuid.uuid4().hex[:8]

    async def initialize(self):
        """Called once, before the first sample is graded."""
        # AsyncDaytona reads DAYTONA_API_KEY (and optionally DAYTONA_SERVER_URL)
        # from the environment automatically.
        self._daytona = AsyncDaytona()
        logger.info("AsyncDaytona client initialised")

    async def cleanup(self):
        """Close the AsyncDaytona HTTP session. Call when grading is done."""
        if self._daytona is not None:
            await self._daytona.close()
            self._daytona = None

    # ------------------------------------------------------------------
    # Core grading logic
    # ------------------------------------------------------------------
    def _parse_retry_after(self, headers: dict, default: float = 60.0) -> float:
        """Extract the smallest Retry-After-* value from rate-limit headers, or return *default*."""
        values = []
        for key, val in headers.items():
            if key.lower().startswith("retry-after-"):
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    pass
        return min(values) if values else default

    def sandbox_params(self, name: str | None = None) -> CreateSandboxFromSnapshotParams:
        """Return the parameters used to create each Daytona sandbox.

        Override in subclasses to customise the language, image, resources, etc.
        """
        return CreateSandboxFromSnapshotParams(
            language="python",
            ephemeral=True,
            name=name,
            labels={"app": "eval360"},
        )

    def ground_truth_to_test_list(ground_truth: str) -> list[dict]:
        raise NotImplementedError()

    def build_test_harness(self, parsed_response: str, ground_truth: Any) -> str:
        """
        Combine the generated *parsed_response* with *ground_truth* into a self-contained
        Python script that exits 0 on success and non-zero on failure.
        """
        raise NotImplementedError()

    def _sandbox_name(self, row: int, gen_index: int) -> str:
        """Build a human-readable sandbox name for identification in the Daytona dashboard."""
        user = getpass.getuser()
        model = self._event_instance.model if self._event_instance else "unknown"
        dataset = self.task.dataset_name if self.task else "unknown"
        return f"{user}-eval360-{model}-{dataset}-row-{row}-gen-{gen_index}-{self._run_id}"

    def _build_sandbox_script(self, parsed_response: str, test_cases: list) -> str:
        """Return the Python script to execute inside the Daytona sandbox.

        Override to change how the solution is run (e.g. to collect raw outputs
        instead of using exit-code pass/fail).  The default delegates to
        :meth:`build_test_harness`.
        """
        return self.build_test_harness(parsed_response, test_cases)

    def _grade_sandbox_result(self, response: Any, test_cases: list) -> tuple[str, list | None]:
        """Determine the grade from the sandbox response.

        Returns a ``(grade, details)`` tuple where *grade* is one of
        "pass" / "wrong_answer" / "timeout" and *details* is an optional
        list of per-test-case result dicts (stored in the grades file when
        present).  The default treats exit-code 0 as "pass" with no details.
        """
        grade = "pass" if response.exit_code == 0 else "wrong_answer"
        return grade, None

    async def _run_in_sandbox(self, parsed_response: str | None, ground_truth: Any, row: int, gen_index: int) -> str:
        """
        Execute *parsed_response* against all test cases in *ground_truth*
        in a single Daytona sandbox.

        Retries indefinitely on rate-limit and resource-limit errors.
        Returns one of: "pass", "wrong_answer", "timeout", "no_output".
        """
        if not parsed_response or not parsed_response.strip():
            return "no_output", None

        test_cases = self.ground_truth_to_test_list(ground_truth)
        _MAX_RETRIES = 3
        async with self._semaphore:
            await asyncio.sleep(5)
            attempts = 0
            while True:
                sandbox = None
                try:
                    sandbox = await self._daytona.create(self.sandbox_params(name=self._sandbox_name(row, gen_index)))

                    harness = self._build_sandbox_script(parsed_response, test_cases)
                    await sandbox.fs.upload_file(harness.encode(), "/tmp/harness.py")
                    try:
                        response = await sandbox.process.exec(
                            "python /tmp/harness.py",
                            timeout=30 * len(test_cases),
                        )
                    except DaytonaError as e:
                        if getattr(e, "status_code", None) == 408:
                            logger.warning("Sandbox execution timed out (408) — treating as failed")
                            return "timeout", None
                        raise

                    return self._grade_sandbox_result(response, test_cases)

                except DaytonaRateLimitError as e:
                    retry_after = self._parse_retry_after(e.headers)
                    logger.warning("Rate limited by Daytona, retrying in %ss", retry_after)
                    await asyncio.sleep(retry_after)
                    # sandbox was never fully created, no cleanup needed
                    continue

                except DaytonaError as e:
                    msg = str(e).lower()
                    status_code = getattr(e, "status_code", None)
                    headers = getattr(e, "headers", None)
                    if "disk limit exceeded" in msg:
                        logger.error(
                            "Daytona disk limit exceeded (status_code=%s, headers=%s): %s",
                            status_code, headers, e,
                        )
                        raise
                    if "limit exceeded" in msg:
                        logger.warning("Daytona resource limit hit, retrying in 60s: %s", e)
                        await asyncio.sleep(60)
                        # sandbox was never fully created, no cleanup needed
                        continue
                    attempts += 1
                    if attempts >= _MAX_RETRIES:
                        logger.error(
                            "DaytonaError after %d attempts (status_code=%s, headers=%s):\n%s",
                            attempts, status_code, headers, traceback.format_exc(),
                        )
                        raise
                    logger.warning(
                        "DaytonaError (attempt %d/%d), retrying: %s"
                        " (status_code=%s, headers=%s)",
                        attempts, _MAX_RETRIES, e, status_code, headers,
                    )
                    continue

                except Exception as e:
                    attempts += 1
                    status_code = getattr(e, "status_code", None)
                    headers = getattr(e, "headers", None)
                    if attempts >= _MAX_RETRIES:
                        raise
                    logger.warning(
                        "Sandbox execution failed (attempt %d/%d), retrying: %s"
                        " (status_code=%s, headers=%s)\n%s",
                        attempts, _MAX_RETRIES, e, status_code, headers,
                        traceback.format_exc(),
                    )
                    continue
                finally:
                    if sandbox is not None:
                        sandbox_id = getattr(sandbox, "id", "?")
                        try:
                            # asyncio.shield ensures the stop coroutine keeps running
                            # even if this task is cancelled, giving it the best chance
                            # of completing. We still catch CancelledError here so we
                            # can log the leak before re-raising.
                            await asyncio.shield(self._daytona.stop(sandbox))
                        except asyncio.CancelledError:
                            logger.error(
                                "LEAKED SANDBOX: %s could not be stopped. "
                                "Please log in to Daytona and stop/delete it manually.",
                                sandbox_id,
                            )
                            raise
                        except Exception:
                            logger.warning("Failed to stop sandbox %s", sandbox_id)
                            logger.error(
                                "LEAKED SANDBOX: %s could not be stopped. "
                                "Please log in to Daytona and stop/delete it manually.",
                                sandbox_id,
                            )

    async def grade_sample(self, sample: Any) -> dict:
        """
        Grade a single sample by running each generated solution
        in an isolated Daytona sandbox.

        Parameters
        ----------
        sample : dict
            Must contain at minimum:
              - ``parsed_generations`` : list[str | None]  (added by GraderBase before this call)
              - ``ground_truth``       : str or dict with test-case data

            May also contain:
              - ``completion_prefix`` : str — when present, this string is
                prepended to each parsed generation before execution.  This
                is used by base/completion models whose ``completion_input``
                ends with a partial ``def`` stub (e.g. ``def func(``); the
                model's raw completion (e.g. ``a, b):\\n    return a+b``)
                must be rejoined with the stub to form runnable code.

        Returns
        -------
        dict
            A copy of *sample* augmented with:
              - ``correct`` : list[int]  (1 = passed, 0 = failed), one entry per generation
        """
        prefix = sample.get("completion_prefix", "")
        generations = sample["parsed_generations"]
        if prefix:
            # Prepend only when the generation does not already contain the
            # function stub.  Using ``in`` instead of ``startswith`` so that
            # instruct-model outputs with imports before the def are not
            # broken (e.g. "import heapq\n\ndef func(..." already has the
            # prefix further down).
            generations = [
                f"{prefix}{g}" if (g is not None and prefix.strip() not in g) else g
                for g in generations
            ]

        sample["correct"] = []
        sample["evaluation_details"] = []

        # Grade each generation concurrently within the sample.
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._run_in_sandbox(generation, sample["ground_truth"], row=sample["row"], gen_index=i))
                for i, generation in enumerate(generations)
            ]

        sandbox_results = []
        for t in tasks:
            grade, details = t.result()
            sample["correct"].append(1 if grade == "pass" else 0)
            sample["evaluation_details"].append(grade)
            sandbox_results.append(details)

        if any(d is not None for d in sandbox_results):
            sample["sandbox_results"] = sandbox_results

        return sample

    # ------------------------------------------------------------------
    # Streaming run (mirrors AccuracyGraderBase.run but owned here so
    # we can customise scoring, checkpointing, etc.)
    # ------------------------------------------------------------------

    async def _async_grade_all_samples_nonblocking(
        self,
        start: int,
        grade_fn: Callable[[Tuple[Any, int]], Awaitable[Tuple[int, Any]]],
        concurrency: int = _DEFAULT_CONCURRENCY,
        **kwargs: Any
    ):
        """
        Grade all samples in self.samples_generator, but allow for later
        samples to be started while earlier ones are still running
        """
        async def grade_asyncio_task(index, sample, queue):
            try:
                sample["parsed_generations"] = self.parse_generations(sample)
                if not self._initialized:
                    await self.initialize()
                    self._initialized = True
                await queue.put((index, await grade_fn(sample)))
            # TODO: handle keyboard interrupt, cancelation
            except Exception as e:
                tb = "".join(traceback.TracebackException.from_exception(e).format())
                error = ExceptionWrapper(exception=e, trace=tb, instance=sample)
                await queue.put((index, error))

        tasks = {}
        queue = asyncio.Queue()
        done_reading = asyncio.Event()

        async def consumer():
            index = 0
            async for sample in self.samples_generator:
                # Fast forward over the ones we are skipping
                if index < start:
                    index += 1
                else:
                    if sample == Sentinel.COMPLETED or isinstance(sample, ExceptionWrapper):
                        await queue.put((index, sample))
                    else:
                        task = asyncio.create_task(grade_asyncio_task(index, sample, queue))
                        tasks[index] = task
                    index += 1
            done_reading.set()

        consumer_task = asyncio.create_task(consumer())

        buffer = {}
        next_index_to_yield = start

        while not done_reading.is_set() or tasks or buffer:
            i, result = await queue.get()
            buffer[i] = result

            while next_index_to_yield in buffer:
                result = buffer.pop(next_index_to_yield)
                yield result

                task = tasks.pop(next_index_to_yield, None)
                if task:
                    task.result()

                next_index_to_yield += 1

        consumer_task.result()

    async def run(
        self,
        existing: AsyncIterator[dict],
        average_over: list[int] | None,
        pass_at: list[int] | None,
    ):
        """
        Stream grades for all samples, then emit aggregate Score objects.

        Yields
        ------
        Grade | Score | ExceptionWrapper | Sentinel.COMPLETED
        """
        if not pass_at:
            pass_at = [1]
        if not average_over:
            average_over = [1]

        all_correct: list[list[int]] = []
        all_evaluation_details: Counter[str] = Counter()
        completed = False

        # --- replay already-graded results from the checkpoint file ---
        count = 0
        async for record in existing:
            if "correct" in record:
                all_correct.append(record["correct"])
            elif "exception" in record:
                # Generation failed for this sample — count it as incorrect so
                # the denominator includes all dataset samples, not just the ones
                # that were successfully graded.
                all_correct.append([0])
            if "evaluation_details" in record:
                all_evaluation_details.update(record["evaluation_details"])
            count += 1

        # --- grade new samples ---
        try:
            async for result in self._async_grade_all_samples_nonblocking(
                start=count,
                grade_fn=self.grade_sample,
                concurrency=_DEFAULT_CONCURRENCY,
            ):
                if result == Sentinel.COMPLETED:
                    completed = True
                    break
                elif isinstance(result, ExceptionWrapper):
                    # Generation or grading failed for this sample — count it as
                    # incorrect so the denominator includes all dataset samples.
                    all_correct.append([0])
                    yield result
                elif isinstance(result, dict) and "correct" in result:
                    all_correct.append(result["correct"])
                    if "evaluation_details" in result:
                        all_evaluation_details.update(result["evaluation_details"])
                    yield Grade(element=result)
        finally:
            await self.cleanup()

        if not completed:
            return

        # --- aggregate scores ---
        if not all_correct or not any(c for row in all_correct for c in row):
            for n in average_over:
                yield Score(name=f"accuracy (avg over {n})", value=float("nan"))
                yield Score(name=f"bootstrap_std (avg over {n})", value=float("nan"))
            for n in pass_at:
                yield Score(name=f"accuracy (pass@{n})", value=float("nan"))
        else:
            for n in average_over:
                yield Score(
                    name=f"accuracy (avg over {n})",
                    value=get_accuracy([row[:n] for row in all_correct]),
                )
                if n > 1:
                    yield Score(
                        name=f"bootstrap_std (avg over {n})",
                        value=get_bootstrap_accuracy_std([row[:n] for row in all_correct]),
                    )
            for n in pass_at:
                per_problem_counts = [(len(row), sum(row)) for row in all_correct]
                yield Score(
                    name=f"accuracy (pass@{n})",
                    value=mean_pass_at_k(per_problem_counts, k=n),
                )

        for reason, cnt in sorted(all_evaluation_details.items()):
            yield Score(name=f"breakdown ({reason})", value=cnt)

        yield Sentinel.COMPLETED
