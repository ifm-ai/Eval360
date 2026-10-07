"""
This file defines the base class for graders.
"""
from openai import AsyncOpenAI
import abc
import aiohttp
import asyncio
import pydantic
import re
from typing import Any, Awaitable, AsyncGenerator, Callable, Dict, Tuple, AsyncIterator

from ..external_requests import (
    RetryingAsyncOpenAI,
    get_external_rate_limiter,
    validate_and_canonicalize_external_endpoint,
)
from ..utils import Sentinel, ExceptionWrapper
from ..metrics import get_accuracy, get_bootstrap_accuracy_std, mean_pass_at_k
from .parser_registry import get_parser
import logging
logger = logging.getLogger("Grader")
logging.basicConfig(level=logging.INFO)




class Grade(pydantic.BaseModel):
    element: dict[str, Any]

    @pydantic.model_validator(mode='after')
    def check_score_is_present(self):
        # TODO: generalize from correct
        if "correct" not in self.element:
            raise ValueError(f'"correct" not in graded datapoint {self.element}')
        return self


class Score(pydantic.BaseModel):
    name: str
    value: Any


# TODO: throw in a utils class
class GradingOpenAIConnection:
    def __init__(self, event_instance, task, job_manager):
        self._event_instance = event_instance
        self._task = task
        self._job_manager = job_manager
        self._client = None
        self._client_proxy = None
        self._url = None
        self._client_lock = asyncio.Lock()

    async def get_client(self) -> AsyncOpenAI | RetryingAsyncOpenAI:
        """Return a client for a live judge URL, blocking until one is available."""
        judge = self._task.grader.llm_as_judge
        if getattr(judge, "is_external", False):
            from ..model import ExternalRetryPolicy
            from ..openai_interface import (
                get_or_create_connection_pool,
            )

            # External judges have no Slurm job to wait for. The canonical
            # endpoint is shared by the SDK client and generation's pool.
            url = validate_and_canonicalize_external_endpoint(
                judge.base_url
            )
            async with self._client_lock:
                if (
                    url != self._url
                    or self._client_proxy is None
                ):
                    retry_policy = getattr(
                        judge,
                        "external_retry_policy",
                        None,
                    )
                    if not isinstance(
                        retry_policy,
                        ExternalRetryPolicy,
                    ):
                        retry_policy = ExternalRetryPolicy()

                    pool = get_or_create_connection_pool(
                        judge.serving_key,
                        judge.max_simultaneous_requests,
                    )
                    client = AsyncOpenAI(
                        base_url=url,
                        api_key=(
                            getattr(judge, "api_key", None)
                            or "no-key"
                        ),
                        max_retries=0,
                        timeout=(
                            retry_policy.request_timeout_seconds
                        ),
                    )
                    await pool.add_url(url)

                    async def acquire_pool_lease():
                        return await pool.acquire(1)

                    async def release_pool_lease(lease):
                        lease_url, connections = lease
                        await pool.release(
                            lease_url,
                            connections,
                        )

                    client_proxy = RetryingAsyncOpenAI(
                        client,
                        policy=retry_policy,
                        rate_limiter=(
                            get_external_rate_limiter(judge)
                        ),
                        acquire_factory=acquire_pool_lease,
                        release_factory=release_pool_lease,
                    )
                    self._url = url
                    self._client = client
                    self._client_proxy = client_proxy
                return self._client_proxy
        url = await self._job_manager.get_live_url(judge.name)
        if url != self._url:
            self._url = url
            self._client = AsyncOpenAI(
                base_url=f"{url}/v1",
                api_key="fake key",
                max_retries=0,
            )
        return self._client

class GraderBase(abc.ABC):
    """
    Grader classes generally should override two methods:
    `grade_sample`: Takes in a test sample and returns the metrics of interest.
    `run`: Runs the grading. Generally, most `run`
        methods will follow this same pattern: loading the data, calling
        `grade_sample`, and aggregating the results.
    """

    def __init__(
        self,
        samples_generator: AsyncGenerator[Dict[str, any], Dict[str, any]],
        # TODO: remove default
        event_manager,
        job_manager,
        event,
        task=None,
    ):
        self.samples_generator = samples_generator
        self.task = task
        self._event_instance = event
        self._event_manager = event_manager
        self._job_manager = job_manager
        self.openai_connection = None
        self._parser = get_parser(event.parser_type)
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    def parse_generations(self, sample: dict[str, Any]) -> list[str] | None:
        parsed = self._parser.parse_generations(sample["generations"])
        # Extract per-generation reasoning/tool metadata. Run for all generations
        # first so we know which keys appear at all, then build parallel lists
        # (one entry per generation, None for generations without that field).
        all_meta = [self._parser.extract_reasoning_and_tools(g) for g in sample["generations"]]
        all_keys = {key for meta in all_meta for key in meta}
        for key in all_keys:
            sample[f"parsed_{key}"] = [meta.get(key) for meta in all_meta]
        return parsed

    async def parse_generations_async(self, sample: dict[str, Any]) -> list[str] | None:
        parsed = self.parse_generations(sample)
        # Yield after each sample so resume-time grading does not monopolize the
        # event loop while generation jobs are still being polled and launched.
        await asyncio.sleep(0)
        return parsed

    @abc.abstractmethod
    def grade_sample(self, sample: Any):
        raise NotImplementedError()

    @abc.abstractmethod
    async def run(self) -> Dict[str, float]:
        """Run the grading."""
        raise NotImplementedError()

    async def initialize(self):
        return

    async def pregrade_map(self, aiter: AsyncGenerator[Dict[str, any], None]):
        # identity by default
        for elem in aiter:
            yield aiter

    def request_openai_connection(self, model):
        self._event_manager.add_desired_model(self._event_instance, is_grading=True)
        return GradingOpenAIConnection(self._event_instance, self.task, self._job_manager)

    async def async_grade_all_samples(
        self,
        start: int = 0,
        grade_fn: Callable[[Tuple[Any, int]], Awaitable[Tuple[int, Any]]] = None,
        concurrency: int = 32,
        skip_rows: set[int] | None = None,
        **kwargs: Any,
    ):
        i = 0
        async for sample in self.samples_generator:
            # Skip already-graded samples: by row set (preferred) or by position (legacy)
            if skip_rows is not None:
                if isinstance(sample, dict) and sample.get("row") in skip_rows:
                    continue
            elif i < start:
                i += 1
                continue
            try:
                if sample == Sentinel.COMPLETED:
                    yield Sentinel.COMPLETED
                    return
                if isinstance(sample, ExceptionWrapper):
                    yield sample
                    continue
                if "exception" in sample:
                    sample["correct"] = [0]
                    yield sample
                    continue
                if sample.get("eval360_input_too_long"):
                    sample["correct"] = [0] * len(sample["generations"])
                    yield sample
                    continue
                sample["parsed_generations"] = await self.parse_generations_async(sample)
                if not self._initialized:
                    await self.initialize()
                    self._initialized = True
                yield await grade_fn(sample)
            # TODO: handle keyboard interrupt, cancelation
            except Exception as e:
                yield ExceptionWrapper.from_exception(e, sample)

    async def async_grade_all_samples_nonblocking(
        self,
        start: int = 0,
        grade_fn: Callable[[Tuple[Any, int]], Awaitable[Tuple[int, Any]]] = None,
        skip_rows: set[int] | None = None,
        **kwargs: Any,
    ):
        """Grade samples concurrently: later samples start before earlier ones finish.

        Yields results in the original sample order for deterministic output.
        Unlike async_grade_all_samples, this starts all sample gradings as concurrent
        asyncio tasks rather than awaiting each one before starting the next.
        """
        async def _grade_task(index: int, sample: dict, queue: asyncio.Queue):
            try:
                sample["parsed_generations"] = await self.parse_generations_async(sample)
                if not self._initialized:
                    async with self._initialize_lock:
                        if not self._initialized:
                            await self.initialize()
                            self._initialized = True
                await queue.put((index, await grade_fn(sample)))
            except asyncio.CancelledError as error:
                # CancelledError is a BaseException, so the normal error path
                # below cannot report it.  Publish it before re-raising so the
                # ordered consumer cannot wait forever on a child that has
                # already terminated.
                queue.put_nowait((index, error))
                raise
            except Exception as e:
                await queue.put(
                    (
                        index,
                        ExceptionWrapper.from_exception(e, sample),
                    )
                )

        tasks: dict[int, asyncio.Task] = {}
        queue: asyncio.Queue = asyncio.Queue()
        done_reading = asyncio.Event()

        async def _consumer():
            index = 0
            # The drain loop can stop waiting only after it knows no producer
            # can add more work, including when input iteration is cancelled
            # or fails before reaching its natural end.
            try:
                async for sample in self.samples_generator:
                    # Skip already-graded samples
                    if skip_rows is not None:
                        if (
                            isinstance(sample, dict)
                            and sample.get("row") in skip_rows
                        ):
                            continue
                    elif index < start:
                        index += 1
                        continue
                    if (
                        sample == Sentinel.COMPLETED
                        or isinstance(sample, ExceptionWrapper)
                    ):
                        await queue.put((index, sample))
                    elif "exception" in sample:
                        sample["correct"] = [0]
                        await queue.put((index, sample))
                    elif sample.get("eval360_input_too_long"):
                        sample["correct"] = [0] * len(
                            sample["generations"]
                        )
                        await queue.put((index, sample))
                    else:
                        tasks[index] = asyncio.create_task(
                            _grade_task(index, sample, queue)
                        )
                    index += 1
            finally:
                done_reading.set()

        consumer_task = asyncio.create_task(_consumer())

        buffer: dict[int, Any] = {}
        next_yield = start

        # Closing or cancelling this async generator must also stop the input
        # consumer and every grading task; otherwise they can outlive the
        # caller and leave queue waiters or external requests behind.
        try:
            while not done_reading.is_set() or tasks or buffer:
                if not tasks and queue.empty():
                    queue_item = asyncio.create_task(queue.get())
                    reader_done = asyncio.create_task(
                        done_reading.wait()
                    )
                    waiters = (queue_item, reader_done)
                    try:
                        completed, _ = await asyncio.wait(
                            waiters,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        for waiter in waiters:
                            if not waiter.done():
                                waiter.cancel()
                        await asyncio.gather(
                            *waiters,
                            return_exceptions=True,
                        )
                    if queue_item not in completed:
                        continue
                    idx, result = queue_item.result()
                else:
                    idx, result = await queue.get()

                buffer[idx] = result
                while next_yield in buffer:
                    result = buffer.pop(next_yield)
                    tasks.pop(next_yield, None)
                    next_yield += 1
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    yield result

            await consumer_task
        finally:
            consumer_task.cancel()
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(
                consumer_task,
                *tasks.values(),
                return_exceptions=True,
            )


class AccuracyGraderBase(GraderBase):
    def _grading_generator(self, start: int = 0, skip_rows: set[int] | None = None):
        """Return the async generator used to grade samples.

        Subclasses may override this to swap in a concurrent (nonblocking)
        generator when sample-level parallelism is beneficial, e.g. for
        LLM-as-judge graders where each sample makes network calls.
        """
        return self.async_grade_all_samples(start=start, grade_fn=self.grade_sample, skip_rows=skip_rows)

    async def run(self, existing: AsyncIterator[Dict[str, any]],
                  average_over: list[int], pass_at: list[int]):
        if not pass_at:
            pass_at = [1]
        if not average_over:
            average_over = [1]
        all_correct = []
        completed = False
        graded_rows: set[int] = set()
        async for result in existing:
            if "correct" in result:
                all_correct.append(result["correct"])
            if "row" in result:
                graded_rows.add(result["row"])
        async for result in self._grading_generator(skip_rows=graded_rows):
            if result == Sentinel.COMPLETED:
                completed = True
                break
            elif isinstance(result, ExceptionWrapper):
                yield result
            elif isinstance(result, dict) and "correct" in result:
                all_correct.append(result["correct"])
                yield Grade(element=result)
        if not completed:
            return

        if not all_correct or not any(all_correct):
            for n in average_over:
                yield Score(name=f"accuracy (avg over {n})", value=float("nan"))
                yield Score(name=f"bootstrap_std (avg over {n})", value=float("nan"))
            for n in pass_at:
                yield Score(name=f"accuracy (pass@{n})", value=float("nan"))
                yield Score(name=f"bootstrap_std (pass@{n})", value=float("nan"))
            yield Sentinel.COMPLETED
        else:
            for n in average_over:
                yield Score(name=f"accuracy (avg over {n})", value=get_accuracy([sample[:n] for sample in all_correct]))
                if n > 1:
                    yield Score(name=f"bootstrap_std (avg over {n})", value=get_bootstrap_accuracy_std([sample[:n] for sample in all_correct]))
            for n in pass_at:
                per_problem_counts = [(len(sample), sum(sample)) for sample in all_correct]
                yield Score(name=f"accuracy (pass@{n})", value=mean_pass_at_k(per_problem_counts, k=n))
            yield Sentinel.COMPLETED
