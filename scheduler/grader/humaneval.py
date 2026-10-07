"""HumanEval grader — evaluates code completions by executing them against test suites.

The sandboxed execution logic is inlined from the OpenAI HumanEval evaluation
harness so that this grader has no external dependency on the human_eval package.
"""
import asyncio
import contextlib
import copy
import faulthandler
import functools
import io
import logging
import multiprocessing
import os
import platform
import signal
import tempfile
from typing import Any, Dict, Optional

from ..utils import Sentinel
from .base import AccuracyGraderBase
from .registry import register

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Sandboxed code execution (adapted from OpenAI HumanEval)
# ---------------------------------------------------------------------------

class _TimeoutException(Exception):
    pass


class _WriteOnlyStringIO(io.StringIO):
    """StringIO that raises on read — prevents executed code from reading stdout."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class _RedirectStdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def _time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise _TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _RedirectStdin(stream):
                yield


@contextlib.contextmanager
def _chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def _create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with _chdir(dirname):
            yield dirname


def _reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """Disable destructive functions to prevent generated code from interfering
    with the host system. NOT a security sandbox — use container isolation for
    untrusted code in production."""
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"
    for attr in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir",
        "fchdir", "setuid", "fork", "forkpty", "killpg", "rename", "renames",
        "truncate", "replace", "unlink", "fchmod", "fchown", "chmod",
        "chown", "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        if hasattr(os, attr):
            setattr(os, attr, None)

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _unsafe_execute(problem: Dict, completion: str, timeout: float, result):
    with _create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        _reliability_guard()

        check_program = (
            problem["prompt"]
            + completion
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )

        try:
            exec_globals = {}
            with _swallow_io():
                with _time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except _TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_correctness(problem: Dict, completion: str, timeout: float) -> Dict:
    """Run a completion against the problem's test suite in a sandboxed subprocess.

    NOTE: Subprocess-based execution is temporarily disabled while a more robust
    sandbox is being implemented. This will raise NotImplementedError until then.
    """
    raise NotImplementedError(
        "Subprocess-based code execution is temporarily disabled. "
        "A more robust sandbox is being integrated — see the humaneval grader "
        "for the planned execution interface."
    )


def _strip_function_signature(gen: str, entry_point: str) -> str:
    """Strip a re-emitted function signature so only the body remains.

    Instruct/chat models often output the full function definition instead of
    just the body.  Since check_correctness prepends the prompt (which already
    contains the signature), we need to remove the duplicate.
    """
    import re
    # Match "def entry_point(...):  " possibly across the first line(s)
    pattern = rf"^\s*def\s+{re.escape(entry_point)}\s*\(.*?\)\s*(?:->.*?)?:\s*\n"
    m = re.match(pattern, gen, flags=re.DOTALL)
    if m:
        body = gen[m.end():]
        # Also strip a repeated docstring if present
        doc_match = re.match(r'\s*""".*?"""\s*\n', body, flags=re.DOTALL)
        if doc_match:
            body = body[doc_match.end():]
        return body
    return gen


# ---------------------------------------------------------------------------
# HumanEval Grader
# ---------------------------------------------------------------------------

@register("humaneval", "human_eval", "human-eval")
class HumanEval(AccuracyGraderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timeout = DEFAULT_TIMEOUT

    async def initialize(self):
        if self.task and self.task.meta and "timeout" in self.task.meta:
            self._timeout = float(self.task.meta["timeout"])

    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict), "sample must be a dict"
        assert "parsed_generations" in sample, "sample must have 'parsed_generations'"
        assert "ground_truth" in sample, "sample must have 'ground_truth'"

        gt = sample["ground_truth"]
        prompt = sample["completion_input"]
        problem = {
            "task_id": sample.get("task_id", f"row_{sample['row']}"),
            "prompt": prompt,
            "test": gt["test"],
            "entry_point": gt["entry_point"],
        }

        entry_point = gt["entry_point"]
        correct = []
        loop = asyncio.get_running_loop()
        for i, gen in enumerate(sample["parsed_generations"]):
            if gen is None:
                gen = sample["generations"][i]

            # Strip prompt if model re-emitted it (avoids duplicate when
            # check_correctness prepends prompt + completion)
            if gen.startswith(prompt):
                gen = gen[len(prompt):]

            # Strip re-emitted function signature (instruct models often
            # output the full function instead of just the body)
            gen = _strip_function_signature(gen, entry_point)

            result = await loop.run_in_executor(
                None,
                functools.partial(check_correctness, problem, gen, self._timeout),
            )
            correct.append(1 if result["passed"] else 0)

        graded = copy.deepcopy(sample)
        graded["correct"] = correct
        return graded
