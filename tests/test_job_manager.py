"""
Unit tests for SlurmManager (scheduler/slurm_manager.py).

All Slurm interactions (squeue, sinfo, sbatch, scancel) are mocked via
FakeSlurmProcess so no real cluster is needed.
"""
import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest

import scheduler.slurm_manager as slurm_manager_module
from scheduler.model import ServingSlurmResources
from scheduler.slurm_manager import SlurmManager


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

# Fixed instance_id used by _jm() so job names are deterministic in tests.
_TEST_IID = "deadbeef"


def _test_sk(label: str) -> str:
    """Deterministic 12-char serving key for tests (sha256 of label)."""
    import hashlib
    return hashlib.sha256(label.encode()).hexdigest()[:12]


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", name.lower())[:20].strip("-")


def _test_job_name(label: str, replica: int = 0) -> str:
    """Build a valid eval360 job name from a test label."""
    return f"eval360-{_TEST_IID}-{_sanitize_name(label)}-{_test_sk(label)}-r{replica}"


def _fake_job_name(model, replica: int = 0) -> str:
    """Build the expected job name for a _fake_model instance."""
    return f"eval360-{_TEST_IID}-{_sanitize_name(model.name)}-{model.serving_key}-r{replica}"


class FakeSlurmProcess:
    """Simulates the object returned by asyncio.create_subprocess_exec."""

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        *,
        hang: bool = False,
        kill_raises: bool = False,
    ):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self._kill_raises = kill_raises
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.Event().wait()  # never resolves
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode

    def kill(self):
        if self._kill_raises:
            raise ProcessLookupError()
        self.killed = True

    @classmethod
    def success(cls, stdout_text: str) -> "FakeSlurmProcess":
        return cls(stdout=stdout_text.encode(), returncode=0)

    @classmethod
    def failure(cls, stderr_text: str = "error", rc: int = 1) -> "FakeSlurmProcess":
        return cls(stderr=stderr_text.encode(), returncode=rc)

    @classmethod
    def timeout(cls) -> "FakeSlurmProcess":
        return cls(hang=True)

    @classmethod
    def timeout_already_exited(cls) -> "FakeSlurmProcess":
        return cls(hang=True, kill_raises=True)


class _VirtualClock:
    """A monotonic clock that only moves when the code under test reads it.

    `wait_for_terminal_job_outcomes` bounds its two lag graces with
    `time.monotonic()`. Left on the wall clock, a test of the expiry branch has
    to sit through the real 30-second grace, and a test of the RECOVERY
    branches is one slow machine away from expiring instead of recovering.
    Stepping the clock per read makes both outcomes a property of the script
    rather than of the machine: `step=0` can never expire, and a step large
    enough to cross the grace always expires, in no real time at all.
    """

    def __init__(self, *, step: float = 0.0, start: float = 1000.0):
        self._step = step
        self._now = start
        self.start = start

    def monotonic(self) -> float:
        """Stand in for `time.monotonic`, advancing one step per read."""
        now = self._now
        self._now += self._step
        return now

    @property
    def elapsed(self) -> float:
        return self._now - self.start


def _virtual_clock(*, step: float = 0.0):
    """Patch the `time` that slurm_manager's graces read, and hand it back.

    `create=True` so the same test body also runs against an implementation
    that has no grace to bound and therefore never imported `time` — that is
    what lets the recovery tests be pointed at the parent to prove they bite.
    """
    return patch.object(
        slurm_manager_module,
        "time",
        _VirtualClock(step=step),
        create=True,
    )


class _ScriptedSacct:
    """Serve one pre-written `sacct` response per call, in order.

    The lag this covers is a SEQUENCE — no row, then a placeholder row, then
    the real one — so a single canned response cannot express it. `then`
    repeats forever, which is how a row that never settles is written.
    """

    def __init__(self, *responses: str, then: str | None = None):
        self._responses = list(responses)
        self._then = then
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv, *, timeout):
        assert argv[0] == "sacct", f"unexpected command {argv[0]!r}"
        self.calls.append(tuple(argv))
        if self._responses:
            return _command_result("sacct", self._responses.pop(0))
        if self._then is not None:
            return _command_result("sacct", self._then)
        raise AssertionError(
            f"sacct was called {len(self.calls)} times, more than scripted"
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _sacct_row(job_id: int, job_name: str, state: str, exit_code: str = "0:0") -> str:
    """One root accounting row in `--parsable2` form, trailing empty field and all."""
    return f"{job_id}|{job_name}|{state}|{exit_code}|None|\n"


async def _submit_into_ledger(jm: SlurmManager, model, job_id: int):
    """Put one real submission in the ledger and return it.

    The ledger is what `wait_for_terminal_job_outcomes` judges "settled"
    against, so these tests populate it the way production does — through
    `update_allocation` — rather than by writing into `_submitted_jobs`.
    """
    async def fake_exec(*args, **kwargs):
        if args[0] == "squeue":
            return FakeSlurmProcess.success("")
        return FakeSlurmProcess.success(f"{job_id};test-cluster")

    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
        _patch_squeue(jm),
    ):
        await jm.update_allocation([(model, 1)], unneeded_models=[])

    ledger = {job.job_id: job for job in jm.get_submitted_jobs()}
    assert job_id in ledger, f"submission of {job_id} did not reach the ledger"
    return ledger[job_id]


def _jm(log_dir=".", poll_interval=5.0) -> SlurmManager:
    return SlurmManager(
        log_dir=log_dir,
        instance_id=_TEST_IID,
        poll_interval=poll_interval,
    )


def _fake_model(name="mymodel", max_time_to_deploy=600, revision=None,
                vllm_cli_args=None, venv_path="/venv/bin/activate",
                path="org/model", allow_long_max_model_len=True,
                vllm_logging_level="WARNING",
                api_model_name=None,
                conda_env=None,
                container_image=None, container_mounts=None,
                serving_slurm_resources=None):
    m = MagicMock()
    m.name = name
    m.max_time_to_deploy = max_time_to_deploy
    m.revision = revision
    m.vllm_cli_args = vllm_cli_args or []
    m.venv_path = venv_path
    m.conda_env = conda_env
    m.path = path
    m.allow_long_max_model_len = allow_long_max_model_len
    m.vllm_logging_level = vllm_logging_level
    m.api_model_name = api_model_name
    m.container_image = container_image
    m.container_mounts = container_mounts
    m.serving_slurm_resources = (
        serving_slurm_resources or ServingSlurmResources()
    )
    m.uses_container = container_image is not None
    m.uses_conda = conda_env is not None
    # Compute serving_key the same way ModelInstance does
    import hashlib, json as _json
    payload = _json.dumps({
        "path": path,
        "revision": revision,
        "vllm_cli_args": sorted(vllm_cli_args or []),
        "venv_path": venv_path,
        "conda_env": conda_env,
        "container_image": container_image,
        "serving_slurm_resources": m.serving_slurm_resources.model_dump(
            mode="json"
        ),
    }, sort_keys=True)
    m.serving_key = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return m


def _squeue_line(job_name, job_id, state, elapsed, nodelist):
    """Build one line of squeue --format %j|%i|%T|%M|%N output."""
    return f"{job_name}|{job_id}|{state}|{elapsed}|{nodelist}"


def _squeue_current_line(job_id, job_name, state, user="testuser"):
    """Build one line of get_current_model_names squeue output (%.18i|%.256j|%.10T|%.20u)."""
    return f"{job_id:>18}|{job_name:<256}|{state:>10}|{user:<20}"


def _command_result(
    command: str,
    stdout_text: str = "",
    *,
    stderr_text: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[bytes]:
    """Build the result returned by one bounded synchronous Slurm command."""
    return subprocess.CompletedProcess(
        args=(command,),
        returncode=returncode,
        stdout=stdout_text.encode(),
        stderr=stderr_text.encode(),
    )


def _squeue_result(
    stdout_text: str = "",
    *,
    stderr_text: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[bytes]:
    """Build the result returned by the bounded synchronous squeue query."""
    return _command_result(
        "squeue",
        stdout_text,
        stderr_text=stderr_text,
        returncode=returncode,
    )


def _patch_squeue(jm: SlurmManager, *stdout: str):
    """Patch the bounded squeue seam with an ordered output sequence."""
    if not stdout:
        return patch.object(
            jm,
            "_query_squeue",
            return_value=_squeue_result(),
        )
    return patch.object(
        jm,
        "_query_squeue",
        side_effect=[_squeue_result(value) for value in stdout],
    )


# ---------------------------------------------------------------------------
# TestToSeconds
# ---------------------------------------------------------------------------

class TestToSeconds:
    def setup_method(self):
        self.jm = _jm()

    def test_mm_ss(self):
        assert self.jm._to_seconds("02:30") == 150

    def test_hh_mm_ss(self):
        assert self.jm._to_seconds("01:02:03") == 3723

    def test_dd_hh_mm_ss(self):
        assert self.jm._to_seconds("1-02:03:04") == 86400 + 7200 + 180 + 4

    def test_leading_trailing_whitespace(self):
        assert self.jm._to_seconds("  01:02:03  ") == 3723

    def test_zero_values(self):
        assert self.jm._to_seconds("0:00") == 0

    def test_invalid_single_part_raises(self):
        with pytest.raises(ValueError):
            self.jm._to_seconds("300")

    def test_invalid_four_parts_raises(self):
        with pytest.raises(ValueError):
            self.jm._to_seconds("1:2:3:4")

    def test_extra_spaces_between_components_parsed(self):
        # Gap 5: "1 : 2 : 3" has spaces around colons.  Python's int() strips
        # surrounding whitespace, so this parses as HH:MM:SS = 1h 2m 3s = 3723s.
        assert self.jm._to_seconds("1 : 2 : 3") == 3723

    def test_empty_string_raises(self):
        # Gap 5: empty string → after strip() still empty → split(':') → [''] → 1 part
        with pytest.raises(ValueError):
            self.jm._to_seconds("")

    def test_two_component_parses_as_mm_ss(self):
        # Gap 5: "5:00" is the 2-part (MM:SS) branch → h=0, m=5, sec=0 → 300 seconds
        assert self.jm._to_seconds("5:00") == 300

    def test_just_number_no_separator_raises(self):
        # Gap 5: "123" → no colon → split(':') → ['123'] → 1 part → ValueError
        with pytest.raises(ValueError):
            self.jm._to_seconds("123")


def test_venv_serving_template_has_no_fixed_slurm_resources():
    script = (
        Path(__file__).parent.parent / "scheduler/slurm/sbatch_script.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH" not in script
    assert "--exclusive" not in script


# ---------------------------------------------------------------------------
# TestJobNameConversions
# ---------------------------------------------------------------------------

class TestJobNameConversions:
    def setup_method(self):
        self.jm = _jm()

    def test_round_trip(self):
        sk = "abc123def456"
        name = self.jm._to_job_name("mymodel", sk, 0)
        assert self.jm._from_job_name(name) == (sk, 0)

    def test_to_job_name_format(self):
        assert self.jm._to_job_name("mymodel", "abc123def456", 0) == f"eval360-{_TEST_IID}-mymodel-abc123def456-r0"
        assert self.jm._to_job_name("mymodel", "abc123def456", 3) == f"eval360-{_TEST_IID}-mymodel-abc123def456-r3"

    def test_to_job_name_sanitizes_model_name(self):
        assert self.jm._to_job_name("My/Model Name!", "abc123def456", 0) == f"eval360-{_TEST_IID}-my-model-name-abc123def456-r0"

    def test_from_job_name_parses_serving_key_and_replica(self):
        assert self.jm._from_job_name(f"eval360-{_TEST_IID}-mymodel-abc123def456-r0") == ("abc123def456", 0)
        assert self.jm._from_job_name(f"eval360-{_TEST_IID}-mymodel-abc123def456-r7") == ("abc123def456", 7)

    def test_from_job_name_multi_digit_replica(self):
        assert self.jm._from_job_name(f"eval360-{_TEST_IID}-mymodel-abc123def456-r12") == ("abc123def456", 12)

    def test_from_job_name_no_prefix_returns_none(self):
        assert self.jm._from_job_name("other-job") is None

    def test_from_job_name_old_format_returns_none(self):
        # Old format without instance_id no longer recognised
        assert self.jm._from_job_name("eval360-my-model") is None
        assert self.jm._from_job_name("eval360-mymodel-abc123def456-r0") is None

    def test_from_job_name_wrong_instance_id_returns_none(self):
        assert self.jm._from_job_name("eval360-aabbccdd-mymodel-abc123def456-r0") is None

    def test_from_job_name_missing_replica_returns_none(self):
        assert self.jm._from_job_name(f"eval360-{_TEST_IID}-abc123def456") is None

    def test_from_job_name_wrong_key_length_returns_none(self):
        # serving key must be exactly 12 hex chars
        assert self.jm._from_job_name(f"eval360-{_TEST_IID}-abc123-r0") is None

    def test_from_job_name_imported_dataset_job_returns_none(self):
        assert self.jm._from_job_name(f"eval360id-{_TEST_IID}-abcd1234") is None


# ---------------------------------------------------------------------------
# TestGetCurrentModelNames
# ---------------------------------------------------------------------------

class TestGetCurrentModelNames:
    """get_current_model_names uses the registry to resolve model names from serving keys."""

    @pytest.mark.asyncio
    async def test_empty_output(self):
        jm = _jm()
        with patch.object(jm, "_query_squeue", return_value=_squeue_result()):
            result = await jm.get_current_model_names()
        assert result == []

    @pytest.mark.asyncio
    async def test_single_running_job(self):
        jm = _jm()
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        line = _squeue_line(_test_job_name("mymodel"), 1, "RUNNING", "01:00:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_current_model_names()
        assert result == ["mymodel"]

    @pytest.mark.asyncio
    async def test_single_pending_job(self):
        jm = _jm()
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        line = _squeue_line(_test_job_name("mymodel"), 2, "PENDING", "0:00", "")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_current_model_names()
        assert result == ["mymodel"]

    @pytest.mark.asyncio
    async def test_non_eval360_job_filtered(self):
        jm = _jm()
        line = _squeue_line("other-job", 3, "RUNNING", "01:00:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_current_model_names()
        assert result == []

    @pytest.mark.asyncio
    async def test_job_without_registry_entry_excluded(self):
        """A valid eval360 job name that has no registry entry yields no model name."""
        jm = _jm()
        line = _squeue_line(_test_job_name("unknown"), 1, "RUNNING", "01:00:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_current_model_names()
        assert result == []

    @pytest.mark.asyncio
    async def test_completed_state_filtered(self):
        jm = _jm()
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        line = _squeue_line(_test_job_name("mymodel"), 4, "COMPLETED", "01:00:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_current_model_names()
        assert result == []

    @pytest.mark.asyncio
    async def test_mixed_lines(self):
        jm = _jm()
        for m in ["modelA", "modelB", "modelC"]:
            jm._serving_key_registry[_test_sk(m)] = m
        lines = "\n".join([
            _squeue_line(_test_job_name("modelA"), 1, "RUNNING", "01:00:00", "node01"),
            _squeue_line("other-job", 2, "RUNNING", "00:30:00", "node02"),
            _squeue_line(_test_job_name("modelB"), 3, "PENDING", "0:00", ""),
            _squeue_line(_test_job_name("modelC"), 4, "RUNNING", "00:15:00", "node04"),
            _squeue_line(_test_job_name("modelD"), 5, "RUNNING", "00:15:00", "node05"),  # no registry → excluded
        ])
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(lines)
        ):
            result = await jm.get_current_model_names()
        assert sorted(result) == ["modelA", "modelB", "modelC"]

    @pytest.mark.asyncio
    async def test_squeue_failure_raises(self):
        jm = _jm()
        with (
            patch.object(
                jm,
                "_query_squeue",
                return_value=_squeue_result(stderr_text="bad", returncode=1),
            ),
            pytest.raises(RuntimeError, match="squeue failed"),
        ):
            await jm.get_current_model_names()

    @pytest.mark.asyncio
    async def test_squeue_timeout_retries_and_returns_model_names(self):
        jm = _jm(poll_interval=0)
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        line = _squeue_line(
            _test_job_name("mymodel"),
            1,
            "RUNNING",
            "01:00:00",
            "node01",
        )
        timeout = subprocess.TimeoutExpired("squeue", 0.01)
        query = AsyncMock(side_effect=[timeout, _squeue_result(line)])
        with patch.object(jm, "_query_squeue", query):
            result = await jm.get_current_model_names()

        assert result == ["mymodel"]
        assert query.await_count == 2


# ---------------------------------------------------------------------------
# TestGetAllJobs
# ---------------------------------------------------------------------------

class TestGetAllJobs:

    @pytest.mark.asyncio
    async def test_empty_output(self):
        jm = _jm()
        with patch.object(jm, "_query_squeue", return_value=_squeue_result()):
            result = await jm.get_all_jobs()
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_running_job(self):
        jm = _jm()
        jname = _test_job_name("mymodel")
        line = _squeue_line(jname, 42, "RUNNING", "01:00:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert jname in result
        job = result[jname]
        assert job["job_id"] == 42
        assert job["state"] == "RUNNING"
        assert job["nodelist"] == "node01"
        assert job["elapsed_time"] == 3600
        assert job["serving_key"] == _test_sk("mymodel")
        assert job["replica_index"] == 0

    @pytest.mark.asyncio
    async def test_pending_job_included(self):
        jm = _jm()
        jname = _test_job_name("mymodel")
        line = _squeue_line(jname, 99, "PENDING", "0:00", "")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert jname in result

    @pytest.mark.asyncio
    async def test_completed_state_excluded(self):
        jm = _jm()
        line = _squeue_line(_test_job_name("mymodel"), 10, "COMPLETED", "01:00:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert result == {}

    @pytest.mark.asyncio
    async def test_non_eval360_job_excluded(self):
        jm = _jm()
        line = _squeue_line("other-job", 5, "RUNNING", "00:30:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert result == {}

    @pytest.mark.asyncio
    async def test_old_format_job_excluded(self):
        """Old-format eval360-{model_name} jobs are no longer recognised."""
        jm = _jm()
        line = _squeue_line("eval360-mymodel", 5, "RUNNING", "00:30:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert result == {}

    @pytest.mark.asyncio
    async def test_elapsed_time_parsed_correctly(self):
        jm = _jm()
        jname = _test_job_name("m")
        line = _squeue_line(jname, 1, "RUNNING", "1-02:03:04", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert result[jname]["elapsed_time"] == 86400 + 7200 + 180 + 4

    @pytest.mark.asyncio
    async def test_multiple_jobs_mixed(self):
        jm = _jm()
        jA = _test_job_name("A")
        jB = _test_job_name("B")
        lines = "\n".join([
            _squeue_line(jA, 1, "RUNNING", "01:00:00", "node01"),
            _squeue_line(jB, 2, "PENDING", "0:00", ""),
            _squeue_line(_test_job_name("C"), 3, "COMPLETED", "02:00:00", "node02"),
            _squeue_line("other-job", 4, "RUNNING", "00:30:00", "node03"),
        ])
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(lines)
        ):
            result = await jm.get_all_jobs()
        assert set(result.keys()) == {jA, jB}

    @pytest.mark.asyncio
    async def test_replica_index_parsed(self):
        """Multiple replicas of the same model are all returned with correct replica_index."""
        jm = _jm()
        j0 = _test_job_name("mymodel", 0)
        j1 = _test_job_name("mymodel", 1)
        lines = "\n".join([
            _squeue_line(j0, 1, "RUNNING", "01:00:00", "node01"),
            _squeue_line(j1, 2, "RUNNING", "00:30:00", "node02"),
        ])
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(lines)
        ):
            result = await jm.get_all_jobs()
        assert set(result.keys()) == {j0, j1}
        assert result[j0]["replica_index"] == 0
        assert result[j1]["replica_index"] == 1
        assert result[j0]["serving_key"] == result[j1]["serving_key"]

    @pytest.mark.asyncio
    async def test_model_name_resolved_from_registry(self):
        jm = _jm()
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        jname = _test_job_name("mymodel")
        line = _squeue_line(jname, 1, "RUNNING", "01:00:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert result[jname]["model_name"] == "mymodel"

    @pytest.mark.asyncio
    async def test_model_name_none_when_not_in_registry(self):
        jm = _jm()
        jname = _test_job_name("mymodel")
        line = _squeue_line(jname, 1, "RUNNING", "01:00:00", "node01")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert result[jname]["model_name"] is None

    @pytest.mark.asyncio
    async def test_squeue_failure_raises(self):
        jm = _jm()
        with (
            patch.object(
                jm,
                "_query_squeue",
                return_value=_squeue_result(stderr_text="error", returncode=1),
            ),
            pytest.raises(RuntimeError, match="squeue failed"),
        ):
            await jm.get_all_jobs()

    @pytest.mark.asyncio
    async def test_squeue_timeout_retries_then_returns_jobs(self):
        jm = _jm(poll_interval=0)
        jname = _test_job_name("mymodel")
        line = _squeue_line(jname, 42, "RUNNING", "01:00:00", "node01")
        query = AsyncMock(
            side_effect=[
                subprocess.TimeoutExpired("squeue", 0.01),
                _squeue_result(line),
            ]
        )

        with patch.object(jm, "_query_squeue", query):
            result = await jm.get_all_jobs()

        assert result[jname]["job_id"] == 42
        assert query.await_count == 2

    @pytest.mark.asyncio
    async def test_squeue_timeout_exhausts_exact_attempt_bound(self):
        jm = _jm(poll_interval=0)
        attempt_count = slurm_manager_module._SQUEUE_QUERY_MAX_ATTEMPTS
        query = AsyncMock(
            side_effect=[
                subprocess.TimeoutExpired("squeue", 0.01)
                for _ in range(attempt_count)
            ]
        )

        with (
            patch.object(jm, "_query_squeue", query),
            pytest.raises(
                RuntimeError,
                match=rf"squeue timed out after {attempt_count} attempts",
            ),
        ):
            await jm.get_all_jobs()

        assert query.await_count == attempt_count

    @pytest.mark.asyncio
    async def test_nodelist_with_brackets(self):
        """Node ranges like node[01-02] are stored as-is."""
        jm = _jm()
        jname = _test_job_name("m")
        line = _squeue_line(jname, 7, "RUNNING", "00:01:00", "node[01-02]")
        with patch.object(
            jm, "_query_squeue", return_value=_squeue_result(line)
        ):
            result = await jm.get_all_jobs()
        assert result[jname]["nodelist"] == "node[01-02]"


class TestBoundedCommand:
    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX waitpid")
    def test_timeout_kills_and_reaps_child(self, tmp_path):
        pid_path = tmp_path / "child.pid"
        child = (
            "import os, signal, sys; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(str(os.getpid())); "
            "signal.pause()"
        )

        with pytest.raises(subprocess.TimeoutExpired):
            slurm_manager_module._run_bounded_command(
                (sys.executable, "-c", child, str(pid_path)),
                timeout=1,
            )

        child_pid = int(pid_path.read_text())
        with pytest.raises(ChildProcessError):
            os.waitpid(child_pid, os.WNOHANG)

    @pytest.mark.asyncio
    async def test_squeue_query_runs_bounded_command_in_worker_thread(self):
        jm = _jm()
        main_thread = threading.get_ident()
        worker_threads = []

        def fake_bounded_command(argv, *, timeout):
            worker_threads.append(threading.get_ident())
            assert argv == (
                "squeue",
                "--noheader",
                "--me",
                "--format",
                "%j|%i|%T|%M|%N",
            )
            assert timeout == slurm_manager_module._SQUEUE_QUERY_TIMEOUT_SECONDS
            return _squeue_result()

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            assert (await jm._query_squeue()).stdout == b""

        assert worker_threads and worker_threads[0] != main_thread


# ---------------------------------------------------------------------------
# TestGetAvailableNodes
# ---------------------------------------------------------------------------

class TestGetAvailableNodes:

    @pytest.mark.asyncio
    async def test_sinfo_uses_configured_partition(self, tmp_path):
        jm = SlurmManager(log_dir=str(tmp_path), partition="custom-partition")
        with patch(
            "asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success("idle|3\n")
        ) as exec_mock:
            assert await jm.get_available_nodes() == 3
        assert exec_mock.call_args.args == (
            "sinfo", "-h", "-p", "custom-partition", "-o", "%T|%D",
        )

    @pytest.mark.asyncio
    async def test_sinfo_omits_partition_when_unset(self, tmp_path):
        jm = SlurmManager(log_dir=str(tmp_path))
        with patch(
            "asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success("idle|3\n")
        ) as exec_mock:
            assert await jm.get_available_nodes() == 3
        assert exec_mock.call_args.args == ("sinfo", "-h", "-o", "%T|%D")

    @pytest.mark.asyncio
    async def test_idle_nodes_returned(self):
        jm = _jm()
        with patch("asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success("idle|3\n")):
            assert await jm.get_available_nodes() == 3

    @pytest.mark.asyncio
    async def test_idle_line_among_others(self):
        jm = _jm()
        stdout = "alloc|2\ndown|1\nidle|5\n"
        with patch("asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success(stdout)):
            assert await jm.get_available_nodes() == 5

    @pytest.mark.asyncio
    async def test_no_idle_line_returns_zero(self):
        jm = _jm()
        with patch("asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success("alloc|2\n")):
            assert await jm.get_available_nodes() == 0

    @pytest.mark.asyncio
    async def test_empty_output_returns_zero(self):
        jm = _jm()
        with patch("asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success("")):
            assert await jm.get_available_nodes() == 0

    @pytest.mark.asyncio
    async def test_malformed_line_skipped(self):
        jm = _jm()
        stdout = "garbled\nidle|4\n"
        with patch("asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success(stdout)):
            assert await jm.get_available_nodes() == 4

    @pytest.mark.asyncio
    async def test_idle_first_line(self):
        """idle appearing as the first line (lines follow after it)."""
        jm = _jm()
        stdout = "idle|7\nalloc|2\ndown|1\n"
        with patch("asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success(stdout)):
            assert await jm.get_available_nodes() == 7

    @pytest.mark.asyncio
    async def test_idle_in_middle(self):
        """idle sandwiched between other states."""
        jm = _jm()
        stdout = "alloc|2\nidle|6\ndown|1\nmix|3\n"
        with patch("asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.success(stdout)):
            assert await jm.get_available_nodes() == 6

    @pytest.mark.asyncio
    async def test_sinfo_failure_raises(self):
        jm = _jm()
        with patch("asyncio.create_subprocess_exec", return_value=FakeSlurmProcess.failure("err")):
            with pytest.raises(RuntimeError, match="sinfo failed"):
                await jm.get_available_nodes()

    @pytest.mark.asyncio
    async def test_sinfo_timeout_raises_and_kills(self):
        jm = _jm()
        proc = FakeSlurmProcess.timeout()
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(asyncio.TimeoutError):
                await jm.get_available_nodes()
        assert proc.killed


# ---------------------------------------------------------------------------
# TestGetUnneededModels
# ---------------------------------------------------------------------------

def _make_job_state(names_states):
    """Build a plain job dict from a list of (job_name, state) tuples.

    For test convenience, model_name is inferred from old-format 'eval360-{name}'
    job names so that get_unneeded_models tests need not change.
    """
    job_dict = {}
    for i, (name, state) in enumerate(names_states):
        model_name = name[len("eval360-"):] if name.startswith("eval360-") else None
        job_dict[name] = {
            "name": name, "state": state, "job_id": i,
            "nodelist": "node01", "elapsed_time": 0,
            "model_name": model_name, "serving_key": None, "replica_index": 0,
        }
    return job_dict


class TestGetUnneededModels:

    @pytest.mark.asyncio
    async def test_all_desired_under_limit(self):
        jm = _jm()
        jobs = _make_job_state([("eval360-A", "RUNNING"), ("eval360-B", "RUNNING")])
        with patch.object(jm, "get_all_jobs", return_value=jobs):
            unneeded, excess = await jm.get_unneeded_models(
                created_models=["A", "B"],
                active_models=["A", "B"],
                desired_models=["A", "B"],
                maximum_nodes=4,
            )
        assert unneeded == [] and excess == []

    @pytest.mark.asyncio
    async def test_one_job_not_desired(self):
        jm = _jm()
        jobs = _make_job_state([("eval360-A", "RUNNING"), ("eval360-stale", "RUNNING")])
        with patch.object(jm, "get_all_jobs", return_value=jobs):
            unneeded, excess = await jm.get_unneeded_models(
                created_models=["A"],
                active_models=["A"],
                desired_models=["A"],
                maximum_nodes=4,
            )
        assert "stale" in unneeded and excess == []

    @pytest.mark.asyncio
    async def test_over_limit_inactive_trimmed_first(self):
        jm = _jm()
        jobs = _make_job_state([
            ("eval360-A", "RUNNING"),
            ("eval360-B", "RUNNING"),
            ("eval360-C", "RUNNING"),
        ])
        # B is inactive (not in active_models)
        with patch.object(jm, "get_all_jobs", return_value=jobs):
            unneeded, excess = await jm.get_unneeded_models(
                created_models=["A", "B", "C"],
                active_models=["A", "C"],
                desired_models=["A", "B", "C"],
                maximum_nodes=2,
            )
        assert "B" in excess and unneeded == []

    @pytest.mark.asyncio
    async def test_no_slurm_jobs(self):
        jm = _jm()
        with patch.object(jm, "get_all_jobs", return_value={}):
            unneeded, excess = await jm.get_unneeded_models([], [], ["A"], 4)
        assert unneeded == [] and excess == []

    @pytest.mark.asyncio
    async def test_non_eval360_jobs_ignored(self):
        jm = _jm()
        jobs = _make_job_state([("other-job", "RUNNING")])
        with patch.object(jm, "get_all_jobs", return_value=jobs):
            unneeded, excess = await jm.get_unneeded_models([], [], [], 4)
        assert unneeded == [] and excess == []

    @pytest.mark.asyncio
    async def test_excess_loop_exits_early(self):
        """Excess loop stops as soon as remaining <= maximum_nodes; not all inactive trimmed."""
        jm = _jm()
        jobs = _make_job_state([
            ("eval360-A", "RUNNING"),
            ("eval360-B", "RUNNING"),
            ("eval360-C", "RUNNING"),
        ])
        # A and B are inactive; maximum_nodes=2 means only 1 needs to be removed
        with patch.object(jm, "get_all_jobs", return_value=jobs):
            unneeded, excess = await jm.get_unneeded_models(
                created_models=["A", "B", "C"],
                active_models=["C"],
                desired_models=["A", "B", "C"],
                maximum_nodes=2,
            )
        # Only one excess model needed to get from 3 → 2
        assert len(excess) == 1

    @pytest.mark.asyncio
    async def test_unneeded_and_excess_both_nonempty_inactive_trimmed(self):
        """One stale job (unneeded) and one over-limit inactive job (excess)."""
        jm = _jm()
        jobs = _make_job_state([
            ("eval360-A",     "RUNNING"),
            ("eval360-B",     "RUNNING"),
            ("eval360-C",     "RUNNING"),
            ("eval360-stale", "RUNNING"),  # not in desired_models
        ])
        with patch.object(jm, "get_all_jobs", return_value=jobs):
            unneeded, excess = await jm.get_unneeded_models(
                created_models=["A", "B", "C"],
                active_models=["A", "C"],       # B is inactive
                desired_models=["A", "B", "C"],
                maximum_nodes=2,
            )
        assert "stale" in unneeded
        assert "B" in excess

    @pytest.mark.asyncio
    async def test_multiple_unneeded_and_multiple_excess(self):
        """Two stale jobs (unneeded) and two over-limit inactive jobs (excess)."""
        jm = _jm()
        jobs = _make_job_state([
            ("eval360-A",      "RUNNING"),
            ("eval360-B",      "RUNNING"),
            ("eval360-C",      "RUNNING"),
            ("eval360-D",      "RUNNING"),
            ("eval360-stale1", "RUNNING"),
            ("eval360-stale2", "RUNNING"),
        ])
        with patch.object(jm, "get_all_jobs", return_value=jobs):
            unneeded, excess = await jm.get_unneeded_models(
                created_models=["A", "B", "C", "D"],
                active_models=["A", "D"],        # B and C are inactive
                desired_models=["A", "B", "C", "D"],
                maximum_nodes=2,
            )
        assert sorted(unneeded) == ["stale1", "stale2"]
        assert sorted(excess) == ["B", "C"]

    @pytest.mark.asyncio
    async def test_unneeded_and_excess_all_active(self):
        """One stale job (unneeded) and excess trimmed from active models (no inactive)."""
        jm = _jm()
        jobs = _make_job_state([
            ("eval360-A",     "RUNNING"),
            ("eval360-B",     "RUNNING"),
            ("eval360-C",     "RUNNING"),
            ("eval360-D",     "RUNNING"),
            ("eval360-stale", "RUNNING"),
        ])
        with patch.object(jm, "get_all_jobs", return_value=jobs):
            unneeded, excess = await jm.get_unneeded_models(
                created_models=["A", "B", "C", "D"],
                active_models=["A", "B", "C", "D"],  # all active
                desired_models=["A", "B", "C", "D"],
                maximum_nodes=2,
            )
        assert "stale" in unneeded
        assert len(excess) == 2

    @pytest.mark.asyncio
    async def test_job_with_unregistered_serving_key_no_key_error(self):
        # Gap 6: squeue returns a job whose serving_key is not in the registry
        # (model_name is None).  get_unneeded_models must not raise KeyError.
        jm = _jm()
        # Build a job dict with a valid eval360 job name but model_name=None
        sk = _test_sk("ghost")
        jname = _test_job_name("ghost")
        unknown_job = {
            jname: {
                "name": jname, "state": "RUNNING", "job_id": 77,
                "nodelist": "node01", "elapsed_time": 0,
                "model_name": None,  # serving_key not in registry
                "serving_key": sk, "replica_index": 0,
            }
        }
        with patch.object(jm, "get_all_jobs", return_value=unknown_job):
            # Should not raise KeyError; the job is simply ignored in model_name tracking
            unneeded, excess = await jm.get_unneeded_models(
                created_models=[],
                active_models=[],
                desired_models=[],
                maximum_nodes=4,
            )
        assert unneeded == []
        assert excess == []


# ---------------------------------------------------------------------------
# TestCheckLive
# ---------------------------------------------------------------------------

class TestCheckLive:

    @pytest.mark.asyncio
    async def test_200_response_returned(self):
        jm = _jm()
        job_state = {"nodelist": "node01", "elapsed_time": 10}
        mock_resp = AsyncMock()
        mock_resp.status = 200

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_get)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            resp, returned_state = await jm.check_live(job_state)

        assert resp.status == 200
        assert returned_state is job_state

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        jm = _jm()
        job_state = {"nodelist": "node01", "elapsed_time": 10}

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
        mock_get.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_get)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            resp, returned_state = await jm.check_live(job_state)

        assert resp is None
        assert returned_state is job_state

    @pytest.mark.asyncio
    async def test_tries_exactly_once(self):
        """while count < 1 means exactly one attempt regardless of outcome."""
        jm = _jm()
        job_state = {"nodelist": "node01", "elapsed_time": 0}
        call_count = 0

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        async def fail_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("fail")

        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(side_effect=fail_get)
        mock_get.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_get)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await jm.check_live(job_state)

        assert call_count == 1


# ---------------------------------------------------------------------------
# TestKillAll
# ---------------------------------------------------------------------------

class TestKillAll:

    @pytest.mark.asyncio
    async def test_empty_model_names_no_scancel(self):
        jm = _jm()
        jm._serving_key_registry[_test_sk("A")] = "A"
        squeue_out = _squeue_line(_test_job_name("A"), 1, "RUNNING", "01:00:00", "node01")
        call_log = []

        def fake_bounded_command(argv, *, timeout):
            call_log.append(argv[0])
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all([])

        assert "scancel" not in call_log

    @pytest.mark.asyncio
    async def test_matching_running_job_cancelled(self):
        jm = _jm()
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        squeue_out = _squeue_line(_test_job_name("mymodel"), 42, "RUNNING", "01:00:00", "node01")
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(argv)
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["mymodel"])

        assert len(scancel_calls) == 1
        assert "42" in scancel_calls[0]

    @pytest.mark.asyncio
    async def test_matching_pending_job_cancelled(self):
        jm = _jm()
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        squeue_out = _squeue_line(_test_job_name("mymodel"), 99, "PENDING", "0:00", "")
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(argv)
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["mymodel"])

        assert len(scancel_calls) == 1

    @pytest.mark.asyncio
    async def test_job_not_in_model_names_not_cancelled(self):
        jm = _jm()
        jm._serving_key_registry[_test_sk("other")] = "other"
        squeue_out = _squeue_line(_test_job_name("other"), 5, "RUNNING", "01:00:00", "node01")
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(argv)
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["mymodel"])  # "mymodel" != "other"

        assert scancel_calls == []

    @pytest.mark.asyncio
    async def test_non_eval360_job_not_cancelled(self):
        jm = _jm()
        squeue_out = _squeue_line("other-job", 7, "RUNNING", "01:00:00", "node01")
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(argv)
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["other-job"])

        assert scancel_calls == []

    @pytest.mark.asyncio
    async def test_multiple_matching_jobs_all_cancelled(self):
        jm = _jm()
        for m in ["A", "B", "C"]:
            jm._serving_key_registry[_test_sk(m)] = m
        squeue_out = "\n".join([
            _squeue_line(_test_job_name("A"), 1, "RUNNING", "01:00:00", "node01"),
            _squeue_line(_test_job_name("B"), 2, "PENDING", "0:00", ""),
            _squeue_line(_test_job_name("C"), 3, "RUNNING", "00:30:00", "node02"),
        ])
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(list(argv))
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["A", "B", "C"])

        cancelled_ids = {call[1] for call in scancel_calls}
        assert cancelled_ids == {"1", "2", "3"}

    @pytest.mark.asyncio
    async def test_multiple_matching_jobs_one_cancelled(self):
        jm = _jm()
        for m in ["A", "B", "C"]:
            jm._serving_key_registry[_test_sk(m)] = m
        squeue_out = "\n".join([
            _squeue_line(_test_job_name("A"), 1, "RUNNING", "01:00:00", "node01"),
            _squeue_line(_test_job_name("B"), 2, "PENDING", "0:00", ""),
            _squeue_line(_test_job_name("C"), 3, "RUNNING", "00:30:00", "node02"),
        ])
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(list(argv))
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["B"])

        cancelled_ids = {call[1] for call in scancel_calls}
        assert cancelled_ids == {"2"}

    @pytest.mark.asyncio
    async def test_multiple_matching_jobs_two_cancelled(self):
        jm = _jm()
        for m in ["A", "B", "C"]:
            jm._serving_key_registry[_test_sk(m)] = m
        squeue_out = "\n".join([
            _squeue_line(_test_job_name("A"), 1, "RUNNING", "01:00:00", "node01"),
            _squeue_line(_test_job_name("B"), 2, "PENDING", "0:00", ""),
            _squeue_line(_test_job_name("C"), 3, "RUNNING", "00:30:00", "node02"),
        ])
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(list(argv))
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["A", "C"])

        cancelled_ids = {call[1] for call in scancel_calls}
        assert cancelled_ids == {"1", "3"}

    @pytest.mark.asyncio
    async def test_all_replicas_of_model_cancelled(self):
        """kill_all cancels all replicas of a model, not just the first one."""
        jm = _jm()
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        squeue_out = "\n".join([
            _squeue_line(_test_job_name("mymodel", 0), 1, "RUNNING", "01:00:00", "node01"),
            _squeue_line(_test_job_name("mymodel", 1), 2, "RUNNING", "00:30:00", "node02"),
            _squeue_line(_test_job_name("mymodel", 2), 3, "PENDING", "0:00", ""),
        ])
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(list(argv))
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["mymodel"])

        cancelled_ids = {call[1] for call in scancel_calls}
        assert cancelled_ids == {"1", "2", "3"}

    @pytest.mark.asyncio
    async def test_running_and_pending_cancelled_together(self):
        """A RUNNING and a PENDING job can both be killed in the same call."""
        jm = _jm()
        for m in ["A", "B", "C"]:
            jm._serving_key_registry[_test_sk(m)] = m
        squeue_out = "\n".join([
            _squeue_line(_test_job_name("A"), 1, "RUNNING", "01:00:00", "node01"),
            _squeue_line(_test_job_name("B"), 2, "PENDING", "0:00", ""),
            _squeue_line(_test_job_name("C"), 3, "RUNNING", "00:30:00", "node02"),
        ])
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            scancel_calls.append(list(argv))
            return _command_result(argv[0])

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["A", "B"])

        cancelled_ids = {call[1] for call in scancel_calls}
        assert cancelled_ids == {"1", "2"}

    @pytest.mark.asyncio
    async def test_failing_scancel_does_not_propagate(self):
        """scancel errors are swallowed — kill_all should complete without raising."""
        jm = _jm()
        jm._serving_key_registry[_test_sk("mymodel")] = "mymodel"
        squeue_out = _squeue_line(_test_job_name("mymodel"), 1, "RUNNING", "01:00:00", "node01")

        def fake_bounded_command(argv, *, timeout):
            return _command_result(
                argv[0],
                stderr_text="permission denied",
                returncode=1,
            )

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            _patch_squeue(jm, squeue_out),
        ):
            await jm.kill_all(["mymodel"])  # should not raise


# ---------------------------------------------------------------------------
# TestUpdateAllocation
# ---------------------------------------------------------------------------

class TestUpdateAllocation:

    @pytest.fixture(autouse=True)
    def _empty_squeue(self, monkeypatch):
        async def query_squeue(_manager):
            return _squeue_result()

        monkeypatch.setattr(SlurmManager, "_query_squeue", query_squeue)

    def _make_calls(self, squeue_for_kill="", squeue_for_names="", sbatch_job_id=100):
        """Returns a side_effect list: [squeue (kill_all), squeue (get_current), sbatch]."""
        return [
            FakeSlurmProcess.success(squeue_for_kill),   # kill_all -> get_all_jobs
            FakeSlurmProcess.success(squeue_for_names),  # get_current_model_names
            FakeSlurmProcess.success(f"Submitted batch job {sbatch_job_id}"),  # sbatch
        ]

    @pytest.mark.asyncio
    async def test_sbatch_called_for_new_model(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="newmodel", vllm_cli_args=["--tensor-parallel-size 8"])
        sbatch_args = []

        async def fake_exec(*args, **kwargs):
            sbatch_args.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        # Patch squeue calls to return empty, then sbatch
        call_n = 0
        async def ordered_exec(*args, **kwargs):
            nonlocal call_n
            call_n += 1
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert "sbatch" in sbatch_args

    @pytest.mark.asyncio
    async def test_sbatch_job_name_flag(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="targetmodel")
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert f"--job-name={_fake_job_name(model)}" in sbatch_args_captured

    @pytest.mark.asyncio
    async def test_sbatch_log_dir_output_flag(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m")
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert f"--output={tmp_path}/slurm-%j.out" in sbatch_args_captured

    @pytest.mark.asyncio
    async def test_default_partition_omitted(self, tmp_path):
        jm = SlurmManager(log_dir=str(tmp_path))  # no partition arg
        model = _fake_model(name="m")
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert "sbatch" in sbatch_args_captured
        # No partition configured: the site's default partition applies.
        assert not [a for a in sbatch_args_captured if str(a).startswith("--partition")]

    @pytest.mark.asyncio
    async def test_custom_partition_passed_to_sbatch(self, tmp_path):
        jm = SlurmManager(log_dir=str(tmp_path), partition="custom-partition")
        model = _fake_model(name="m")
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert "--partition=custom-partition" in sbatch_args_captured
        assert [a for a in sbatch_args_captured if str(a).startswith("--partition")] == [
            "--partition=custom-partition"
        ]

    @pytest.mark.asyncio
    async def test_model_serving_resources_are_explicit_sbatch_arguments(
        self, tmp_path
    ):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(
            name="k2-horizon-7b-base",
            serving_slurm_resources=ServingSlurmResources(
                # Obviously arbitrary placeholders, distinct from the 12345 defaults.
                gpus_per_node=23456,
                cpus_per_task=34567,
                memory_gb=45678,
                time_limit="12:34:56",
            ),
        )
        captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        for argument in (
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:23456",
            "--cpus-per-task=34567",
            "--mem=45678G",
            "--time=12:34:56",
        ):
            assert argument in captured
        assert "--exclusive" not in captured

    @pytest.mark.asyncio
    async def test_vllm_args_encoded_in_export(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m", vllm_cli_args=["--tensor-parallel-size 8"])
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        export_arg = next(a for a in sbatch_args_captured if a.startswith("--export="))
        for part in export_arg.split(","):
            if part.startswith("vllm_args="):
                b64 = part.split("=", 1)[1]
                decoded = json.loads(base64.b64decode(b64).decode())
                normalize = lambda args: [" ".join(s.split()) for s in args]
                assert normalize(decoded) == normalize([
                    "--tensor-parallel-size", "8",
                    "--served-model-name", "m",
                ])
                break
        else:
            pytest.fail("vllm_args not found in --export")

    @pytest.mark.asyncio
    async def test_incremental_siblings_use_initial_stable_served_name(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        stable_name = "shared-base-model"
        model = _fake_model(
            name="shared-base-model-eval-first",
            api_model_name=stable_name,
        )
        later_sibling = _fake_model(
            name="shared-base-model-eval-later",
            api_model_name=stable_name,
        )
        assert later_sibling.serving_key == model.serving_key

        # The first allocation cycle knows only about the first active pair.
        # A later sibling cannot mutate this already-running vLLM process.
        currently_known_siblings = [model.name]
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation(
                [(model, 1)],
                unneeded_models=[],
                sibling_names_by_sk={
                    model.serving_key: currently_known_siblings
                },
            )

        export_arg = next(
            argument
            for argument in sbatch_args_captured
            if argument.startswith("--export=")
        )
        encoded = next(
            part.split("=", 1)[1]
            for part in export_arg.split(",")
            if part.startswith("vllm_args=")
        )
        decoded = json.loads(base64.b64decode(encoded).decode())
        parser = argparse.ArgumentParser()
        parser.add_argument("--served-model-name", nargs="+")

        assert decoded.count("--served-model-name") == 1
        advertised_names = parser.parse_args(decoded).served_model_name
        assert advertised_names == [stable_name, model.name]
        assert later_sibling.api_model_name in advertised_names

    @pytest.mark.asyncio
    async def test_revision_included_in_vllm_args(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m", revision="abc123")
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        export_arg = next(a for a in sbatch_args_captured if a.startswith("--export="))
        for part in export_arg.split(","):
            if part.startswith("vllm_args="):
                b64 = part.split("=", 1)[1]
                decoded = json.loads(base64.b64decode(b64).decode())
                normalize = lambda args: [" ".join(s.split()) for s in args]
                assert normalize(decoded) == normalize([
                    "--revision", "abc123",
                    "--served-model-name", "m",
                ])
                break
        else:
            pytest.fail("vllm_args not found in --export")

    @pytest.mark.asyncio
    async def test_no_revision_no_revision_flag(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m", revision=None)
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        export_arg = next(a for a in sbatch_args_captured if a.startswith("--export="))
        for part in export_arg.split(","):
            if part.startswith("vllm_args="):
                b64 = part.split("=", 1)[1]
                decoded = json.loads(base64.b64decode(b64).decode())
                normalize = lambda args: [" ".join(s.split()) for s in args]
                assert normalize(decoded) == normalize([
                    "--served-model-name", "m",
                ])
                break
        else:
            pytest.fail("vllm_args not found in --export")

    @pytest.mark.asyncio
    async def test_allow_long_max_model_len_exported_when_enabled(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m", allow_long_max_model_len=True)
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        export_arg = next(a for a in sbatch_args_captured if a.startswith("--export="))
        assert "vllm_allow_long_max_model_len=1" in export_arg

    @pytest.mark.asyncio
    async def test_allow_long_max_model_len_can_be_disabled(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m", allow_long_max_model_len=False)
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        export_arg = next(a for a in sbatch_args_captured if a.startswith("--export="))
        assert "vllm_allow_long_max_model_len=0" in export_arg

    @pytest.mark.asyncio
    async def test_logs_when_allow_long_max_model_len_enabled(self, tmp_path, caplog):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m", allow_long_max_model_len=True)

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert "Enabling VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for m" in caplog.text

    @pytest.mark.asyncio
    async def test_vllm_logging_level_exported(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m", vllm_logging_level="DEBUG")
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        export_arg = next(a for a in sbatch_args_captured if a.startswith("--export="))
        assert "vllm_logging_level=DEBUG" in export_arg

    @pytest.mark.asyncio
    async def test_vllm_logging_level_defaults_to_warning_in_export(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m")  # default vllm_logging_level="WARNING"
        sbatch_args_captured = []

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        export_arg = next(a for a in sbatch_args_captured if a.startswith("--export="))
        assert "vllm_logging_level=WARNING" in export_arg

    @pytest.mark.asyncio
    async def test_model_already_exists_sbatch_not_called(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="existing")
        # update_allocation calls squeue twice: kill_all and get_all_jobs
        existing_line = _squeue_line(_fake_job_name(model), 1, "RUNNING", "01:00:00", "node01")
        sbatch_called = []
        squeue_call_n = [0]

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                squeue_call_n[0] += 1
                if squeue_call_n[0] == 1:
                    return FakeSlurmProcess.success("")  # kill_all: no jobs
                return FakeSlurmProcess.success(existing_line)  # update_allocation: get_all_jobs
            sbatch_called.append(args)
            return FakeSlurmProcess.success("Submitted batch job 1")

        with (
            patch("asyncio.create_subprocess_exec", side_effect=ordered_exec),
            _patch_squeue(jm, "", existing_line),
        ):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert sbatch_called == []

    @pytest.mark.asyncio
    async def test_second_model_deployed_when_first_already_exists(self, tmp_path):
        """When the first model already exists, the second should still be sbatched."""
        jm = _jm(log_dir=str(tmp_path))
        existing_model = _fake_model(name="existing", path="org/existing-model")
        new_model = _fake_model(name="newmodel", path="org/new-model")
        existing_line = _squeue_line(_fake_job_name(existing_model), 1, "RUNNING", "01:00:00", "node01")
        sbatch_args_captured = []
        squeue_call_n = [0]

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                squeue_call_n[0] += 1
                if squeue_call_n[0] == 1:
                    return FakeSlurmProcess.success("")  # kill_all: no jobs
                return FakeSlurmProcess.success(existing_line)  # update_allocation: get_all_jobs
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 2")

        with (
            patch("asyncio.create_subprocess_exec", side_effect=ordered_exec),
            _patch_squeue(jm, "", existing_line),
        ):
            await jm.update_allocation([(existing_model, 1), (new_model, 1)], unneeded_models=[])

        assert "sbatch" in sbatch_args_captured
        assert f"--job-name={_fake_job_name(new_model)}" in sbatch_args_captured

    @pytest.mark.asyncio
    async def test_first_model_deployed_when_second_already_exists(self, tmp_path):
        """When the second model already exists, the first should still be sbatched."""
        jm = _jm(log_dir=str(tmp_path))
        new_model = _fake_model(name="newmodel", path="org/new-model")
        existing_model = _fake_model(name="existing", path="org/existing-model")
        existing_line = _squeue_line(_fake_job_name(existing_model), 1, "RUNNING", "01:00:00", "node01")
        sbatch_args_captured = []
        squeue_call_n = [0]

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                squeue_call_n[0] += 1
                if squeue_call_n[0] == 1:
                    return FakeSlurmProcess.success("")  # kill_all: no jobs
                return FakeSlurmProcess.success(existing_line)  # update_allocation: get_all_jobs
            sbatch_args_captured.extend(args)
            return FakeSlurmProcess.success("Submitted batch job 2")

        with (
            patch("asyncio.create_subprocess_exec", side_effect=ordered_exec),
            _patch_squeue(jm, "", existing_line),
        ):
            await jm.update_allocation([(new_model, 1), (existing_model, 1)], unneeded_models=[])

        assert "sbatch" in sbatch_args_captured
        assert f"--job-name={_fake_job_name(new_model)}" in sbatch_args_captured
        assert f"--job-name={_fake_job_name(existing_model)}" not in sbatch_args_captured

    @pytest.mark.asyncio
    async def test_sbatch_failure_raises(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m")

        async def ordered_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            return FakeSlurmProcess.failure("partition not found", rc=1)

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            with pytest.raises(RuntimeError, match="sbatch failed"):
                await jm.update_allocation([(model, 1)], unneeded_models=[])

    @pytest.mark.asyncio
    async def test_sbatch_timeout_raises(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="m")
        call_n = [0]

        async def ordered_exec(*args, **kwargs):
            call_n[0] += 1
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            return FakeSlurmProcess.timeout()

        with patch("asyncio.create_subprocess_exec", side_effect=ordered_exec):
            with pytest.raises(asyncio.TimeoutError):
                await jm.update_allocation([(model, 1)], unneeded_models=[])


# ---------------------------------------------------------------------------
# TestSubmitImportedDatasetJob
# ---------------------------------------------------------------------------

class TestSubmitImportedDatasetJob:

    @pytest.mark.asyncio
    async def test_returns_slurm_job_id(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model()
        with patch("asyncio.create_subprocess_exec",
                   return_value=FakeSlurmProcess.success("Submitted batch job 42")):
            job_id = await jm.submit_imported_dataset_job(
                "abcd1234-uuid", model, "my-runner", "echo setup", "echo benchmark", tmp_path
            )
        assert job_id == 42

    @pytest.mark.asyncio
    async def test_job_name_uses_event_uuid_prefix(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model()
        captured = []
        with patch("asyncio.create_subprocess_exec",
                   side_effect=lambda *a, **kw: captured.extend(a) or
                               FakeSlurmProcess.success("Submitted batch job 1")):
            await jm.submit_imported_dataset_job(
                "abcd1234-5678", model, "runner", "echo s", "echo b", tmp_path
            )
        assert f"--job-name=eval360id-{_TEST_IID}-runner-mymodel-abcd1234" in captured

    @pytest.mark.asyncio
    async def test_imported_job_name_not_detected_as_grading_job(self):
        jm = _jm()
        assert jm._from_job_name(f"eval360id-{_TEST_IID}-abcd1234") is None

    @pytest.mark.asyncio
    async def test_setup_and_benchmark_scripts_base64_encoded_in_export(self, tmp_path):
        import base64
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model()
        captured = []
        with patch("asyncio.create_subprocess_exec",
                   side_effect=lambda *a, **kw: captured.extend(a) or
                               FakeSlurmProcess.success("Submitted batch job 1")):
            await jm.submit_imported_dataset_job(
                "uuid-1234", model, "runner", "echo setup", "echo benchmark", tmp_path
            )
        export_arg = next(a for a in captured if a.startswith("--export="))
        assert base64.b64encode(b"echo setup").decode() in export_arg
        assert base64.b64encode(b"echo benchmark").decode() in export_arg

    @pytest.mark.asyncio
    async def test_runner_name_in_export(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model()
        captured = []
        with patch("asyncio.create_subprocess_exec",
                   side_effect=lambda *a, **kw: captured.extend(a) or
                               FakeSlurmProcess.success("Submitted batch job 1")):
            await jm.submit_imported_dataset_job(
                "uuid-1234", model, "bfcl", "echo s", "echo b", tmp_path
            )
        export_arg = next(a for a in captured if a.startswith("--export="))
        assert "runner_name=bfcl" in export_arg

    @pytest.mark.asyncio
    async def test_bootstrap_python_in_export(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model()
        captured = []
        with patch("asyncio.create_subprocess_exec",
                   side_effect=lambda *a, **kw: captured.extend(a) or
                               FakeSlurmProcess.success("Submitted batch job 1")):
            await jm.submit_imported_dataset_job(
                "uuid-1234", model, "runner", "echo s", "echo b", tmp_path
            )
        export_arg = next(a for a in captured if a.startswith("--export="))
        assert f"bootstrap_python={sys.executable}" in export_arg

    @pytest.mark.asyncio
    async def test_container_imported_job_discovers_bootstrap_python_in_job(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(container_image="/images/vllm.sqsh")
        captured = []
        with patch("asyncio.create_subprocess_exec",
                   side_effect=lambda *a, **kw: captured.extend(a) or
                               FakeSlurmProcess.success("Submitted batch job 1")):
            await jm.submit_imported_dataset_job(
                "uuid-1234", model, "runner", "echo s", "echo b", tmp_path
            )
        export_arg = next(a for a in captured if a.startswith("--export="))
        assert f"bootstrap_python={sys.executable}" not in export_arg
        assert "bootstrap_python=," in export_arg
        assert any(a == "--container-image=/images/vllm.sqsh" for a in captured)
        mount_args = [a for a in captured if isinstance(a, str) and a.startswith("--container-mounts=")]
        assert mount_args == ["--container-mounts=org/model:org/model"]
        assert any("imported_dataset_script_container.sh" in str(a) for a in captured)

    @pytest.mark.asyncio
    async def test_container_imported_job_combines_model_and_user_mounts(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(
            container_image="/images/vllm.sqsh",
            path="/data/checkpoints/checkpoint_050000",
            container_mounts=["/secrets/search.env:/secrets/search.env:ro"],
        )
        captured = []
        with patch("asyncio.create_subprocess_exec",
                   side_effect=lambda *a, **kw: captured.extend(a) or
                               FakeSlurmProcess.success("Submitted batch job 1")):
            await jm.submit_imported_dataset_job(
                "uuid-1234", model, "runner", "echo s", "echo b", tmp_path
            )

        mount_args = [a for a in captured if isinstance(a, str) and a.startswith("--container-mounts=")]
        assert len(mount_args) == 1
        assert mount_args[0] == (
            "--container-mounts="
            "/data/checkpoints/checkpoint_050000:/data/checkpoints/checkpoint_050000,"
            "/secrets/search.env:/secrets/search.env:ro"
        )

    @pytest.mark.asyncio
    async def test_sbatch_failure_raises(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model()
        with patch("asyncio.create_subprocess_exec",
                   return_value=FakeSlurmProcess.failure("sbatch error")):
            with pytest.raises(RuntimeError, match="sbatch failed"):
                await jm.submit_imported_dataset_job(
                    "uuid-1234", model, "runner", "echo s", "echo b", tmp_path
                )

    @pytest.mark.asyncio
    async def test_sbatch_timeout_raises_and_kills(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model()
        proc = FakeSlurmProcess.timeout()
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(asyncio.TimeoutError):
                await jm.submit_imported_dataset_job(
                    "uuid-1234", model, "runner", "echo s", "echo b", tmp_path
                )
        assert proc.killed

    @pytest.mark.asyncio
    async def test_uses_imported_dataset_script(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model()
        captured = []
        with patch("asyncio.create_subprocess_exec",
                   side_effect=lambda *a, **kw: captured.extend(a) or
                               FakeSlurmProcess.success("Submitted batch job 1")):
            await jm.submit_imported_dataset_job(
                "uuid-1234", model, "runner", "echo s", "echo b", tmp_path
            )
        assert any("imported_dataset_script.sh" in str(a) for a in captured)


class TestTerminalJobAccounting:
    """Authoritative root-job evidence must come from sacct, not squeue."""

    @pytest.mark.asyncio
    async def test_submitted_serving_job_is_retained_in_ledger(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        jm.begin_terminal_result_capture()
        model = _fake_model(name="ledger-model")

        async def fake_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            assert "--parsable" in args
            return FakeSlurmProcess.success("123;test-cluster")

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            _patch_squeue(jm, "", ""),
        ):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        jobs = jm.get_submitted_jobs()
        assert len(jobs) == 1
        assert jobs[0].job_id == 123
        assert jobs[0].kind == "model_serving"
        assert jobs[0].submission_origin == "submitted"
        assert jobs[0].serving_key == model.serving_key

    @pytest.mark.asyncio
    async def test_release_uses_ledgered_serving_job_ids_without_squeue(self):
        jm = _jm()
        jm.begin_terminal_result_capture()
        jobs = (
            slurm_manager_module.SubmittedSlurmJob(
                job_id=41,
                job_name="eval360-deadbeef-model-a-123456789abc-r0",
                kind="model_serving",
                submission_origin="submitted",
                model_name="model-a",
                serving_key="123456789abc",
            ),
            slurm_manager_module.SubmittedSlurmJob(
                job_id=42,
                job_name="eval360-deadbeef-model-b-abcdef123456-r0",
                kind="model_serving",
                submission_origin="submitted",
                model_name="model-b",
                serving_key="abcdef123456",
            ),
            slurm_manager_module.SubmittedSlurmJob(
                job_id=43,
                job_name="eval360id-deadbeef-runner-model-event",
                kind="imported_dataset",
                submission_origin="submitted",
                model_name="model-a",
                serving_key="123456789abc",
            ),
        )
        for job in jobs:
            jm._remember_job(job)
        commands = []

        def fake_bounded_command(argv, *, timeout):
            commands.append(argv)
            return _command_result(argv[0])

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            await jm.release_submitted_model_serving_jobs()

        assert all(command[0] != "squeue" for command in commands)
        assert {
            command[:2] for command in commands if command[0] == "scancel"
        } == {
            ("scancel", "41"),
            ("scancel", "42"),
        }
        released = {job.job_id: job for job in jm.get_submitted_jobs()}
        assert released[41].cancellation_intent == "scheduler_release"
        assert released[42].cancellation_intent == "scheduler_release"
        assert released[43].cancellation_intent is None

    @pytest.mark.asyncio
    async def test_release_does_not_recancel_terminal_serving_jobs(self):
        jm = _jm()
        jm.begin_terminal_result_capture()
        for job_id in (41, 42):
            jm._remember_job(
                slurm_manager_module.SubmittedSlurmJob(
                    job_id=job_id,
                    job_name=(
                        f"eval360-deadbeef-model-{job_id}-123456789abc-r0"
                    ),
                    kind="model_serving",
                    submission_origin="submitted",
                    model_name=f"model-{job_id}",
                    serving_key="123456789abc",
                )
            )
        commands = []

        def fake_bounded_command(argv, *, timeout):
            commands.append(argv)
            if argv[0] == "sacct":
                return _command_result(
                    "sacct",
                    "41|eval360-deadbeef-model-41-123456789abc-r0|"
                    "COMPLETED|0:0|None\n"
                    "42|eval360-deadbeef-model-42-123456789abc-r0|"
                    "RUNNING|0:0|None\n"
                )
            return _command_result(argv[0])

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            await jm.release_submitted_model_serving_jobs()

        assert [command[:2] for command in commands if command[0] == "scancel"] == [
            ("scancel", "42")
        ]
        released = {job.job_id: job for job in jm.get_submitted_jobs()}
        assert released[41].cancellation_intent is None
        assert released[42].cancellation_intent == "scheduler_release"

    @pytest.mark.asyncio
    async def test_release_attempts_every_active_job_before_raising(self):
        jm = _jm()
        jm.begin_terminal_result_capture()
        for job_id in (41, 42):
            jm._remember_job(
                slurm_manager_module.SubmittedSlurmJob(
                    job_id=job_id,
                    job_name=(
                        f"eval360-deadbeef-model-{job_id}-123456789abc-r0"
                    ),
                    kind="model_serving",
                    submission_origin="submitted",
                    model_name=f"model-{job_id}",
                    serving_key="123456789abc",
                )
            )
        cancelled = []

        def fake_bounded_command(argv, *, timeout):
            if argv[0] == "sacct":
                return _command_result("sacct")
            cancelled.append(argv[1])
            if argv[1] == "41":
                return _command_result(
                    "scancel",
                    stderr_text="permission denied",
                    returncode=1,
                )
            return _command_result("scancel")

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            pytest.raises(RuntimeError, match="scancel failed.*permission denied"),
        ):
            await jm.release_submitted_model_serving_jobs()

        assert set(cancelled) == {"41", "42"}

    @pytest.mark.asyncio
    async def test_accounting_failure_still_releases_every_serving_job(self):
        jm = _jm()
        jm.begin_terminal_result_capture()
        for job_id in (41, 42):
            jm._remember_job(
                slurm_manager_module.SubmittedSlurmJob(
                    job_id=job_id,
                    job_name=(
                        f"eval360-deadbeef-model-{job_id}-123456789abc-r0"
                    ),
                    kind="model_serving",
                    submission_origin="submitted",
                    model_name=f"model-{job_id}",
                    serving_key="123456789abc",
                )
            )
        cancelled = []

        def fake_bounded_command(argv, *, timeout):
            if argv[0] == "sacct":
                raise RuntimeError("accounting unavailable")
            cancelled.append(argv[1])
            return _command_result("scancel")

        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ),
            pytest.raises(RuntimeError, match="accounting unavailable"),
        ):
            await jm.release_submitted_model_serving_jobs()

        assert set(cancelled) == {"41", "42"}

    @pytest.mark.asyncio
    async def test_ambiguous_sbatch_success_fails_terminal_capture(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        jm.begin_terminal_result_capture()
        model = _fake_model(name="ambiguous-job-id")

        async def fake_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            assert "--parsable" in args
            return FakeSlurmProcess.success("Submitted batch job 123")

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            _patch_squeue(jm, "", ""),
            pytest.raises(RuntimeError, match="parseable submitted job ID"),
        ):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert jm.get_submitted_jobs() == ()

    @pytest.mark.asyncio
    async def test_role_completion_is_bound_to_active_serving_child(self, tmp_path):
        jm = _jm(log_dir=str(tmp_path))
        jm.begin_terminal_result_capture()
        model = _fake_model(name="role-model")
        job_name = _fake_job_name(model)
        squeue_calls = 0

        async def fake_exec(*args, **kwargs):
            nonlocal squeue_calls
            if args[0] == "squeue":
                squeue_calls += 1
                if squeue_calls == 1:
                    return FakeSlurmProcess.success("")
                return FakeSlurmProcess.success(
                    f"{job_name}|123|RUNNING|00:01|node-a\n"
                )
            return FakeSlurmProcess.success("123;test-cluster")

        active_job = f"{job_name}|123|RUNNING|00:01|node-a\n"
        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            _patch_squeue(jm, "", active_job, active_job),
        ):
            await jm.update_allocation([(model, 1)], unneeded_models=[])
            await jm.record_role_completion(
                model.serving_key,
                "event-a",
                "generation",
            )

        assert jm.get_submitted_jobs()[0].completed_event_roles == (
            ("event-a", "generation"),
        )

    @pytest.mark.asyncio
    async def test_failed_release_cancellation_propagates_in_terminal_mode(self):
        jm = _jm()
        jm.begin_terminal_result_capture()
        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                return_value=_command_result(
                    "scancel",
                    stderr_text="permission denied",
                    returncode=1,
                ),
            ),
            pytest.raises(RuntimeError, match="scancel failed.*permission denied"),
        ):
            await jm.cancel_job(123, cancellation_intent="scheduler_release")

    @pytest.mark.asyncio
    async def test_legacy_submission_does_not_require_evidence_job_id_parsing(
        self,
        tmp_path,
    ):
        jm = _jm(log_dir=str(tmp_path))
        model = _fake_model(name="legacy-model")

        async def fake_exec(*args, **kwargs):
            if args[0] == "squeue":
                return FakeSlurmProcess.success("")
            return FakeSlurmProcess.success("site-specific success text")

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            _patch_squeue(jm, "", ""),
        ):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert jm.get_submitted_jobs() == ()

    @pytest.mark.asyncio
    async def test_sacct_uses_only_exact_root_rows(self):
        jm = _jm()
        stdout = (
            "42|eval360-root|COMPLETED|0:0|None|\n"
            "42.batch|batch|CANCELLED|0:0|\n"
        )
        captured = []

        def fake_bounded_command(argv, *, timeout):
            captured.append(argv)
            return _command_result(argv[0], stdout)

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            outcomes = await jm.get_job_accounting([42])

        assert list(outcomes) == [42]
        assert outcomes[42].is_clean_completion
        assert captured[0][0] == "sacct"
        assert "--format=JobIDRaw,JobName%256,State,ExitCode,Reason" in captured[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stdout",
        (
            "42|eval360-root|COMPLETED|0:0|\n",
            "42|eval360-root|COMPLETED|0:0||\n",
        ),
    )
    async def test_sacct_accepts_an_empty_reason(self, stdout):
        jm = _jm()

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            return_value=_command_result("sacct", stdout),
        ):
            outcomes = await jm.get_job_accounting([42])

        assert outcomes[42].is_clean_completion
        assert outcomes[42].reason == ""

    @pytest.mark.asyncio
    async def test_waits_on_sacct_state_without_using_squeue(self):
        jm = SlurmManager(instance_id=_TEST_IID, poll_interval=0)
        responses = iter(
            [
                "42|job|RUNNING|0:0|None|\n",
                "42|job|COMPLETED|0:0|None|\n",
            ]
        )
        commands = []

        def fake_bounded_command(argv, *, timeout):
            commands.append(argv[0])
            return _command_result(argv[0], next(responses))

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            outcomes = await jm.wait_for_terminal_job_outcomes([42])

        assert outcomes[42].state == "COMPLETED"
        assert commands == ["sacct", "sacct"]

    @pytest.mark.asyncio
    async def test_missing_required_sacct_root_row_fails(self):
        # The publish grace is real time, so this used to spend the whole
        # 30 seconds asleep on every run of the suite. The virtual clock walks
        # the production grace to its end without waiting for it.
        jm = _jm(poll_interval=0)
        with (
            _virtual_clock(step=10.0),
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                return_value=_command_result(
                    "sacct",
                    "42.batch|batch|COMPLETED|0:0|None|\n"
                ),
            ),
            pytest.raises(RuntimeError, match="no root accounting row.*42"),
        ):
            await jm.wait_for_terminal_job_outcomes([42])

    @pytest.mark.asyncio
    async def test_unknown_sacct_state_fails_instead_of_polling_forever(self):
        jm = _jm()
        with (
            patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                return_value=_command_result(
                    "sacct",
                    "42|job|FUTURE_STATE|0:0|None|\n"
                ),
            ),
            pytest.raises(RuntimeError, match="unknown sacct state.*42"),
        ):
            await jm.wait_for_terminal_job_outcomes([42])


# ---------------------------------------------------------------------------
# TestSacctPublishLag
# ---------------------------------------------------------------------------

class TestSacctPublishLag:
    """`wait_for_terminal_job_outcomes` against slurmdbd's staged publish.

    What this covers:
        slurmdbd does not write a job's accounting row once. Measured against a
        real cluster, the same job is successively: absent, then present under
        the placeholder name `allocation` and possibly ALREADY TERMINAL, then
        present under its real name. A caller that reads the middle stage gets
        a terminal row whose name is not the job's, and
        `Scheduler._reconcile_terminal_jobs` fails the whole evaluation with
        "sacct job name ... does not match the submitted child identity".

    Why it is here and not in the cluster suite:
        The cluster regression cancels a job and races the window, so it can
        pass without ever entering the settlement loop it exists to cover, and
        it cannot reach the expiry branches at all without a 30-second wait.
        These drive the same method with the response sequence and the clock
        both written down, so every branch is reached on purpose.

    What makes them bite:
        The recovery cases are written to run unchanged against the previous
        implementation, where they fail — the first on the raise it had no
        retry for, the second on the placeholder name it returned.
    """

    @pytest.mark.asyncio
    async def test_a_row_that_is_not_published_yet_is_waited_for(self, tmp_path):
        """Stage one: no row at all is a lag, not a missing job."""
        jm = _jm(log_dir=str(tmp_path), poll_interval=0)
        jm.begin_terminal_result_capture()
        child = await _submit_into_ledger(jm, _fake_model(name="lag-publish"), 123)

        sacct = _ScriptedSacct(
            "",                                       # nothing published yet
            "123.batch|batch|COMPLETED|0:0|None|\n",  # a step row, still no root
            _sacct_row(123, child.job_name, "COMPLETED"),
        )
        with (
            _virtual_clock(),
            patch.object(
                slurm_manager_module, "_run_bounded_command", side_effect=sacct
            ),
        ):
            outcomes = await jm.wait_for_terminal_job_outcomes([123])

        assert sacct.call_count == 3, "gave up before the row was published"
        assert outcomes[123].is_terminal
        assert outcomes[123].job_name == child.job_name

    @pytest.mark.asyncio
    async def test_a_terminal_placeholder_name_is_waited_out(self, tmp_path):
        """Stage two: a terminal row under the placeholder name is not the answer.

        This is the silent half of the bug. Returning here is not an error at
        the time — the row is genuinely terminal — it just carries a name the
        scheduler will reject several steps later.
        """
        jm = _jm(log_dir=str(tmp_path), poll_interval=0)
        jm.begin_terminal_result_capture()
        child = await _submit_into_ledger(jm, _fake_model(name="lag-settle"), 123)
        assert child.job_name != "allocation"

        sacct = _ScriptedSacct(
            _sacct_row(123, "allocation", "CANCELLED"),
            _sacct_row(123, "allocation", "CANCELLED"),
            _sacct_row(123, child.job_name, "CANCELLED"),
        )
        with (
            _virtual_clock(),
            patch.object(
                slurm_manager_module, "_run_bounded_command", side_effect=sacct
            ),
        ):
            outcomes = await jm.wait_for_terminal_job_outcomes([123])

        assert sacct.call_count == 3, "returned while the row was still a placeholder"
        assert outcomes[123].job_name == child.job_name, (
            "a terminal row under the placeholder name was returned as the "
            "job's outcome; Scheduler raises 'sacct job name does not match "
            "the submitted child identity' on exactly this"
        )

    @pytest.mark.asyncio
    async def test_a_name_that_never_settles_expires_instead_of_hanging(
        self, tmp_path
    ):
        """The settle grace is a ceiling: a permanent mismatch fails, loudly.

        Tolerating the lag must not become tolerating a row that is simply
        someone else's. The clock steps 10s per read, so the production 30s
        grace is crossed here in no real time — and a regression that made the
        wait unbounded would hang this test rather than quietly passing it.
        """
        jm = _jm(log_dir=str(tmp_path), poll_interval=0)
        jm.begin_terminal_result_capture()
        child = await _submit_into_ledger(jm, _fake_model(name="lag-stuck"), 123)

        sacct = _ScriptedSacct(then=_sacct_row(123, "allocation", "CANCELLED"))
        clock = _virtual_clock(step=10.0)
        with (
            clock as fake_time,
            patch.object(
                slurm_manager_module, "_run_bounded_command", side_effect=sacct
            ),
            pytest.raises(RuntimeError) as raised,
        ):
            await jm.wait_for_terminal_job_outcomes([123])

        message = str(raised.value)
        assert "placeholder job name" in message
        assert "'allocation'" in message and repr(child.job_name) in message
        assert sacct.call_count > 1, "expired without ever retrying"
        assert fake_time.elapsed >= 30.0, (
            "gave up before the documented 30s settle grace elapsed"
        )

    @pytest.mark.asyncio
    async def test_every_job_is_waited_out_even_when_they_settle_apart(
        self, tmp_path
    ):
        """Completeness: the slowest job sets the pace, per stage.

        Three children, each stuck at a different stage, and none of them may
        be reported until it carries its own submitted name. A wait that
        settled only the job it happened to look at first would pass every
        single-job test and still corrupt this one.
        """
        jm = _jm(log_dir=str(tmp_path), poll_interval=0)
        jm.begin_terminal_result_capture()
        first = await _submit_into_ledger(jm, _fake_model(name="lag-multi-a"), 101)
        second = await _submit_into_ledger(jm, _fake_model(name="lag-multi-b"), 102)
        third = await _submit_into_ledger(jm, _fake_model(name="lag-multi-c"), 103)

        settled_first = _sacct_row(101, first.job_name, "COMPLETED")
        placeholder_second = _sacct_row(102, "allocation", "CANCELLED")
        settled_second = _sacct_row(102, second.job_name, "CANCELLED")
        sacct = _ScriptedSacct(
            settled_first,                                        # 102, 103 absent
            settled_first + placeholder_second,                   # 103 still absent
            settled_first
            + placeholder_second
            + _sacct_row(103, third.job_name, "RUNNING"),          # published, running
            settled_first
            + placeholder_second
            + _sacct_row(103, third.job_name, "COMPLETED"),        # terminal, unsettled
            settled_first
            + settled_second
            + _sacct_row(103, third.job_name, "COMPLETED"),        # all settled
        )
        with (
            _virtual_clock(),
            patch.object(
                slurm_manager_module, "_run_bounded_command", side_effect=sacct
            ),
        ):
            outcomes = await jm.wait_for_terminal_job_outcomes([101, 102, 103])

        assert sacct.call_count == 5
        assert set(outcomes) == {101, 102, 103}
        assert all(outcome.is_terminal for outcome in outcomes.values())
        assert {job_id: outcome.job_name for job_id, outcome in outcomes.items()} == {
            101: first.job_name,
            102: second.job_name,
            103: third.job_name,
        }

    @pytest.mark.asyncio
    async def test_a_row_lost_while_waiting_for_terminality_fails(self, tmp_path):
        """A published row that disappears is not a lag — the grace is over.

        Re-entering the publish grace on a row that already existed would turn
        a purged or mis-queried job into a 30-second wait and then the wrong
        error. It has to fail on the loss.
        """
        jm = _jm(log_dir=str(tmp_path), poll_interval=0)
        jm.begin_terminal_result_capture()
        child = await _submit_into_ledger(jm, _fake_model(name="lag-lost-run"), 123)

        sacct = _ScriptedSacct(_sacct_row(123, child.job_name, "RUNNING"), "")
        with (
            _virtual_clock(),
            patch.object(
                slurm_manager_module, "_run_bounded_command", side_effect=sacct
            ),
            pytest.raises(RuntimeError, match="lost the root accounting row.*123"),
        ):
            await jm.wait_for_terminal_job_outcomes([123])

        assert sacct.call_count == 2

    @pytest.mark.asyncio
    async def test_a_row_lost_while_waiting_to_settle_fails(self, tmp_path):
        """Same loss, reached from the other grace."""
        jm = _jm(log_dir=str(tmp_path), poll_interval=0)
        jm.begin_terminal_result_capture()
        await _submit_into_ledger(jm, _fake_model(name="lag-lost-settle"), 123)

        sacct = _ScriptedSacct(_sacct_row(123, "allocation", "CANCELLED"), "")
        with (
            _virtual_clock(),
            patch.object(
                slurm_manager_module, "_run_bounded_command", side_effect=sacct
            ),
            pytest.raises(RuntimeError, match="lost the root accounting row.*123"),
        ):
            await jm.wait_for_terminal_job_outcomes([123])

        assert sacct.call_count == 2

    @pytest.mark.asyncio
    async def test_an_adopted_job_has_no_name_to_settle_against(self):
        """A job absent from the ledger is terminal-and-done, not unsettled.

        Nothing will compare an adopted job's name to anything, so waiting for
        one to match would be a wait for an event that cannot happen.
        """
        jm = _jm(poll_interval=0)
        sacct = _ScriptedSacct(_sacct_row(42, "allocation", "COMPLETED"))
        with (
            _virtual_clock(step=10.0),
            patch.object(
                slurm_manager_module, "_run_bounded_command", side_effect=sacct
            ),
        ):
            outcomes = await jm.wait_for_terminal_job_outcomes([42])

        assert sacct.call_count == 1
        assert outcomes[42].job_name == "allocation"


# ---------------------------------------------------------------------------
# TestWaitForJobCompletion
# ---------------------------------------------------------------------------

class TestWaitForJobCompletion:

    @pytest.mark.asyncio
    async def test_returns_immediately_when_job_not_found(self):
        jm = _jm()
        with patch("asyncio.create_subprocess_exec",
                   return_value=FakeSlurmProcess.success("")):
            await jm.wait_for_job_completion(42, poll_interval=0)

    @pytest.mark.asyncio
    async def test_polls_until_job_disappears(self):
        jm = _jm()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FakeSlurmProcess.success("42  RUNNING")
            return FakeSlurmProcess.success("")

        with patch("asyncio.create_subprocess_exec", side_effect=side_effect):
            await jm.wait_for_job_completion(42, poll_interval=0)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_nonzero_exit_treated_as_complete(self):
        jm = _jm()
        with patch("asyncio.create_subprocess_exec",
                   return_value=FakeSlurmProcess.failure("Invalid job id")):
            await jm.wait_for_job_completion(99, poll_interval=0)

    @pytest.mark.asyncio
    async def test_timeout_raises_and_kills(self):
        jm = _jm()
        proc = FakeSlurmProcess.timeout()
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(asyncio.TimeoutError):
                await jm.wait_for_job_completion(42, poll_interval=0)
        assert proc.killed

    @pytest.mark.asyncio
    async def test_whitespace_only_stdout_treated_as_complete(self):
        # Gap 7: squeue stdout of "\n   \n" must be treated as "job not found"
        # (i.e. complete) because stdout.decode().strip() is falsy.
        jm = _jm()
        with patch("asyncio.create_subprocess_exec",
                   return_value=FakeSlurmProcess.success("\n   \n")):
            # Should return immediately, not loop forever
            await jm.wait_for_job_completion(42, poll_interval=0)

# ---------------------------------------------------------------------------
# TestGetJobNode
# ---------------------------------------------------------------------------

class TestGetJobNode:

    @pytest.mark.asyncio
    async def test_returns_node_when_running(self):
        jm = _jm()
        with patch("asyncio.create_subprocess_exec",
                   return_value=FakeSlurmProcess.success("gpu-node-01")):
            node = await jm.get_job_node(42, poll_interval=0)
        assert node == "gpu-node-01"

    @pytest.mark.asyncio
    async def test_polls_until_node_assigned(self):
        jm = _jm()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FakeSlurmProcess.success("(None)")
            return FakeSlurmProcess.success("gpu-node-01")

        with patch("asyncio.create_subprocess_exec", side_effect=side_effect):
            node = await jm.get_job_node(42, poll_interval=0)
        assert node == "gpu-node-01"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_polls_while_a_pending_row_has_no_node_yet(self):
        """A PENDING job is one row whose %N is empty — keep waiting.

        THIS FIXTURE WAS WRONG, and it was defending a bug. It previously fed
        back `""` for the pending case and asserted that polling continued,
        which is what let `get_job_node` loop forever on a job that had already
        left the queue.

        Real `squeue --jobs N --format=%N --noheader`, measured against a live
        cluster, distinguishes the two by ROW COUNT and not by content:

            pending         rc=0, one row, empty  -> b"\\n"
            left the queue  rc=0, no rows         -> b""

        Both `.strip()` to the empty string, which is why the old fixture and
        the old code agreed with each other and disagreed with Slurm. The
        pending case is modelled here; the absent case is the test below.
        """
        jm = _jm()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return FakeSlurmProcess.success("\n")
            return FakeSlurmProcess.success("gpu-node-02")

        with patch("asyncio.create_subprocess_exec", side_effect=side_effect):
            node = await jm.get_job_node(42, poll_interval=0)
        assert node == "gpu-node-02"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_when_the_job_has_left_the_queue(self):
        """No rows means the job is gone; polling on would never end.

        `handle_imported_dataset_event` awaits `get_job_node` with no timeout,
        so returning to the loop here hangs the event forever rather than
        failing it.
        """
        jm = _jm()
        with patch("asyncio.create_subprocess_exec",
                   return_value=FakeSlurmProcess.success("")):
            with pytest.raises(RuntimeError, match="no longer queued"):
                await jm.get_job_node(42, poll_interval=0)

    @pytest.mark.asyncio
    async def test_raises_when_squeue_rejects_the_job_id(self):
        """A purged job ID exits non-zero — also nothing left to wait for."""
        jm = _jm()
        with patch("asyncio.create_subprocess_exec",
                   return_value=FakeSlurmProcess.failure(
                       "slurm_load_jobs error: Invalid job id specified")):
            with pytest.raises(RuntimeError, match="no longer queued"):
                await jm.get_job_node(42, poll_interval=0)

    @pytest.mark.asyncio
    async def test_timeout_raises_and_kills(self):
        jm = _jm()
        proc = FakeSlurmProcess.timeout()
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(asyncio.TimeoutError):
                await jm.get_job_node(42, poll_interval=0)
        assert proc.killed


# ---------------------------------------------------------------------------
# TestWaitForVllmHealth
# ---------------------------------------------------------------------------

class TestWaitForVllmHealth:

    def _make_session(self, status: int):
        """Return a mock aiohttp.ClientSession that yields a response with the given status."""
        resp = MagicMock()
        resp.status = status
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.get = MagicMock(return_value=resp)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return session

    @pytest.mark.asyncio
    async def test_returns_true_on_200(self):
        jm = _jm()
        session = self._make_session(200)
        with patch("aiohttp.ClientSession", return_value=session):
            result = await jm.wait_for_vllm_health("gpu-node-01", max_time_to_deploy=60, poll_interval=0)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        jm = _jm()
        session = self._make_session(503)
        with patch("aiohttp.ClientSession", return_value=session):
            result = await jm.wait_for_vllm_health("gpu-node-01", max_time_to_deploy=0, poll_interval=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        jm = _jm()
        call_count = 0

        resp_ok = MagicMock()
        resp_ok.status = 200
        resp_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        resp_ok.__aexit__ = AsyncMock(return_value=False)

        def get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise aiohttp.ClientConnectionError()
            return resp_ok

        session = MagicMock()
        session.get = MagicMock(side_effect=get_side_effect)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session):
            result = await jm.wait_for_vllm_health("gpu-node-01", max_time_to_deploy=60, poll_interval=0)
        assert result is True
        assert call_count == 3


# ---------------------------------------------------------------------------
# TestCancelJob
# ---------------------------------------------------------------------------

class TestCancelJob:

    @pytest.mark.asyncio
    async def test_scancel_called_with_correct_job_id(self):
        """cancel_job must invoke scancel with the job ID as a string argument."""
        jm = _jm()
        calls = []

        def fake_bounded_command(argv, *, timeout):
            calls.append(argv)
            return _command_result(argv[0])

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            await jm.cancel_job(42)

        assert calls[0][0] == "scancel"
        assert calls[0][1] == "42"

    @pytest.mark.asyncio
    async def test_returns_none(self):
        jm = _jm()

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            return_value=_command_result("scancel"),
        ):
            result = await jm.cancel_job(99)

        assert result is None

    @pytest.mark.asyncio
    async def test_subprocess_exception_swallowed(self):
        """Missing scancel remains a swallowed best-effort failure."""
        jm = _jm()

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=FileNotFoundError("scancel not found"),
        ):
            await jm.cancel_job(7)  # must not raise

    @pytest.mark.asyncio
    async def test_scancel_nonzero_exit_does_not_raise(self):
        """Non-zero exit from scancel (e.g. job already gone) must be swallowed."""
        jm = _jm()

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            return_value=_command_result(
                "scancel",
                stderr_text="Invalid job id",
                returncode=1,
            ),
        ):
            await jm.cancel_job(123)  # must not raise

    @pytest.mark.asyncio
    async def test_different_job_ids_passed_as_string(self):
        """Job IDs of various sizes are always forwarded as str(job_id)."""
        jm = _jm()
        for job_id in [1, 100, 999999]:
            calls = []

            def fake_bounded_command(argv, *, timeout):
                calls.append(argv)
                return _command_result(argv[0])

            with patch.object(
                slurm_manager_module,
                "_run_bounded_command",
                side_effect=fake_bounded_command,
            ):
                await jm.cancel_job(job_id)

            assert calls[0][1] == str(job_id)


# ---------------------------------------------------------------------------
# TestCancelAllOwnedJobs
# ---------------------------------------------------------------------------

class TestCancelAllOwnedJobs:

    @pytest.mark.asyncio
    async def test_cancels_vllm_jobs(self):
        """eval360- prefixed jobs with matching instance_id are cancelled."""
        jm = _jm()
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            if argv[0] == "squeue":
                return _command_result(
                    "squeue",
                    f"eval360-{_TEST_IID}-mymodel-abc123def456-r0|99",
                )
            scancel_calls.append(argv[1])
            return _command_result("scancel")

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            await jm.cancel_all_owned_jobs()

        assert "99" in scancel_calls

    @pytest.mark.asyncio
    async def test_cancels_imported_dataset_jobs(self):
        """eval360id- prefixed jobs with matching instance_id are cancelled."""
        jm = _jm()
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            if argv[0] == "squeue":
                return _command_result(
                    "squeue",
                    f"eval360id-{_TEST_IID}-runner-mymodel-abcd1234|77",
                )
            scancel_calls.append(argv[1])
            return _command_result("scancel")

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            await jm.cancel_all_owned_jobs()

        assert "77" in scancel_calls

    @pytest.mark.asyncio
    async def test_cancels_all_matching_jobs(self):
        """All eval360 jobs with matching instance_id are cancelled."""
        jm = _jm()
        scancel_calls = []
        squeue_output = "\n".join([
            f"eval360-{_TEST_IID}-model-abc123def456-r0|55",
            f"eval360id-{_TEST_IID}-runner-model-uuid1234|66",
            "some-other-job|99",  # must NOT be cancelled
        ])

        def fake_bounded_command(argv, *, timeout):
            if argv[0] == "squeue":
                return _command_result("squeue", squeue_output)
            scancel_calls.append(argv[1])
            return _command_result("scancel")

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            await jm.cancel_all_owned_jobs()

        assert set(scancel_calls) == {"55", "66"}
        assert "99" not in scancel_calls

    @pytest.mark.asyncio
    async def test_no_op_when_no_matching_jobs(self):
        """No scancel calls if squeue returns no eval360 jobs."""
        jm = _jm()
        scancel_calls = []

        def fake_bounded_command(argv, *, timeout):
            if argv[0] == "squeue":
                return _command_result("squeue")
            scancel_calls.append(argv[1])
            return _command_result("scancel")

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            await jm.cancel_all_owned_jobs()

        assert scancel_calls == []

    @pytest.mark.asyncio
    async def test_ignores_jobs_from_other_instance(self):
        """Jobs with a different instance_id are NOT cancelled."""
        jm = _jm()
        scancel_calls = []
        squeue_output = "\n".join([
            f"eval360-{_TEST_IID}-mymodel-abc123def456-r0|10",     # ours
            "eval360-aabbccdd-othermodel-abc123def456-r0|20",       # other instance
            f"eval360id-{_TEST_IID}-runner-mymodel-abcd1234|30",   # ours (imported)
            "eval360id-aabbccdd-runner-othermodel-abcd1234|40",     # other instance (imported)
        ])

        def fake_bounded_command(argv, *, timeout):
            if argv[0] == "squeue":
                return _command_result("squeue", squeue_output)
            scancel_calls.append(argv[1])
            return _command_result("scancel")

        with patch.object(
            slurm_manager_module,
            "_run_bounded_command",
            side_effect=fake_bounded_command,
        ):
            await jm.cancel_all_owned_jobs()

        assert set(scancel_calls) == {"10", "30"}

    @pytest.mark.asyncio
    async def test_get_all_jobs_ignores_other_instance(self):
        """get_all_jobs only returns jobs with matching instance_id."""
        jm = _jm()
        squeue_output = "\n".join([
            f"eval360-{_TEST_IID}-mymodel-abc123def456-r0|1|RUNNING|01:00:00|node01",
            "eval360-aabbccdd-othermodel-abc123def456-r0|2|RUNNING|01:00:00|node02",
        ])
        with patch.object(
            jm,
            "_query_squeue",
            return_value=_squeue_result(squeue_output),
        ):
            result = await jm.get_all_jobs()

        assert len(result) == 1
        assert f"eval360-{_TEST_IID}-mymodel-abc123def456-r0" in result


# ---------------------------------------------------------------------------
# TestInstanceId
# ---------------------------------------------------------------------------

class TestInstanceId:

    def test_auto_generated_when_not_provided(self):
        jm = SlurmManager()
        assert len(jm.instance_id) == 8
        assert all(c in "0123456789abcdef" for c in jm.instance_id)

    def test_uses_provided_instance_id(self):
        jm = SlurmManager(instance_id="abcd1234")
        assert jm.instance_id == "abcd1234"

    def test_two_instances_independent(self):
        jm1 = SlurmManager(instance_id="aaaaaaaa")
        jm2 = SlurmManager(instance_id="bbbbbbbb")
        sk = "abc123def456"
        name1 = jm1._to_job_name("model", sk, 0)
        name2 = jm2._to_job_name("model", sk, 0)
        assert name1 != name2
        assert jm1._from_job_name(name1) == (sk, 0)
        assert jm1._from_job_name(name2) is None  # wrong instance_id
        assert jm2._from_job_name(name2) == (sk, 0)
        assert jm2._from_job_name(name1) is None


# ---------------------------------------------------------------------------
# TestGetModelState
# ---------------------------------------------------------------------------

def _deployment_info(model_name, max_time_to_deploy=600):
    """Build a fake DeploymentInfo with a model attribute."""
    model = MagicMock()
    model.name = model_name
    model.max_time_to_deploy = max_time_to_deploy
    info = MagicMock()
    info.model = model
    return info


def _running_job_state(model_name, elapsed_time=10, nodelist="gpu-001"):
    model = _fake_model(name=model_name, path=f"org/{model_name}")
    job_name = _fake_job_name(model)
    return {
        job_name: {
            "name": job_name,
            "state": "RUNNING",
            "nodelist": nodelist,
            "elapsed_time": elapsed_time,
            "job_id": 1,
            "model_name": model_name,
            "serving_key": model.serving_key,
            "replica_index": 0,
        }
    }


def _pending_job_state(model_name):
    model = _fake_model(name=model_name, path=f"org/{model_name}")
    job_name = _fake_job_name(model)
    return {
        job_name: {
            "name": job_name,
            "state": "PENDING",
            "nodelist": "",
            "elapsed_time": 0,
            "job_id": 1,
            "model_name": model_name,
            "serving_key": model.serving_key,
            "replica_index": 0,
        }
    }


def _make_health_resp(status: int):
    resp = MagicMock()
    resp.status = status
    return resp


class TestGetModelState:

    @pytest.mark.asyncio
    async def test_empty_desired_models_returns_all_empty(self):
        jm = _jm()
        jm.get_all_jobs = AsyncMock(return_value={})

        pending, deploying, live, dead, replica_counts = await jm.get_model_state({})

        assert pending == []
        assert deploying == []
        assert live == []
        assert dead == []

    @pytest.mark.asyncio
    async def test_pending_model_goes_to_pending_list(self):
        jm = _jm()
        jm.get_all_jobs = AsyncMock(return_value=_pending_job_state("modelA"))
        desired = {"modelA": _deployment_info("modelA")}

        pending, deploying, live, dead, replica_counts = await jm.get_model_state(desired)

        assert pending == ["modelA"]
        assert deploying == live == dead == []

    @pytest.mark.asyncio
    async def test_running_healthy_model_goes_to_live(self):
        jm = _jm()
        jobs = _running_job_state("modelA")
        job_state = next(iter(jobs.values()))
        jm.get_all_jobs = AsyncMock(return_value=jobs)
        jm.check_live = AsyncMock(return_value=(_make_health_resp(200), job_state))
        desired = {"modelA": _deployment_info("modelA", max_time_to_deploy=600)}

        pending, deploying, live, dead, replica_counts = await jm.get_model_state(desired)

        assert live == [("modelA", "http://gpu-001:8000")]
        assert pending == deploying == dead == []

    @pytest.mark.asyncio
    async def test_running_unhealthy_within_deploy_time_goes_to_deploying(self):
        jm = _jm()
        jobs = _running_job_state("modelA", elapsed_time=10)
        job_state = next(iter(jobs.values()))
        jm.get_all_jobs = AsyncMock(return_value=jobs)
        jm.check_live = AsyncMock(return_value=(None, job_state))
        desired = {"modelA": _deployment_info("modelA", max_time_to_deploy=600)}

        pending, deploying, live, dead, replica_counts = await jm.get_model_state(desired)

        assert deploying == ["modelA"]
        assert pending == live == dead == []

    @pytest.mark.asyncio
    async def test_running_unhealthy_past_deploy_time_goes_to_dead(self):
        jm = _jm()
        jobs = _running_job_state("modelA", elapsed_time=700)
        job_state = next(iter(jobs.values()))
        jm.get_all_jobs = AsyncMock(return_value=jobs)
        jm.check_live = AsyncMock(return_value=(None, job_state))
        desired = {"modelA": _deployment_info("modelA", max_time_to_deploy=600)}

        pending, deploying, live, dead, replica_counts = await jm.get_model_state(desired)

        assert dead == ["modelA"]
        assert pending == deploying == live == []

    @pytest.mark.asyncio
    async def test_model_not_in_slurm_silently_omitted(self):
        """A desired model with no corresponding Slurm job must not appear in any list."""
        jm = _jm()
        jm.get_all_jobs = AsyncMock(return_value={})
        desired = {"modelA": _deployment_info("modelA")}

        pending, deploying, live, dead, replica_counts = await jm.get_model_state(desired)

        assert pending == deploying == live == dead == []

    @pytest.mark.asyncio
    async def test_unknown_slurm_state_raises(self):
        """A job in a state other than PENDING/RUNNING must raise RuntimeError."""
        jm = _jm()
        model = _fake_model(name="modelA", path="org/modelA")
        job_name = _fake_job_name(model)
        jm.get_all_jobs = AsyncMock(return_value={
            job_name: {"name": job_name, "state": "COMPLETING", "nodelist": "", "elapsed_time": 0,
                       "job_id": 1, "model_name": "modelA", "serving_key": model.serving_key, "replica_index": 0}
        })
        desired = {"modelA": _deployment_info("modelA")}

        with pytest.raises(RuntimeError, match="COMPLETING"):
            await jm.get_model_state(desired)

    @pytest.mark.asyncio
    async def test_running_health_non_200_within_time_is_deploying(self):
        """Health response with status 503 (not None, but not 200) + within time → deploying."""
        jm = _jm()
        jobs = _running_job_state("modelA", elapsed_time=10)
        job_state = next(iter(jobs.values()))
        jm.get_all_jobs = AsyncMock(return_value=jobs)
        jm.check_live = AsyncMock(return_value=(_make_health_resp(503), job_state))
        desired = {"modelA": _deployment_info("modelA", max_time_to_deploy=600)}

        pending, deploying, live, dead, replica_counts = await jm.get_model_state(desired)

        assert deploying == ["modelA"]

    @pytest.mark.asyncio
    async def test_multiple_models_classified_independently(self):
        """Three models: one PENDING, one live, one deploying — all correctly classified."""
        jm = _jm()
        job_dict = {}
        job_dict.update(_pending_job_state("m-pending"))
        job_dict.update(_running_job_state("m-live", elapsed_time=10))
        job_dict.update(_running_job_state("m-deploying", elapsed_time=10))

        jm.get_all_jobs = AsyncMock(return_value=job_dict)

        m_live_name = _fake_job_name(_fake_model(name='m-live', path='org/m-live'))

        def check_live_side_effect(job_state):
            if job_state["name"] == m_live_name:
                return _make_health_resp(200), job_state
            return None, job_state

        jm.check_live = AsyncMock(side_effect=check_live_side_effect)

        desired = {
            "m-pending": _deployment_info("m-pending"),
            "m-live": _deployment_info("m-live", max_time_to_deploy=600),
            "m-deploying": _deployment_info("m-deploying", max_time_to_deploy=600),
        }

        pending, deploying, live, dead, replica_counts = await jm.get_model_state(desired)

        assert pending == ["m-pending"]
        assert live == [("m-live", "http://gpu-001:8000")]
        assert deploying == ["m-deploying"]
        assert dead == []
        assert replica_counts == {"m-pending": 1, "m-live": 1, "m-deploying": 1}

    @pytest.mark.asyncio
    async def test_replica_counts_includes_all_states(self):
        """replica_counts must include PENDING + RUNNING replicas regardless of health."""
        jm = _jm()
        # modelA: 1 PENDING + 2 RUNNING (1 live, 1 deploying) = 3 total
        job_dict = {}
        job_dict.update(_pending_job_state("modelA"))
        live_jobs = _running_job_state("modelA", elapsed_time=10)
        deploying_jobs = _running_job_state("modelA", elapsed_time=10)
        # Give unique job names so they don't collide
        live_key = f"eval360-{_TEST_IID}-modela-{_test_sk('modelA')}-r1"
        deploying_key = f"eval360-{_TEST_IID}-modela-{_test_sk('modelA')}-r2"
        live_job = next(iter(live_jobs.values()))
        live_job["name"] = live_key
        live_job["replica_index"] = 1
        deploying_job = next(iter(deploying_jobs.values()))
        deploying_job["name"] = deploying_key
        deploying_job["replica_index"] = 2
        job_dict[live_key] = live_job
        job_dict[deploying_key] = deploying_job

        def check_live_side_effect(job_state):
            if job_state["name"] == live_key:
                return _make_health_resp(200), job_state
            return None, job_state

        jm.get_all_jobs = AsyncMock(return_value=job_dict)
        jm.check_live = AsyncMock(side_effect=check_live_side_effect)
        desired = {"modelA": _deployment_info("modelA", max_time_to_deploy=600)}

        pending, deploying, live, dead, replica_counts = await jm.get_model_state(desired)

        assert replica_counts == {"modelA": 3}
        # live replica still present; deploying one filtered from deploying_final since model is live
        assert live == [("modelA", "http://gpu-001:8000")]
        assert pending == []
        assert deploying == []
