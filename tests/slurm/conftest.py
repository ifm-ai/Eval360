"""Fixtures for the cluster suite.

What this provides:
    `drained_cluster` — autouse for EVERY test in the package. It drains the
    Slurm queue and reaps orphaned stub servers at both setup and teardown.
    This is the ONLY copy of that logic; three modules used to carry their own
    and no longer do, so comments elsewhere pointing here for it are pointing
    at the right place. The primitives it drives live in `cluster_harness` and
    all fail closed — a `squeue` that could not be asked raises rather than
    reading as an empty queue, because this fixture SIGKILLs on the strength of
    the queue being empty. `test_cluster_teardown_guard.py` proves it.

    `cluster_evidence` — module-scoped, and used only by
    test_slurm_manager_cluster.py. It submits ONE model-serving job through the
    real `SlurmManager` against a real slurmd, drives it to live and then to a
    terminal accounting state, and records what each stage produced. Tests then
    assert on that record.

Why it is shaped this way:
    Submitting is slow (seconds), so a job per test would add minutes to a
    suite that asserts on one lifecycle over and over. Collecting once and
    asserting many times keeps it fast without weakening any individual
    assertion. Capacity is not the constraint — the cluster advertises gpu:4,
    and other modules deliberately run three replicas at once.

    Each stage stores its RESULT OR ITS EXCEPTION rather than raising during
    collection. A raise in a module-scoped fixture errors every test in the
    module with the same traceback, which hides which stage actually broke;
    storing per stage means the squeue test fails for squeue reasons and the
    sacct test fails for sacct reasons.

    The fixture is deliberately synchronous and wraps one `asyncio.run()`.
    `SlurmManager._deployment_lock` is an `asyncio.Lock` that binds to the first
    loop that uses it, so spreading these calls across several `asyncio.run()`
    invocations would bind the lock to a dead loop and fail on the second
    `update_allocation`. One loop for the whole interaction avoids that without
    depending on a particular pytest-asyncio fixture-loop-scope setting.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from scheduler.event import DeploymentInfo
from scheduler.model import ModelInstance, ModelType, ServingSlurmResources
from scheduler.slurm_manager import SlurmManager

from .cluster_harness import (
    cancel_all_user_jobs,
    drain_queue,
    queued_job_ids,
    reap_orphaned_stubs,
)

MODEL_NAME = "ci-stub-model"

# Generous because it covers Slurm scheduling plus the stub server's startup,
# and a CI runner is slower than a laptop. Not a target: the suite normally
# finishes each stage in single-digit seconds.
DEPLOY_TIMEOUT_SECONDS = int(os.environ.get("DEPLOY_TIMEOUT_SECONDS", "180"))
TERMINAL_TIMEOUT_SECONDS = int(os.environ.get("TERMINAL_TIMEOUT_SECONDS", "120"))


@dataclass
class ClusterEvidence:
    """What each stage produced, or the exception that stopped it."""

    stages: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, BaseException] = field(default_factory=dict)
    instance_id: str = ""
    serving_key: str = ""
    resource_args: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record(self, stage: str, value: Any) -> None:
        self.stages[stage] = value

    def fail(self, stage: str, error: BaseException) -> None:
        self.failures[stage] = error

    def get(self, stage: str) -> Any:
        """Return a stage's result, failing THIS test if that stage broke."""
        if stage in self.failures:
            pytest.fail(
                f"stage {stage!r} did not complete: "
                f"{type(self.failures[stage]).__name__}: {self.failures[stage]}"
            )
        if stage not in self.stages:
            pytest.fail(
                f"stage {stage!r} never ran — an earlier stage must have "
                f"stopped the sequence. Completed: {sorted(self.stages)}; "
                f"failed: {sorted(self.failures)}"
            )
        return self.stages[stage]


def _build_model() -> ModelInstance:
    """A model whose resource request the CI node can actually satisfy.

    `_serving_resource_args` always emits `--gres=gpu:N` and `gpus_per_node` is
    a StrictPositiveInt, so a serving job can never request zero GPUs. The CI
    cluster therefore has to advertise at least one — see `scripts/ci/
    slurm_up.sh`, which creates fake GPU device files because a file-less GPU
    GRES is silently dropped by slurmd.
    """
    return ModelInstance(
        name=MODEL_NAME,
        path=os.environ.get("CI_MODEL_PATH", "/tmp/model"),
        venv_path=os.environ.get("CI_SERVING_VENV", "/tmp/serving-venv"),
        max_simultaneous_requests=4,
        max_time_to_deploy=DEPLOY_TIMEOUT_SECONDS,
        vllm_cli_args=["--max-model-len 256"],
        openai_kwargs={"temperature": 0.0},
        parser_type="noop",
        model_type=ModelType.CHAT,
        owner="eval360-ci",
        output_path=os.environ.get("CI_OUTPUT_PATH", "/tmp/output"),
        serving_slurm_resources=ServingSlurmResources(
            gpus_per_node=1,
            cpus_per_task=1,
            memory_gb=1,
            time_limit="0:20:00",
        ),
    )


async def _collect(evidence: ClusterEvidence) -> None:
    model = _build_model()
    manager = SlurmManager(
        log_dir=os.environ.get("CI_SLURM_LOG_DIR", "."),
        partition="ci",
        poll_interval=2.0,
    )
    # Without this, `update_allocation` does not even pass `--parsable` to
    # sbatch and never calls `_parse_sbatch_job_id` — both are guarded at the
    # call site on `_terminal_evidence_enabled`, not merely inside
    # `_remember_job` — so nothing is ledgered. Production enables capture the
    # same way before submitting children it intends to account for.
    manager.begin_terminal_result_capture()

    evidence.instance_id = manager.instance_id
    evidence.serving_key = model.serving_key
    evidence.resource_args = SlurmManager._serving_resource_args(model)

    deployment = {
        MODEL_NAME: DeploymentInfo(
            model=model, priority=0, generation_events=set(), grader_events=set()
        )
    }

    # ---- sinfo ----------------------------------------------------------
    try:
        evidence.record("sinfo", await manager.get_available_nodes())
    except Exception as error:  # noqa: BLE001
        evidence.fail("sinfo", error)

    # ---- sbatch ---------------------------------------------------------
    try:
        await manager.update_allocation([(model, 1)], [])
        evidence.record("sbatch", manager.get_submitted_jobs())
    except Exception as error:  # noqa: BLE001
        evidence.fail("sbatch", error)
        # Nothing downstream can mean anything without a submitted job.
        return

    # ---- squeue ---------------------------------------------------------
    try:
        deadline = time.monotonic() + DEPLOY_TIMEOUT_SECONDS
        record = None
        while time.monotonic() < deadline:
            jobs = await manager.get_all_jobs()
            if jobs:
                candidate = next(iter(jobs.values()))
                if candidate["state"] == "RUNNING":
                    record = candidate
                    break
            await asyncio.sleep(2)
        if record is None:
            raise TimeoutError(
                f"no RUNNING job within {DEPLOY_TIMEOUT_SECONDS}s. A job stuck "
                "PENDING usually means the node cannot satisfy the request — "
                "check `scontrol show job` and the GRES configuration."
            )
        evidence.record("squeue", record)
    except Exception as error:  # noqa: BLE001
        evidence.fail("squeue", error)
        await manager.cancel_all_owned_jobs()
        return

    # ---- health ---------------------------------------------------------
    try:
        deadline = time.monotonic() + DEPLOY_TIMEOUT_SECONDS
        live: list = []
        while time.monotonic() < deadline:
            pending, deploying, live, dead, replicas = await manager.get_model_state(
                deployment
            )
            if live or dead:
                break
            await asyncio.sleep(3)
        evidence.record("health", live)
    except Exception as error:  # noqa: BLE001
        evidence.fail("health", error)

    job_id = evidence.stages["squeue"]["job_id"]

    # ---- sacct, job still active ---------------------------------------
    try:
        outcomes = await manager.get_job_accounting([job_id])
        evidence.record("sacct_active", outcomes)
        active = outcomes.get(job_id)
        # slurmdbd populates the root row in two steps. For a few seconds after
        # submission it carries a placeholder (JobName='allocation',
        # State='PENDING') while squeue already reports RUNNING. Recorded as a
        # note, never asserted: nothing in production reads the name before the
        # job is terminal. See
        # test_terminal_row_converges_on_the_submitted_job_name.
        if active is not None and active.job_name != evidence.stages["squeue"]["name"]:
            evidence.notes.append(
                f"sacct placeholder row observed: job_name={active.job_name!r} "
                f"state={active.raw_state!r} while squeue reported RUNNING"
            )
    except Exception as error:  # noqa: BLE001
        evidence.fail("sacct_active", error)

    # ---- scancel + terminal accounting ----------------------------------
    try:
        submitted_name = evidence.stages["squeue"]["name"]
        await manager.cancel_all_owned_jobs()
        outcomes = await asyncio.wait_for(
            manager.wait_for_terminal_job_outcomes([job_id]),
            timeout=TERMINAL_TIMEOUT_SECONDS,
        )
        outcome = outcomes[job_id]

        # KNOWN HAZARD, and the reason this extra wait exists.
        #
        # `wait_for_terminal_job_outcomes` returns as soon as `is_terminal` is
        # true, but slurmdbd populates a row's STATE and its NAME independently.
        # For up to ~1.5s after a scancel, sacct serves a row that is terminal
        # AND still carries the placeholder name:
        #
        #   <id>|allocation|CANCELLED by 0
        #
        # Reproduced 3/3 on this cluster with a submit-then-immediately-cancel
        # loop. Production does NOT wait this out: `Scheduler` compares
        # `outcome.job_name` to the submitted name right after the terminal wait
        # and raises "sacct job name ... does not match the submitted child
        # identity", so a short-lived child can fail a terminal result for a
        # reason that has nothing to do with the job.
        #
        # This wait is HERE and not in the assertion so that the test stays
        # deterministic while still asserting the real contract — the name does
        # converge — rather than being weakened to accept 'allocation'. The
        # observation is recorded as a note so it shows up in every CI run
        # instead of being silently smoothed over.
        if outcome.job_name != submitted_name:
            evidence.notes.append(
                "TERMINAL PLACEHOLDER OBSERVED: sacct reported "
                f"state={outcome.raw_state!r} with job_name={outcome.job_name!r} "
                f"instead of {submitted_name!r}. Scheduler would have raised "
                "'sacct job name does not match the submitted child identity'."
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and outcome.job_name != submitted_name:
                await asyncio.sleep(0.5)
                outcome = (await manager.get_job_accounting([job_id]))[job_id]

        evidence.record("sacct_terminal", outcome)
        evidence.record("ledger", {j.job_id: j for j in manager.get_submitted_jobs()})
    except Exception as error:  # noqa: BLE001
        evidence.fail("sacct_terminal", error)
    finally:
        # Belt and braces: a failure above must not leave a job holding
        # cluster GPUs for the next run.
        try:
            await manager.cancel_all_owned_jobs()
        except Exception:  # noqa: BLE001, S110
            pass


@pytest.fixture(scope="module")
def cluster_evidence() -> ClusterEvidence:
    """Run one full job lifecycle against the cluster and return what happened."""
    evidence = ClusterEvidence()
    asyncio.run(_collect(evidence))
    for note in evidence.notes:
        print(f"NOTE: {note}")
    return evidence


@pytest.fixture(autouse=True)
def drained_cluster():
    """Every test starts and ends with a CONFIRMED empty queue.

    WHY THIS IS AUTOUSE AND UNCONDITIONAL. These tests share one small cluster,
    and several of them deliberately fill it — a blocker job holding every GPU
    to force a PENDING state, replicas that sit RUNNING for a while. `scancel`
    returns as soon as the controller accepts the request, not when the job is
    gone, so a test that tidies up with a bare scancel can still be occupying
    the cluster when the next test starts. That next test then sits PENDING and
    fails on a timeout that has nothing to do with what it was testing.

    That is exactly what happened: the whole suite was green module by module
    and red when run together, because `test_slurm_failure_modes` sorts before
    `test_slurm_isolation` and its blocker had not been reaped yet. Longer
    timeouts would have hidden it rather than fixed it; ordering-dependent
    tests are worth eliminating, not tuning.

    Cancelling EVERY job the user owns — which TEARDOWN does, and setup
    deliberately does not — is safe here in a way it would never be on a real
    cluster: this container holds nothing but the test cluster, and it is
    thrown away afterwards. Setup only waits and reaps, so a queue that will
    not drain fails loudly rather than being silently cleared.

    EVERY STEP BELOW FAILS CLOSED, and this fixture is why that matters. It
    reaps by SIGKILL on the strength of one claim — the queue is empty — so a
    `squeue` that FAILED must not be readable as "nothing is queued". The
    helpers it calls raise `ClusterControlError` rather than guessing, and
    nothing here catches it: an unanswerable cluster errors the test that asked,
    which is a diagnosis. The alternative, silently reaping anyway, kills a live
    job's server and fails some unrelated test minutes later with no clue why.
    """
    # SETUP: inherit a clean cluster, not whatever the last test left.
    if shutil.which("squeue"):
        drain_queue()
        reap_orphaned_stubs()
    yield
    # TEARDOWN: leave a clean cluster for the next test.
    if not shutil.which("squeue"):
        return
    # `queued_job_ids` raises if it cannot tell, so the scancel is skipped only
    # when the queue is KNOWN to be empty, never because the question went
    # unanswered.
    if queued_job_ids():
        cancel_all_user_jobs()
    drain_queue()
    # Reaping re-confirms the drain itself; see `reap_orphaned_stubs`.
    reap_orphaned_stubs()


def pytest_collection_modifyitems(config, items):
    """Skip the cluster suite when there is no cluster to talk to.

    A skip here is safe ONLY because the CI job that owns this suite enforces an
    execution floor (`scripts/ci/assert_pytest_executed.py`): pytest exits 0
    when everything skips, so without that floor a missing cluster would report
    green having tested nothing. Locally, skipping is the right behaviour —
    `pytest tests/` should not fail because Docker is not running.
    """
    if shutil.which("sinfo") and shutil.which("sbatch"):
        return
    skip = pytest.mark.skip(
        reason="no Slurm on PATH — run inside the test cluster: "
               "scripts/slurm_cluster.sh up && scripts/slurm_cluster.sh test tests/slurm"
    )
    for item in items:
        item.add_marker(skip)
