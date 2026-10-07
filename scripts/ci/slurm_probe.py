#!/usr/bin/env python3
"""Drive the REAL SlurmManager against the cluster and report what happened.

NOTHING HERE IS PATCHED. `tests/test_job_manager.py` patches
`asyncio.create_subprocess_exec` and hands the parsers strings written by the
same person who wrote the parsers; `tests/fake_slurm.py` replaces the class
outright. This module imports `scheduler.slurm_manager.SlurmManager` and lets it
shell out to the binaries installed by `scripts/ci/slurm_up.sh`.

Every check below is a parser meeting output it did not author:

  sbatch   `_parse_sbatch_job_id` on real `sbatch --parsable` stdout
  squeue   `get_all_jobs` on real `--format %j|%i|%T|%M|%N`, including
           `_to_seconds` on a real elapsed field and `_JOB_NAME_RE` on a job
           name that made the round trip through Slurm
  sinfo    `get_available_nodes` on real `sinfo -h -p ci -o %T|%D`
  sacct    `get_job_accounting` on real `--parsable2 --format=JobIDRaw,
           JobName%256,State,ExitCode,Reason` — the newest of these and the one
           with no cluster coverage at all
  health   `get_model_state` resolving http://{nodelist}:8000/health against a
           node name Slurm chose

Exit status is the verdict; every stage also prints what it saw. This is the
by-hand counterpart to `tests/slurm/`: the automated suite asserts, this prints
the raw Slurm output next to what the parsers made of it, which is what you want
when something on the real cluster disagrees with CI.

CLEANUP IS UNCONDITIONAL ONCE ANYTHING HAS BEEN SUBMITTED. Everything after the
sbatch stage runs inside a `try/finally` whose `finally` cancels this probe's
jobs and then WAITS for them to leave the queue. That structure is the point:
the stages below give up early on their own terms — a job that never reaches
RUNNING, accounting that never answers — and every one of those early exits used
to step straight past the cancel. The job left behind is real, it holds a GPU,
and on this cluster it holds it for the full 20-minute time limit, so the next
probe or test run sits PENDING and fails for a reason that has nothing to do
with it. `report()` is called after the cleanup rather than inside it, so a
cleanup that could not confirm the drain shows up in the exit status instead of
being announced to nobody.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scheduler.event import DeploymentInfo  # noqa: E402
from scheduler.model import ModelInstance, ModelType, ServingSlurmResources  # noqa: E402
from scheduler.slurm_manager import SlurmManager  # noqa: E402

MODEL_NAME = "ci-stub-model"
DEPLOY_TIMEOUT_SECONDS = int(os.environ.get("DEPLOY_TIMEOUT_SECONDS", "180"))

# How long the cleanup will wait for a cancelled job to actually leave the
# queue. Comfortably longer than the seconds a scancel takes on this cluster,
# and far shorter than the 20-minute time limit a leaked job would otherwise
# hold a GPU for — the window this bound exists to close.
CLEANUP_DRAIN_TIMEOUT_SECONDS = float(
    os.environ.get("CLEANUP_DRAIN_TIMEOUT_SECONDS", "90")
)

failures: list[str] = []


def stage(title: str) -> None:
    print(f"\n\033[1;36m=== {title}\033[0m", flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "\033[1;32mPASS\033[0m" if ok else "\033[1;31mFAIL\033[0m"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(label)
    return ok


def build_model(venv_path: str, model_path: str, output_path: str) -> ModelInstance:
    """A model whose resource request the CI node can actually satisfy.

    `_serving_resource_args` always emits `--gres=gpu:N` and N cannot be zero,
    so the ONLY way this schedules is if the fictional GRES from
    `slurm_up.sh` is real enough for the scheduler — which is why those devices
    have to be real files rather than a bare count.
    """
    return ModelInstance(
        name=MODEL_NAME,
        path=model_path,
        venv_path=venv_path,
        max_simultaneous_requests=4,
        max_time_to_deploy=DEPLOY_TIMEOUT_SECONDS,
        vllm_cli_args=["--max-model-len 256"],
        openai_kwargs={"temperature": 0.0},
        parser_type="noop",
        model_type=ModelType.CHAT,
        owner="eval360-ci",
        output_path=output_path,
        serving_slurm_resources=ServingSlurmResources(
            gpus_per_node=1,
            cpus_per_task=1,
            memory_gb=1,
            time_limit="0:20:00",
        ),
    )


async def main() -> int:
    venv_path = os.environ["CI_SERVING_VENV"]
    model_path = os.environ["CI_MODEL_PATH"]
    output_path = os.environ["CI_OUTPUT_PATH"]
    log_dir = os.environ.get("CI_SLURM_LOG_DIR", ".")

    model = build_model(venv_path, model_path, output_path)
    manager = SlurmManager(log_dir=log_dir, partition="ci", poll_interval=2.0)
    # Without this, `_remember_job` returns early (`if not
    # self._terminal_evidence_enabled: return`) and nothing is ledgered, so
    # `_parse_sbatch_job_id` never gets exercised and `get_submitted_jobs()` is
    # empty. Production enables capture the same way.
    manager.begin_terminal_result_capture()
    print(f"instance_id = {manager.instance_id}")
    print(f"serving_key = {model.serving_key}")
    print(f"resource args = {SlurmManager._serving_resource_args(model)}")

    deployment = {
        MODEL_NAME: DeploymentInfo(
            model=model, priority=0, generation_events=set(), grader_events=set()
        )
    }

    # ---------------------------------------------------------------- sinfo
    stage("sinfo — get_available_nodes() against a real partition")
    try:
        idle_nodes = await manager.get_available_nodes()
        check("sinfo parsed", isinstance(idle_nodes, int), f"idle nodes = {idle_nodes}")
        check("at least one idle node", idle_nodes >= 1, f"got {idle_nodes}")
    except Exception as error:  # noqa: BLE001
        check("sinfo parsed", False, repr(error))

    # ---------------------------------------------------------------- sbatch
    stage("sbatch — update_allocation() submits the real sbatch_script.sh")
    try:
        await manager.update_allocation([(model, 1)], [])
    except Exception as error:  # noqa: BLE001
        check("sbatch accepted the submission", False, repr(error))
        # Nothing was submitted, so there is nothing to clean up. This is the
        # ONLY early return past this point for that reason.
        return report()
    check("sbatch accepted the submission", True)

    # A REAL JOB NOW EXISTS. Every path out of here goes through the cleanup.
    try:
        await probe_submitted_job(manager, model, deployment, log_dir)
    except Exception as error:  # noqa: BLE001
        # Caught rather than propagated so that `report()` still runs and the
        # verdict lists this alongside any check that failed first. The
        # traceback is printed because an unexpected error here is a bug in the
        # probe, and `repr()` alone would not locate it.
        check("probe ran to completion without an unexpected error", False, repr(error))
        traceback.print_exc()
    finally:
        await cancel_and_confirm_drained(manager)

    return report()


async def probe_submitted_job(
    manager: SlurmManager,
    model: ModelInstance,
    deployment: dict,
    log_dir: str,
) -> None:
    """Everything that can only be checked once a job has been submitted.

    Split out of `main` purely so the submitted job has an enclosing
    `try/finally` — see the module docstring. Note that it returns rather than
    calling `report()` on its early exits: the verdict is printed once, by
    `main`, AFTER the cleanup has had its say.
    """
    submitted = manager.get_submitted_jobs()
    check(
        "_parse_sbatch_job_id extracted an ID from real --parsable stdout",
        len(submitted) == 1 and submitted[0].job_id > 0,
        f"submitted = {submitted}",
    )

    # ---------------------------------------------------------------- squeue
    #
    # The fictional GRES is decided here. A job requesting a GPU that does not exist sits
    # PENDING forever, so "did it ever reach RUNNING" is the whole question. The
    # PENDING reason is printed on timeout because it names the cause exactly
    # ("Requested node configuration is not available" vs anything else).
    stage("squeue — get_all_jobs() until the job runs")
    deadline = time.monotonic() + DEPLOY_TIMEOUT_SECONDS
    jobs: dict = {}
    running = False
    while time.monotonic() < deadline:
        jobs = await manager.get_all_jobs()
        if jobs:
            state = next(iter(jobs.values()))["state"]
            if state == "RUNNING":
                running = True
                break
        await asyncio.sleep(2)

    check("get_all_jobs() found our job", bool(jobs), f"jobs = {jobs}")
    if not check("job reached RUNNING (the fictional GRES schedules)", running):
        os.system("squeue -o '%.10i %.20j %.8T %.20R' || true")
        os.system("scontrol show job || true")
        # THE leak this structure exists for. A job stuck PENDING is still a
        # submitted job: it keeps its resource reservation and, once the
        # scheduler can satisfy it, runs for its full time limit. Returning here
        # hands it to the cleanup rather than abandoning it.
        return

    job = next(iter(jobs.values()))
    print(f"  parsed job record: {job}")
    check("job name round-tripped through _JOB_NAME_RE", job["serving_key"] == model.serving_key,
          f"serving_key = {job['serving_key']}")
    check("model name resolved from the registry", job["model_name"] == MODEL_NAME)
    check("_to_seconds parsed a real elapsed field", isinstance(job["elapsed_time"], int),
          f"elapsed = {job['elapsed_time']}s")
    check("nodelist is a real node", bool(job["nodelist"]) and not job["nodelist"].startswith("("),
          f"nodelist = {job['nodelist']}")

    # ---------------------------------------------------------------- health
    #
    # The end of the chain the stub exists to make reachable: sbatch_script.sh
    # activated the venv, decoded $vllm_args, exec'd the fake vllm, and the
    # scheduler found it by the node name Slurm assigned.
    stage("health — get_model_state() reaches http://{nodelist}:8000/health")
    live: list = []
    deadline = time.monotonic() + DEPLOY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        pending, deploying, live, dead, replica_counts = await manager.get_model_state(deployment)
        print(f"  pending={pending} deploying={deploying} live={live} dead={dead} "
              f"replicas={replica_counts}")
        if live or dead:
            break
        await asyncio.sleep(5)

    if not check("model became live", bool(live), f"live = {live}"):
        print("  --- slurm job output ---")
        os.system(f"cat {log_dir}/slurm-*.out 2>/dev/null || true")

    # ---------------------------------------------------------------- sacct
    #
    # Accounting. This runs whether or not the model went live: a row exists
    # for a submitted job regardless, and whether sacct answers at all is the
    # thing being measured.
    stage("sacct — get_job_accounting() on a live job")
    job_id = job["job_id"]
    try:
        outcomes = await manager.get_job_accounting([job_id])
        if check("sacct returned a root row (accounting works)", job_id in outcomes,
                 f"outcomes = {outcomes}"):
            outcome = outcomes[job_id]
            print(f"  parsed outcome: {outcome}")
            check("state recognised by _normalize_sacct_state",
                  outcome.state in ("RUNNING", "PENDING"),
                  f"state = {outcome.state!r} (raw {outcome.raw_state!r})")
            # NOT a pass/fail. slurmdbd populates the root row in two steps, and
            # for a few seconds after submission it carries a PLACEHOLDER:
            # JobName='allocation', State='PENDING', while squeue already reports
            # RUNNING. Measured on this cluster: no row at all for ~1s, then
            # 'allocation' for ~3s, then the real name.
            #
            # This is reported rather than asserted because the name is NOT
            # required to be correct at this point. Production reads it only in
            # `Scheduler._reconcile...`, after
            # `wait_for_terminal_job_outcomes()` has looped until every job is
            # terminal — and a terminal row always carries the real name. An
            # assertion here would be testing a contract nothing relies on.
            if outcome.job_name != job["name"]:
                print(f"  NOTE: sacct is still showing a placeholder row — "
                      f"job_name={outcome.job_name!r} state={outcome.raw_state!r} "
                      f"(squeue says {job['state']}). Expected transient.")
    except Exception as error:  # noqa: BLE001
        check("sacct returned a root row (accounting works)", False, repr(error))
        os.system(f"sacct -j {job_id} --parsable2 "
                  "--format=JobIDRaw,JobName%256,State,ExitCode,Reason || true")

    # ---------------------------------------------------------- scancel/sacct
    stage("scancel + sacct — the job reaches a terminal accounting state")
    try:
        await manager.cancel_all_owned_jobs()
        check("cancel_all_owned_jobs() ran", True)
    except Exception as error:  # noqa: BLE001
        check("cancel_all_owned_jobs() ran", False, repr(error))

    try:
        final = await asyncio.wait_for(
            manager.wait_for_terminal_job_outcomes([job_id]), timeout=120
        )
        outcome = final[job_id]
        print(f"  terminal outcome: {outcome}")
        check("wait_for_terminal_job_outcomes() resolved", outcome.is_terminal,
              f"state = {outcome.state}")
        check("cancelled job reports CANCELLED", outcome.state == "CANCELLED",
              f"state = {outcome.state}")
        # THIS is where the name matters, and the only place production reads
        # it: `Scheduler` raises "sacct job name ... does not match the
        # submitted child identity" on a mismatch here, after the terminal wait
        # above. A terminal row must carry the real submitted name.
        check("terminal sacct row carries the submitted job name",
              outcome.job_name == job["name"],
              f"sacct={outcome.job_name!r} submitted={job['name']!r}")
        # The ledgered identity and the accounting identity must agree — this is
        # the pairing `Scheduler` relies on to attribute outcomes to children.
        ledgered = {j.job_id: j for j in manager.get_submitted_jobs()}
        check("ledgered job ID matches the sacct root row",
              job_id in ledgered and ledgered[job_id].job_name == outcome.job_name,
              f"ledgered = {ledgered.get(job_id)}")
    except Exception as error:  # noqa: BLE001
        check("wait_for_terminal_job_outcomes() resolved", False, repr(error))
        os.system(f"sacct -j {job_id} --parsable2 "
                  "--format=JobIDRaw,JobName%256,State,ExitCode,Reason || true")


async def cancel_and_confirm_drained(
    manager: SlurmManager, *, timeout: float = CLEANUP_DRAIN_TIMEOUT_SECONDS
) -> None:
    """Cancel this probe's jobs and WAIT until Slurm agrees they are gone.

    Cancelling is not finishing. `scancel` returns as soon as the controller
    accepts the request, so a cleanup that stops there reports success while the
    job is still winding down and still holding its GPU — which is the same
    false all-clear as not cleaning up at all, just shorter. Hence the drain,
    and hence the bound on it: a job that will not leave the queue is a finding,
    not something to wait on forever.

    Both outcomes go through `check()` rather than `print()`, so "the probe
    leaked a job" reaches the exit status. A probe that passes every parser
    check and strands a GPU has not passed.

    Runs on every path out of the submitted-job stages, including the successful
    one where `cancel_all_owned_jobs` has already been called — re-cancelling
    nothing is free, and making the cleanup conditional on which stage we left
    from is exactly the reasoning that leaked jobs in the first place.
    """
    stage("cleanup — cancel this probe's jobs and confirm the queue drained")
    try:
        await manager.cancel_all_owned_jobs()
        check("cleanup cancelled the probe's jobs", True)
    except Exception as error:  # noqa: BLE001
        # Report it, then still try to drain: the cancel may have reached the
        # controller before whatever went wrong, and the queue is the authority
        # on whether anything is left.
        check("cleanup cancelled the probe's jobs", False, repr(error))

    deadline = time.monotonic() + timeout
    while True:
        try:
            remaining = await manager.get_all_jobs()
        except Exception as error:  # noqa: BLE001
            check("cleanup confirmed the queue drained", False, repr(error))
            return
        if not remaining:
            check("cleanup confirmed the queue drained", True)
            return
        if time.monotonic() >= deadline:
            check("cleanup confirmed the queue drained", False,
                  f"still queued after {timeout}s: {sorted(remaining)} — these "
                  "hold GPUs until their time limit expires")
            os.system("squeue -o '%.10i %.20j %.8T %.20R' || true")
            return
        await asyncio.sleep(1)


def report() -> int:
    stage("Verdict")
    if failures:
        print(f"\033[1;31m{len(failures)} check(s) failed:\033[0m")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("\033[1;32mAll checks passed — the real SlurmManager drove a real "
          "Slurm cluster end to end.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
