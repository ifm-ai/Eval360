import asyncio
import base64
import copy
import json
import logging
import re
import subprocess
import sys
import time
import uuid as _uuid
from dataclasses import dataclass, replace
from pathlib import Path

import aiohttp

logger = logging.getLogger("SlurmManager")
_SCANCEL_TIMEOUT_SECONDS = 2.0
_SACCT_QUERY_TIMEOUT_SECONDS = 5.0
_SQUEUE_QUERY_TIMEOUT_SECONDS = 5.0
_SQUEUE_QUERY_MAX_ATTEMPTS = 3
# slurmdbd publishes an accounting row in stages: nothing for ~1s, then a
# placeholder row named `allocation`, then the real name. Both windows are
# short, so these graces are generous ceilings that turn a genuinely absent row
# into a loud failure rather than an unbounded wait.
_SACCT_PUBLISH_GRACE_SECONDS = 30.0
_SACCT_SETTLE_GRACE_SECONDS = 30.0
# The lag is measured in hundreds of milliseconds, so polling it at the
# scheduler's ordinary interval (seconds) would spend most of the grace asleep.
_SACCT_LAG_POLL_SECONDS = 0.5
_NONTERMINAL_SLURM_STATES = frozenset(
    {
        "COMPLETING",
        "CONFIGURING",
        "PENDING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "RESV_DEL_HOLD",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "SUSPENDED",
    }
)
_TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


class SlurmJobVanished(RuntimeError):
    """A job left the queue before it was ever allocated a node.

    WHY THIS IS ITS OWN TYPE. This is an ORDINARY outcome, not a fault: a job
    can be preempted, cancelled, or run and end between two polls. `get_job_node`
    has to stop waiting for it — that is the hang fixed in this branch — but the
    caller has to be able to tell "the job is gone" apart from "Slurm, or this
    code, is broken". A bare `RuntimeError` cannot be caught precisely, so the
    only way to handle the expected case would be `except RuntimeError`, which
    would also swallow a genuine defect and turn it into a quietly failed event.

    Subclasses `RuntimeError` deliberately: every existing caller and test that
    catches or asserts `RuntimeError` keeps working unchanged.
    """


def _run_bounded_command(
    argv: tuple[str, ...],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Run a short command with synchronous timeout and child reaping."""
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


@dataclass(frozen=True)
class SubmittedSlurmJob:
    """One Slurm child submitted or adopted by this scheduler instance."""

    job_id: int
    job_name: str
    kind: str
    submission_origin: str
    model_name: str | None
    serving_key: str | None
    event_uuid: str | None = None
    runner_name: str | None = None
    cancellation_intent: str | None = None
    completed_event_roles: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SlurmJobOutcome:
    """Authoritative root-job outcome returned by ``sacct``."""

    job_id: int
    job_id_raw: str
    job_name: str
    raw_state: str
    state: str
    exit_code: int
    signal: int
    reason: str

    @property
    def is_terminal(self) -> bool:
        """Return whether Slurm reports a terminal scheduler state."""
        return self.state in _TERMINAL_SLURM_STATES

    @property
    def is_clean_completion(self) -> bool:
        """Return whether the batch job itself completed with exit code 0:0."""
        return (
            self.state == "COMPLETED"
            and self.exit_code == 0
            and self.signal == 0
        )


class SlurmManager:
    """Slurm subprocess interface. Handles all squeue/sbatch/scancel calls.

    This class is intentionally a thin wrapper around Slurm CLI tools with no
    business logic. Future backends (K8s, etc.) would implement the same
    interface.

    Each SlurmManager instance has a unique ``instance_id`` (8-char hex)
    embedded in every job name it creates.  Only jobs whose name contains
    this instance_id are considered "ours", allowing multiple schedulers
    to share the same Slurm account without interference.
    """

    # Job name format: eval360-{instance_id}-{model}-{serving_key}-r{replica}
    _JOB_NAME_RE = re.compile(
        r"^eval360-([0-9a-f]{8})-.+-([0-9a-f]{12})-r(\d+)$"
    )
    # Imported dataset job name: eval360id-{instance_id}-{runner}-{model}-{event_uuid}
    _IMPORTED_JOB_PREFIX = "eval360id-"

    def __init__(self, log_dir=".", partition=None, instance_id=None, poll_interval=5.0):
        self._deployment_lock = asyncio.Lock()
        self._log_dir = str(log_dir)
        self._partition = partition
        self._instance_id = (
            _uuid.uuid4().hex[:8] if instance_id is None else instance_id
        )
        self._poll_interval = poll_interval
        # Maps serving_key (12-char hex) → model_name.
        # Populated in get_model_state, get_unneeded_models, and update_allocation.
        self._serving_key_registry: dict[str, str] = {}
        self._submitted_jobs: dict[int, SubmittedSlurmJob] = {}
        self._terminal_evidence_enabled = False
        logger.info("SlurmManager instance_id=%s", self._instance_id)

    # ------------------------------------------------------------------
    # Job name helpers
    # ------------------------------------------------------------------

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def _to_job_name(self, model_name: str, serving_key: str, replica_index: int) -> str:
        safe = re.sub(r"[^a-z0-9-]", "-", model_name.lower())[:20].strip("-")
        return f"eval360-{self._instance_id}-{safe}-{serving_key}-r{replica_index}"

    def _from_job_name(self, job_name: str) -> tuple[str, int] | None:
        """Parse (serving_key, replica_index) from job name, or None if not ours."""
        m = self._JOB_NAME_RE.match(job_name)
        if not m:
            return None
        iid, serving_key, replica_index = m.group(1), m.group(2), int(m.group(3))
        if iid != self._instance_id:
            return None
        return serving_key, replica_index

    @staticmethod
    def _parse_sbatch_job_id(stdout: bytes) -> int:
        """Parse the one root job ID returned by ``sbatch --parsable``."""
        match = re.fullmatch(
            rb"\s*([1-9][0-9]*)(?:;[^\s;]+)?\s*",
            stdout,
        )
        if match is None:
            raise RuntimeError(
                "sbatch succeeded without a parseable submitted job ID"
            )
        return int(match.group(1))

    def _remember_job(self, job: SubmittedSlurmJob) -> None:
        if not self._terminal_evidence_enabled:
            return
        existing = self._submitted_jobs.get(job.job_id)
        if existing is None:
            self._submitted_jobs[job.job_id] = job
            return
        if existing.job_name != job.job_name or existing.kind != job.kind:
            raise RuntimeError(
                f"Slurm job ID {job.job_id} changed identity within one scheduler"
            )
        self._submitted_jobs[job.job_id] = replace(
            existing,
            model_name=existing.model_name or job.model_name,
            serving_key=existing.serving_key or job.serving_key,
            event_uuid=existing.event_uuid or job.event_uuid,
            runner_name=existing.runner_name or job.runner_name,
            completed_event_roles=tuple(
                sorted(
                    set(existing.completed_event_roles)
                    | set(job.completed_event_roles)
                )
            ),
        )

    def begin_terminal_result_capture(self) -> None:
        """Enable invocation-scoped child capture for terminal evidence."""
        if self._terminal_evidence_enabled:
            raise RuntimeError("terminal-result child capture is already active")
        self._submitted_jobs.clear()
        self._terminal_evidence_enabled = True

    def get_submitted_jobs(self) -> tuple[SubmittedSlurmJob, ...]:
        """Return every remembered child in stable numeric job-ID order."""
        return tuple(
            self._submitted_jobs[job_id]
            for job_id in sorted(self._submitted_jobs)
        )

    async def release_submitted_model_serving_jobs(self) -> None:
        """Release ledgered model-serving children directly by job ID."""
        jobs = [
            job
            for job in self.get_submitted_jobs()
            if job.kind == "model_serving"
            and job.cancellation_intent != "scheduler_release"
        ]
        if not jobs:
            return
        accounting_error: BaseException | None = None
        try:
            outcomes = await self.get_job_accounting(
                [job.job_id for job in jobs]
            )
        except BaseException as error:
            accounting_error = error
            outcomes = {}
        active_jobs = [
            job
            for job in jobs
            if job.job_id not in outcomes
            or not outcomes[job.job_id].is_terminal
        ]
        results = await asyncio.gather(
            *(
                self.cancel_job(
                    job.job_id,
                    cancellation_intent="scheduler_release",
                )
                for job in active_jobs
            ),
            return_exceptions=True,
        )
        errors = [
            result
            for result in results
            if isinstance(result, BaseException)
        ]
        if accounting_error is not None:
            raise accounting_error
        if errors:
            raise errors[0]

    async def record_role_completion(
        self,
        serving_key: str,
        event_uuid: str,
        role: str,
    ) -> None:
        """Bind a successful event role to every then-active serving child."""
        if not self._terminal_evidence_enabled:
            return
        if role not in {"generation", "grading"}:
            raise ValueError(f"unsupported serving role completion: {role!r}")
        active_jobs = await self.get_all_jobs()
        matching_ids = sorted(
            job["job_id"]
            for job in active_jobs.values()
            if job.get("serving_key") == serving_key
        )
        if not matching_ids:
            raise RuntimeError(
                f"cannot record {role} completion for event {event_uuid}: "
                "no active serving child"
            )
        completion = (event_uuid, role)
        for job_id in matching_ids:
            job = self._submitted_jobs.get(job_id)
            if job is None:
                raise RuntimeError(
                    f"active serving child {job_id} is absent from the evidence ledger"
                )
            self._submitted_jobs[job_id] = replace(
                job,
                completed_event_roles=tuple(
                    sorted(set(job.completed_event_roles) | {completion})
                ),
            )

    async def _kill_timed_out_process(self, proc) -> None:
        """Best-effort cleanup for a timed-out Slurm CLI subprocess."""
        try:
            proc.kill()
        except ProcessLookupError:
            return
        await proc.wait()

    # ------------------------------------------------------------------
    # Container helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _container_mounts(model_instance) -> list[str]:
        """Build the full mount list for a container job.

        Always includes an identity mount for the model path so VLLM can
        access the weights at their original absolute path inside the
        container.  Conflicting user mounts are rejected at ModelSpec
        validation time.
        """
        model_path = model_instance.path
        user_mounts = model_instance.container_mounts or []
        identity_mount = f"{model_path}:{model_path}"
        return [identity_mount] + list(user_mounts)

    @classmethod
    def _container_mounts_arg(cls, model_instance) -> str:
        """Return a single pyxis --container-mounts argument.

        Pyxis expects a comma-separated mount list. Passing the flag multiple
        times can cause later values to override earlier ones, which would drop
        the auto-injected model-path mount.
        """
        return "--container-mounts=" + ",".join(cls._container_mounts(model_instance))

    @staticmethod
    def _serving_resource_args(model_instance) -> list[str]:
        """Return the exact bounded Slurm request for one serving replica."""
        resources = model_instance.serving_slurm_resources
        arguments = [
            "--nodes=1",
            "--ntasks=1",
            f"--gres=gpu:{resources.gpus_per_node}",
            f"--cpus-per-task={resources.cpus_per_task}",
            f"--time={resources.time_limit}",
        ]
        if resources.memory_gb is not None:
            arguments.append(f"--mem={resources.memory_gb}G")
        return arguments

    # ------------------------------------------------------------------
    # Time parsing
    # ------------------------------------------------------------------

    def _to_seconds(self, s):
        """Parse Slurm time strings (MM:SS / HH:MM:SS / DD-HH:MM:SS)."""
        s = s.strip()
        days = 0
        if '-' in s:
            days_part, s = s.split('-', 1)
            days = int(days_part)
        parts = s.split(':')
        if len(parts) == 2:
            h, m, sec = 0, *map(int, parts)
        elif len(parts) == 3:
            h, m, sec = map(int, parts)
        else:
            raise ValueError(f"Invalid Slurm time format: {s!r}")
        return days * 86400 + h * 3600 + m * 60 + sec

    # ------------------------------------------------------------------
    # squeue helpers
    # ------------------------------------------------------------------

    async def _query_squeue(self) -> subprocess.CompletedProcess[bytes]:
        """Run the high-frequency queue query outside asyncio subprocess I/O."""
        return await asyncio.to_thread(
            _run_bounded_command,
            (
                "squeue",
                "--noheader",
                "--me",
                "--format",
                "%j|%i|%T|%M|%N",
            ),
            timeout=_SQUEUE_QUERY_TIMEOUT_SECONDS,
        )

    async def get_all_jobs(self):
        """Return {job_name: {name, nodelist, elapsed_time, job_id, state,
        serving_key, replica_index, model_name}} for all active eval360 jobs."""
        active_states = ("RUNNING", "PENDING")
        for attempt in range(1, _SQUEUE_QUERY_MAX_ATTEMPTS + 1):
            try:
                completed = await self._query_squeue()
                break
            except subprocess.TimeoutExpired as error:
                if attempt == _SQUEUE_QUERY_MAX_ATTEMPTS:
                    raise RuntimeError(
                        "squeue timed out after "
                        f"{_SQUEUE_QUERY_MAX_ATTEMPTS} attempts"
                    ) from error
                logger.warning(
                    "squeue timed out after %.1fs (attempt %d/%d); retrying",
                    _SQUEUE_QUERY_TIMEOUT_SECONDS,
                    attempt,
                    _SQUEUE_QUERY_MAX_ATTEMPTS,
                )
                await asyncio.sleep(self._poll_interval)
        if completed.returncode != 0:
            raise RuntimeError(
                f"squeue failed: {completed.stderr.decode().strip()}"
            )

        result = {}
        for line in completed.stdout.decode().splitlines():
            job_name, job_id, job_state, elapsed_time, nodelist = line.split("|", 5)
            if job_state not in active_states:
                continue
            parsed = self._from_job_name(job_name)
            if not parsed:
                continue
            serving_key, replica_index = parsed
            model_name = self._serving_key_registry.get(serving_key)
            result[job_name] = {
                "name": job_name,
                "nodelist": nodelist,
                "elapsed_time": self._to_seconds(elapsed_time),
                "job_id": int(job_id),
                "state": job_state,
                "serving_key": serving_key,
                "replica_index": replica_index,
                "model_name": model_name,
            }
            self._remember_job(
                SubmittedSlurmJob(
                    job_id=int(job_id),
                    job_name=job_name,
                    kind="model_serving",
                    submission_origin="adopted",
                    model_name=model_name,
                    serving_key=serving_key,
                )
            )
        return result

    async def get_current_model_names(self):
        """Return list of unique model names with active (RUNNING or PENDING) jobs."""
        jobs = await self.get_all_jobs()
        model_names = set()
        for job in jobs.values():
            mn = job.get("model_name")
            if mn:
                model_names.add(mn)
        return list(model_names)

    async def get_unneeded_models(self, created_models, active_models, desired_models, maximum_nodes):
        """Return (unneeded, excess) model name lists.

        desired_models may be a dict (model_name → DeploymentInfo) or a
        collection of model names — both support `in` membership testing.
        If it is a dict with DeploymentInfo values, the serving_key registry
        is pre-populated from it so that get_all_jobs() can resolve model names.
        """
        # Pre-populate registry if desired_models is a full deployment dict
        if isinstance(desired_models, dict):
            for model_name, deployment_info in desired_models.items():
                try:
                    sk = deployment_info.model.serving_key
                    self._serving_key_registry[sk] = model_name
                except AttributeError:
                    pass

        created_jobs = await self.get_all_jobs()

        # Get model_name for each job (uses registry populated above or previously)
        all_job_model_names = []
        for job in created_jobs.values():
            mn = job.get("model_name")
            if mn:
                all_job_model_names.append(mn)

        unneeded = list({mn for mn in all_job_model_names if mn not in desired_models})
        remaining_jobs = [mn for mn in all_job_model_names if mn in desired_models]
        remaining_model_set = set(remaining_jobs)

        excess = []
        if len(remaining_jobs) > maximum_nodes:
            for model in created_models:
                if model not in active_models and model in remaining_model_set:
                    excess.append(model)
                remaining_after = sum(1 for mn in remaining_jobs if mn not in excess)
                if remaining_after <= maximum_nodes:
                    return unneeded, excess
            # Need to trim active models too
            remaining_after = sum(1 for mn in remaining_jobs if mn not in excess)
            for model in active_models:
                if remaining_after <= maximum_nodes:
                    break
                if model in remaining_model_set and model not in excess:
                    excess.append(model)
                    remaining_after = sum(1 for mn in remaining_jobs if mn not in excess)
        return unneeded, excess

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    async def check_live(self, job_state):
        """HTTP health-check a running job. Returns (response, job_state)."""
        url = f"http://{job_state['nodelist']}:8000/health"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=3.0) as resp:
                    return resp, job_state
            except Exception as e:
                logger.info(f"Health check failed for {url}: {e}")
                return None, job_state

    async def is_url_healthy(self, url: str) -> bool:
        """Single health check for a specific URL."""
        health_url = f"{url}/health"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(health_url, timeout=3.0) as resp:
                    return resp.status == 200
            except Exception:
                return False

    # ------------------------------------------------------------------
    # Model state (used by Scheduler.handle_job_update)
    # ------------------------------------------------------------------

    async def get_model_state(self, desired_models_dict):
        """Return (pending, deploying, live, dead) model name lists.

        pending and deploying contain one entry per job (may have duplicates for
        the same model) so that len(pending) + len(deploying) + len(live) equals
        the total number of occupied Slurm nodes.

        live entries are (model_name, url) tuples — one per healthy replica.
        dead contains unique model names where ALL replicas have timed out.
        """
        # Pre-populate registry from desired models
        for model_name, deployment_info in desired_models_dict.items():
            sk = deployment_info.model.serving_key
            self._serving_key_registry[sk] = model_name

        created_jobs = await self.get_all_jobs()

        # Group jobs by model_name
        jobs_by_model: dict[str, list] = {}
        for job in created_jobs.values():
            mn = job.get("model_name")
            if mn and mn in desired_models_dict:
                jobs_by_model.setdefault(mn, []).append(job)

        pending_models = []
        models_to_check = []  # (model_instance, job_state)

        for model_name, deployment_info in desired_models_dict.items():
            jobs = jobs_by_model.get(model_name, [])
            if not jobs:
                continue
            for job_state in jobs:
                if job_state["state"] == "PENDING":
                    pending_models.append(model_name)
                elif job_state["state"] == "RUNNING":
                    models_to_check.append((deployment_info.model, job_state))
                else:
                    raise RuntimeError(
                        f"unexpected job state for {model_name}: {job_state['state']}"
                    )

        # Health-check all RUNNING jobs concurrently
        task_list = []
        async with asyncio.TaskGroup() as tg:
            for model_instance, job_state in models_to_check:
                task_list.append(
                    (model_instance, job_state, tg.create_task(self.check_live(job_state)))
                )

        live_models: list[tuple[str, str]] = []
        live_model_names: set[str] = set()
        deploying_models: list[str] = []
        dead_replica_counts: dict[str, int] = {}
        running_replica_counts: dict[str, int] = {}

        for model_instance, job_state in models_to_check:
            mn = model_instance.name
            running_replica_counts[mn] = running_replica_counts.get(mn, 0) + 1

        for model_instance, job_state, task in task_list:
            resp, _ = task.result()
            mn = model_instance.name
            url = f"http://{job_state['nodelist']}:8000"
            if resp and resp.status == 200:
                live_models.append((mn, url))
                live_model_names.add(mn)
            elif (job_state["elapsed_time"] and
                  model_instance.max_time_to_deploy < job_state["elapsed_time"]):
                dead_replica_counts[mn] = dead_replica_counts.get(mn, 0) + 1
            else:
                deploying_models.append(mn)

        # Model is dead only when ALL running replicas have exceeded max_time_to_deploy
        # and none are live.
        dead_models = []
        for mn, dead_count in dead_replica_counts.items():
            if mn not in live_model_names and dead_count >= running_replica_counts.get(mn, 0):
                dead_models.append(mn)
            else:
                deploying_models.append(mn)

        # Exclude models that have live replicas from pending/deploying lists
        pending_final = [m for m in pending_models if m not in live_model_names]
        deploying_final = [m for m in deploying_models if m not in live_model_names]

        # Total squeue job count per model (all states: PENDING + RUNNING), used by
        # the scheduler to accurately count occupied nodes including mixed-state replicas.
        replica_counts = {mn: len(jobs) for mn, jobs in jobs_by_model.items()}

        return pending_final, deploying_final, live_models, dead_models, replica_counts

    # ------------------------------------------------------------------
    # Job allocation
    # ------------------------------------------------------------------

    async def _retire_superseded_deployments(
        self,
        created_jobs: dict,
        desired_allocations,
        sibling_names_by_sk: dict | None,
    ) -> set[int]:
        """Cancel jobs serving a wanted model under a config it no longer has.

        Returns the job IDs cancelled, so the caller can discount them without
        waiting for squeue to catch up.

        `unneeded_models` only ever names models that are no longer wanted AT
        ALL, so nothing else retires a deployment whose model is still desired
        but whose CONFIG has changed. Without this, editing a model's YAML in
        place left the previous deployment running and serving traffic under the
        new config's name — see the comment in `update_allocation`.

        A job is superseded when its model is still desired but its serving key
        is not one of the keys now wanted. Sibling names matter here: several
        models can share one deployment, and only the representative appears in
        `desired_allocations`, so a job could otherwise look unowned and be left
        alone. When the caller supplies no sibling map (as tests may), the
        representative names are the best available answer and the check simply
        covers less rather than misfiring.
        """
        desired_keys = {model.serving_key for model, _ in desired_allocations}
        desired_names = {model.name for model, _ in desired_allocations}
        if sibling_names_by_sk:
            for key in desired_keys:
                desired_names.update(sibling_names_by_sk.get(key, []))

        superseded = {
            job["job_id"]: job
            for job in created_jobs.values()
            if job.get("model_name") in desired_names
            and job.get("serving_key") not in desired_keys
        }
        if not superseded:
            return set()

        for job_id, job in superseded.items():
            logger.info(
                "Retiring superseded deployment of %s: job %s serves config %s, "
                "which is no longer wanted",
                job.get("model_name"), job_id, job.get("serving_key"),
            )
        async with asyncio.TaskGroup() as tg:
            for job_id in superseded:
                tg.create_task(
                    self.cancel_job(job_id, cancellation_intent="superseded_config")
                )
        return set(superseded)

    async def update_allocation(self, desired_allocations, unneeded_models, sibling_names_by_sk=None):
        async with self._deployment_lock:
            # Pre-populate registry before any get_all_jobs() call so that
            # model names are resolved correctly in the results.
            for model_instance, _ in desired_allocations:
                self._serving_key_registry[model_instance.serving_key] = model_instance.name

            await self.kill_all(unneeded_models)
            created_jobs = await self.get_all_jobs()

            superseded_job_ids = await self._retire_superseded_deployments(
                created_jobs, desired_allocations, sibling_names_by_sk
            )

            # Replicas are counted per SERVING KEY, not per model name.
            #
            # Counting by name was a silent-wrong-results bug. A serving key is
            # a content hash of the deployment config (path, revision, vllm args,
            # venv/conda/container), so editing a model's config in place keeps
            # the NAME and changes the KEY. The old job then satisfied the new
            # config's replica target —
            #
            #     Skipping ci-edited-model, already has 1/1 replicas
            #
            # — so the new config was never submitted. And because
            # `_serving_key_registry` maps BOTH keys to that one name, the stale
            # job still resolved back to it, so `get_model_state` reported the
            # model `live` and generation ran against the previous config. The
            # eval finished green with results from a config nobody asked for.
            #
            # Counting by key is safe against sibling models (several names, one
            # deployment) because `Scheduler.get_desired_allocation` already
            # collapses `desired_allocations` to one entry per serving key.
            #
            # Jobs retired just above are excluded rather than re-queried:
            # scancel returns before the row leaves squeue, so counting them
            # would re-create the very skip this fixes for one more poll.
            replicas_per_key: dict[str, int] = {}
            max_replica_idx: dict[str, int] = {}
            for job in created_jobs.values():
                if job["job_id"] in superseded_job_ids:
                    continue
                sk = job.get("serving_key")
                if sk:
                    replicas_per_key[sk] = replicas_per_key.get(sk, 0) + 1
                    ri = job["replica_index"]
                    max_replica_idx[sk] = max(max_replica_idx.get(sk, -1), ri)

            for model_instance, replica_count in desired_allocations:
                model_name = model_instance.name
                serving_key = model_instance.serving_key
                self._serving_key_registry[serving_key] = model_name

                existing_count = replicas_per_key.get(serving_key, 0)
                if existing_count >= replica_count:
                    logger.info(
                        f"Skipping {model_name} ({serving_key}), already has "
                        f"{existing_count}/{replica_count} replicas"
                    )
                    continue

                next_index = max_replica_idx.get(serving_key, -1) + 1
                for replica_index in range(next_index, next_index + (replica_count - existing_count)):
                    vllm_args = copy.deepcopy(model_instance.vllm_cli_args)
                    split_vllm_args = []
                    for arg in vllm_args:
                        for part in arg.split(None, 1):
                            split_vllm_args.append(part.strip().strip("'").strip('"'))
                    if model_instance.revision:
                        split_vllm_args += ["--revision", model_instance.revision]
                    all_names = (
                        sibling_names_by_sk.get(serving_key, [model_name])
                        if sibling_names_by_sk else [model_name]
                    )
                    request_model_name = (
                        model_instance.api_model_name or model_name
                    )
                    all_names = list(
                        dict.fromkeys([request_model_name, *all_names])
                    )
                    split_vllm_args += ["--served-model-name", *all_names]
                    if model_instance.allow_long_max_model_len:
                        logger.info("Enabling VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for %s", model_name)

                    vllm_args_b64 = base64.b64encode(json.dumps(split_vllm_args).encode()).decode()
                    job_name = self._to_job_name(model_name, serving_key, replica_index)

                    command = [
                        "sbatch",
                        f"--job-name={job_name}",
                        *([f"--partition={self._partition}"] if self._partition else []),
                        f"--output={self._log_dir}/slurm-%j.out",
                        *self._serving_resource_args(model_instance),
                    ]
                    if self._terminal_evidence_enabled:
                        command.append("--parsable")

                    if model_instance.uses_container:
                        slurm_path = Path(__file__).resolve().parent / "slurm/sbatch_script_container.sh"
                        command += [
                            f"--container-image={model_instance.container_image}",
                            "--container-writable",
                        ]
                        command.append(self._container_mounts_arg(model_instance))
                        command.append(
                            f"--export=vllm_args={vllm_args_b64},"
                            f"model_path={model_instance.path},"
                            f"vllm_allow_long_max_model_len={int(model_instance.allow_long_max_model_len)},"
                            f"vllm_logging_level={model_instance.vllm_logging_level}"
                        )
                    elif model_instance.uses_conda:
                        slurm_path = Path(__file__).resolve().parent / "slurm/sbatch_script_conda.sh"
                        command.append(
                            f"--export=conda_env={model_instance.conda_env},"
                            f"vllm_args={vllm_args_b64},"
                            f"model_path={model_instance.path},"
                            f"vllm_allow_long_max_model_len={int(model_instance.allow_long_max_model_len)},"
                            f"vllm_logging_level={model_instance.vllm_logging_level}"
                        )
                    else:
                        slurm_path = Path(__file__).resolve().parent / "slurm/sbatch_script.sh"
                        command.append(
                            f"--export=venv_path={model_instance.venv_path},"
                            f"vllm_args={vllm_args_b64},"
                            f"model_path={model_instance.path},"
                            f"vllm_allow_long_max_model_len={int(model_instance.allow_long_max_model_len)},"
                            f"vllm_logging_level={model_instance.vllm_logging_level}"
                        )

                    command.append(str(slurm_path))
                    logger.info(f"Submitting job replica {replica_index} for {model_name}: {command}")
                    proc = await asyncio.create_subprocess_exec(
                        *command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
                    except asyncio.TimeoutError:
                        await self._kill_timed_out_process(proc)
                        raise
                    if proc.returncode != 0:
                        raise RuntimeError(f"sbatch failed: {stderr.decode()}")
                    if self._terminal_evidence_enabled:
                        job_id = self._parse_sbatch_job_id(stdout)
                        self._remember_job(
                            SubmittedSlurmJob(
                                job_id=job_id,
                                job_name=job_name,
                                kind="model_serving",
                                submission_origin="submitted",
                                model_name=model_name,
                                serving_key=serving_key,
                            )
                        )

    # ------------------------------------------------------------------
    # Imported-dataset jobs
    # ------------------------------------------------------------------

    async def submit_imported_dataset_job(
        self,
        event_uuid: str,
        model_instance,
        runner_name: str,
        setup_script: str,
        benchmark_script: str,
        output_dir: Path,
    ) -> int:
        setup_b64 = base64.b64encode(setup_script.encode()).decode()
        benchmark_b64 = base64.b64encode(benchmark_script.encode()).decode()

        vllm_args = copy.deepcopy(model_instance.vllm_cli_args)
        split_vllm_args = []
        for arg in vllm_args:
            for part in arg.split(None, 1):
                split_vllm_args.append(part.strip().strip("'").strip('"'))
        if model_instance.revision:
            split_vllm_args += ["--revision", model_instance.revision]
        split_vllm_args += ["--served-model-name", model_instance.name]
        if model_instance.allow_long_max_model_len:
            logger.info("Enabling VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for %s", model_instance.name)

        vllm_args_b64 = base64.b64encode(json.dumps(split_vllm_args).encode()).decode()
        repo_root = Path(__file__).resolve().parent.parent
        safe_model = re.sub(r"[^a-z0-9-]", "-", model_instance.name.lower())[:20].strip("-")
        safe_runner = re.sub(r"[^a-z0-9-]", "-", runner_name.lower())[:15].strip("-")
        job_name = f"eval360id-{self._instance_id}-{safe_runner}-{safe_model}-{event_uuid[:8]}"
        bootstrap_python = "" if model_instance.uses_container else sys.executable

        common_export = (
            f"model_path={model_instance.path},"
            f"vllm_args={vllm_args_b64},"
            f"vllm_allow_long_max_model_len={int(model_instance.allow_long_max_model_len)},"
            f"bootstrap_python={bootstrap_python},"
            f"runner_name={runner_name},"
            f"repo_root={repo_root},"
            f"max_time_to_deploy={model_instance.max_time_to_deploy},"
            f"setup_script_b64={setup_b64},"
            f"benchmark_script_b64={benchmark_b64},"
            f"output_dir={output_dir}"
        )
        command = [
            "sbatch",
            f"--job-name={job_name}",
            *([f"--partition={self._partition}"] if self._partition else []),
            f"--output={self._log_dir}/slurm-%j.out",
            *self._serving_resource_args(model_instance),
        ]
        if self._terminal_evidence_enabled:
            command.append("--parsable")

        if model_instance.uses_container:
            slurm_path = Path(__file__).resolve().parent / "slurm/imported_dataset_script_container.sh"
            command += [
                f"--container-image={model_instance.container_image}",
                "--container-writable",
            ]
            command.append(self._container_mounts_arg(model_instance))
            command.append(f"--export={common_export}")
        elif model_instance.uses_conda:
            slurm_path = Path(__file__).resolve().parent / "slurm/imported_dataset_script_conda.sh"
            command.append(f"--export=conda_env={model_instance.conda_env},{common_export}")
        else:
            slurm_path = Path(__file__).resolve().parent / "slurm/imported_dataset_script.sh"
            command.append(f"--export=venv_path={model_instance.venv_path},{common_export}")

        command.append(str(slurm_path))
        logger.info(f"Submitting imported dataset job: {job_name}")
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        except asyncio.TimeoutError:
            await self._kill_timed_out_process(proc)
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {stderr.decode()}")
        job_id = (
            self._parse_sbatch_job_id(stdout)
            if self._terminal_evidence_enabled
            else int(stdout.decode().strip().split()[-1])
        )
        self._remember_job(
            SubmittedSlurmJob(
                job_id=job_id,
                job_name=job_name,
                kind="imported_dataset",
                submission_origin="submitted",
                model_name=model_instance.name,
                serving_key=model_instance.serving_key,
                event_uuid=event_uuid,
                runner_name=runner_name,
            )
        )
        return job_id

    async def get_job_node(self, job_id: int, poll_interval: float = 10.0) -> str:
        """Poll until the job is running and return its node name.

        Raises `SlurmJobVanished` (a `RuntimeError`) if the job leaves the queue
        without ever being allocated a node. That is an EXPECTED outcome, which
        is why it has a type of its own — see the class docstring — and
        `Scheduler.handle_imported_dataset_event` catches exactly it. Anything
        else raised from here (a squeue timeout, say) is a genuine fault and
        must keep propagating.

        WHY THAT RAISE EXISTS. This used to loop while stdout was empty and
        never inspect ``returncode``, so a job that vanished before it was first
        observed RUNNING polled forever. ``handle_imported_dataset_event``
        awaits this with no timeout, so that hung the event silently: no error,
        no progress, nothing in the log.

        The subtlety is that "no node yet" and "no job any more" look almost
        identical once the output is stripped. Measured against a real cluster:

            pending         rc=0, ONE row whose %N is empty
            recently ended  rc=0, ZERO rows
            purged          rc=1, "Invalid job id specified"

        So the signal to keep waiting is "a row exists but has no node", not
        "the output is empty" — counting rows is what separates the first case
        from the second, and ``returncode`` covers the third.
        """
        while True:
            proc = await asyncio.create_subprocess_exec(
                "squeue", "--jobs", str(job_id), "--format=%N", "--noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            except asyncio.TimeoutError:
                await self._kill_timed_out_process(proc)
                raise

            if proc.returncode != 0:
                # squeue rejects an unknown job ID outright once Slurm has
                # forgotten it. There is nothing left to wait for.
                detail = stderr.decode().strip() or f"squeue exited {proc.returncode}"
                raise SlurmJobVanished(
                    f"Slurm job {job_id} is no longer queued: {detail}"
                )

            rows = stdout.decode().splitlines()
            if not rows:
                raise SlurmJobVanished(
                    f"Slurm job {job_id} is no longer queued; it left the queue "
                    "before a node was allocated"
                )

            node = rows[0].strip()
            if node and not node.startswith("("):
                return node
            await asyncio.sleep(poll_interval)

    async def wait_for_vllm_health(self, node: str, max_time_to_deploy: int, poll_interval: float = 10.0) -> bool:
        url = f"http://{node}:8000/health"
        elapsed = 0
        async with aiohttp.ClientSession() as session:
            while elapsed < max_time_to_deploy:
                try:
                    async with session.get(url, timeout=3.0) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
        return False

    async def wait_for_job_completion(self, job_id: int, poll_interval: float = 30.0) -> None:
        while True:
            proc = await asyncio.create_subprocess_exec(
                "squeue", "--jobs", str(job_id), "--noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            except asyncio.TimeoutError:
                await self._kill_timed_out_process(proc)
                raise
            if proc.returncode != 0 or not stdout.decode().strip():
                return
            await asyncio.sleep(poll_interval)

    async def cancel_job(
        self,
        job_id: int,
        *,
        cancellation_intent: str | None = None,
    ) -> None:
        try:
            completed = await asyncio.to_thread(
                _run_bounded_command,
                ("scancel", str(job_id)),
                timeout=_SCANCEL_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0:
                message = (
                    f"scancel failed for Slurm job {job_id}: "
                    f"{completed.stderr.decode().strip()}"
                )
                if self._terminal_evidence_enabled:
                    raise RuntimeError(message)
                logger.warning(message)
                return
            if cancellation_intent is not None and job_id in self._submitted_jobs:
                self._submitted_jobs[job_id] = replace(
                    self._submitted_jobs[job_id],
                    cancellation_intent=cancellation_intent,
                )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Timed out after %.1fs while cancelling Slurm job %s",
                _SCANCEL_TIMEOUT_SECONDS,
                job_id,
            )
            if self._terminal_evidence_enabled:
                raise
        except Exception:
            if self._terminal_evidence_enabled:
                raise
            logger.exception(f"Failed to scancel job {job_id}")

    async def kill_all(self, model_names):
        model_name_set = set(model_names)
        # Find serving_keys for these model names using the registry
        wanted_serving_keys = {
            sk for sk, mn in self._serving_key_registry.items() if mn in model_name_set
        }
        jobs = await self.get_all_jobs()
        async with asyncio.TaskGroup() as tg:
            for job in jobs.values():
                if job.get("serving_key") in wanted_serving_keys:
                    tg.create_task(
                        self.cancel_job(
                            job["job_id"],
                            cancellation_intent="scheduler_release",
                        )
                    )

    async def cancel_all_owned_jobs(self) -> None:
        """Cancel all active Slurm jobs belonging to this scheduler instance."""
        try:
            completed = await asyncio.to_thread(
                _run_bounded_command,
                ("squeue", "--noheader", "--me", "--format", "%j|%i"),
                timeout=_SQUEUE_QUERY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return
        vllm_prefix = f"eval360-{self._instance_id}-"
        imported_prefix = f"eval360id-{self._instance_id}-"
        job_ids = []
        for line in completed.stdout.decode().splitlines():
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            job_name, job_id = parts
            if job_name.startswith(vllm_prefix) or job_name.startswith(imported_prefix):
                job_ids.append(int(job_id))
        if not job_ids:
            return
        logger.info("Cancelling %d eval360 Slurm job(s) (instance %s) on interrupt: %s",
                     len(job_ids), self._instance_id, job_ids)
        async with asyncio.TaskGroup() as tg:
            for job_id in job_ids:
                tg.create_task(
                    self.cancel_job(
                        job_id,
                        cancellation_intent="controller_interrupt",
                    )
                )

    @staticmethod
    def _normalize_sacct_state(raw_state: str) -> str:
        """Normalize decorations while retaining ``raw_state`` as evidence."""
        return raw_state.split(maxsplit=1)[0].removesuffix("+")

    async def get_job_accounting(
        self,
        job_ids: list[int] | tuple[int, ...],
    ) -> dict[int, SlurmJobOutcome]:
        """Query root-job scheduler outcomes from ``sacct`` exactly once."""
        requested = tuple(sorted(set(job_ids)))
        if not requested:
            return {}
        completed = await asyncio.to_thread(
            _run_bounded_command,
            (
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                ",".join(str(job_id) for job_id in requested),
                "--format=JobIDRaw,JobName%256,State,ExitCode,Reason",
            ),
            timeout=_SACCT_QUERY_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"sacct failed: {completed.stderr.decode().strip()}"
            )

        outcomes: dict[int, SlurmJobOutcome] = {}
        requested_set = set(requested)
        for raw_line in completed.stdout.decode().splitlines():
            if not raw_line:
                continue
            fields = raw_line.split("|")
            if len(fields) == 6 and fields[-1] == "":
                fields.pop()
            if len(fields) != 5:
                raise RuntimeError(f"malformed sacct row: {raw_line!r}")
            job_id_raw, job_name, raw_state, raw_exit_code, reason = fields
            if not job_id_raw.isdecimal():
                continue
            job_id = int(job_id_raw)
            if job_id not in requested_set:
                continue
            exit_match = re.fullmatch(r"([0-9]+):([0-9]+)", raw_exit_code)
            if exit_match is None:
                raise RuntimeError(
                    f"malformed sacct exit code for job {job_id}: {raw_exit_code!r}"
                )
            outcome = SlurmJobOutcome(
                job_id=job_id,
                job_id_raw=job_id_raw,
                job_name=job_name,
                raw_state=raw_state,
                state=self._normalize_sacct_state(raw_state),
                exit_code=int(exit_match.group(1)),
                signal=int(exit_match.group(2)),
                reason=reason,
            )
            if (
                outcome.state not in _TERMINAL_SLURM_STATES
                and outcome.state not in _NONTERMINAL_SLURM_STATES
            ):
                raise RuntimeError(
                    f"unknown sacct state for job {job_id}: {raw_state!r}"
                )
            if job_id in outcomes:
                raise RuntimeError(f"duplicate root sacct row for job {job_id}")
            outcomes[job_id] = outcome
        return outcomes

    async def wait_for_terminal_job_outcomes(
        self,
        job_ids: list[int] | tuple[int, ...],
    ) -> dict[int, SlurmJobOutcome]:
        """Wait without an elapsed-time deadline for authoritative terminals.

        slurmdbd publishes a job's accounting record in STAGES, and this used to
        tolerate neither of them. Measured against a real cluster:

            t+0.0s   no row at all
            t+1.0s   a row, named `allocation`, possibly ALREADY TERMINAL
            t+1.5s   the same row, carrying the real job name

        The first stage made this raise "no root accounting row" on its very
        first query, with no retry. The second is worse, because it is silent:
        the method returned as soon as ``is_terminal`` was true, so the caller
        could receive a terminal row whose name is not the job's, and
        ``Scheduler._reconcile_terminal_jobs`` then raised "sacct job name ...
        does not match the submitted child identity" — failing an otherwise
        successful evaluation for a reason unrelated to the job. A short-lived
        child is all it takes.

        Both lag windows now get a BOUNDED grace; the wait for terminality
        itself stays unbounded, because a real job may run for hours and that
        was never the problem. Exceeding either grace still raises, so an
        accounting row that genuinely never arrives fails as loudly as before —
        terminal evidence that quietly degrades would be worse than none.

        "Settled" is judged against the submission ledger rather than against
        the literal string ``allocation``: the ledger is what the caller will
        compare to, so matching it is the actual contract, and nothing here has
        to know what placeholder Slurm happens to use. Jobs absent from the
        ledger (adopted rather than submitted) have no expected name and are
        accepted as soon as they are terminal.
        """
        requested = tuple(sorted(set(job_ids)))
        if not requested:
            return {}

        outcomes = await self._await_published_rows(requested)

        while not all(outcome.is_terminal for outcome in outcomes.values()):
            await asyncio.sleep(self._poll_interval)
            outcomes = await self._refetch(requested)

        return await self._await_settled_rows(requested, outcomes)

    def _row_is_settled(self, outcome: SlurmJobOutcome) -> bool:
        """Whether a row carries the name its job was submitted with."""
        submitted = self._submitted_jobs.get(outcome.job_id)
        if submitted is None:
            return True
        return outcome.job_name == submitted.job_name

    async def _refetch(
        self, requested: tuple[int, ...]
    ) -> dict[int, SlurmJobOutcome]:
        """Re-query, treating a row that has vanished as a hard failure."""
        outcomes = await self.get_job_accounting(requested)
        missing = sorted(set(requested) - set(outcomes))
        if missing:
            raise RuntimeError(
                "sacct lost the root accounting row for required Slurm job(s): "
                + ", ".join(str(job_id) for job_id in missing)
            )
        return outcomes

    async def _await_published_rows(
        self, requested: tuple[int, ...]
    ) -> dict[int, SlurmJobOutcome]:
        """Wait, briefly, for slurmdbd to publish a row for every job."""
        deadline = time.monotonic() + _SACCT_PUBLISH_GRACE_SECONDS
        while True:
            outcomes = await self.get_job_accounting(requested)
            missing = sorted(set(requested) - set(outcomes))
            if not missing:
                return outcomes
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "sacct returned no root accounting row for required Slurm "
                    "job(s) after "
                    f"{_SACCT_PUBLISH_GRACE_SECONDS:.0f}s: "
                    + ", ".join(str(job_id) for job_id in missing)
                )
            await asyncio.sleep(min(self._poll_interval, _SACCT_LAG_POLL_SECONDS))

    async def _await_settled_rows(
        self,
        requested: tuple[int, ...],
        outcomes: dict[int, SlurmJobOutcome],
    ) -> dict[int, SlurmJobOutcome]:
        """Wait, briefly, for terminal rows to carry their real job names."""
        deadline = time.monotonic() + _SACCT_SETTLE_GRACE_SECONDS
        while True:
            unsettled = sorted(
                job_id
                for job_id, outcome in outcomes.items()
                if not self._row_is_settled(outcome)
            )
            if not unsettled:
                return outcomes
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "sacct still reports a placeholder job name after "
                    f"{_SACCT_SETTLE_GRACE_SECONDS:.0f}s for Slurm job(s): "
                    + ", ".join(
                        f"{job_id} ({outcomes[job_id].job_name!r} != "
                        f"{self._submitted_jobs[job_id].job_name!r})"
                        for job_id in unsettled
                    )
                )
            await asyncio.sleep(min(self._poll_interval, _SACCT_LAG_POLL_SECONDS))
            outcomes = await self._refetch(requested)

    async def get_available_nodes(self):
        proc = await asyncio.create_subprocess_exec(
            "sinfo", "-h",
            *(["-p", self._partition] if self._partition else []),
            "-o", "%T|%D",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        except asyncio.TimeoutError:
            await self._kill_timed_out_process(proc)
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"sinfo failed: {stderr.decode()}")
        for line in stdout.decode().splitlines():
            try:
                state, num = line.split("|")
            except ValueError:
                continue
            if state == "idle":
                return int(num)
        return 0

    # ------------------------------------------------------------------
    # Polling loop (source for the scheduler's job update queue)
    # ------------------------------------------------------------------

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self._poll_interval)
        return True
