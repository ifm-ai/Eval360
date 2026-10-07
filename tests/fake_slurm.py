"""
tests/fake_slurm.py — TestManager: drop-in replacement for SlurmManager.

Jobs are tracked in memory and transition through states automatically
(PENDING → RUNNING → healthy) or under explicit test control.

Usage::

    s = _scheduler(tmp_path)
    test_mgr = FakeSlurmManager()
    s.slurm_manager = test_mgr          # inject before running

    # After run, inspect state:
    test_mgr.get_active_job_count()     # PENDING + RUNNING jobs
    test_mgr.get_jobs("my-model")       # list[FakeJob] for one model

    # Test control:
    test_mgr.preempt("my-model")        # simulate Slurm preemption
    test_mgr.set_healthy("serving-key") # mark VLLM healthy immediately
"""

import asyncio
import time
from dataclasses import dataclass, field

from scheduler.slurm_manager import (
    SlurmJobOutcome,
    SlurmJobVanished,
    SubmittedSlurmJob,
)


@dataclass
class FakeJob:
    """A single fake Slurm job with mutable state."""
    job_id: int
    job_name: str
    model_name: str
    serving_key: str
    replica_index: int
    state: str = "PENDING"           # "PENDING", "RUNNING", "CANCELLED"
    node: str = "fake-node"
    created_at: float = field(default_factory=time.monotonic)
    run_started_at: float | None = None
    kind: str = "model_serving"
    submission_origin: str = "submitted"
    event_uuid: str | None = None
    runner_name: str | None = None
    cancellation_intent: str | None = None
    completed_event_roles: set[tuple[str, str]] = field(default_factory=set)
    exit_code: int = 0
    signal: int = 0
    reason: str = "None"


class FakeSlurmManager:
    """Drop-in replacement for SlurmManager backed by in-memory state.

    Jobs transition PENDING → RUNNING automatically after ``run_delay``
    seconds (when ``auto_run=True``).  RUNNING jobs become "healthy" after
    ``healthy_delay`` seconds (when ``auto_healthy=True``), causing
    ``get_model_state`` to include them in the ``live`` list.

    The async iterator fires every ``poll_interval`` seconds (default 50 ms)
    and advances job states before returning ``True``, which triggers the
    scheduler's ``handle_job_update`` loop.
    """

    def __init__(
        self,
        *,
        vllm_url: str = "http://fake-node:8000",
        auto_run: bool = True,
        run_delay: float = 0.0,
        auto_healthy: bool = True,
        healthy_delay: float = 0.0,
        poll_interval: float = 0.05,
        instance_id: str = "faketest",
    ):
        self._jobs: dict[int, FakeJob] = {}
        self._next_job_id = 1000
        self._serving_key_registry: dict[str, str] = {}
        self._deployment_lock = asyncio.Lock()
        self._instance_id = instance_id

        self._vllm_url = vllm_url
        self._auto_run = auto_run
        self._run_delay = run_delay
        self._auto_healthy = auto_healthy
        self._healthy_delay = healthy_delay
        self._poll_interval = poll_interval
        self._healthy_sks: set[str] = set()  # serving keys whose VLLM is healthy
        self._submit_count: int = 0  # total jobs ever created (for assertions)

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def begin_terminal_result_capture(self) -> None:
        """Mirror the production opt-in capture boundary."""

    async def record_role_completion(
        self,
        serving_key: str,
        event_uuid: str,
        role: str,
    ) -> None:
        matching = [
            job
            for job in self._jobs.values()
            if job.kind == "model_serving"
            and job.serving_key == serving_key
            and job.state in ("PENDING", "RUNNING")
        ]
        if not matching:
            raise RuntimeError(
                f"cannot record {role} completion for event {event_uuid}: "
                "no active serving child"
            )
        for job in matching:
            job.completed_event_roles.add((event_uuid, role))

    async def release_submitted_model_serving_jobs(self) -> None:
        """Release ledgered serving children without active-job discovery."""
        active_jobs = [
            job
            for job in self._jobs.values()
            if job.kind == "model_serving"
            and job.state in ("PENDING", "RUNNING")
            and job.cancellation_intent != "scheduler_release"
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
        if errors:
            raise errors[0]

    # ──────────────────────────────────────────────────────────────
    # Async iterator — drives handle_job_update
    # ──────────────────────────────────────────────────────────────

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self._poll_interval)
        self._tick()
        return True

    def _tick(self):
        """Advance job states based on elapsed time and auto-transition settings."""
        now = time.monotonic()
        for job in self._jobs.values():
            if job.state == "PENDING" and self._auto_run:
                if now - job.created_at >= self._run_delay:
                    job.state = "RUNNING"
                    job.run_started_at = now
            if job.state == "RUNNING" and self._auto_healthy:
                run_since = now - (job.run_started_at or job.created_at)
                if run_since >= self._healthy_delay:
                    self._healthy_sks.add(job.serving_key)

    # ──────────────────────────────────────────────────────────────
    # Core interface — mirrors SlurmManager
    # ──────────────────────────────────────────────────────────────

    async def get_model_state(self, desired_models_dict):
        """Return (pending, deploying, live, dead, replica_counts).

        ``live`` is a list of (model_name, url) tuples.
        """
        for model_name, deployment_info in desired_models_dict.items():
            sk = deployment_info.model.serving_key
            self._serving_key_registry[sk] = model_name

        self._tick()

        jobs_by_model: dict[str, list[FakeJob]] = {}
        for job in self._jobs.values():
            if (
                job.kind == "model_serving"
                and job.state in ("PENDING", "RUNNING")
                and job.model_name in desired_models_dict
            ):
                jobs_by_model.setdefault(job.model_name, []).append(job)

        pending: list[str] = []
        deploying: list[str] = []
        live: list[tuple[str, str]] = []
        dead: list[str] = []
        live_names: set[str] = set()
        replica_counts: dict[str, int] = {mn: len(jobs) for mn, jobs in jobs_by_model.items()}

        now = time.monotonic()
        for model_name, jobs in jobs_by_model.items():
            di = desired_models_dict[model_name]
            max_ttd = di.model.max_time_to_deploy
            sk = di.model.serving_key

            for job in jobs:
                if job.state == "PENDING":
                    pending.append(model_name)
                elif job.state == "RUNNING":
                    if sk in self._healthy_sks:
                        live.append((model_name, self._vllm_url))
                        live_names.add(model_name)
                    else:
                        elapsed = now - job.created_at
                        if elapsed >= max_ttd:
                            # This replica has exceeded max_time_to_deploy
                            # Counted as "dead" below if all replicas are in this state
                            pass
                        else:
                            deploying.append(model_name)

            # Model is "dead" when every running replica exceeded max_ttd and none are live
            if model_name not in live_names:
                running_jobs = [j for j in jobs if j.state == "RUNNING"]
                timed_out = [j for j in running_jobs if now - j.created_at >= max_ttd]
                if running_jobs and len(timed_out) == len(running_jobs):
                    dead.append(model_name)

        pending_final = [m for m in pending if m not in live_names]
        deploying_final = [m for m in deploying if m not in live_names]
        return pending_final, deploying_final, live, dead, replica_counts

    async def get_all_jobs(self):
        """Return {job_name: info} for all PENDING and RUNNING jobs."""
        result = {}
        now = time.monotonic()
        for job in self._jobs.values():
            if (
                job.kind != "model_serving"
                or job.state not in ("PENDING", "RUNNING")
            ):
                continue
            result[job.job_name] = {
                "name": job.job_name,
                "nodelist": job.node,
                "elapsed_time": int(now - job.created_at),
                "job_id": job.job_id,
                "state": job.state,
                "serving_key": job.serving_key,
                "replica_index": job.replica_index,
                "model_name": job.model_name,
            }
        return result

    async def get_unneeded_models(self, created_models, active_models, desired_models, maximum_nodes):
        """Return (unneeded, excess) model name lists. Mirrors SlurmManager logic."""
        if isinstance(desired_models, dict):
            for model_name, deployment_info in desired_models.items():
                try:
                    sk = deployment_info.model.serving_key
                    self._serving_key_registry[sk] = model_name
                except AttributeError:
                    pass

        active_job_model_names = [
            job.model_name for job in self._jobs.values()
            if (
                job.kind == "model_serving"
                and job.state in ("PENDING", "RUNNING")
                and job.model_name
            )
        ]

        unneeded = list({mn for mn in active_job_model_names if mn not in desired_models})
        remaining_jobs = [mn for mn in active_job_model_names if mn in desired_models]
        remaining_model_set = set(remaining_jobs)

        excess = []
        if len(remaining_jobs) > maximum_nodes:
            for model in created_models:
                if model not in active_models and model in remaining_model_set:
                    excess.append(model)
                remaining_after = sum(1 for mn in remaining_jobs if mn not in excess)
                if remaining_after <= maximum_nodes:
                    return unneeded, excess
            remaining_after = sum(1 for mn in remaining_jobs if mn not in excess)
            for model in active_models:
                if remaining_after <= maximum_nodes:
                    break
                if model in remaining_model_set and model not in excess:
                    excess.append(model)
                    remaining_after = sum(1 for mn in remaining_jobs if mn not in excess)

        return unneeded, excess

    async def update_allocation(self, desired_allocations, unneeded_models, sibling_names_by_sk=None):
        """Cancel unneeded jobs, submit new replicas for desired models."""
        async with self._deployment_lock:
            for model_instance, _ in desired_allocations:
                self._serving_key_registry[model_instance.serving_key] = model_instance.name

            # Cancel unneeded
            for job in list(self._jobs.values()):
                if (
                    job.kind == "model_serving"
                    and job.model_name in unneeded_models
                    and job.state in ("PENDING", "RUNNING")
                ):
                    job.state = "CANCELLED"
                    job.cancellation_intent = "scheduler_release"
                    job.signal = 15

            # Count existing active replicas
            replicas_per_model: dict[str, int] = {}
            max_replica_idx: dict[str, int] = {}
            for job in self._jobs.values():
                if (
                    job.kind == "model_serving"
                    and job.state in ("PENDING", "RUNNING")
                ):
                    if job.model_name:
                        replicas_per_model[job.model_name] = replicas_per_model.get(job.model_name, 0) + 1
                    max_replica_idx[job.serving_key] = max(
                        max_replica_idx.get(job.serving_key, -1), job.replica_index
                    )

            # Submit new replicas to cover shortfall
            for model_instance, replica_count in desired_allocations:
                model_name = model_instance.name
                serving_key = model_instance.serving_key
                existing = replicas_per_model.get(model_name, 0)
                if existing >= replica_count:
                    continue
                next_index = max_replica_idx.get(serving_key, -1) + 1
                for replica_index in range(next_index, next_index + (replica_count - existing)):
                    self._create_job(model_instance, replica_index)

    @property
    def submit_count(self) -> int:
        """Total number of jobs ever submitted (including cancelled/completed)."""
        return self._submit_count

    def _create_job(
        self,
        model_instance,
        replica_index: int,
        *,
        kind: str = "model_serving",
        event_uuid: str | None = None,
        runner_name: str | None = None,
    ) -> FakeJob:
        job_id = self._next_job_id
        self._next_job_id += 1
        self._submit_count += 1
        sk = model_instance.serving_key
        safe = model_instance.name.lower()[:20]
        job_name = f"eval360-{self._instance_id}-{safe}-{sk[:12]}-r{replica_index}"
        job = FakeJob(
            job_id=job_id,
            job_name=job_name,
            model_name=model_instance.name,
            serving_key=sk,
            replica_index=replica_index,
            kind=kind,
            event_uuid=event_uuid,
            runner_name=runner_name,
        )
        self._jobs[job_id] = job
        self._serving_key_registry[sk] = model_instance.name
        return job

    async def kill_all(self, model_names):
        model_name_set = set(model_names)
        for job in self._jobs.values():
            if (
                job.kind == "model_serving"
                and job.model_name in model_name_set
                and job.state in ("PENDING", "RUNNING")
            ):
                job.state = "CANCELLED"
                job.cancellation_intent = "scheduler_release"
                job.signal = 15

    async def cancel_all_owned_jobs(self):
        for job in self._jobs.values():
            if job.state in ("PENDING", "RUNNING"):
                job.state = "CANCELLED"
                job.cancellation_intent = "controller_interrupt"
                job.signal = 15

    async def cancel_job(
        self,
        job_id: int,
        *,
        cancellation_intent: str | None = None,
    ):
        if job_id in self._jobs:
            self._jobs[job_id].state = "CANCELLED"
            self._jobs[job_id].cancellation_intent = cancellation_intent
            self._jobs[job_id].signal = 15

    async def get_available_nodes(self):
        return 8

    async def is_url_healthy(self, url: str) -> bool:
        return url == self._vllm_url and bool(self._healthy_sks)

    # ──────────────────────────────────────────────────────────────
    # Imported dataset stubs
    # ──────────────────────────────────────────────────────────────

    async def submit_imported_dataset_job(
        self, event_uuid, model_instance, runner_name,
        setup_script, benchmark_script, output_dir,
    ) -> int:
        job = self._create_job(
            model_instance,
            0,
            kind="imported_dataset",
            event_uuid=event_uuid,
            runner_name=runner_name,
        )
        return job.job_id

    def get_submitted_jobs(self) -> tuple[SubmittedSlurmJob, ...]:
        """Return the fake ledger through the production evidence interface."""
        return tuple(
            SubmittedSlurmJob(
                job_id=job.job_id,
                job_name=job.job_name,
                kind=job.kind,
                submission_origin=job.submission_origin,
                model_name=job.model_name,
                serving_key=job.serving_key,
                event_uuid=job.event_uuid,
                runner_name=job.runner_name,
                cancellation_intent=job.cancellation_intent,
                completed_event_roles=tuple(sorted(job.completed_event_roles)),
            )
            for job in sorted(self._jobs.values(), key=lambda item: item.job_id)
        )

    async def wait_for_terminal_job_outcomes(
        self,
        job_ids: list[int] | tuple[int, ...],
    ) -> dict[int, SlurmJobOutcome]:
        """Return authoritative-looking fake root accounting rows."""
        outcomes: dict[int, SlurmJobOutcome] = {}
        for job_id in sorted(set(job_ids)):
            job = self._jobs.get(job_id)
            if job is None:
                raise RuntimeError(
                    f"sacct returned no root accounting row for required Slurm job(s): {job_id}"
                )
            if job.state in ("PENDING", "RUNNING"):
                raise RuntimeError(
                    f"required fake Slurm job {job_id} is not terminal"
                )
            outcomes[job_id] = SlurmJobOutcome(
                job_id=job_id,
                job_id_raw=str(job_id),
                job_name=job.job_name,
                raw_state=job.state,
                state=job.state,
                exit_code=job.exit_code,
                signal=job.signal,
                reason=job.reason,
            )
        return outcomes

    async def get_job_node(self, job_id: int, poll_interval: float = 0.05) -> str:
        """Mirror `SlurmManager.get_job_node`, including how it ENDS.

        THIS USED TO LOOP FOREVER on a job that was gone — an unknown id or a
        cancelled/preempted one — because it only ever returned on RUNNING. Real
        `squeue` shows queued jobs only, so both of those are "no row for this
        job", and the real method raises `SlurmJobVanished` for them. A fake that
        instead polls on cannot be used to test what the scheduler does when a
        job disappears, which is the one thing about this method that has a
        blast radius.
        """
        while True:
            job = self._jobs.get(job_id)
            if job is None:
                raise SlurmJobVanished(
                    f"Slurm job {job_id} is no longer queued: "
                    "slurm_load_jobs error: Invalid job id specified"
                )
            if job.state == "RUNNING":
                return job.node
            if job.state != "PENDING":
                # Terminal: squeue would return zero rows for it.
                raise SlurmJobVanished(
                    f"Slurm job {job_id} is no longer queued; it left the queue "
                    "before a node was allocated"
                )
            await asyncio.sleep(poll_interval)
            self._tick()

    async def wait_for_vllm_health(
        self, node: str, max_time_to_deploy: int, poll_interval: float = 0.05
    ) -> bool:
        deadline = time.monotonic() + max_time_to_deploy
        while time.monotonic() < deadline:
            for job in self._jobs.values():
                if (job.node == node and job.state == "RUNNING"
                        and job.serving_key in self._healthy_sks):
                    return True
            await asyncio.sleep(poll_interval)
            self._tick()
        return False

    async def wait_for_job_completion(self, job_id: int, poll_interval: float = 0.05):
        while True:
            job = self._jobs.get(job_id)
            if not job or job.state not in ("PENDING", "RUNNING"):
                return
            await asyncio.sleep(poll_interval)

    # ──────────────────────────────────────────────────────────────
    # Test control API (not part of SlurmManager interface)
    # ──────────────────────────────────────────────────────────────

    def vanish(self, job_id: int):
        """Make ONE job leave the queue, as a preemption or an operator scancel would.

        `preempt` selects by model name, which cannot express "this imported
        job disappeared while that model's serving job carried on" — and that
        is exactly the situation in which a mishandled disappearance takes the
        other job down with it.
        """
        job = self._jobs[job_id]
        job.state = "CANCELLED"
        job.signal = 9

    def preempt(self, model_name: str):
        """Simulate Slurm preemption — jobs disappear from squeue."""
        for job in self._jobs.values():
            if job.model_name == model_name and job.state in ("PENDING", "RUNNING"):
                job.state = "CANCELLED"
                job.signal = 9

    def set_healthy(self, serving_key: str, healthy: bool = True):
        """Directly control whether a serving key's VLLM is considered healthy."""
        if healthy:
            self._healthy_sks.add(serving_key)
        else:
            self._healthy_sks.discard(serving_key)

    def get_jobs(self, model_name: str | None = None) -> list[FakeJob]:
        """Return all jobs, optionally filtered by model name."""
        jobs = list(self._jobs.values())
        if model_name is not None:
            jobs = [j for j in jobs if j.model_name == model_name]
        return jobs

    def get_active_job_count(self, model_name: str | None = None) -> int:
        """Count PENDING + RUNNING jobs."""
        return sum(
            1 for j in self.get_jobs(model_name)
            if j.state in ("PENDING", "RUNNING")
        )
