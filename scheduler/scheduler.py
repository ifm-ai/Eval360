import errno
import hashlib
import logging
import os
logger = logging.getLogger("Scheduler")
_EVALUATE_NOW_QUEUE_POLL_TIMEOUT = 1.0
_EVALUATE_NOW_FINAL_JOB_UPDATE_DELAY = 0.1

from .job import JobManager
from .slurm_manager import SlurmJobVanished, SlurmManager
from .database import DatabaseManager
from .event import EventManager, EventInstance
from .generation import GenerationManager
from .grading import GradingManager
from .grader import Grade, Score, get_grader
from .imported_dataset import get_runner
from .event import GradingEventInstance, ImportedDatasetEventInstance
from .filesystem_manager import FSManager, DirType
from .progress import ProgressManager
from .eval_config import EvalConfigParser, LoadedEvalConfig, LoadedModelSpec, LoadedTask
from .model import (
    CacheSaltConfig,
    ExternalRetryPolicy,
    ModelInstance,
    ModelParser,
    validate_external_total_deadline_override,
)
from .task import Task, ImportedDatasetTask, AsyncGenerationTask
from .evaluation_request import EvaluationRequest, SuiteTask
from .terminal_result import (
    FileBinding,
    TERMINAL_RESULT_CONTRACT,
    TERMINAL_RESULT_SCHEMA_VERSION,
    YAML_TERMINAL_RESULT_CONTRACT,
    YAML_TERMINAL_RESULT_SCHEMA_VERSION,
    YamlTerminalInvocation,
    bind_regular_file,
    canonical_json,
)
from . import openai_interface
from .utils import dataset_iterator, Sentinel, ExceptionWrapper, is_hf_uri, resolve_hf_path, count_jsonl_records, load_completed_row_ids, normalize_eval_input_record, expand_data_path
import asyncio
import copy
import concurrent.futures
import json
import pathlib
import yaml
from collections import defaultdict
from dataclasses import dataclass
from watchdog import events


def _attach_structured_error(
    instance: dict,
    error: ExceptionWrapper,
) -> None:
    """Persist stable external failure evidence without changing legacy fields."""
    if error.error_code is None:
        return
    metadata = {
        "code": error.error_code,
        "attempts": error.attempts,
        "elapsed_seconds": error.elapsed_seconds,
        "retriable": error.retriable,
        "http_status": error.http_status,
    }
    instance["eval360_error"] = {
        key: value
        for key, value in metadata.items()
        if value is not None
    }


@dataclass
class _ExternalRegistrationRollback:
    """Undo only process-scoped resources created by one registration call."""

    serving_key: str | None = None
    pool: openai_interface.ModelConnectionPool | None = None
    endpoint: str | None = None
    remove_pool: bool = False
    remove_endpoint: bool = False
    quota_identity: str | None = None
    rate_limiter: object | None = None
    remove_rate_limiter: bool = False
    _rolled_back: bool = False

    async def rollback(self) -> None:
        if self._rolled_back:
            return
        self._rolled_back = True

        if (
            self.remove_endpoint
            and self.pool is not None
            and self.endpoint is not None
        ):
            await self.pool.remove_url(self.endpoint)

        if (
            self.remove_pool
            and self.serving_key is not None
            and openai_interface.LOCKED_CONNECTIONS.get(self.serving_key)
            is self.pool
            and self.pool is not None
            and not self.pool.urls
        ):
            openai_interface.LOCKED_CONNECTIONS.pop(self.serving_key, None)

        if (
            self.remove_rate_limiter
            and self.quota_identity is not None
            and self.quota_identity in openai_interface.RATE_LIMITERS
            and openai_interface.RATE_LIMITERS[self.quota_identity]
            is self.rate_limiter
        ):
            openai_interface.RATE_LIMITERS.pop(self.quota_identity, None)


@dataclass(frozen=True)
class _SelectedEvent:
    """Invocation-scoped event plus its evaluator-owned task closure."""

    event: EventInstance
    suite_task: SuiteTask | None
    dataset_bindings: tuple[FileBinding, ...]
    task_closure_sha256: str
    generation_required: bool
    grading_required: bool
    aggregation_required: bool
    candidate_serving_required: bool
    grader_serving_required: bool
    imported_job_required: bool
    model_config_binding: FileBinding | None = None
    data_config_binding: FileBinding | None = None
    eval_config_binding: FileBinding | None = None
    eval_group_name: str | None = None
    model_definition_sha256: str | None = None


class Scheduler:
    def __init__(self, model_directory, dataset_directory, max_generation_jobs=10, max_grading_parallelism=10, log_dir=".", hf_cache_dir=None, force_logprobs=False, debug=False, slurm_partition=None, instance_id=None, no_slurm=False, salt_cache=False, external_total_deadline_seconds=None):
        self._no_slurm = no_slurm
        self.task_queue = asyncio.Queue()
        self.job_manager = JobManager()
        self.slurm_manager = SlurmManager(log_dir=log_dir, partition=slurm_partition, instance_id=instance_id) if not no_slurm else None
        self.db_manager = DatabaseManager()
        self.event_manager = EventManager(self.db_manager)
        self.filesystem_manager = FSManager(model_directory, dataset_directory)
        self.model_parser = ModelParser()
        self.generation_manager = GenerationManager()
        self.grading_manager = GradingManager()
        self.progress_manager = ProgressManager(db_manager=self.db_manager)
        self._queue = asyncio.Queue()
        self._event_lock = asyncio.Lock()
        self._job_lock = asyncio.Lock()
        self._registration_lock = asyncio.Lock()
        # Durable model rows survive a process restart, whereas external pools
        # and rate limiters do not.  Track the registrations this scheduler has
        # actually installed so only its own live external resources are
        # immutable.
        self._owned_external_registrations: set[str] = set()
        self._file_mutex = defaultdict(asyncio.Lock)
        self._max_generation_jobs = max_generation_jobs
        self._max_grading_parallelism = max_grading_parallelism
        self._grading_semaphore = (
            asyncio.Semaphore(value=max_grading_parallelism)
            if max_grading_parallelism is not None
            else None
        )
        self._active_imported_dataset_jobs = 0
        self._flush_frequency = 100000 # in characters
        self._hf_cache_dir = hf_cache_dir
        self._force = False
        self._force_logprobs = force_logprobs
        self._debug = debug
        self._salt_cache = salt_cache
        self._external_total_deadline_seconds = (
            validate_external_total_deadline_override(
                external_total_deadline_seconds
            )
            if external_total_deadline_seconds is not None
            else None
        )
        self._openai_connections: dict = {}  # event_instance → OpenAIConnection
        self._file_flush_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._evaluation_request: EvaluationRequest | None = None
        self._terminal_evidence_enabled = False
        self._bound_dataset_paths: dict[str, tuple[str, ...]] = {}
        self._request_event_metadata: dict[str, dict[str, str]] = {}
        self._yaml_event_metadata: dict[str, dict[str, object]] = {}

    def _parse_model_spec(self, path, *, eval_mode=False):
        return ModelParser.parse_yaml(
            path,
            eval_mode=eval_mode,
            external_total_deadline_seconds=(
                self._external_total_deadline_seconds
            ),
        )

    def _get_file_flush_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._file_flush_executor is None:
            self._file_flush_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="eval360-file-flush",
            )
        return self._file_flush_executor

    async def _flush_file(self, file_obj):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._get_file_flush_executor(), file_obj.flush)

    def _shutdown_file_flush_executor(self) -> None:
        if self._file_flush_executor is not None:
            self._file_flush_executor.shutdown(wait=True)
            self._file_flush_executor = None

    def _cache_salt_metadata(self, model_instance: ModelInstance) -> dict:
        cache_salt = model_instance.cache_salt
        if cache_salt is None:
            raise ValueError("cache_salt must be a CacheSaltConfig; use mode='disabled'")
        mode = cache_salt.mode
        enabled = mode != "disabled"
        metadata = {
            "enabled": enabled,
            "mode": mode,
            "source": "cli" if enabled and self._salt_cache else ("model_config" if enabled else None),
            "provider_field": "extra_body.cache_salt",
        }
        if mode == "static":
            metadata["value"] = "<redacted>"
            metadata["value_sha256_12"] = hashlib.sha256(cache_salt.salt.encode()).hexdigest()[:12]
        elif mode == "unique":
            metadata["value"] = "<per-request unique>"
        return metadata

    def _run_metadata_path(self, event_instance: EventInstance, task: AsyncGenerationTask) -> pathlib.Path:
        return pathlib.Path(event_instance.path_to_scores).with_name(
            f"{task.dataset_name}_run_metadata.yaml"
        )

    def _run_metadata(
        self,
        model_instance: ModelInstance,
        task: AsyncGenerationTask,
        event_instance: EventInstance | None = None,
    ) -> dict:
        metadata = {
            "model": model_instance.name,
            "dataset": task.dataset_name,
            "cache_salt": self._cache_salt_metadata(model_instance),
        }
        request_metadata = (
            self._request_event_metadata.get(event_instance.uuid)
            if event_instance is not None
            else None
        )
        if request_metadata is not None:
            metadata["evaluation_request"] = request_metadata
        yaml_metadata = (
            self._yaml_event_metadata.get(event_instance.uuid)
            if event_instance is not None
            else None
        )
        if yaml_metadata is not None:
            metadata["evaluation_yaml"] = yaml_metadata
        return metadata

    def _validate_run_metadata_compatible(
        self,
        event_instance: EventInstance,
        model_instance: ModelInstance,
        task: AsyncGenerationTask,
        completed_rows: set[int],
    ) -> None:
        if not completed_rows:
            return

        metadata_path = self._run_metadata_path(event_instance, task)
        current_metadata = self._run_metadata(
            model_instance,
            task,
            event_instance,
        )
        current_cache_salt = current_metadata["cache_salt"]
        if not metadata_path.exists():
            if current_cache_salt["enabled"]:
                raise ValueError(
                    f"{event_instance.path_to_generations} contains existing generations "
                    "without run metadata; rerun with --force before enabling cache_salt"
                )
            return

        try:
            previous_metadata = yaml.safe_load(metadata_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Could not read run metadata at {metadata_path}; rerun with --force"
            ) from exc
        previous_cache_salt = previous_metadata.get("cache_salt")
        if previous_cache_salt != current_cache_salt:
            raise ValueError(
                f"Existing generations in {event_instance.path_to_generations} were "
                "created with different cache_salt metadata; rerun with --force"
            )
        if "evaluation_request" in current_metadata and previous_metadata.get(
            "evaluation_request"
        ) != current_metadata["evaluation_request"]:
            raise ValueError(
                f"Existing generations in {event_instance.path_to_generations} were "
                "created for a different evaluation request; rerun with --force"
            )
        if "evaluation_yaml" in current_metadata and previous_metadata.get(
            "evaluation_yaml"
        ) != current_metadata["evaluation_yaml"]:
            raise ValueError(
                f"Existing generations in {event_instance.path_to_generations} were "
                "created for different evaluation YAML inputs; rerun with --force"
            )

    def _write_run_metadata(self, event_instance: EventInstance, model_instance: ModelInstance, task: AsyncGenerationTask) -> None:
        metadata_path = self._run_metadata_path(event_instance, task)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._run_metadata(model_instance, task, event_instance)
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))

    def _validate_request_resume_binding(
        self,
        event_instance: GradingEventInstance,
        task: AsyncGenerationTask,
        *,
        force: bool,
    ) -> None:
        """Reject output reuse unless it was created by this exact request/task."""
        if force or event_instance.uuid not in self._request_event_metadata:
            return
        metadata_path = self._run_metadata_path(event_instance, task)
        artifacts = (
            pathlib.Path(event_instance.path_to_generations),
            pathlib.Path(event_instance.path_to_grades),
            pathlib.Path(event_instance.path_to_scores),
            metadata_path,
        )
        if not any(path.exists() for path in artifacts):
            return
        if not metadata_path.is_file():
            raise ValueError(
                "existing evaluation outputs have no request binding; rerun with --force"
            )
        try:
            previous = yaml.safe_load(metadata_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Could not read run metadata at {metadata_path}; rerun with --force"
            ) from exc
        if previous.get("evaluation_request") != self._request_event_metadata[
            event_instance.uuid
        ]:
            raise ValueError(
                "existing evaluation outputs belong to a different evaluation "
                "request; rerun with --force"
            )

    def _validate_yaml_resume_binding(
        self,
        event_instance: GradingEventInstance,
        task: AsyncGenerationTask,
        *,
        force: bool,
    ) -> None:
        """Reject reuse unless outputs bind to this exact parsed YAML event."""
        if force or event_instance.uuid not in self._yaml_event_metadata:
            return
        metadata_path = self._run_metadata_path(event_instance, task)
        artifacts = (
            pathlib.Path(event_instance.path_to_generations),
            pathlib.Path(event_instance.path_to_grades),
            pathlib.Path(event_instance.path_to_scores),
            metadata_path,
        )
        if not any(path.exists() for path in artifacts):
            return
        if not metadata_path.is_file():
            raise ValueError(
                "existing evaluation outputs have no YAML binding; rerun with --force"
            )
        try:
            previous = yaml.safe_load(metadata_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Could not read run metadata at {metadata_path}; rerun with --force"
            ) from exc
        if previous.get("evaluation_yaml") != self._yaml_event_metadata[
            event_instance.uuid
        ]:
            raise ValueError(
                "existing evaluation outputs belong to different evaluation "
                "YAML inputs; rerun with --force"
            )

    async def _bound_input_iterator(
        self,
        task: AsyncGenerationTask,
        completed_rows: set[int],
    ):
        """Read an evidence invocation's exact ordered dataset-file closure."""
        row_index = 0
        for path in self._bound_dataset_paths[task.uuid]:
            async for record in dataset_iterator(path, logger, read_only=True):
                record = normalize_eval_input_record(record, row_index)
                row_index += 1
                if record.get("row") in completed_rows:
                    continue
                yield record
        yield Sentinel.COMPLETED

    async def _record_serving_role_completion(
        self,
        event_instance: GradingEventInstance,
        *,
        is_grading: bool,
    ) -> None:
        """Record which active Slurm serving children completed one role."""
        if not self._terminal_evidence_enabled or self._no_slurm:
            return
        if is_grading:
            task = self.db_manager.get_task(event_instance.task_uuid)
            model = task.grader.llm_as_judge
            if model is None or model.is_external:
                return
            role = "grading"
        else:
            model = self.db_manager.get_model(event_instance.model)
            if model.is_external:
                return
            role = "generation"
        await self.slurm_manager.record_role_completion(
            model.serving_key,
            event_instance.uuid,
            role,
        )

    async def handle_update(self, source_name, update):
        if source_name == "fs":
            await self.handle_fs_update(update)
        # TODO: remove if db source not needed
        # elif source_name == "db":
        elif source_name == "job":
            await self.handle_job_update(update)
        elif source_name == "event":
            await self.handle_event(update)
        elif source_name == "generation":
            event_instance, aiter = update
            await self.handle_generation(event_instance, aiter)
        elif source_name == "grading":
            logger.info(f"found grading update: {update}")
            event_instance, aiter = update
            await self.handle_grading(event_instance, aiter)
        else:
            raise RuntimeError(f"Invalid source: {source_name}")

    async def handle_fs_update(self, update):
        dirtype, event, info = update
        # TODO: a single file update could be caught by both spec and instance watchers
        if dirtype == DirType.MODELSPEC:
            if (isinstance(event, events.FileModifiedEvent) or
                    isinstance(event, events.FileCreatedEvent)):
                try:
                    logger.info(pathlib.Path(event.src_path).suffix)
                    if pathlib.Path(event.src_path).suffix in [".yaml", ".json"]:
                        model_spec = self._parse_model_spec(event.src_path)
                        await self.register_model_spec(model_spec)
                except Exception as e:
                    logger.exception(f"Encountered error: {e}")
        if dirtype == DirType.MODELINSTANCE:
            if isinstance(event, str):
                try:
                    path = pathlib.Path(event)
                    if path.stem == "done":
                        model_instance = ModelParser.model_instance_from_path(
                            path=path, model_spec=info)
                        await self.register_model_instance(model_instance)
                except Exception as e:
                    logger.exception(f"Encountered error: {e}")
                    raise e
            elif (isinstance(event, events.FileModifiedEvent) or
                    isinstance(event, events.FileCreatedEvent)):
                try:
                    path = pathlib.Path(event.src_path)
                    if path.stem == "done":
                        model_instance = ModelParser.model_instance_from_path(
                            path=path, model_spec=info)
                        await self.register_model_instance(model_instance)
                except Exception as e:
                    logger.exception(f"Encountered error: {e}")
                    raise e
        if dirtype == DirType.DATA:
            if (isinstance(event, events.FileModifiedEvent) or
                    isinstance(event, events.FileCreatedEvent)):
                logger.info(f"Encountered {event}")
                try:
                    # TODO POC: implement done properly
                    # if path.stem == "done":
                    if pathlib.Path(event.src_path).suffix in [".yaml", ".json"]:
                        task = Task.parse_yaml(event.src_path)
                        await self.register_task(task)
                except Exception as e:
                    logger.exception(f"Encountered error: {e}")
                logger.info(f"received data update: {update}")

    async def register_model_spec(self, model_spec):
        if model_spec.local_model is not None:
            await self.filesystem_manager.ensures_exists_register_path(
                model_spec.local_model.path_glob, DirType.MODELINSTANCE, register_existing=model_spec.local_model.enqueue_existing, info=model_spec)
            self.db_manager.register_model_family(model_spec)
        elif model_spec.remote_model is not None:
            model_instance = ModelParser.model_instance_from_path(
                path=model_spec.remote_model.path, model_spec=model_spec)
            await self.register_model_instance(model_instance)
        elif model_spec.external_model is not None:
            model_instance = ModelParser.model_instance_from_path(
                path=model_spec.external_model.base_url, model_spec=model_spec)
            await self.register_model_instance(model_instance)
        logger.info(f"registering model spec: {model_spec}")

    async def _prepare_external_registration(
        self,
        model_instance: ModelInstance,
    ) -> _ExternalRegistrationRollback:
        """Validate and eagerly install one immutable external resource policy."""
        rollback = _ExternalRegistrationRollback()
        if not getattr(model_instance, "is_external", False):
            return rollback

        serving_key = model_instance.serving_key
        openai_interface.validate_connection_pool_registration(
            serving_key,
            model_instance.max_simultaneous_requests,
        )
        openai_interface.validate_external_rate_limiter_registration(
            model_instance
        )

        # Both checks above are pure. Mutate the process-scoped registries only
        # after every shared policy has passed, while _registration_lock is held.
        endpoint = (
            openai_interface.validate_and_canonicalize_external_endpoint(
                model_instance.base_url
            )
        )
        quota_identity = openai_interface.external_quota_identity(
            model_instance
        )
        pool_existed = serving_key in openai_interface.LOCKED_CONNECTIONS
        limiter_existed = quota_identity in openai_interface.RATE_LIMITERS

        try:
            pool = openai_interface.get_or_create_connection_pool(
                serving_key,
                model_instance.max_simultaneous_requests,
            )
            rollback.serving_key = serving_key
            rollback.pool = pool
            rollback.remove_pool = not pool_existed

            rate_limiter = openai_interface.get_external_rate_limiter(
                model_instance
            )
            rollback.quota_identity = quota_identity
            rollback.rate_limiter = rate_limiter
            rollback.remove_rate_limiter = not limiter_existed

            rollback.endpoint = endpoint
            rollback.remove_endpoint = endpoint not in pool.urls
            await pool.add_url(endpoint)
        except BaseException:
            await rollback.rollback()
            raise

        return rollback

    @staticmethod
    def _external_runtime_policy(
        model_instance: ModelInstance,
    ) -> dict[str, object]:
        """Return the complete secret-safe immutable registration state."""
        return model_instance.registration_snapshot()

    def _validate_immutable_external_registration(
        self,
        model_instance: ModelInstance,
    ) -> None:
        """Reject drift only for an external resource owned by this process."""
        if model_instance.name not in self._owned_external_registrations:
            return
        existing = self.db_manager.get_model_if_exists(model_instance.name)
        if existing is None:
            return
        if not existing.is_external and not model_instance.is_external:
            return

        hydrated_policy = self._external_runtime_policy(existing)
        persisted_policy = (
            self.db_manager.get_model_registration_snapshot(
                model_instance.name
            )
        )
        # Snapshots written by an older release may not contain fields added
        # later. Fill only those missing keys from the migrated DB row, while
        # retaining the original values for every field it did capture.
        existing_policy = {
            field: (
                persisted_policy[field]
                if persisted_policy is not None
                and field in persisted_policy
                else value
            )
            for field, value in hydrated_policy.items()
        }
        requested_policy = self._external_runtime_policy(model_instance)
        changed_fields = sorted(
            field
            for field, existing_value in existing_policy.items()
            if existing_value != requested_policy[field]
        )
        if changed_fields:
            raise ValueError(
                "immutable external registration for model "
                f"{model_instance.name!r} cannot change until scheduler "
                f"restart: {', '.join(changed_fields)}"
            )

    async def register_model_instance(self, model_instance):
        assert (isinstance(model_instance, ModelInstance))
        if (
            model_instance.is_external
            and self._external_total_deadline_seconds is not None
        ):
            policy = model_instance.external_retry_policy.model_dump()
            policy["total_deadline_seconds"] = (
                self._external_total_deadline_seconds
            )
            model_instance = model_instance.model_copy(
                update={
                    "external_retry_policy": (
                        ExternalRetryPolicy.model_validate(policy)
                    )
                }
            )
        async with self._registration_lock:
            self._validate_immutable_external_registration(model_instance)
            rollback = await self._prepare_external_registration(
                model_instance
            )
            pending_events = []
            try:
                with (
                    self.db_manager.transaction(),
                    self.event_manager.defer_enqueues() as pending_events,
                ):
                    self.db_manager.register_model(model_instance)
                    logger.info(
                        f"registering model instance: {model_instance}"
                    )
                    await self.event_manager.create_events_for_new_model(
                        model_instance
                    )
            except BaseException:
                await rollback.rollback()
                raise

            if model_instance.is_external:
                self._owned_external_registrations.add(model_instance.name)
            else:
                self._owned_external_registrations.discard(model_instance.name)

            for event_instance in pending_events:
                self.event_manager.enqueue_nowait(event_instance)

    async def register_task(self, task: "ImportedDatasetTask | AsyncGenerationTask"):
        if isinstance(task, ImportedDatasetTask):
            await self._register_imported_dataset_task(task)
        else:
            await self._register_standard_task(task)

    async def _register_imported_dataset_task(self, task: ImportedDatasetTask):
        async with self._registration_lock:
            pending_events = []
            with (
                self.db_manager.transaction(),
                self.event_manager.defer_enqueues() as pending_events,
            ):
                self.db_manager.register_task(task)
                logger.info(f"registering imported dataset task: {task}")
                await self.event_manager.create_events_for_new_tasks(
                    [task]
                )

            for event_instance in pending_events:
                self.event_manager.enqueue_nowait(event_instance)

    async def _register_standard_task(self, task: AsyncGenerationTask):
        async with self._registration_lock:
            task = self._resolve_hf_task_for_registration(task)
            if task.grader and task.grader.llm_as_judge is not None:
                judge = task.grader.llm_as_judge
                if self._salt_cache:
                    judge = judge.model_copy(
                        update={
                            "cache_salt": CacheSaltConfig(mode="unique")
                        }
                    )
                    task.grader.llm_as_judge = judge
            else:
                judge = None

            if judge is not None:
                self._validate_immutable_external_registration(judge)
            rollback = (
                await self._prepare_external_registration(judge)
                if judge is not None
                else _ExternalRegistrationRollback()
            )
            pending_events = []
            try:
                with (
                    self.db_manager.transaction(),
                    self.event_manager.defer_enqueues() as pending_events,
                ):
                    if judge is not None:
                        self.db_manager.register_model(
                            judge,
                            grading_only=True,
                        )
                    self.db_manager.register_task(task)
                    logger.info(f"registering task: {task}")
                    await self.event_manager.create_events_for_new_tasks(
                        [task]
                    )
            except BaseException:
                await rollback.rollback()
                raise

            if judge is not None:
                if judge.is_external:
                    self._owned_external_registrations.add(judge.name)
                else:
                    self._owned_external_registrations.discard(judge.name)

            for event_instance in pending_events:
                self.event_manager.enqueue_nowait(event_instance)

    def _resolve_hf_task_for_registration(
        self,
        task: AsyncGenerationTask,
    ) -> AsyncGenerationTask:
        """Resolve and validate HF cardinality before publishing task state."""
        if not is_hf_uri(task.data_path):
            return task

        # On restart, retain a count already committed by an earlier run and
        # avoid a redundant download for an auto-counted task.
        if task.num_generations is None:
            db_task = self.db_manager.get_task(task.uuid)
            if db_task is not None and db_task.num_generations is not None:
                logger.info(
                    "task %s already has num_generations=%d from a "
                    "previous run, skipping HF download",
                    task.uuid,
                    db_task.num_generations,
                )
                return task.model_copy(
                    update={
                        "num_generations": db_task.num_generations,
                    }
                )

        local_path = resolve_hf_path(
            task.data_path,
            cache_dir=self._hf_cache_dir,
        )
        count = count_jsonl_records(local_path)
        if task.num_generations is None:
            logger.info(
                "auto-counted %d records from %s",
                count,
                task.data_path,
            )
            return task.model_copy(
                update={"num_generations": count},
            )
        if task.num_generations != count:
            raise ValueError(
                f"num_generations={task.num_generations} does not match "
                f"the actual record count of {count} in {task.data_path!r}"
            )
        return task

    async def _resolve_incomplete_imported_dataset_event(
        self,
        event_instance: ImportedDatasetEventInstance,
        job_id: int,
        output_dir: pathlib.Path,
    ) -> None:
        """Decide the outcome of an imported-dataset job that left no results.

        The three branches are the ones that have always lived at the end of
        `handle_imported_dataset_event`; they are here so that the two ways of
        arriving at "this job produced no `.job_complete`" — it ended, or it
        vanished from the queue before it was ever allocated a node — reach one
        set of outcome semantics instead of two that can drift apart.

        Requeue is the branch with positive evidence behind it: the job wrote
        `.setup_complete_job` and then stopped without writing `.job_failed`,
        which is what a preemption mid-benchmark looks like. Everything else
        fails the event, which is also what a vanished job gets when it left
        nothing behind — the alternative, requeueing on no evidence, would
        resubmit a permanently unschedulable job forever, and there is no retry
        budget here to stop it.
        """
        if not (output_dir / ".setup_complete_job").exists():
            logger.error(f"Imported dataset job {job_id} failed during setup — marking event failed")
            self.progress_manager.set_status(event_instance, "Failed")
            await self.event_manager.register_processing_event(event_instance, -1)
        elif (output_dir / ".job_failed").exists():
            logger.error(f"Imported dataset job {job_id} failed internally — marking event failed")
            self.progress_manager.set_status(event_instance, "Failed")
            await self.event_manager.register_processing_event(event_instance, -1)
        else:
            logger.warning(f"Imported dataset job {job_id} was preempted — rescheduling {event_instance}")
            self.progress_manager.set_status(event_instance, "Queued")
            await self.event_manager.enqueue(event_instance)

    async def handle_imported_dataset_event(self, event_instance: ImportedDatasetEventInstance):
        task = self.db_manager.get_task(event_instance.task_uuid)
        model_instance = self.db_manager.get_model(event_instance.model)

        runner_name = task.imported_dataset.name
        runner = get_runner(runner_name)()

        output_dir = (pathlib.Path(event_instance.path_to_scores).parent / f"{task.dataset_name}_output").resolve()

        if self._force:
            for sentinel in (".job_complete", ".job_failed", ".setup_complete_job"):
                (output_dir / sentinel).unlink(missing_ok=True)
            logger.info(f"--force: cleared sentinels in {output_dir}")

        # Resumption: job ran to completion on a previous run — parse results and mark complete.
        if (output_dir / ".job_complete").exists():
            scores = runner.parse_results(output_dir, task, model_instance)
            parent_dir = pathlib.Path(event_instance.path_to_scores).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            with open(event_instance.path_to_scores, "w") as f:
                for score in scores:
                    f.write(f'"{score.name}": {score.value}\n')
            self.progress_manager.set_status(event_instance, "Complete")
            await self.event_manager.register_processing_event(event_instance, 2)
            return
        repo_root = pathlib.Path(__file__).resolve().parent.parent

        setup_script = runner.build_setup_script(repo_root)
        benchmark_script = runner.build_benchmark_script(
            model_instance, task, output_dir
        )

        self.progress_manager.set_status(event_instance, "Running")

        job_id = await self.slurm_manager.submit_imported_dataset_job(
            event_instance.uuid,
            model_instance,
            runner_name,
            setup_script,
            benchmark_script,
            output_dir,
        )
        self._active_imported_dataset_jobs += 1

        try:
            try:
                node = await self.slurm_manager.get_job_node(job_id)
            except SlurmJobVanished as vanished:
                # THE JOB DISAPPEARING IS ORDINARY, AND IT IS THIS EVENT'S
                # PROBLEM ALONE. Preemption, an operator scancel, or a job that
                # starts and ends between two 10s polls all land here. Letting
                # the raise out would leave `handle_update`'s task to fail,
                # which takes down the main loop's TaskGroup and makes it run
                # `cancel_all_owned_jobs` — cancelling the Slurm jobs of every
                # OTHER evaluation in flight. One vanished imported job must not
                # do that.
                #
                # The outcome is decided by the same sentinel triage a job that
                # ended without `.job_complete` gets, rather than by a rule
                # invented here: usually nothing was written, so this fails the
                # event; if the job did get far enough to write
                # `.setup_complete_job` before vanishing, it requeues, exactly
                # as a preemption observed later would.
                logger.warning(
                    f"Imported dataset job {job_id} left the queue before a node "
                    f"was allocated ({vanished}) — deciding this event's outcome "
                    f"from its sentinels"
                )
                await self._resolve_incomplete_imported_dataset_event(
                    event_instance, job_id, output_dir
                )
                return

            healthy = await self.slurm_manager.wait_for_vllm_health(node, model_instance.max_time_to_deploy)
            if not healthy:
                logger.error(f"VLLM failed to start for imported dataset job {job_id} on {node} — marking event failed")
                self.progress_manager.set_status(event_instance, "Failed")
                await self.event_manager.register_processing_event(event_instance, -1)
                return

            await self.slurm_manager.wait_for_job_completion(job_id)

            if not (output_dir / ".job_complete").exists():
                await self._resolve_incomplete_imported_dataset_event(
                    event_instance, job_id, output_dir
                )
                return

            scores = runner.parse_results(output_dir, task, model_instance)

            parent_dir = pathlib.Path(event_instance.path_to_scores).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            with open(event_instance.path_to_scores, "w") as f:
                for score in scores:
                    f.write(f'"{score.name}": {score.value}\n')

            self.progress_manager.set_status(event_instance, "Complete")
            await self.event_manager.register_processing_event(event_instance, 2)
        finally:
            self._active_imported_dataset_jobs -= 1
            if not (output_dir / ".job_complete").exists():
                await self.slurm_manager.cancel_job(job_id)

    async def handle_event(self, event_instance: EventInstance):
        if isinstance(event_instance, ImportedDatasetEventInstance):
            await self.handle_imported_dataset_event(event_instance)
            return

        task = self.db_manager.get_task(event_instance.task_uuid)
        model_instance = self.db_manager.get_model(event_instance.model)
        data_path = resolve_hf_path(task.data_path, cache_dir=self._hf_cache_dir) if is_hf_uri(task.data_path) else task.data_path
        self.progress_manager.set_status(event_instance, "Queued")

        if self._force:
            for p in (
                event_instance.path_to_generations,
                event_instance.path_to_grades,
                event_instance.path_to_scores,
                self._run_metadata_path(event_instance, task),
            ):
                pathlib.Path(p).unlink(missing_ok=True)
            logger.info(f"--force: cleared output files for {event_instance}")

        completed_rows = load_completed_row_ids(event_instance.path_to_generations)
        self._validate_run_metadata_compatible(
            event_instance,
            model_instance,
            task,
            completed_rows,
        )

        await asyncio.sleep(1)  # a hacky way to clear dead async tasks
        async with self._event_lock:
            self.event_manager.remove_dead_async_tasks(event_instance)
            active_async_tasks = self.event_manager.get_async_tasks(event_instance)
            for task_type, async_task in active_async_tasks:
                if task_type in ("generation", "grading"):
                    logging.info(f"Task {async_task} type={task_type} for event already exists: exiting")
                    return
            self.event_manager.add_async_task(event_instance, "generation", asyncio.current_task())

        # Request deployment as soon as the event starts so Slurm submission is
        # not delayed by resume/grading work on pre-existing outputs.
        self.event_manager.add_desired_model(event_instance, is_grading=False)
        if not self._no_slurm:
            logger.info(f"Triggering immediate job update for {event_instance.model}")
            await self.handle_job_update(None)

        async def combined_iterator():
            logger.info("entered generation iterator")

            #  Yield all pre-existing results (lazily, in file order — downstream doesn't need sorting)
            lock = self._file_mutex[event_instance.path_to_generations]
            if lock.locked():
                raise RuntimeError(f"Double access to {event_instance.path_to_generations}")
            index = 0
            async with lock:
                async for item in dataset_iterator(event_instance.path_to_generations, logger, read_only=False):
                    self.progress_manager.update(event=event_instance, index=index, completed=max(task.average_over + task.pass_at), new_elems=None, mode="Generation")
                    yield item
                    index += 1
                logger.info(f"found {index} pre-existing completed generations ({len(completed_rows)} unique rows)")

                parent_dir = pathlib.Path(event_instance.path_to_generations).parent
                parent_dir.mkdir(parents=True, exist_ok=True)
                self._write_run_metadata(event_instance, model_instance, task)

                openai_connection = openai_interface.OpenAIConnection(model=model_instance,
                                                                      event_instance=event_instance,
                                                                      task=task,
                                                                      job_manager=self.job_manager,
                                                                      progress_manager=self.progress_manager,
                                                                      new_field_name="generations",
                                                                      force_logprobs=self._force_logprobs,
                                                                      debug=self._debug)
                self._openai_connections[event_instance] = openai_connection
                async def completion_hook():
                    if self._terminal_evidence_enabled:
                        await self._record_serving_role_completion(
                            event_instance,
                            is_grading=False,
                        )
                    self.event_manager.remove_desired_model(event_instance, is_grading=False)
                try:
                    with open(event_instance.path_to_generations, mode="a") as f:
                        count = 0
                        input_iterator = (
                            self._bound_input_iterator(task, completed_rows)
                            if task.uuid in self._bound_dataset_paths
                            else dataset_iterator(
                                data_path,
                                logger,
                                read_only=True,
                                skip_rows=completed_rows,
                                sentinel=True,
                                record_transform=normalize_eval_input_record,
                            )
                        )
                        async for item in openai_connection.launch_requests(
                                input_iterator,
                                offset=index, completion_hook=completion_hook):
                            if isinstance(item, ExceptionWrapper):
                                if (
                                    self._terminal_evidence_enabled
                                    or os.environ.get(
                                        "EVAL360_IGNORE_ERRORS", "true"
                                    )
                                    == "false"
                                ):
                                    raise item.exception
                                logger.warning(
                                    f"Generation error for item (continuing): {item.exception}\n{item.trace}"
                                )
                                instance = copy.deepcopy(item.instance)
                                instance["exception"] = str(item.exception)
                                instance["trace"] = item.trace
                                _attach_structured_error(instance, item)
                                line = json.dumps(instance, separators=(",", ":"))
                                f.write(line + "\n")
                                await self._flush_file(f)
                                yield item
                            elif item != Sentinel.COMPLETED:
                                line = json.dumps(item, separators=(",", ":"))
                                f.write(line + "\n")
                                count += len(line)
                                if count > self._flush_frequency:
                                    await self._flush_file(f)
                                    count = 0
                                yield item
                            elif item == Sentinel.COMPLETED:
                                await self._flush_file(f)
                                self.progress_manager.set_status(event_instance, "Complete")
                                await self.event_manager.register_processing_event(event_instance, 1)
                                yield item
                                return
                except OSError as e:
                    if e.errno in (errno.ENOSPC, errno.EDQUOT):
                        logger.error(
                            f"DISK QUOTA/SPACE EXCEEDED writing to "
                            f"{event_instance.path_to_generations}: {e}. "
                            f"Free up space and restart."
                        )
                    self.event_manager.remove_desired_model(event_instance, is_grading=False)
                    await self.event_manager.register_processing_event(event_instance, -1)
                    raise
                except Exception as e:
                    # TODO POSTPOC: make this only for an openai connection error
                    logger.exception(f"Found exception {e}")
                    self.event_manager.remove_desired_model(event_instance, is_grading=False)
                    # TODO: dont hard exit here
                    raise e
                finally:
                    self._openai_connections.pop(event_instance, None)

        iterator = combined_iterator()
        async def prefetch(aiter):
            queue = asyncio.Queue(maxsize=16)
            done = object()
            failure = object()
 
            async def producer():
                try:
                    async for item in aiter:
                        await queue.put(item)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    await queue.put((failure, exc))
                else:
                    await queue.put(done)


            asyncio_task = asyncio.create_task(producer())

            try:
                while True:
                    item = await queue.get()
                    if item is done:
                        break
                    if isinstance(item, tuple) and len(item) == 2 and item[0] is failure:
                        raise item[1]
                    yield item
            except asyncio.exceptions.CancelledError:
                logger.info("Generation stopped")
                raise
            finally:
                if not asyncio_task.done():
                    asyncio_task.cancel()
                try:
                    await asyncio_task
                except asyncio.CancelledError:
                    pass

        await self.generation_manager.enqueue(event_instance, prefetch(iterator))

    async def handle_generation(self, event_instance, generation_aiter):
        self.event_manager.add_async_task(event_instance, "grading", asyncio.current_task())

        await asyncio.sleep(1)  # a hacky way to clear dead async tasks

        # TODO: factor this combined iterator logic out into its own file
        #       these are two combined iterators are subtly different
        async def combined_iterator():
            #  We need to keep track of the last index of the output file that exists
            #  so that we know where to start reading on the input file
            logger.info("entered grading iterator")
            #  Yield existing grades lazily and build the set of already-graded rows
            async for item in dataset_iterator(event_instance.path_to_grades, logger, read_only=False):
                self.progress_manager.update(event_instance, None, None, 1, mode="Grading")
                yield item

            parent_dir = pathlib.Path(event_instance.path_to_grades).parent
            parent_dir = parent_dir.resolve(strict=False)
            parent_dir.mkdir(parents=True, exist_ok=True)

            lock = self._file_mutex[event_instance.path_to_scores]
            if lock.locked():
                raise RuntimeError(f"Double access to {event_instance.path_to_scores}")
            grader = get_grader(event_instance.grader_type)
            task = self.db_manager.get_task(event_instance.task_uuid)
            async with lock:
                g = grader(generation_aiter, event_manager=self.event_manager, job_manager=self.job_manager,  event=event_instance, task=task)
                existing = dataset_iterator(event_instance.path_to_grades, logger, read_only=False)
                try:
                    with (open(event_instance.path_to_grades, mode="a") as f1,
                          open(event_instance.path_to_scores, mode="w") as f2):
                        count = 0
                        async for item in g.run(existing=existing, average_over=task.average_over, pass_at=task.pass_at):
                            if isinstance(item, Grade):
                                self.progress_manager.update(event_instance, None, None, 1, "Grading")
                                line = json.dumps(item.element, separators=(",", ":"))
                                f1.write(line + "\n")
                                count += len(line)
                                if count > self._flush_frequency:
                                    await self._flush_file(f1)
                                    count = 0
                                yield item
                            elif isinstance(item, Score):
                                f2.write(f"\"{item.name}\": {item.value}\n")
                                await self._flush_file(f2)
                                yield item
                            elif isinstance(item, ExceptionWrapper):
                                if (
                                    self._terminal_evidence_enabled
                                    or os.environ.get(
                                        "EVAL360_IGNORE_ERRORS", "true"
                                    )
                                    == "false"
                                ):
                                    raise item.exception
                                self.progress_manager.update(event_instance, None, None, 1, "Grading")
                                instance = copy.deepcopy(item.instance)
                                instance["exception"] = str(item.exception)
                                instance["trace"] = item.trace
                                _attach_structured_error(instance, item)
                                line = json.dumps(instance, separators=(",", ":"))
                                logger.info(f"about to dump {line}")
                                f1.write(line + "\n")
                                await self._flush_file(f1)
                                yield item
                            elif item == Sentinel.COMPLETED:
                                # we have reached the end of the data: we can now
                                # update the event phase to done grading
                                await self._flush_file(f1)
                                await self._flush_file(f2)
                                self.progress_manager.set_status(event_instance, "Complete", mode="Grading")
                                if self._terminal_evidence_enabled:
                                    await self._record_serving_role_completion(
                                        event_instance,
                                        is_grading=True,
                                    )
                                self.event_manager.remove_desired_model(event_instance, is_grading=True)
                                await self.event_manager.register_processing_event(event_instance, 2)
                                yield item
                                # this return probably is redundant
                                return
                except Exception as e:
                    # TODO: this is not preventing double access to scores/grades
                    # TODO POSTPOC: make this only for an openai connection error
                    logger.exception(f"Found exception {e}")
                    self.event_manager.remove_desired_model(event_instance, is_grading=True)
                    # TODO: dont hard exit here
                    raise e
        iterator = combined_iterator()
        await self.grading_manager.enqueue(event_instance, iterator)

    async def handle_grading(self, event_instance, generation_aiter):
        self.event_manager.add_async_task(event_instance, "grading", asyncio.current_task())
        # Just a read over the input aiter to force evaluation
        if self._grading_semaphore is None:
            raise RuntimeError("grading parallelism was not resolved")
        async with self._grading_semaphore:
            async for elem in generation_aiter:
                pass

    def _resolve_evaluate_now_parallelism(self, total_events: int) -> None:
        """Resolve omitted one-shot limits from the finite parsed request."""
        if self._max_generation_jobs is None:
            serving_models = list(self.db_manager.get_all_models().values())
            for task in self.db_manager.get_all_tasks():
                grader = getattr(task, "grader", None)
                judge = getattr(grader, "llm_as_judge", None)
                if judge is not None:
                    serving_models.append(judge)
            serving_keys = {
                model.serving_key
                for model in serving_models
                if not model.is_external
            }
            self._max_generation_jobs = len(serving_keys)
            logger.info(
                "Derived max_generation_jobs=%d from %d unique serving keys",
                self._max_generation_jobs,
                len(serving_keys),
            )
        if self._max_grading_parallelism is None:
            self._max_grading_parallelism = total_events
            self._grading_semaphore = asyncio.Semaphore(value=total_events)
            logger.info(
                "Derived max_grading_parallelism=%d from the finite evaluation request",
                total_events,
            )

    async def handle_job_update(self, job_state):
        # We are ignoring job_state here:
        # This is actually the desired behavior because job_state could be out of state
        # Currently job_state is on a 1 second sleep loop.
        # TODO (POSTPOC): find a way to make this only update on changes to slurm state

        # get all events that need generation and grading


        # TODO: lock this
        try:
            await asyncio.wait_for(self._job_lock.acquire(), 5.0)
        except asyncio.TimeoutError:
            return
        try:
            logger.info(f"------- current models: {list(self.event_manager.desired_models_dict.keys())} -------")

            desired_models = self.event_manager.get_desired_models()
            pending_models, deploying_models, live_models, dead_models, replica_counts = \
                await self.slurm_manager.get_model_state(desired_models)
            logger.info(f"pending={pending_models} deploying={deploying_models} live={live_models} dead={dead_models}")

            # Build sibling map: serving_key → [model_names].
            # When k2-v2 and k2-v2-low share a VLLM deployment, only one of them
            # appears in pending/deploying/live (whichever _serving_key_registry points to).
            # We expand to include all siblings so status display and created_models
            # are correct for all models sharing a deployment.
            sk_to_siblings: dict[str, list[str]] = {}
            for _mn, _di in desired_models.items():
                sk_to_siblings.setdefault(_di.model.serving_key, []).append(_mn)

            def _expand_siblings(names):
                result = set(names)
                for _mn in list(names):
                    _di = desired_models.get(_mn)
                    if _di:
                        result.update(sk_to_siblings.get(_di.model.serving_key, []))
                return result

            # Update the live URL pool. live_models is a list of (model_name, url) tuples,
            # one entry per healthy replica.
            live_model_names = []
            for model_name, url in live_models:
                await self.job_manager.register_live_url(model_name, url)
                _sk = desired_models[model_name].model.serving_key if model_name in desired_models else None
                if _sk and _sk in openai_interface.LOCKED_CONNECTIONS:
                    await openai_interface.LOCKED_CONNECTIONS[_sk].add_url(url)
                live_model_names.append(model_name)

            # Evict URLs for models whose jobs have timed out (exceeded max_time_to_deploy).
            # These are permanent failures — no new job will be submitted for them.
            for model_name in dead_models:
                _sk = desired_models[model_name].model.serving_key if model_name in desired_models else None
                for url in self.job_manager.get_live_urls(model_name):
                    await self.job_manager.evict_url(model_name, url)
                    if _sk and _sk in openai_interface.LOCKED_CONNECTIONS:
                        await openai_interface.LOCKED_CONNECTIONS[_sk].remove_url(url)

            # Evict URLs for desired models whose Slurm jobs have completely disappeared
            # (e.g. externally cancelled via scancel). get_model_state skips these with
            # `continue`, so they never appear in dead_models. Without this, get_live_url()
            # returns a permanently-dead URL indefinitely.
            all_known_models = _expand_siblings(set(pending_models + deploying_models + [n for n, _ in live_models]))
            for model_name in desired_models:
                if model_name not in all_known_models:
                    _sk = desired_models[model_name].model.serving_key
                    for url in self.job_manager.get_live_urls(model_name):
                        logger.info(f"Evicting URL for {model_name}: job no longer in squeue")
                        await self.job_manager.evict_url(model_name, url)
                        if _sk in openai_interface.LOCKED_CONNECTIONS:
                            await openai_interface.LOCKED_CONNECTIONS[_sk].remove_url(url)
            await self.event_manager.fail_all_events_with_models(dead_models, progress_manager=self.progress_manager)

            # Cancel in-flight generation connections for dead serving keys so
            # make_request tasks don't block forever in pool.acquire().
            dead_sks = {desired_models[m].model.serving_key for m in dead_models if m in desired_models}
            if dead_sks:
                for event_instance, conn in list(self._openai_connections.items()):
                    di = desired_models.get(event_instance.model)
                    if di and di.model.serving_key in dead_sks:
                        cancel = getattr(conn, "cancel", None)
                        if cancel is not None:
                            cancel()

            # Update status based on Slurm job state.
            # PENDING = waiting for a node → "Queued"; RUNNING but unhealthy → "Deploying".
            for model_name, status in (
                [(m, "Queued") for m in _expand_siblings(pending_models)] +
                [(m, "Deploying") for m in _expand_siblings(deploying_models)]
            ):
                deployment_info = desired_models.get(model_name)
                if deployment_info is None:
                    continue
                for event in deployment_info.generation_events:
                    self.progress_manager.set_status(event, status, mode="Generation")
                for event in deployment_info.grader_events:
                    self.progress_manager.set_status(event, status, mode="Grading")

            created_models = _expand_siblings(set(pending_models + deploying_models + live_model_names))

            # Count total occupied Slurm nodes (one per replica, not per unique model).
            # Use replica_counts (raw squeue totals) rather than the filtered pending/deploying
            # lists, which exclude replicas of already-live models and would undercount.
            total_occupied = sum(replica_counts.values())
            available_nodes = max(0, self._max_generation_jobs - total_occupied - self._active_imported_dataset_jobs)

            logger.info(f"-------- available nodes: {available_nodes} --------")
            allocation = self.get_desired_allocation(desired_models, created_models, max(available_nodes, 0), replica_counts=replica_counts)
            logger.info(f"allocation: {allocation}")
            unneeded_models, excess_models = await self.slurm_manager.get_unneeded_models(
                list(created_models), live_model_names, desired_models, self._max_generation_jobs)

            logger.info(f"desired_models: {list(desired_models.keys())}")
            logger.info(f"unneeded_models: {unneeded_models}")
            logger.info(f"excess_models: {excess_models}")

            sibling_names_by_sk: dict[str, list[str]] = {}
            for mn, di in desired_models.items():
                sibling_names_by_sk.setdefault(di.model.serving_key, []).append(mn)

            unneeded_models += excess_models
            await self.slurm_manager.update_allocation(allocation, unneeded_models, sibling_names_by_sk)
        except Exception as e:
            logger.exception(f"Encountered exception {e}")
            if self._terminal_evidence_enabled:
                raise
        finally:
            self._job_lock.release()

    def get_desired_allocation(self, desired_model_dict, created_models, available_nodes, replica_counts=None):
        # TODO: sort by priority

        # Group desired models by serving_key so that sibling models (same VLLM
        # deployment, different openai_kwargs) share a single Slurm job entry.
        # Each serving_key maps to one representative ModelInstance plus a flag
        # for whether it has generation vs grader events.
        #
        # Structure: sk -> {"model": ModelInstance, "has_gen": bool, "has_grader": bool}
        sk_to_deployment: dict = {}
        for model_name, deployment_info in desired_model_dict.items():
            logger.info(f"{model_name}: {deployment_info}")
            sk = deployment_info.model.serving_key
            if sk not in sk_to_deployment:
                sk_to_deployment[sk] = {
                    "model": deployment_info.model,
                    "has_gen": bool(deployment_info.generation_events),
                    "has_grader": bool(deployment_info.grader_events),
                }
            else:
                sk_to_deployment[sk]["has_gen"] |= bool(deployment_info.generation_events)
                sk_to_deployment[sk]["has_grader"] |= bool(deployment_info.grader_events)

        # Serving keys that already have a running Slurm job. These count toward
        # available_nodes but are still eligible for extra replicas (expansion).
        created_sks = {
            desired_model_dict[name].model.serving_key
            for name in created_models
            if name in desired_model_dict
        }

        generation_deployments = []
        grader_deployments = []
        for sk, info in sk_to_deployment.items():
            if getattr(info["model"], "is_external", False):
                continue  # external models don't need Slurm allocation
            if info["has_gen"]:
                generation_deployments.append(info["model"])
            elif info["has_grader"]:
                grader_deployments.append(info["model"])

        logger.info(f"generation_deployments (unique serving keys): {[m.name for m in generation_deployments]}, created_sks: {created_sks}")

        if available_nodes <= 0:
            return []

        # Step 1: give each generation deployment at least 1 replica.
        # New deployments consume a node slot; already-running ones don't (their
        # node is already counted in total_occupied), so they are always included.
        new_gen = [m for m in generation_deployments if m.serving_key not in created_sks]
        existing_gen = [m for m in generation_deployments if m.serving_key in created_sks]

        new_gen_to_deploy = new_gen[:available_nodes]
        nodes_used = len(new_gen_to_deploy)
        remaining = available_nodes - nodes_used
        gen_to_deploy = new_gen_to_deploy + existing_gen

        # Step 2: give grader deployments 1 replica each from remaining nodes
        new_graders = [m for m in grader_deployments if m.serving_key not in created_sks]
        graders_to_deploy = new_graders[:remaining]
        remaining -= len(graders_to_deploy)

        # Step 3: distribute leftover nodes as extra replicas across generation deployments.
        # When replica_counts is provided, start from each model's actual current replica
        # count rather than 1, so that models already at N replicas get desired=N+extra
        # and update_allocation doesn't skip them (existing >= desired).
        logger.info(f"generation_deployments to deploy: {[m.name for m in gen_to_deploy]}, remaining nodes for extra replicas: {remaining}")
        if gen_to_deploy and remaining > 0:
            base = {m.name: replica_counts.get(m.name, 1) for m in gen_to_deploy} \
                if replica_counts else {m.name: 1 for m in gen_to_deploy}
            extra_per_gen = remaining // len(gen_to_deploy)
            bonus = remaining % len(gen_to_deploy)
            sorted_gens = sorted(gen_to_deploy, key=lambda m: base[m.name])
            result = [
                (m, base[m.name] + extra_per_gen + (1 if i < bonus else 0))
                for i, m in enumerate(sorted_gens)
            ]
        else:
            result = [(m, 1) for m in gen_to_deploy]

        result += [(m, 1) for m in graders_to_deploy]
        return result

    async def loop(self):
        def log_task_result(task: asyncio.Task, name: str):
            try:
                task.result()
            except asyncio.CancelledError:
                pass  # expected during shutdown
            except Exception as e:
                logger.exception(e)
                logger.info(f"[{name}] task failed: {e}")

        async def enqueue_source(aiter, source_name):
            async for item in aiter:
                await self._queue.put((source_name, item))

        logger.info("about to create sources")

        tasks = [
            asyncio.create_task(enqueue_source(self.filesystem_manager, "fs")),
            asyncio.create_task(enqueue_source(self.slurm_manager, "job")),
            asyncio.create_task(enqueue_source(self.event_manager, "event")),
            asyncio.create_task(enqueue_source(self.generation_manager, "generation")),
            asyncio.create_task(enqueue_source(self.grading_manager, "grading"))
        ]
        logger.info("created sources")
        async with asyncio.TaskGroup() as tg:
            # TODO POSTPOC: do a passthrough of model spec and data directory to ensure db is up to date
            for model_spec in self.db_manager.get_all_model_families():
                tg.create_task(self.register_model_spec(model_spec))
            for name, model_instance in self.db_manager.get_all_models().items():
                assert(isinstance(model_instance, ModelInstance))
                tg.create_task(self.register_model_instance(model_instance))

        #  TODO: on startup, enqueue all incomplete events
        names = ["fs", "event", "job", "generation", "grading"]
        for t, name in zip(tasks, names):
            t.add_done_callback(lambda t, n=name: log_task_result(t, n))

        _interrupted = False
        try:
            async with asyncio.TaskGroup() as tg:
                try:
                    while True:
                        source, update = await self._queue.get()
                        task = tg.create_task(self.handle_update(source, update))
                        task.add_done_callback(lambda t: log_task_result(t, source))
                except asyncio.CancelledError:
                    # TaskGroup will cancel all child tasks here
                    raise
                except Exception as e:
                    # TODO REFACTOR: make this a soft error
                    logger.exception(f"encountered exception {e} for {task}")
                    raise
        except BaseException:
            _interrupted = True
            raise
        finally:
            if _interrupted:
                await self.slurm_manager.cancel_all_owned_jobs()

    def _capture_selected_event(
        self,
        *,
        model_name: str,
        task_uuid: str,
        suite_task: SuiteTask,
        dataset_bindings: tuple[FileBinding, ...],
        task_closure_sha256: str,
        force: bool,
    ) -> _SelectedEvent:
        events = self.db_manager.get_event_by_model_and_task(
            model_name,
            task_uuid,
            include_completed=True,
        )
        if len(events) != 1:
            raise RuntimeError(
                "terminal-result selection requires exactly one event for "
                f"model={model_name!r}, task={task_uuid!r}; found {len(events)}"
            )
        event = events[0]
        task = self.db_manager.get_task(task_uuid)
        model = self.db_manager.get_model(model_name)
        event_was_complete = self.db_manager.get_event_phase(event.uuid) == 2
        if isinstance(event, ImportedDatasetEventInstance):
            raise ValueError(
                "evaluation-request schema 1.0 supports only explicit "
                "generation-task dataset closures"
            )

        assert isinstance(event, GradingEventInstance)
        assert isinstance(task, AsyncGenerationTask)
        self._validate_request_resume_binding(event, task, force=force)
        completed_generation_rows = load_completed_row_ids(
            event.path_to_generations
        )
        completed_grade_rows = load_completed_row_ids(event.path_to_grades)
        expected_rows = task.num_generations or 0
        judge = task.grader.llm_as_judge
        generation_required = (
            force or len(completed_generation_rows) < expected_rows
        )
        grading_required = force or len(completed_grade_rows) < expected_rows
        return _SelectedEvent(
            event=event,
            suite_task=suite_task,
            dataset_bindings=dataset_bindings,
            task_closure_sha256=task_closure_sha256,
            generation_required=generation_required,
            grading_required=grading_required,
            aggregation_required=force or not event_was_complete,
            candidate_serving_required=(
                not model.is_external
                and generation_required
            ),
            grader_serving_required=(
                judge is not None
                and not judge.is_external
                and grading_required
            ),
            imported_job_required=False,
        )

    @staticmethod
    def _definition_sha256(value: dict) -> str:
        """Hash one deterministic, secret-safe resolved definition."""
        return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

    def _yaml_dataset_bindings(
        self,
        task: AsyncGenerationTask,
    ) -> tuple[FileBinding, ...]:
        """Bind every concrete dataset file selected by an ordinary YAML task."""
        data_path = (
            resolve_hf_path(task.data_path, cache_dir=self._hf_cache_dir)
            if is_hf_uri(task.data_path)
            else task.data_path
        )
        paths = expand_data_path(data_path)
        if not paths:
            raise ValueError(
                f"task {task.uuid!r} data_path matched no dataset files"
            )
        return tuple(
            bind_regular_file(path, label=f"task {task.uuid} dataset {ordinal}")
            for ordinal, path in enumerate(paths)
        )

    @staticmethod
    def _safe_task_definition(task: AsyncGenerationTask) -> dict:
        """Return resolved task fields without exposing an external judge key."""
        definition = task.model_dump(mode="json")
        judge = task.grader.llm_as_judge
        if judge is not None:
            definition["grader"]["llm_as_judge"] = judge.registration_snapshot()
        return definition

    def _capture_yaml_selected_event(
        self,
        *,
        invocation: YamlTerminalInvocation,
        model_name: str,
        task_uuid: str,
        model_config_path: str,
        data_config_path: str,
        eval_config_path: str,
        eval_group_name: str,
        force: bool,
    ) -> _SelectedEvent:
        """Bind one exact resolved eval-config pair to its durable DB event."""
        events = self.db_manager.get_event_by_model_and_task(
            model_name,
            task_uuid,
            include_completed=True,
        )
        if len(events) != 1:
            raise RuntimeError(
                "YAML terminal selection requires exactly one event for "
                f"model={model_name!r}, task={task_uuid!r}; found {len(events)}"
            )
        event = events[0]
        if isinstance(event, ImportedDatasetEventInstance):
            raise TypeError(
                "YAML terminal evidence does not support imported-dataset tasks"
            )
        if not isinstance(event, GradingEventInstance):
            raise TypeError("YAML terminal selection resolved an unsupported event")

        task = self.db_manager.get_task(task_uuid)
        model = self.db_manager.get_model(model_name)
        if not isinstance(task, AsyncGenerationTask):
            raise TypeError(
                "YAML terminal evidence supports generation/grading tasks only"
            )
        model_binding = invocation.config_binding("model", model_config_path)
        data_binding = invocation.config_binding("data", data_config_path)
        eval_binding = invocation.config_binding("eval", eval_config_path)
        dataset_bindings = self._yaml_dataset_bindings(task)
        self._bound_dataset_paths[task.uuid] = tuple(
            item.resolved_path for item in dataset_bindings
        )
        model_definition_sha256 = self._definition_sha256(
            model.registration_snapshot()
        )
        task_definition_sha256 = self._definition_sha256(
            self._safe_task_definition(task)
        )
        yaml_metadata = {
            "dataset_files": [item.as_dict() for item in dataset_bindings],
            "model": {
                "definition_sha256": model_definition_sha256,
                "name": model.name,
                "parser_type": model.parser_type,
            },
            "source": {
                "data_config": data_binding.as_dict(),
                "eval_config": eval_binding.as_dict(),
                "eval_group": eval_group_name,
                "model_config": model_binding.as_dict(),
            },
            "task": {
                "dataset_name": task.dataset_name,
                "definition_sha256": task_definition_sha256,
                "grader_type": task.grader.type,
                "id": task.uuid,
                "mode": task.mode,
                "semantic_version": task.semantic_version,
            },
        }
        self._yaml_event_metadata[event.uuid] = yaml_metadata
        self._validate_yaml_resume_binding(event, task, force=force)

        event_was_complete = self.db_manager.get_event_phase(event.uuid) == 2
        completed_generation_rows = load_completed_row_ids(
            event.path_to_generations
        )
        completed_grade_rows = load_completed_row_ids(event.path_to_grades)
        expected_rows = task.num_generations or 0
        judge = task.grader.llm_as_judge
        generation_required = force or len(completed_generation_rows) < expected_rows
        grading_required = force or len(completed_grade_rows) < expected_rows
        return _SelectedEvent(
            event=event,
            suite_task=None,
            dataset_bindings=dataset_bindings,
            task_closure_sha256=task_definition_sha256,
            generation_required=generation_required,
            grading_required=grading_required,
            aggregation_required=force or not event_was_complete,
            candidate_serving_required=(
                not model.is_external and generation_required
            ),
            grader_serving_required=(
                judge is not None
                and not judge.is_external
                and grading_required
            ),
            imported_job_required=False,
            model_config_binding=model_binding,
            data_config_binding=data_binding,
            eval_config_binding=eval_binding,
            eval_group_name=eval_group_name,
            model_definition_sha256=model_definition_sha256,
        )

    @staticmethod
    def _terminal_output_binding(path, role: str) -> dict:
        binding = bind_regular_file(
            path,
            label=f"required {role} output",
        )
        return {"role": role, **binding.as_dict()}

    @staticmethod
    def _terminal_unit(
        role: str,
        *,
        required: bool,
        executor: str,
        job_ids: list[int] | None = None,
        serving_required: bool | None = None,
    ) -> dict:
        unit = {
            "executor": executor if required else "none",
            "job_ids": job_ids or [],
            "outcome": "succeeded" if required else "not_required",
            "required": required,
            "role": role,
        }
        if serving_required is not None:
            unit["serving_required"] = serving_required
        return unit

    async def _reconcile_terminal_jobs(
        self,
        *,
        selected_events: list[_SelectedEvent],
    ) -> tuple[list[dict], dict[str, dict[int, set[str]]]]:
        """Strictly reconcile selected events with ledgered sacct children."""
        non_successful = [
            selected.event.uuid
            for selected in selected_events
            if self.db_manager.get_event_phase(selected.event.uuid) != 2
        ]
        if non_successful:
            raise RuntimeError(
                "terminal result requires successful selected events: "
                + ", ".join(non_successful)
            )

        submitted_jobs = (
            () if self._no_slurm else self.slurm_manager.get_submitted_jobs()
        )
        job_associations: dict[int, list[dict]] = {
            job.job_id: [] for job in submitted_jobs
        }
        event_job_roles: dict[str, dict[int, set[str]]] = {
            selected.event.uuid: {} for selected in selected_events
        }

        selected_by_id = {
            selected.event.uuid: selected for selected in selected_events
        }
        for job in submitted_jobs:
            if job.kind != "model_serving":
                raise RuntimeError(
                    "terminal evaluation captured an unsupported Slurm "
                    f"child kind: {job.kind!r}"
                )
            roles_by_event: dict[str, set[str]] = {}
            for event_id, role in job.completed_event_roles:
                selected = selected_by_id.get(event_id)
                if selected is None:
                    raise RuntimeError(
                        f"Slurm child {job.job_id} names unselected event {event_id}"
                    )
                event = selected.event
                task = self.db_manager.get_task(event.task_uuid)
                if role == "generation":
                    model = self.db_manager.get_model(event.model)
                    valid = (
                        not model.is_external
                        and job.serving_key == model.serving_key
                    )
                elif role == "grading":
                    judge = task.grader.llm_as_judge
                    valid = (
                        judge is not None
                        and not judge.is_external
                        and job.serving_key == judge.serving_key
                    )
                else:
                    raise RuntimeError(
                        f"Slurm child {job.job_id} has unsupported role {role!r}"
                    )
                if not valid:
                    raise RuntimeError(
                        f"Slurm child {job.job_id} has an invalid {role} "
                        f"completion for event {event_id}"
                    )
                roles_by_event.setdefault(event_id, {"serving"}).add(role)

            for event_id in sorted(roles_by_event):
                roles = roles_by_event[event_id]
                ordered_roles = [
                    role
                    for role in ("serving", "generation", "grading")
                    if role in roles
                ]
                job_associations[job.job_id].append(
                    {"event_id": event_id, "roles": ordered_roles}
                )
                event_job_roles[event_id][job.job_id] = roles

        unassociated = [
            str(job_id)
            for job_id, associations in job_associations.items()
            if not associations
        ]
        if unassociated:
            raise RuntimeError(
                "terminal result found unassociated Eval360 Slurm child job(s): "
                + ", ".join(unassociated)
            )

        for selected in selected_events:
            associated = event_job_roles[selected.event.uuid]
            if selected.candidate_serving_required and not any(
                "generation" in roles for roles in associated.values()
            ):
                raise RuntimeError(
                    f"selected event {selected.event.uuid} is missing a required generation-serving Slurm child"
                )
            if selected.grader_serving_required and not any(
                "grading" in roles for roles in associated.values()
            ):
                raise RuntimeError(
                    f"selected event {selected.event.uuid} is missing a required grading-serving Slurm child"
                )

        outcomes = (
            {}
            if not submitted_jobs
            else await self.slurm_manager.wait_for_terminal_job_outcomes(
                [job.job_id for job in submitted_jobs]
            )
        )
        if set(outcomes) != {job.job_id for job in submitted_jobs}:
            raise RuntimeError(
                "terminal result is missing authoritative Slurm child outcomes"
            )

        jobs = []
        for job in submitted_jobs:
            outcome = outcomes[job.job_id]
            if outcome.job_name != job.job_name:
                raise RuntimeError(
                    f"sacct job name for {job.job_id} does not match the "
                    "submitted child identity"
                )
            expected_release = (
                job.kind == "model_serving"
                and job.cancellation_intent == "scheduler_release"
                and outcome.state == "CANCELLED"
                and bool(job.completed_event_roles)
            )
            if not outcome.is_clean_completion and not expected_release:
                raise RuntimeError(
                    f"required Slurm child job {job.job_id} was unsuccessful: "
                    f"state={outcome.raw_state!r} exit={outcome.exit_code}:{outcome.signal}"
                )
            jobs.append(
                {
                    "associations": job_associations[job.job_id],
                    "cancellation_intent": job.cancellation_intent,
                    "event_id": job.event_uuid,
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "kind": job.kind,
                    "model": job.model_name,
                    "outcome": "succeeded",
                    "runner": job.runner_name,
                    "role_completions": [
                        {"event_id": event_id, "role": role}
                        for event_id, role in job.completed_event_roles
                    ],
                    "scheduler": {
                        "exit_code": outcome.exit_code,
                        "job_id_raw": outcome.job_id_raw,
                        "job_name": outcome.job_name,
                        "raw_state": outcome.raw_state,
                        "reason": outcome.reason,
                        "signal": outcome.signal,
                        "source": "sacct",
                        "state": outcome.state,
                    },
                    "serving_key": job.serving_key,
                    "submission_origin": job.submission_origin,
                }
            )

        return jobs, event_job_roles

    async def _build_terminal_scheduler_result(
        self,
        *,
        request: EvaluationRequest,
        selected_events: list[_SelectedEvent],
        selection_mode: str,
    ) -> dict:
        """Build the strict request-native result after common reconciliation."""
        request.verify_inputs()
        jobs, event_job_roles = await self._reconcile_terminal_jobs(
            selected_events=selected_events
        )

        events = []
        for ordinal, selected in enumerate(selected_events):
            if selected.suite_task is None:
                raise RuntimeError(
                    "request-native terminal event has no suite task"
                )
            event = selected.event
            task = self.db_manager.get_task(event.task_uuid)
            model = self.db_manager.get_model(event.model)
            inputs = {
                "dataset_files": [
                    item.as_dict() for item in selected.dataset_bindings
                ],
                "task_closure_sha256": selected.task_closure_sha256,
            }
            associated_ids = event_job_roles[event.uuid]
            if isinstance(event, ImportedDatasetEventInstance):
                output_dir = pathlib.Path(event.path_to_scores).parent / (
                    f"{task.dataset_name}_output"
                )
                outputs = [
                    self._terminal_output_binding(event.path_to_scores, "scores"),
                    self._terminal_output_binding(
                        output_dir / ".job_complete",
                        "job_completion_sentinel",
                    ),
                ]
                generation_jobs = sorted(
                    job_id
                    for job_id, roles in associated_ids.items()
                    if "generation" in roles
                )
                grading_jobs = sorted(
                    job_id
                    for job_id, roles in associated_ids.items()
                    if "grading" in roles
                )
                units = [
                    self._terminal_unit(
                        "generation",
                        required=(
                            selected.generation_required or bool(generation_jobs)
                        ),
                        executor="slurm_child",
                        job_ids=generation_jobs,
                    ),
                    self._terminal_unit(
                        "grading",
                        required=(
                            selected.grading_required or bool(grading_jobs)
                        ),
                        executor="slurm_child",
                        job_ids=grading_jobs,
                    ),
                    self._terminal_unit(
                        "aggregation",
                        required=selected.aggregation_required,
                        executor="controller",
                    ),
                ]
                event_type = "imported_dataset"
                grader_type = None
            else:
                generation_jobs = sorted(
                    job_id
                    for job_id, roles in associated_ids.items()
                    if "generation" in roles
                )
                grading_jobs = sorted(
                    job_id
                    for job_id, roles in associated_ids.items()
                    if "grading" in roles
                )
                outputs = [
                    self._terminal_output_binding(
                        event.path_to_generations,
                        "generations",
                    ),
                    self._terminal_output_binding(event.path_to_grades, "grades"),
                    self._terminal_output_binding(event.path_to_scores, "scores"),
                    self._terminal_output_binding(
                        self._run_metadata_path(event, task),
                        "run_metadata",
                    ),
                ]
                units = [
                    self._terminal_unit(
                        "generation",
                        required=(
                            selected.generation_required or bool(generation_jobs)
                        ),
                        executor="controller",
                        job_ids=generation_jobs,
                    ),
                    self._terminal_unit(
                        "grading",
                        required=(
                            selected.grading_required or bool(grading_jobs)
                        ),
                        executor="controller",
                        job_ids=grading_jobs,
                    ),
                    self._terminal_unit(
                        "aggregation",
                        required=selected.aggregation_required,
                        executor="controller",
                    ),
                ]
                event_type = "generation_grading"
                grader_type = task.grader.type

            events.append(
                {
                    "controller_units": units,
                    "event_id": event.uuid,
                    "event_type": event_type,
                    "inputs": inputs,
                    "model": {
                        "name": model.name,
                        "parser_type": model.parser_type,
                    },
                    "ordinal": ordinal,
                    "outputs": outputs,
                    "task": {
                        "dataset_name": task.dataset_name,
                        "definition": selected.suite_task.as_dict(),
                        "grader_type": grader_type,
                        "id": task.uuid,
                        "mode": task.mode,
                        "semantic_version": task.semantic_version,
                    },
                    "terminal_phase": 2,
                }
            )

        return {
            "bindings": request.bindings_dict(),
            "contract": TERMINAL_RESULT_CONTRACT,
            "execution": request.raw_execution,
            "jobs": jobs,
            "runner": {
                "id": request.runner_definition.runner_id,
                "source": request.runner_definition.source.as_dict(),
            },
            "schema_version": TERMINAL_RESULT_SCHEMA_VERSION,
            "selection": {
                "events": events,
                "mode": selection_mode,
                "suite": {
                    "catalog_id": request.suite_catalog.catalog_id,
                    "id": request.suite.suite_id,
                    "owner": request.suite.owner,
                    "source": request.suite.source.as_dict(),
                    "version": request.suite.version,
                },
            },
            "status": "succeeded",
        }

    async def _build_yaml_terminal_scheduler_result(
        self,
        *,
        invocation: YamlTerminalInvocation,
        selected_events: list[_SelectedEvent],
        selection_mode: str,
    ) -> dict:
        """Build output-only evidence for the parsed ordinary-YAML route."""
        invocation.verify_inputs()
        jobs, event_job_roles = await self._reconcile_terminal_jobs(
            selected_events=selected_events
        )

        events = []
        for ordinal, selected in enumerate(selected_events):
            event = selected.event
            if not isinstance(event, GradingEventInstance):
                raise TypeError(
                    "YAML terminal evidence supports generation/grading tasks only"
                )
            if (
                selected.model_config_binding is None
                or selected.data_config_binding is None
                or selected.eval_config_binding is None
                or selected.eval_group_name is None
                or selected.model_definition_sha256 is None
            ):
                raise RuntimeError(
                    f"selected YAML event {event.uuid} has incomplete source bindings"
                )
            task = self.db_manager.get_task(event.task_uuid)
            model = self.db_manager.get_model(event.model)
            associated_ids = event_job_roles[event.uuid]
            generation_jobs = sorted(
                job_id
                for job_id, roles in associated_ids.items()
                if "generation" in roles
            )
            grading_jobs = sorted(
                job_id
                for job_id, roles in associated_ids.items()
                if "grading" in roles
            )
            events.append(
                {
                    "controller_units": [
                        self._terminal_unit(
                            "generation",
                            required=(
                                selected.generation_required
                                or bool(generation_jobs)
                            ),
                            executor="controller",
                            job_ids=generation_jobs,
                            serving_required=bool(generation_jobs),
                        ),
                        self._terminal_unit(
                            "grading",
                            required=(
                                selected.grading_required or bool(grading_jobs)
                            ),
                            executor="controller",
                            job_ids=grading_jobs,
                            serving_required=bool(grading_jobs),
                        ),
                        self._terminal_unit(
                            "aggregation",
                            required=selected.aggregation_required,
                            executor="controller",
                            serving_required=False,
                        ),
                    ],
                    "event_id": event.uuid,
                    "event_type": "generation_grading",
                    "inputs": {
                        "dataset_files": [
                            item.as_dict()
                            for item in selected.dataset_bindings
                        ],
                    },
                    "model": {
                        "definition_sha256": (
                            selected.model_definition_sha256
                        ),
                        "name": model.name,
                        "parser_type": model.parser_type,
                    },
                    "ordinal": ordinal,
                    "outputs": [
                        self._terminal_output_binding(
                            event.path_to_generations,
                            "generations",
                        ),
                        self._terminal_output_binding(
                            event.path_to_grades,
                            "grades",
                        ),
                        self._terminal_output_binding(
                            event.path_to_scores,
                            "scores",
                        ),
                        self._terminal_output_binding(
                            self._run_metadata_path(event, task),
                            "run_metadata",
                        ),
                    ],
                    "source": {
                        "data_config": selected.data_config_binding.as_dict(),
                        "eval_config": selected.eval_config_binding.as_dict(),
                        "eval_group": selected.eval_group_name,
                        "model_config": selected.model_config_binding.as_dict(),
                    },
                    "task": {
                        "dataset_name": task.dataset_name,
                        "definition_sha256": selected.task_closure_sha256,
                        "grader_type": task.grader.type,
                        "id": task.uuid,
                        "mode": task.mode,
                        "semantic_version": task.semantic_version,
                    },
                    "terminal_phase": 2,
                }
            )

        return {
            "bindings": invocation.bindings_dict(),
            "contract": YAML_TERMINAL_RESULT_CONTRACT,
            "jobs": jobs,
            "schema_version": YAML_TERMINAL_RESULT_SCHEMA_VERSION,
            "selection": {
                "events": events,
                "mode": selection_mode,
            },
            "status": "succeeded",
        }

    async def run_evaluate_now(
        self,
        paths_to_model_specs,
        paths_to_datasets,
        force: bool = False,
        eval_paths=None,
        evaluation_request: EvaluationRequest | None = None,
        yaml_terminal_invocation: YamlTerminalInvocation | None = None,
    ):
        if self._salt_cache and not force:
            raise ValueError(
                "--salt-cache requires force=True/--force to avoid mixing "
                "pre-existing unsalted generations with salted requests"
            )
        self._force = force
        if evaluation_request is not None and yaml_terminal_invocation is not None:
            raise ValueError(
                "request-native and ordinary-YAML terminal evidence are mutually exclusive"
            )
        if yaml_terminal_invocation is not None and (
            not paths_to_model_specs or not paths_to_datasets or not eval_paths
        ):
            raise ValueError(
                "ordinary-YAML terminal evidence requires model, data, "
                "and eval paths"
            )
        if yaml_terminal_invocation is not None:
            yaml_terminal_invocation.verify_config_paths(
                model_paths=paths_to_model_specs,
                data_paths=paths_to_datasets,
                eval_paths=eval_paths,
            )
            yaml_terminal_invocation.verify_inputs()
        self._terminal_evidence_enabled = (
            evaluation_request is not None or yaml_terminal_invocation is not None
        )

        async def enqueue_source(aiter, source_name):
            async for item in aiter:
                await self._queue.put((source_name, item))
        selected_events: list[_SelectedEvent] = []

        if evaluation_request is not None:
            if paths_to_model_specs or paths_to_datasets or eval_paths:
                raise ValueError(
                    "evaluation request is mutually exclusive with legacy YAML inputs"
                )
            if self._salt_cache:
                raise ValueError(
                    "request-native evaluation uses suite-owned cache settings"
                )
            evaluation_request.verify_inputs()
            self._evaluation_request = evaluation_request
            self._bound_dataset_paths = dict(evaluation_request.dataset_paths)
            model_spec = evaluation_request.model_spec
            assert model_spec.remote_model is not None
            model = ModelParser.model_instance_from_path(
                model_spec.remote_model.path,
                model_spec,
            )
            await self.register_model_instance(model)
            for task in evaluation_request.tasks:
                await self.register_task(task)
            for suite_task in evaluation_request.suite.tasks:
                events = self.db_manager.get_event_by_model_and_task(
                    model.name,
                    suite_task.task_id,
                    include_completed=True,
                )
                if len(events) != 1:
                    raise RuntimeError(
                        "evaluation-request selection requires exactly one event "
                        f"for task {suite_task.task_id!r}; found {len(events)}"
                    )
                event = events[0]
                self._request_event_metadata[event.uuid] = (
                    evaluation_request.event_request_metadata(suite_task.task_id)
                )
                selected_events.append(
                    self._capture_selected_event(
                        model_name=model.name,
                        task_uuid=suite_task.task_id,
                        suite_task=suite_task,
                        dataset_bindings=evaluation_request.dataset_bindings[
                            suite_task.task_id
                        ],
                        task_closure_sha256=(
                            evaluation_request.task_closure_sha256[
                                suite_task.task_id
                            ]
                        ),
                        force=force,
                    )
                )
            total_events = len(selected_events)
            selection_mode = "evaluation_request"
        elif eval_paths:
            loaded_models: list[LoadedModelSpec] = []
            for path_to_model_spec in paths_to_model_specs:
                model_spec = self._parse_model_spec(
                    path_to_model_spec,
                    eval_mode=True,
                )
                if self._salt_cache:
                    model_spec.cache_salt = CacheSaltConfig(mode="unique")
                if self._no_slurm:
                    assert model_spec.external_model is not None, \
                        "--no-slurm requires external_model configs (remote_model and local_model need Slurm)"
                loaded_models.append(LoadedModelSpec(path=path_to_model_spec, spec=model_spec))

            loaded_tasks: list[LoadedTask] = []
            for path in paths_to_datasets:
                task = Task.parse_yaml(path)
                if not isinstance(task, AsyncGenerationTask):
                    raise ValueError("Eval-driven mode does not support imported_dataset tasks")
                loaded_tasks.append(LoadedTask(path=path, task=task))

            loaded_eval_configs = [
                LoadedEvalConfig(
                    path=path,
                    config=EvalConfigParser.parse_yaml(path),
                )
                for path in eval_paths
            ]
            resolved_pairs = EvalConfigParser.build_resolved_pairs(
                loaded_models=loaded_models,
                loaded_tasks=loaded_tasks,
                loaded_eval_configs=loaded_eval_configs,
            )

            async with asyncio.TaskGroup() as tg:
                for resolved in resolved_pairs:
                    tg.create_task(self.register_model_instance(resolved.model))
            async with asyncio.TaskGroup() as tg:
                for resolved in resolved_pairs:
                    tg.create_task(self.register_task(resolved.task))

            if yaml_terminal_invocation is not None:
                for resolved in resolved_pairs:
                    selected_events.append(
                        self._capture_yaml_selected_event(
                            invocation=yaml_terminal_invocation,
                            model_name=resolved.model.name,
                            task_uuid=resolved.task.uuid,
                            model_config_path=resolved.model_config_path,
                            data_config_path=resolved.data_config_path,
                            eval_config_path=resolved.eval_config_path,
                            eval_group_name=resolved.eval_group_name,
                            force=force,
                        )
                    )

            total_events = len(resolved_pairs)
            selection_mode = "eval_configs"
        else:
            async with asyncio.TaskGroup() as tg:
                for path_to_model_spec in paths_to_model_specs:
                    model_spec = self._parse_model_spec(path_to_model_spec)
                    if self._salt_cache:
                        model_spec.cache_salt = CacheSaltConfig(mode="unique")
                    assert (model_spec.remote_model or model_spec.external_model), \
                        "`remote_model` or `external_model` required for `evaluate_now`"
                    if self._no_slurm:
                        assert model_spec.external_model is not None, \
                            "--no-slurm requires external_model configs (remote_model and local_model need Slurm)"
                    tg.create_task(self.register_model_spec(model_spec))
            async with asyncio.TaskGroup() as tg:
                for path in paths_to_datasets:
                    task = Task.parse_yaml(path)
                    tg.create_task(self.register_task(task))

            total_events = len(paths_to_model_specs) * len(paths_to_datasets)
            selection_mode = "cartesian"
        self._resolve_evaluate_now_parallelism(total_events)
        selected_event_ids = (
            [selected.event.uuid for selected in selected_events]
            if self._terminal_evidence_enabled
            else None
        )
        if selected_event_ids is not None and len(set(selected_event_ids)) != len(
            selected_event_ids
        ):
            raise RuntimeError(
                "terminal-result selection resolved the same event more than once"
            )
        if self._terminal_evidence_enabled and not self._no_slurm:
            self.slurm_manager.begin_terminal_result_capture()
        source_tasks = [
            asyncio.create_task(enqueue_source(self.event_manager, "event")),
            asyncio.create_task(
                enqueue_source(self.generation_manager, "generation")
            ),
            asyncio.create_task(enqueue_source(self.grading_manager, "grading")),
        ]
        if not self._no_slurm:
            source_tasks.insert(
                0,
                asyncio.create_task(enqueue_source(self.slurm_manager, "job")),
            )

        def count_selected_phases(phases: set[int]) -> int:
            if selected_event_ids is None:
                if phases == {2, -1}:
                    return self.db_manager.count_completed_events()
                if phases == {2}:
                    return self.db_manager.count_successful_events()
                raise RuntimeError("unsupported legacy event phase query")
            return sum(
                self.db_manager.get_event_phase(event_id) in phases
                for event_id in selected_event_ids
            )

        functional_tasks = set()
        job_tasks = set()

        async def raise_completed_tasks(tasks):
            if not tasks:
                return tasks
            done, pending = await asyncio.wait(
                tasks,
                timeout=0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception
            return pending

        async def stop_tasks(tasks):
            if not tasks:
                return []
            for task in tasks:
                task.cancel()
            return await asyncio.gather(*tasks, return_exceptions=True)

        async def collect_task_exceptions(tasks):
            if not tasks:
                return []
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [
                result for result in results if isinstance(result, BaseException)
            ]

        functional_exceptions = []
        job_exceptions = []
        release_exception = None
        completed_normally = False
        try:
            while count_selected_phases({2, -1}) < total_events:
                try:
                    source, update = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=_EVALUATE_NOW_QUEUE_POLL_TIMEOUT,
                    )
                except TimeoutError:
                    # Queue was empty; still drain any completed tasks that may
                    # have raised exceptions before re-checking the while condition.
                    functional_tasks = await raise_completed_tasks(
                        functional_tasks
                    )
                    job_tasks = await raise_completed_tasks(job_tasks)
                    continue
                task = asyncio.create_task(self.handle_update(source, update))
                if source == "job":
                    job_tasks.add(task)
                else:
                    functional_tasks.add(task)

                # check completed tasks
                functional_tasks = await raise_completed_tasks(functional_tasks)
                job_tasks = await raise_completed_tasks(job_tasks)

            completed_normally = count_selected_phases({2}) == total_events
            logger.info(self.db_manager.get_all_incomplete_events())
            await stop_tasks(source_tasks)
            job_exceptions = await collect_task_exceptions(job_tasks)
            if completed_normally:
                functional_exceptions = await collect_task_exceptions(
                    functional_tasks
                )
            else:
                await stop_tasks(functional_tasks)
            self._shutdown_file_flush_executor()

            if self._terminal_evidence_enabled and not self._no_slurm:
                try:
                    await self.slurm_manager.release_submitted_model_serving_jobs()
                except Exception as exception:  # noqa: BLE001
                    # Release is mandatory before any collected handler error.
                    release_exception = exception
            elif (
                not self._no_slurm
                and not functional_exceptions
                and not job_exceptions
            ):
                try:
                    await asyncio.sleep(_EVALUATE_NOW_FINAL_JOB_UPDATE_DELAY)
                    await self.handle_job_update(None)
                except Exception as exception:  # noqa: BLE001
                    # Preserve an earlier handler error after legacy cleanup.
                    release_exception = exception
        except BaseException:
            try:
                await stop_tasks(source_tasks)
            except BaseException:
                logger.exception(
                    "Failed to stop scheduler sources during interruption"
                )
            try:
                await stop_tasks(functional_tasks | job_tasks)
            except BaseException:
                logger.exception(
                    "Failed to stop scheduler handlers during interruption"
                )
            try:
                self._shutdown_file_flush_executor()
            except BaseException:
                logger.exception(
                    "Failed to stop the file flush executor during interruption"
                )
            if not self._no_slurm:
                try:
                    await self.slurm_manager.cancel_all_owned_jobs()
                except BaseException:
                    logger.exception(
                        "Failed to cancel owned Slurm jobs during interruption"
                    )
            raise

        if (
            (functional_exceptions or job_exceptions)
            and not self._terminal_evidence_enabled
            and not self._no_slurm
        ):
            try:
                await self.slurm_manager.cancel_all_owned_jobs()
            except BaseException:
                logger.exception(
                    "Failed to cancel owned Slurm jobs after handler failure"
                )

        if functional_exceptions:
            raise functional_exceptions[0]
        if job_exceptions:
            raise job_exceptions[0]
        if release_exception is not None:
            raise release_exception

        successful_events = count_selected_phases({2})
        if successful_events != total_events:
            failed_events = total_events - successful_events
            raise RuntimeError(
                f"{failed_events} of {total_events} selected evaluation events failed"
            )
        if evaluation_request is None and yaml_terminal_invocation is None:
            return None
        if yaml_terminal_invocation is not None:
            return await self._build_yaml_terminal_scheduler_result(
                invocation=yaml_terminal_invocation,
                selected_events=selected_events,
                selection_mode=selection_mode,
            )
        return await self._build_terminal_scheduler_result(
            request=evaluation_request,
            selected_events=selected_events,
            selection_mode=selection_mode,
        )
