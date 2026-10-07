import asyncio
import json
import os
import sqlite3
from contextlib import contextmanager

from .event import EventInstance, GradingEventInstance, ImportedDatasetEventInstance
from .model import (
    CacheSaltConfig,
    ExternalRetryPolicy,
    LocalModel,
    ModelInstance,
    ModelSpec,
    ModelType,
    ServingSlurmResources,
)
from .task import Task, AsyncGenerationTask, ImportedDatasetTask, GraderConfig, ImportedDatasetConfig
import logging
logger = logging.getLogger("FSManager")
logging.basicConfig(level=logging.INFO)


def _dump_cache_salt(cache_salt: CacheSaltConfig) -> str:
    if cache_salt is None:
        raise ValueError("cache_salt must be a CacheSaltConfig; use mode='disabled'")
    return json.dumps(cache_salt.model_dump())


def _load_cache_salt(cache_salt_json: str | None) -> CacheSaltConfig:
    return CacheSaltConfig.model_validate(json.loads(cache_salt_json)) if cache_salt_json else CacheSaltConfig()


def _dump_serving_slurm_resources(resources: ServingSlurmResources) -> str:
    return json.dumps(resources.model_dump(mode="json"))


def _load_serving_slurm_resources(
    resources_json: str | None,
) -> ServingSlurmResources:
    if resources_json is None:
        return ServingSlurmResources()
    return ServingSlurmResources.model_validate(json.loads(resources_json))


def _add_column_if_missing(
    cursor: sqlite3.Cursor,
    table: str,
    column: str,
    definition: str,
) -> None:
    try:
        cursor.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _dump_external_retry_policy(policy: ExternalRetryPolicy) -> str:
    return json.dumps(policy.model_dump())


def _load_external_retry_policy(
    policy_json: str | None,
) -> ExternalRetryPolicy:
    if policy_json is None:
        return ExternalRetryPolicy()
    return ExternalRetryPolicy.model_validate(json.loads(policy_json))


def _dump_model_registration_snapshot(model: ModelInstance) -> str:
    return json.dumps(
        model.registration_snapshot(),
        sort_keys=True,
        separators=(",", ":"),
    )


class DatabaseManager:
    def __init__(self):
        if os.environ.get("EVAL360_IN_MEMORY_DB", "false") == "true":
            self._connection = sqlite3.connect(":memory:")
        else:
            self._connection = sqlite3.connect("eval360.db")
        self._cursor = self._connection.cursor()
        self._transaction_depth = 0
        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS task (
                uuid TEXT PRIMARY KEY,
                average_over TEXT,
                pass_at TEXT,
                openai_settings TEXT,
                mode TEXT NOT NULL,
                grader_type TEXT,
                grader_model_name TEXT,
                data_path TEXT,
                dataset_name TEXT NOT NULL,
                semantic_version TEXT NOT NULL,
                tag TEXT NOT NULL DEFAULT 'any',
                meta TEXT,
                num_generations int,
                imported_dataset TEXT
            )
        """)
        self._connection.commit()

        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_family (
                model_family_name TEXT PRIMARY KEY,
                path_glob TEXT NOT NULL UNIQUE,
                version_level INT NOT NULL,
                enqueue_existing INT NOT NULL,
                name_modifier TEXT,
                venv_path TEXT,
                conda_env TEXT,
                container_image TEXT,
                container_mounts TEXT,
                serving_slurm_resources TEXT,
                max_simultaneous_requests INT NOT NULL,
                max_time_to_deploy INT NOT NULL,
                allow_long_max_model_len INT NOT NULL DEFAULT 1,
                vllm_cli_args TEXT NOT NULL,
                openai_kwargs TEXT NOT NULL,
                cache_salt TEXT,
                model_type INT NOT NULL,
                owner TEXT NOT NULL,
                output_path TEXT NOT NULL,
                parser_type TEXT NOT NULL DEFAULT 'passthrough',
                tag TEXT NOT NULL DEFAULT 'any',
                prompt_prefix_instructions TEXT
            )
        """)
        self._connection.commit()

        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                name TEXT PRIMARY KEY UNIQUE,
                family_name TEXT,
                path TEXT NOT NULL,
                revision TEXT,
                venv_path TEXT,
                conda_env TEXT,
                container_image TEXT,
                container_mounts TEXT,
                serving_slurm_resources TEXT,
                max_simultaneous_requests INT NOT NULL,
                max_time_to_deploy INT,
                allow_long_max_model_len INT NOT NULL DEFAULT 1,
                vllm_cli_args TEXT,
                vllm_logging_level TEXT NOT NULL DEFAULT 'WARNING',
                openai_kwargs TEXT NOT NULL,
                cache_salt TEXT,
                parser_type TEXT,
                model_type INT NOT NULL,
                owner TEXT NOT NULL,
                output_path TEXT NOT NULL,
                grading_only INT NOT NULL,
                tag TEXT NOT NULL DEFAULT 'any',
                prompt_prefix_instructions TEXT,
                base_url TEXT,
                api_key TEXT,
                requests_per_minute INT,
                is_external INT NOT NULL DEFAULT 0,
                api_model_name TEXT,
                external_retry_policy TEXT,
                registration_snapshot TEXT
            )
        """)

        self._connection.commit()

        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_events (
                uuid TEXT PRIMARY KEY,
                parent_uuid TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                model TEXT NOT NULL,
                task_uuid TEXT NOT NULL,
                path_to_generations TEXT,
                path_to_grades TEXT,
                path_to_scores TEXT,
                grader_type TEXT,
                grader_model_name TEXT,
                parser_type TEXT,
                phase INTEGER NOT NULL,
                event_type TEXT NOT NULL DEFAULT 'grading'
            )
        """)
        # Migrations for existing databases
        for col, definition in [
            ("imported_dataset", "TEXT"),
            ("tag", "TEXT NOT NULL DEFAULT 'any'"),
            ("event_type", "TEXT NOT NULL DEFAULT 'grading'"),
        ]:
            _add_column_if_missing(
                self._cursor,
                "task",
                col,
                definition,
            )
        _add_column_if_missing(
            self._cursor,
            "eval_events",
            "event_type",
            "TEXT NOT NULL DEFAULT 'grading'",
        )
        _add_column_if_missing(
            self._cursor,
            "models",
            "tag",
            "TEXT NOT NULL DEFAULT 'any'",
        )
        for table in ["model_family", "models"]:
            _add_column_if_missing(
                self._cursor,
                table,
                "allow_long_max_model_len",
                "INT NOT NULL DEFAULT 1",
            )
            _add_column_if_missing(
                self._cursor,
                table,
                "cache_salt",
                "TEXT",
            )
            _add_column_if_missing(
                self._cursor,
                table,
                "serving_slurm_resources",
                "TEXT",
            )
        for col, definition in [
            ("parser_type", "TEXT NOT NULL DEFAULT 'passthrough'"),
            ("tag", "TEXT NOT NULL DEFAULT 'any'"),
            ("prompt_prefix_instructions", "TEXT"),
        ]:
            _add_column_if_missing(
                self._cursor,
                "model_family",
                col,
                definition,
            )
        # Migrations for external model fields
        for col, definition in [
            ("base_url", "TEXT"),
            ("api_key", "TEXT"),
            ("requests_per_minute", "INT"),
            ("is_external", "INT NOT NULL DEFAULT 0"),
            ("api_model_name", "TEXT"),
            ("external_retry_policy", "TEXT"),
        ]:
            _add_column_if_missing(
                self._cursor,
                "models",
                col,
                definition,
            )
        for col, definition in [
            ("family_name", "TEXT"),
            (
                "vllm_logging_level",
                "TEXT NOT NULL DEFAULT 'WARNING'",
            ),
            ("registration_snapshot", "TEXT"),
        ]:
            _add_column_if_missing(
                self._cursor,
                "models",
                col,
                definition,
            )
        self._cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_events_phase0
            ON eval_events (phase)
            WHERE phase = 0;
        """)

        self._cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_events_phase1
            ON eval_events (phase)
            WHERE phase = 1;
        """)

        self._cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_events_parent
            ON eval_events (parent_uuid);
        """)

        self._connection.commit()
        logger.info("Database initialized")

    def _commit_if_not_in_transaction(self) -> None:
        if self._transaction_depth == 0:
            self._connection.commit()

    def _rollback_savepoint(self, savepoint: str) -> None:
        """Roll back one scope, falling back to the whole connection."""
        try:
            self._cursor.execute(
                f"ROLLBACK TO SAVEPOINT {savepoint}"
            )
            self._cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
        except BaseException:
            # A failed RELEASE can leave SQLite's savepoint state uncertain.
            # A connection-wide rollback is the only fail-closed recovery.
            self._connection.rollback()

    @contextmanager
    def transaction(self):
        """Atomically group registration writes, including nested callers."""
        savepoint = f"eval360_transaction_{self._transaction_depth}"
        self._cursor.execute(f"SAVEPOINT {savepoint}")
        self._transaction_depth += 1
        try:
            yield
            self._cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
        except BaseException:
            self._rollback_savepoint(savepoint)
            raise
        finally:
            self._transaction_depth -= 1

    def register_event(self, event_instance: EventInstance):
        # TODO: verify model and tasks both exist
        if isinstance(event_instance, GradingEventInstance):
            self._cursor.execute(
                """
                INSERT INTO eval_events (
                    uuid, parent_uuid, model, task_uuid, path_to_generations, path_to_grades,
                    path_to_scores, grader_type, parser_type, phase, event_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'grading')
                """,
                (event_instance.uuid,
                 event_instance.parent_uuid,
                 event_instance.model,
                 event_instance.task_uuid,
                 event_instance.path_to_generations,
                 event_instance.path_to_grades,
                 event_instance.path_to_scores,
                 event_instance.grader_type,
                 event_instance.parser_type,
                 0))
        else:
            self._cursor.execute(
                """
                INSERT INTO eval_events (
                    uuid, parent_uuid, model, task_uuid,
                    path_to_scores, phase, event_type
                )
                VALUES (?, ?, ?, ?, ?, ?, 'imported_dataset')
                """,
                (event_instance.uuid,
                 event_instance.parent_uuid,
                 event_instance.model,
                 event_instance.task_uuid,
                 event_instance.path_to_scores,
                 0))
        self._commit_if_not_in_transaction()

    def update_event(self, uuid: str, new_phase: int):
        """
        Update the phase of an eval_event identified by its UUID.

        Args:
            uuid (str): The UUID of the event to update.
            new_phase (int): The new phase value.
        """
        self._cursor.execute(
            """
            UPDATE eval_events
            SET phase = ?
            WHERE uuid = ?
            """,
            (new_phase, uuid)
        )
        self._commit_if_not_in_transaction()

    def event_exists_for_model_and_task(
        self,
        model: str,
        task_uuid: str,
    ) -> bool:
        """Return whether a model/task pair has any durable event."""
        self._cursor.execute(
            """
            SELECT 1
            FROM eval_events
            WHERE model = ? AND task_uuid = ?
            LIMIT 1
            """,
            (model, task_uuid),
        )
        return self._cursor.fetchone() is not None

    def get_event_by_model_and_task(
        self,
        model,
        task_uuid,
        *,
        include_completed: bool = False,
    ):
        self._cursor.execute(
            """
            SELECT
                uuid, parent_uuid,
                path_to_generations, path_to_grades, path_to_scores,
                grader_type, parser_type, phase, event_type
            FROM eval_events
            WHERE model = ? AND task_uuid = ? AND (? = 1 OR phase != 2)
            """,
            (model, task_uuid, int(include_completed)),
        )
        results = []
        for elem in self._cursor.fetchall():
            uuid, parent_uuid, path_to_generations, path_to_grades, path_to_scores, grader_type, parser_type, phase, event_type = elem
            if event_type == 'imported_dataset':
                results.append(ImportedDatasetEventInstance(
                    uuid=uuid,
                    parent_uuid=parent_uuid,
                    model=model,
                    task_uuid=task_uuid,
                    path_to_scores=path_to_scores,
                ))
            else:
                results.append(GradingEventInstance(
                    uuid=uuid,
                    parent_uuid=parent_uuid,
                    model=model,
                    task_uuid=task_uuid,
                    path_to_generations=path_to_generations,
                    path_to_grades=path_to_grades,
                    path_to_scores=path_to_scores,
                    grader_type=grader_type,
                    parser_type=parser_type,
                ))
        return results

    def _event_from_row(self, row) -> EventInstance:
        uuid, parent_uuid, model, task_uuid, path_to_generations, path_to_grades, path_to_scores, grader_type, parser_type, event_type = row
        if event_type == 'imported_dataset':
            return ImportedDatasetEventInstance(
                uuid=uuid,
                parent_uuid=parent_uuid,
                model=model,
                task_uuid=task_uuid,
                path_to_scores=path_to_scores,
            )
        return GradingEventInstance(
            uuid=uuid,
            parent_uuid=parent_uuid,
            model=model,
            task_uuid=task_uuid,
            path_to_generations=path_to_generations,
            path_to_grades=path_to_grades,
            path_to_scores=path_to_scores,
            grader_type=grader_type,
            parser_type=parser_type,
        )

    def get_all_incomplete_events(self):
        self._cursor.execute(
            """
            SELECT
                uuid, parent_uuid, model, task_uuid,
                path_to_generations, path_to_grades, path_to_scores,
                grader_type, parser_type, event_type
            FROM eval_events
            where phase = 0
            """)
        phase_0_rows = self._cursor.fetchall()
        self._cursor.execute(
            """
            SELECT
                uuid, parent_uuid, model, task_uuid,
                path_to_generations, path_to_grades, path_to_scores,
                grader_type, parser_type, event_type
            FROM eval_events
            where phase = 1
            """)
        phase_1_rows = self._cursor.fetchall()
        return (
            [self._event_from_row(row) for row in phase_0_rows],
            [self._event_from_row(row) for row in phase_1_rows],
        )

    def count_completed_events(self) -> int:
        self._cursor.execute("SELECT COUNT(*) FROM eval_events WHERE phase = 2 OR phase = -1")
        return self._cursor.fetchone()[0]

    def count_successful_events(self) -> int:
        self._cursor.execute("SELECT COUNT(*) FROM eval_events WHERE phase = 2")
        return self._cursor.fetchone()[0]

    def get_event_phase(self, uuid: str):
        self._cursor.execute(
            """
            SELECT phase
            FROM eval_events
            where uuid = ?
            """,
            (uuid,))
        return self._cursor.fetchone()[0]

    def register_task(self, task: Task):
        # TODO POC: check that meta/openai_settings work when null
        if isinstance(task, AsyncGenerationTask):
            grader_type = task.grader.type
            llm_as_judge_name = task.grader.llm_as_judge.name if task.grader.llm_as_judge else None
            average_over = json.dumps(task.average_over)
            pass_at = json.dumps(task.pass_at)
            data_path = task.data_path
            num_generations = task.num_generations
            imported_dataset_json = None
        else:
            assert isinstance(task, ImportedDatasetTask)
            grader_type = None
            llm_as_judge_name = None
            average_over = None
            pass_at = None
            data_path = None
            num_generations = None
            imported_dataset_json = task.imported_dataset.model_dump_json()
        self._cursor.execute(
            """
            INSERT OR IGNORE INTO task (
                uuid, average_over, pass_at, openai_settings, mode, grader_type, grader_model_name, data_path,
                dataset_name, semantic_version, tag, meta, num_generations, imported_dataset
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task.uuid, average_over, pass_at,
                  json.dumps(task.openai_settings), task.mode, grader_type, llm_as_judge_name, data_path,
                  task.dataset_name, task.semantic_version, task.tag, json.dumps(task.meta), num_generations,
                  imported_dataset_json)
        )
        self._commit_if_not_in_transaction()

    def update_task_num_generations(self, uuid: str, count: int):
        self._cursor.execute(
            "UPDATE task SET num_generations = ? WHERE uuid = ?",
            (count, uuid)
        )
        self._commit_if_not_in_transaction()

    def register_model_family(self, model_spec):
        self._cursor.execute(
            """
            INSERT OR REPLACE INTO model_family (
                model_family_name,
                path_glob,
                version_level,
                enqueue_existing,
                name_modifier,
                venv_path,
                conda_env,
                container_image,
                container_mounts,
                serving_slurm_resources,
                max_simultaneous_requests,
                max_time_to_deploy,
                allow_long_max_model_len,
                vllm_cli_args,
                openai_kwargs,
                cache_salt,
                model_type,
                owner,
                output_path,
                parser_type,
                tag,
                prompt_prefix_instructions
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
             model_spec.local_model.model_family_name,
             model_spec.local_model.path_glob,
             model_spec.local_model.version_level,
             int(model_spec.local_model.enqueue_existing),
             model_spec.name_modifier,
             model_spec.venv_path,
             model_spec.conda_env,
             model_spec.container_image,
             json.dumps(model_spec.container_mounts) if model_spec.container_mounts else None,
             _dump_serving_slurm_resources(model_spec.serving_slurm_resources),
             model_spec.max_simultaneous_requests,
             model_spec.max_time_to_deploy,
             int(model_spec.allow_long_max_model_len),
             json.dumps(model_spec.vllm_cli_args),
             json.dumps(model_spec.openai_kwargs),
             _dump_cache_salt(model_spec.cache_salt),
             model_spec.model_type.value,
             model_spec.owner,
             model_spec.output_path,
             model_spec.parser_type,
             model_spec.tag,
             model_spec.prompt_prefix_instructions,
            )
        )
        self._commit_if_not_in_transaction()

    def get_all_model_families(self):
        self._cursor.execute(
            """
            SELECT model_family_name, path_glob, version_level, enqueue_existing,
                   name_modifier, venv_path, conda_env, container_image, container_mounts,
                   serving_slurm_resources,
                   max_simultaneous_requests, max_time_to_deploy,
                   allow_long_max_model_len, vllm_cli_args, openai_kwargs, cache_salt, model_type,
                   owner, output_path, parser_type, tag, prompt_prefix_instructions
            FROM model_family
            """)
        rows = self._cursor.fetchall()
        specs = []
        for row in rows:
            (model_family_name, path_glob, version_level, enqueue_existing,
             name_modifier, venv_path, conda_env, container_image, container_mounts_json,
             serving_slurm_resources_json,
             max_simultaneous_requests, max_time_to_deploy,
             allow_long_max_model_len, vllm_cli_args, openai_kwargs, cache_salt_json, model_type,
             owner, output_path, parser_type, tag, prompt_prefix_instructions) = row
            specs.append(ModelSpec(
                local_model=LocalModel(
                    model_family_name=model_family_name,
                    path_glob=path_glob,
                    version_level=version_level,
                    enqueue_existing=bool(enqueue_existing),
                ),
                name_modifier=name_modifier,
                venv_path=venv_path,
                conda_env=conda_env,
                container_image=container_image,
                container_mounts=json.loads(container_mounts_json) if container_mounts_json else None,
                serving_slurm_resources=_load_serving_slurm_resources(
                    serving_slurm_resources_json
                ),
                max_simultaneous_requests=max_simultaneous_requests,
                max_time_to_deploy=max_time_to_deploy,
                allow_long_max_model_len=bool(allow_long_max_model_len),
                vllm_cli_args=json.loads(vllm_cli_args) if vllm_cli_args else [],
                openai_kwargs=json.loads(openai_kwargs) if openai_kwargs else {},
                cache_salt=_load_cache_salt(cache_salt_json),
                model_type=ModelType(model_type),
                owner=owner,
                ready=True,
                output_path=output_path,
                parser_type=parser_type or "passthrough",
                tag=tag or "any",
                prompt_prefix_instructions=prompt_prefix_instructions,
            ))
        return specs

    @staticmethod
    def _model_row(model: ModelInstance, grading_only: bool) -> dict[str, object]:
        """Serialize a model once for the ``models`` table's named columns."""
        return {
            "name": model.name,
            "family_name": model.family_name,
            "path": model.path,
            "revision": model.revision,
            "venv_path": model.venv_path,
            "conda_env": model.conda_env,
            "container_image": model.container_image,
            "container_mounts": json.dumps(model.container_mounts) if model.container_mounts else None,
            "serving_slurm_resources": _dump_serving_slurm_resources(
                model.serving_slurm_resources
            ),
            "max_simultaneous_requests": model.max_simultaneous_requests,
            "max_time_to_deploy": model.max_time_to_deploy,
            "allow_long_max_model_len": model.allow_long_max_model_len,
            "vllm_cli_args": json.dumps(model.vllm_cli_args) if model.vllm_cli_args is not None else None,
            "vllm_logging_level": model.vllm_logging_level,
            "openai_kwargs": json.dumps(model.openai_kwargs),
            "cache_salt": _dump_cache_salt(model.cache_salt),
            "parser_type": model.parser_type,
            "model_type": model.model_type.value,
            "owner": model.owner,
            "output_path": model.output_path,
            "grading_only": grading_only,
            "tag": model.tag,
            "prompt_prefix_instructions": model.prompt_prefix_instructions,
            "base_url": model.base_url,
            "api_key": model.api_key,
            "requests_per_minute": model.requests_per_minute,
            "is_external": 1 if model.is_external else 0,
            "api_model_name": model.api_model_name,
            "external_retry_policy": _dump_external_retry_policy(
                model.external_retry_policy
            ),
            "registration_snapshot": _dump_model_registration_snapshot(model),
        }

    def register_model(self, model: ModelInstance, grading_only=False):
        row = self._model_row(model, grading_only)
        columns = ", ".join(row)
        placeholders = ", ".join(f":{column}" for column in row)
        self._cursor.execute(
            f"""
            INSERT INTO models ({columns})
            VALUES ({placeholders})
            ON CONFLICT(name) DO UPDATE SET
                family_name = excluded.family_name,
                path = excluded.path,
                revision = excluded.revision,
                venv_path = excluded.venv_path,
                conda_env = excluded.conda_env,
                container_image = excluded.container_image,
                container_mounts = excluded.container_mounts,
                serving_slurm_resources = excluded.serving_slurm_resources,
                max_simultaneous_requests = excluded.max_simultaneous_requests,
                max_time_to_deploy = excluded.max_time_to_deploy,
                allow_long_max_model_len = excluded.allow_long_max_model_len,
                vllm_cli_args = excluded.vllm_cli_args,
                vllm_logging_level = excluded.vllm_logging_level,
                openai_kwargs = excluded.openai_kwargs,
                cache_salt = excluded.cache_salt,
                parser_type = excluded.parser_type,
                model_type = excluded.model_type,
                owner = excluded.owner,
                output_path = excluded.output_path,
                grading_only = models.grading_only AND excluded.grading_only,
                tag = excluded.tag,
                prompt_prefix_instructions = excluded.prompt_prefix_instructions,
                base_url = excluded.base_url,
                api_key = excluded.api_key,
                requests_per_minute = excluded.requests_per_minute,
                is_external = excluded.is_external,
                api_model_name = excluded.api_model_name,
                external_retry_policy = excluded.external_retry_policy,
                registration_snapshot = excluded.registration_snapshot
            """, row,
        )
        self._commit_if_not_in_transaction()

    def get_all_models(self) -> dict[str, ModelInstance]:
        self._cursor.execute(
            """
            SELECT name, family_name, path, revision, venv_path, conda_env, container_image, container_mounts,
               serving_slurm_resources,
               max_simultaneous_requests,
               max_time_to_deploy, allow_long_max_model_len, vllm_cli_args,
               vllm_logging_level, openai_kwargs,
               cache_salt, parser_type, model_type, owner, output_path, tag, prompt_prefix_instructions,
               base_url, api_key, requests_per_minute, is_external, api_model_name,
               external_retry_policy
            FROM models
            where grading_only = 0
            """
        )
        rows = self._cursor.fetchall()

        models = {}
        for row in rows:
            models[row[0]] = (
                ModelInstance(
                    name=row[0],
                    family_name=row[1],
                    path=row[2],
                    revision=row[3],
                    venv_path=row[4],
                    conda_env=row[5],
                    container_image=row[6],
                    container_mounts=json.loads(row[7]) if row[7] else None,
                    serving_slurm_resources=_load_serving_slurm_resources(row[8]),
                    max_simultaneous_requests=row[9],
                    max_time_to_deploy=row[10],
                    allow_long_max_model_len=bool(row[11]),
                    vllm_cli_args=json.loads(row[12]) if row[12] is not None else None,
                    vllm_logging_level=row[13],
                    openai_kwargs=json.loads(row[14]) if row[14] else {},
                    cache_salt=_load_cache_salt(row[15]),
                    parser_type=row[16],
                    model_type=ModelType(row[17]),
                    owner=row[18],
                    output_path=row[19],
                    tag=row[20],
                    prompt_prefix_instructions=row[21],
                    base_url=row[22],
                    api_key=row[23],
                    requests_per_minute=row[24],
                    is_external=bool(row[25]) if row[25] is not None else False,
                    api_model_name=row[26],
                    external_retry_policy=_load_external_retry_policy(row[27]),
                )
            )

        return models

    def get_model(self, name: str):
        self._cursor.execute(
            """
            SELECT family_name, path, revision, venv_path, conda_env, container_image, container_mounts,
               serving_slurm_resources,
               max_simultaneous_requests,
               max_time_to_deploy, allow_long_max_model_len, vllm_cli_args,
               vllm_logging_level, openai_kwargs,
               cache_salt, parser_type, model_type, owner, output_path, tag, prompt_prefix_instructions,
               base_url, api_key, requests_per_minute, is_external, api_model_name,
               external_retry_policy
            FROM models
            where name = ?
            LIMIT 1
            """, (name,))
        row = self._cursor.fetchone()
        model = ModelInstance(
                    name=name,
                    family_name=row[0],
                    path=row[1],
                    revision=row[2],
                    venv_path=row[3],
                    conda_env=row[4],
                    container_image=row[5],
                    container_mounts=json.loads(row[6]) if row[6] else None,
                    serving_slurm_resources=_load_serving_slurm_resources(row[7]),
                    max_simultaneous_requests=row[8],
                    max_time_to_deploy=row[9],
                    allow_long_max_model_len=bool(row[10]),
                    vllm_cli_args=json.loads(row[11]) if row[11] is not None else None,
                    vllm_logging_level=row[12],
                    openai_kwargs=json.loads(row[13]) if row[13] else {},
                    cache_salt=_load_cache_salt(row[14]),
                    parser_type=row[15],
                    model_type=ModelType(row[16]),
                    owner=row[17],
                    output_path=row[18],
                    tag=row[19],
                    prompt_prefix_instructions=row[20],
                    base_url=row[21],
                    api_key=row[22],
                    requests_per_minute=row[23],
                    is_external=bool(row[24]) if row[24] is not None else False,
                    api_model_name=row[25],
                    external_retry_policy=_load_external_retry_policy(row[26]),
                    )

        return model

    def get_model_if_exists(self, name: str) -> ModelInstance | None:
        """Return a registered model by logical name, or ``None``."""
        self._cursor.execute(
            "SELECT 1 FROM models WHERE name = ? LIMIT 1",
            (name,),
        )
        if self._cursor.fetchone() is None:
            return None
        return self.get_model(name)

    def get_model_registration_snapshot(
        self,
        name: str,
    ) -> dict[str, object] | None:
        """Return the persisted secret-safe immutable model snapshot."""
        self._cursor.execute(
            "SELECT registration_snapshot FROM models WHERE name = ? LIMIT 1",
            (name,),
        )
        row = self._cursor.fetchone()
        if row is None or row[0] is None:
            return None
        snapshot = json.loads(row[0])
        if not isinstance(snapshot, dict):
            raise ValueError(
                f"invalid registration snapshot for model {name!r}"
            )
        return snapshot

    def _task_from_row(self, task_uuid, row) -> "ImportedDatasetTask | AsyncGenerationTask":
        average_over, pass_at, openai_settings, mode, grader_type, grader_model_name, data_path, dataset_name, semantic_version, tag, meta, num_generations, imported_dataset = row
        common = dict(
            uuid=task_uuid,
            openai_settings=json.loads(openai_settings) if openai_settings else None,
            mode=mode,
            dataset_name=dataset_name,
            semantic_version=semantic_version,
            tag=tag,
            meta=json.loads(meta) if meta else None,
        )
        if imported_dataset:
            return ImportedDatasetTask(
                **common,
                imported_dataset=ImportedDatasetConfig.model_validate_json(imported_dataset),
            )
        llm_as_judge = self.get_model(grader_model_name) if grader_model_name else None
        return AsyncGenerationTask(
            **common,
            average_over=json.loads(average_over),
            pass_at=json.loads(pass_at),
            grader=GraderConfig(type=grader_type, llm_as_judge=llm_as_judge),
            data_path=data_path,
            num_generations=num_generations,
        )

    def get_all_tasks(self):
        self._cursor.execute("""
            SELECT uuid, average_over, pass_at, openai_settings, mode, grader_type, grader_model_name, data_path, dataset_name, semantic_version, tag, meta, num_generations, imported_dataset
            FROM task
            """)
        elems = self._cursor.fetchall()
        return [self._task_from_row(elem[0], elem[1:]) for elem in elems]

    def get_task(self, task_uuid):
        self._cursor.execute("""
            SELECT average_over, pass_at, openai_settings, mode, grader_type, grader_model_name, data_path, dataset_name, semantic_version, tag, meta, num_generations, imported_dataset
            FROM task
            where uuid = ?
            LIMIT 1
            """, (task_uuid,))
        row = self._cursor.fetchone()
        if row is None:
            return None
        return self._task_from_row(task_uuid, row)

    def __aiter__(self):
        return self

    async def __anext__(self):
        # TODO: figure out if this is even needed
        await asyncio.sleep(3)
        self._cursor.execute("""
            SELECT uuid, task_uuid, model
            FROM eval_events
            WHERE phase = 2
            """)

        rows = [{"UUID": uuid,
                 "messages": json.loads(messages),
                 "model": model}
                for uuid, messages, model in self._cursor.fetchall()]

        return rows
