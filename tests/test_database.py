"""
Unit tests for scheduler/database.py (DatabaseManager).

All tests use an in-memory SQLite database (EVAL360_IN_MEMORY_DB=true is set in conftest.py).
Each test receives a fresh DatabaseManager instance via the `db` fixture.
"""
import sqlite3

import pytest

from scheduler import database as database_module
from scheduler.database import DatabaseManager
from scheduler.event import GradingEventInstance, ImportedDatasetEventInstance
from scheduler.model import (
    CacheSaltConfig,
    ExternalRetryPolicy,
    LocalModel,
    ModelInstance,
    ModelSpec,
    ModelType,
    ServingSlurmResources,
)
from scheduler.task import GraderConfig, AsyncGenerationTask, ImportedDatasetTask, ImportedDatasetConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model_spec(
    family_name: str = "test-family",
    path_glob: str = "/checkpoints/test/**",
    parser_type: str = "noop",
    tag: str = "any",
    prompt_prefix_instructions: str | None = None,
) -> ModelSpec:
    return ModelSpec(
        local_model=LocalModel(
            model_family_name=family_name,
            path_glob=path_glob,
            version_level=-1,
            enqueue_existing=True,
        ),
        name_modifier=None,
        venv_path="/path/to/venv/bin/activate",
        max_simultaneous_requests=100,
        max_time_to_deploy=600,
        allow_long_max_model_len=True,
        vllm_cli_args=["--tensor-parallel-size", "8"],
        openai_kwargs={"temperature": 0.0},
        model_type=ModelType.CHAT,
        owner="test.user",
        ready=True,
        output_path="/output/test-family",
        parser_type=parser_type,
        tag=tag,
        prompt_prefix_instructions=prompt_prefix_instructions,
    )


def make_model(name: str = "test-model", tag: str = "any") -> ModelInstance:
    return ModelInstance(
        name=name,
        path="LLM360/TestModel",
        revision=None,
        venv_path="/path/to/venv/bin/activate",
        max_simultaneous_requests=100,
        max_time_to_deploy=600,
        allow_long_max_model_len=True,
        vllm_cli_args=["--tensor-parallel-size", "1"],
        openai_kwargs={"temperature": 0.0},
        parser_type="noop",
        model_type=ModelType.BASE,
        owner="test.user",
        output_path="/output/test-model",
        tag=tag,
    )


def make_task(uuid: str = "task-uuid-1", tag: str = "any") -> AsyncGenerationTask:
    return AsyncGenerationTask(
        uuid=uuid,
        average_over=[1],
        pass_at=[1],
        mode="base",
        grader=GraderConfig(type="multiple_choice"),
        data_path="/data/mmlu/*.jsonl",
        dataset_name="mmlu",
        semantic_version="1.0.0",
        meta={"split": "test"},
        num_generations=100,
        tag=tag,
    )


def make_event(
    model_name: str = "test-model",
    task_uuid: str = "task-uuid-1",
    suffix: str = "",
) -> GradingEventInstance:
    return GradingEventInstance(
        uuid=f"event-uuid{suffix}",
        parent_uuid=f"parent-uuid{suffix}",
        model=model_name,
        task_uuid=task_uuid,
        path_to_generations=f"/output/test-model/mmlu_generations{suffix}.jsonl",
        path_to_grades=f"/output/test-model/mmlu_grades{suffix}.jsonl",
        path_to_scores=f"/output/test-model/mmlu_scores{suffix}.jsonl",
        grader_type="multiple_choice",
        parser_type="noop",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_required_tables(self, db: DatabaseManager):
        db._cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in db._cursor.fetchall()}
        assert {"task", "models", "eval_events", "model_family"}.issubset(tables)

    def test_creates_phase_index(self, db: DatabaseManager):
        db._cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in db._cursor.fetchall()}
        assert "idx_eval_events_phase0" in indices
        assert "idx_eval_events_phase1" in indices

    @pytest.mark.parametrize(
        "failing_migration",
        [
            "ALTER TABLE task ADD COLUMN imported_dataset TEXT",
            (
                "ALTER TABLE eval_events ADD COLUMN event_type "
                "TEXT NOT NULL DEFAULT 'grading'"
            ),
            "ALTER TABLE models ADD COLUMN tag TEXT NOT NULL DEFAULT 'any'",
            (
                "ALTER TABLE model_family ADD COLUMN "
                "allow_long_max_model_len INT NOT NULL DEFAULT 1"
            ),
            (
                "ALTER TABLE model_family ADD COLUMN "
                "parser_type TEXT NOT NULL DEFAULT 'passthrough'"
            ),
            "ALTER TABLE models ADD COLUMN base_url TEXT",
        ],
    )
    def test_legacy_migrations_reraise_unexpected_sqlite_error(
        self,
        monkeypatch,
        failing_migration,
    ):
        connection = sqlite3.connect(":memory:")

        class CursorProxy:
            def __init__(self, cursor):
                self._cursor = cursor

            def execute(self, sql, *args):
                if " ".join(sql.split()) == failing_migration:
                    raise sqlite3.OperationalError("disk I/O error")
                return self._cursor.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._cursor, name)

        class ConnectionProxy:
            def cursor(self):
                return CursorProxy(connection.cursor())

            def __getattr__(self, name):
                return getattr(connection, name)

        monkeypatch.setattr(
            database_module.sqlite3,
            "connect",
            lambda _path: ConnectionProxy(),
        )

        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            DatabaseManager()

    def test_external_column_migration_reraises_unexpected_sqlite_error(
        self,
        monkeypatch,
    ):
        connection = sqlite3.connect(":memory:")

        class CursorProxy:
            def __init__(self, cursor):
                self._cursor = cursor

            def execute(self, sql, *args):
                normalized_sql = " ".join(sql.split())
                if normalized_sql == (
                    "ALTER TABLE models ADD COLUMN "
                    "external_retry_policy TEXT"
                ):
                    raise sqlite3.OperationalError("disk I/O error")
                return self._cursor.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._cursor, name)

        class ConnectionProxy:
            def cursor(self):
                return CursorProxy(connection.cursor())

            def __getattr__(self, name):
                return getattr(connection, name)

        monkeypatch.setattr(
            database_module.sqlite3,
            "connect",
            lambda _path: ConnectionProxy(),
        )

        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            DatabaseManager()

    def test_external_columns_are_added_to_legacy_models_table(
        self,
        monkeypatch,
    ):
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE models (name TEXT PRIMARY KEY)"
        )
        monkeypatch.setattr(
            database_module.sqlite3,
            "connect",
            lambda _path: connection,
        )

        migrated_db = DatabaseManager()

        migrated_db._cursor.execute("PRAGMA table_info(models)")
        columns = {
            row[1]
            for row in migrated_db._cursor.fetchall()
        }
        assert {
            "base_url",
            "api_key",
            "requests_per_minute",
            "is_external",
            "api_model_name",
            "external_retry_policy",
            "family_name",
            "vllm_logging_level",
            "registration_snapshot",
            "serving_slurm_resources",
        }.issubset(columns)

# ---------------------------------------------------------------------------
# Model CRUD
# ---------------------------------------------------------------------------

class TestModel:
    def test_register_and_get_model(self, db: DatabaseManager):
        model = make_model()
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.name == model.name
        assert retrieved.path == model.path
        assert retrieved.model_type == ModelType.BASE
        assert retrieved.owner == model.owner
        assert retrieved.tag == "any"

    def test_vllm_cli_args_round_trips(self, db: DatabaseManager):
        db.register_model(make_model())
        assert db.get_model("test-model").vllm_cli_args == ["--tensor-parallel-size", "1"]

    def test_openai_kwargs_round_trips(self, db: DatabaseManager):
        db.register_model(make_model())
        assert db.get_model("test-model").openai_kwargs == {"temperature": 0.0}

    def test_serving_slurm_resources_round_trip(self, db: DatabaseManager):
        model = make_model()
        model.serving_slurm_resources = ServingSlurmResources(
            # Obviously arbitrary placeholders, distinct from the 12345 defaults.
            gpus_per_node=23456,
            cpus_per_task=34567,
            memory_gb=45678,
            time_limit="12:34:56",
        )
        db.register_model(model)

        assert (
            db.get_model(model.name).serving_slurm_resources
            == model.serving_slurm_resources
        )
        assert (
            db.get_all_models()[model.name].serving_slurm_resources
            == model.serving_slurm_resources
        )

    def test_legacy_null_serving_slurm_resources_uses_defaults(
        self, db: DatabaseManager
    ):
        model = make_model()
        db.register_model(model)
        db._cursor.execute(
            "UPDATE models SET serving_slurm_resources = NULL WHERE name = ?",
            (model.name,),
        )
        db._connection.commit()

        assert db.get_model(model.name).serving_slurm_resources == (
            ServingSlurmResources()
        )

    def test_complete_registration_snapshot_and_lost_fields_round_trip(
        self,
        db: DatabaseManager,
    ):
        model = make_model()
        model.family_name = "custom-family"
        model.vllm_logging_level = "DEBUG"
        model.api_key = "secret-for-fingerprint"

        db.register_model(model)

        retrieved = db.get_model(model.name)
        snapshot = db.get_model_registration_snapshot(model.name)
        assert retrieved.family_name == "custom-family"
        assert retrieved.vllm_logging_level == "DEBUG"
        assert snapshot == model.registration_snapshot()
        assert "secret-for-fingerprint" not in str(snapshot)

    def test_cache_salt_round_trips(self, db: DatabaseManager):
        model = make_model()
        model.cache_salt = CacheSaltConfig(mode="static", salt="partition-a")
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.cache_salt.mode == "static"
        assert retrieved.cache_salt.salt == "partition-a"

    def test_legacy_null_cache_salt_hydrates_to_disabled(self, db: DatabaseManager):
        model = make_model()
        db.register_model(model)
        db._cursor.execute("UPDATE models SET cache_salt = NULL WHERE name = ?", (model.name,))
        db._connection.commit()
        retrieved = db.get_model(model.name)
        assert retrieved.cache_salt == CacheSaltConfig(mode="disabled")

    def test_get_all_models_cache_salt_round_trips(self, db: DatabaseManager):
        model = make_model()
        model.cache_salt = CacheSaltConfig(mode="unique")
        db.register_model(model)
        retrieved = db.get_all_models()[model.name]
        assert retrieved.cache_salt.mode == "unique"

    def test_external_model_cache_salt_round_trips(self, db: DatabaseManager):
        model = make_model("external-model")
        model.path = "https://api.openai.com/v1"
        model.base_url = "https://api.openai.com/v1"
        model.api_key = "test-key"
        model.api_model_name = "gpt-4o"
        model.is_external = True
        model.cache_salt = CacheSaltConfig(mode="static", salt="external-partition")
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.cache_salt.mode == "static"
        assert retrieved.cache_salt.salt == "external-partition"
        assert retrieved.is_external is True

    def test_allow_long_max_model_len_round_trips(self, db: DatabaseManager):
        db.register_model(make_model())
        assert db.get_model("test-model").allow_long_max_model_len is True

        enabled = make_model("long-model")
        enabled.allow_long_max_model_len = True
        db.register_model(enabled)
        assert db.get_model("long-model").allow_long_max_model_len is True

        disabled = make_model("short-model")
        disabled.allow_long_max_model_len = False
        db.register_model(disabled)
        assert db.get_model("short-model").allow_long_max_model_len is False

    def test_register_model_upserts_on_conflict(self, db: DatabaseManager):
        db.register_model(make_model())
        updated = ModelInstance(
            name="test-model",
            path="updated/path",
            revision="abc123",
            venv_path="/venv/bin/activate",
            max_simultaneous_requests=50,
            max_time_to_deploy=300,
            allow_long_max_model_len=True,
            vllm_cli_args=[],
            openai_kwargs={},
            parser_type="noop",
            model_type=ModelType.BASE,
            owner="test.user",
            output_path="/output/test-model",
            tag="chat",
        )
        db.register_model(updated)
        retrieved = db.get_model("test-model")
        assert retrieved.path == "updated/path"
        assert retrieved.revision == "abc123"
        assert retrieved.max_simultaneous_requests == 50
        assert retrieved.tag == "chat"
        assert retrieved.allow_long_max_model_len is True

    def test_get_all_models_excludes_grading_only(self, db: DatabaseManager):
        db.register_model(make_model("gen-model"), grading_only=False)
        db.register_model(make_model("judge-model"), grading_only=True)
        all_models = db.get_all_models()
        assert "gen-model" in all_models
        assert "judge-model" not in all_models

    def test_grading_only_judge_cache_salt_hydrates_through_task(self, db: DatabaseManager):
        judge = make_model("judge-model")
        judge.cache_salt = CacheSaltConfig(mode="static", salt="judge-partition")
        db.register_model(judge, grading_only=True)
        task = AsyncGenerationTask(
            uuid="judge-task",
            average_over=[1],
            pass_at=[1],
            mode="base",
            grader=GraderConfig(type="multiple_choice", llm_as_judge=judge),
            data_path="/data/mmlu/*.jsonl",
            dataset_name="mmlu",
            semantic_version="1.0.0",
            num_generations=100,
        )
        db.register_task(task)

        hydrated = db.get_task(task.uuid)
        assert hydrated.grader.llm_as_judge.cache_salt.mode == "static"
        assert hydrated.grader.llm_as_judge.cache_salt.salt == "judge-partition"

    def test_get_all_models_returns_all_non_grading(self, db: DatabaseManager):
        for i in range(3):
            db.register_model(make_model(f"model-{i}"))
        assert len(db.get_all_models()) == 3

    def test_model_revision_nullable(self, db: DatabaseManager):
        model = make_model()
        db.register_model(model)
        assert db.get_model(model.name).revision is None

    def test_model_tag_round_trips(self, db: DatabaseManager):
        model = make_model(tag="vision")
        db.register_model(model)
        assert db.get_model(model.name).tag == "vision"

    def test_prompt_prefix_instructions_round_trips(self, db: DatabaseManager):
        """prompt_prefix_instructions must survive register → get_model."""
        model = make_model()
        model.prompt_prefix_instructions = r"Put your answer in \boxed{}"
        db.register_model(model)
        assert db.get_model(model.name).prompt_prefix_instructions == r"Put your answer in \boxed{}"

    def test_prompt_prefix_instructions_none_by_default(self, db: DatabaseManager):
        """Models without prompt_prefix_instructions must return None, not crash."""
        db.register_model(make_model())
        assert db.get_model("test-model").prompt_prefix_instructions is None

    def test_prompt_prefix_instructions_in_get_all_models(self, db: DatabaseManager):
        """prompt_prefix_instructions must also survive register → get_all_models."""
        model = make_model()
        model.prompt_prefix_instructions = "Answer: "
        db.register_model(model)
        retrieved = db.get_all_models()["test-model"]
        assert retrieved.prompt_prefix_instructions == "Answer: "

    def test_prompt_prefix_instructions_updated_on_upsert(self, db: DatabaseManager):
        """Re-registering a model with a new prefix must overwrite the old one."""
        model = make_model()
        model.prompt_prefix_instructions = "old prefix"
        db.register_model(model)

        model.prompt_prefix_instructions = "new prefix"
        db.register_model(model)

        assert db.get_model(model.name).prompt_prefix_instructions == "new prefix"


# ---------------------------------------------------------------------------
# Task CRUD
# ---------------------------------------------------------------------------

class TestTask:
    def test_register_and_get_task(self, db: DatabaseManager):
        task = make_task()
        db.register_task(task)
        retrieved = db.get_task(task.uuid)
        assert retrieved.uuid == task.uuid
        assert retrieved.dataset_name == "mmlu"
        assert retrieved.data_path == "/data/mmlu/*.jsonl"
        assert retrieved.average_over == [1]
        assert retrieved.pass_at == [1]
        assert retrieved.tag == "any"

    def test_register_task_is_idempotent(self, db: DatabaseManager):
        task = make_task()
        db.register_task(task)
        db.register_task(task)  # second insert ignored via INSERT OR IGNORE
        tasks = db.get_all_tasks()
        assert len(tasks) == 1

    def test_meta_round_trips(self, db: DatabaseManager):
        db.register_task(make_task())
        assert db.get_task("task-uuid-1").meta == {"split": "test"}

    def test_get_all_tasks_returns_all(self, db: DatabaseManager):
        for i in range(3):
            db.register_task(make_task(uuid=f"task-{i}"))
        assert len(db.get_all_tasks()) == 3

    def test_task_with_null_meta(self, db: DatabaseManager):
        task = AsyncGenerationTask(
            uuid="task-null-meta",
            average_over=[1],
            pass_at=[1],
            mode="base",
            grader=GraderConfig(type="multiple_choice"),
            data_path="/data/*.jsonl",
            dataset_name="test",
            semantic_version="1.0.0",
            meta=None,
            num_generations=10,
            tag="text",
        )
        db.register_task(task)
        retrieved = db.get_task("task-null-meta")
        assert retrieved.meta is None
        assert retrieved.tag == "text"

    def test_task_tag_round_trips(self, db: DatabaseManager):
        task = make_task(tag="vision")
        db.register_task(task)
        assert db.get_task(task.uuid).tag == "vision"


# ---------------------------------------------------------------------------
# Event CRUD
# ---------------------------------------------------------------------------

class TestEvent:
    def _setup(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task())

    def test_register_event_initial_phase_is_zero(self, db: DatabaseManager):
        self._setup(db)
        event = make_event()
        db.register_event(event)
        assert db.get_event_phase(event.uuid) == 0

    def test_update_event_phase(self, db: DatabaseManager):
        self._setup(db)
        event = make_event()
        db.register_event(event)
        db.update_event(event.uuid, 1)
        assert db.get_event_phase(event.uuid) == 1
        db.update_event(event.uuid, 2)
        assert db.get_event_phase(event.uuid) == 2

    def test_update_event_rolls_back_with_surrounding_transaction(
        self,
        db: DatabaseManager,
    ):
        self._setup(db)
        event = make_event()
        db.register_event(event)

        with pytest.raises(RuntimeError, match="abort transaction"):
            with db.transaction():
                db.update_event(event.uuid, 1)
                raise RuntimeError("abort transaction")

        assert db.get_event_phase(event.uuid) == 0

    def test_release_failure_rolls_back_pending_transaction(
        self,
        db: DatabaseManager,
    ):
        real_cursor = db._cursor

        class ReleaseFailingCursor:
            def __init__(self):
                self.failed = False

            def execute(self, sql, *args):
                if (
                    not self.failed
                    and sql.startswith("RELEASE SAVEPOINT")
                ):
                    self.failed = True
                    raise sqlite3.OperationalError("release failed")
                return real_cursor.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(real_cursor, name)

        db._cursor = ReleaseFailingCursor()
        aborted = make_task(uuid="aborted-release")

        with pytest.raises(sqlite3.OperationalError, match="release failed"):
            with db.transaction():
                db.register_task(aborted)

        assert db._transaction_depth == 0
        assert db.get_task(aborted.uuid) is None

        committed = make_task(uuid="committed-after-release")
        db.register_task(committed)
        assert db.get_task(committed.uuid) == committed
        assert db.get_task(aborted.uuid) is None

    def test_get_event_by_model_and_task(self, db: DatabaseManager):
        self._setup(db)
        event = make_event()
        db.register_event(event)
        results = db.get_event_by_model_and_task("test-model", "task-uuid-1")
        assert len(results) == 1
        assert results[0].uuid == event.uuid
        assert results[0].grader_type == "multiple_choice"

    def test_get_event_by_model_and_task_excludes_completed(self, db: DatabaseManager):
        """Phase-2 (complete) events must not be returned — they should not be re-enqueued on restart."""
        self._setup(db)
        event = make_event()
        db.register_event(event)
        db.update_event(event.uuid, 2)
        results = db.get_event_by_model_and_task("test-model", "task-uuid-1")
        assert results == []

    def test_get_event_by_model_and_task_can_include_completed(self, db: DatabaseManager):
        self._setup(db)
        event = make_event()
        db.register_event(event)
        db.update_event(event.uuid, 2)

        results = db.get_event_by_model_and_task(
            "test-model",
            "task-uuid-1",
            include_completed=True,
        )

        assert [result.uuid for result in results] == [event.uuid]

    def test_get_event_by_model_and_task_includes_failed(self, db: DatabaseManager):
        """Phase -1 (failed) events must still be returned so they can retry on restart."""
        self._setup(db)
        event = make_event()
        db.register_event(event)
        db.update_event(event.uuid, -1)
        results = db.get_event_by_model_and_task("test-model", "task-uuid-1")
        assert len(results) == 1
        assert results[0].uuid == event.uuid

    def test_get_event_by_model_and_task_empty(self, db: DatabaseManager):
        results = db.get_event_by_model_and_task("nonexistent-model", "nonexistent-task")
        assert results == []

    def test_get_event_by_model_filters_by_task(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task("task-1"))
        db.register_task(make_task("task-2"))
        event1 = make_event(task_uuid="task-1", suffix="-t1")
        event2 = make_event(task_uuid="task-2", suffix="-t2")
        db.register_event(event1)
        db.register_event(event2)
        results = db.get_event_by_model_and_task("test-model", "task-1")
        assert len(results) == 1
        assert results[0].uuid == event1.uuid

    def test_get_all_incomplete_events_phase_separation(self, db: DatabaseManager):
        self._setup(db)
        ev_0a = make_event(suffix="-0a")
        ev_0b = make_event(suffix="-0b")
        ev_1 = make_event(suffix="-1")
        ev_done = make_event(suffix="-done")
        for ev in [ev_0a, ev_0b, ev_1, ev_done]:
            db.register_event(ev)
        db.update_event(ev_1.uuid, 1)
        db.update_event(ev_done.uuid, 2)

        phase_0, phase_1 = db.get_all_incomplete_events()
        p0_uuids = {e.uuid for e in phase_0}
        p1_uuids = {e.uuid for e in phase_1}

        assert ev_0a.uuid in p0_uuids
        assert ev_0b.uuid in p0_uuids
        assert ev_1.uuid in p1_uuids
        assert ev_done.uuid not in p0_uuids
        assert ev_done.uuid not in p1_uuids

    def test_get_all_incomplete_events_returns_event_instance_fields(self, db: DatabaseManager):
        self._setup(db)
        event = make_event()
        db.register_event(event)
        phase_0, _ = db.get_all_incomplete_events()
        assert len(phase_0) == 1
        result = phase_0[0]
        assert result.model == "test-model"
        assert result.task_uuid == "task-uuid-1"
        assert result.grader_type == "multiple_choice"
        assert result.path_to_generations == event.path_to_generations


# ---------------------------------------------------------------------------
# get_event_phase (direct coverage)
# ---------------------------------------------------------------------------

class TestGetEventPhase:

    def test_initial_phase_is_zero(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task())
        event = make_event()
        db.register_event(event)
        assert db.get_event_phase(event.uuid) == 0

    def test_phase_after_update_to_one(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task())
        event = make_event()
        db.register_event(event)
        db.update_event(event.uuid, 1)
        assert db.get_event_phase(event.uuid) == 1

    def test_phase_after_update_to_two(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task())
        event = make_event()
        db.register_event(event)
        db.update_event(event.uuid, 2)
        assert db.get_event_phase(event.uuid) == 2

    def test_failed_phase_minus_one(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task())
        event = make_event()
        db.register_event(event)
        db.update_event(event.uuid, -1)
        assert db.get_event_phase(event.uuid) == -1


# ---------------------------------------------------------------------------
# count_completed_events / count_successful_events
# ---------------------------------------------------------------------------

class TestCountEvents:

    def _register(self, db: DatabaseManager, count: int, suffix_start: int = 0):
        for i in range(count):
            db.register_event(make_event(suffix=f"-{suffix_start + i}"))

    def _setup(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task())

    def test_zero_when_no_events(self, db: DatabaseManager):
        assert db.count_completed_events() == 0
        assert db.count_successful_events() == 0

    def test_phase_0_events_not_counted(self, db: DatabaseManager):
        self._setup(db)
        self._register(db, 3)
        assert db.count_completed_events() == 0
        assert db.count_successful_events() == 0

    def test_phase_2_counts_as_completed_and_successful(self, db: DatabaseManager):
        self._setup(db)
        ev = make_event(suffix="-done")
        db.register_event(ev)
        db.update_event(ev.uuid, 2)
        assert db.count_completed_events() == 1
        assert db.count_successful_events() == 1

    def test_phase_minus_one_counts_as_completed_but_not_successful(self, db: DatabaseManager):
        self._setup(db)
        ev = make_event(suffix="-fail")
        db.register_event(ev)
        db.update_event(ev.uuid, -1)
        assert db.count_completed_events() == 1
        assert db.count_successful_events() == 0

    def test_mixed_phases(self, db: DatabaseManager):
        self._setup(db)
        events = [make_event(suffix=f"-{i}") for i in range(5)]
        for ev in events:
            db.register_event(ev)
        # phase 0: events[0] (untouched)
        # phase 1: events[1]
        db.update_event(events[1].uuid, 1)
        # phase 2: events[2], events[3]
        db.update_event(events[2].uuid, 2)
        db.update_event(events[3].uuid, 2)
        # phase -1: events[4]
        db.update_event(events[4].uuid, -1)

        assert db.count_completed_events() == 3   # phase 2×2 + phase -1×1
        assert db.count_successful_events() == 2  # only phase 2


# ---------------------------------------------------------------------------
# update_task_num_generations
# ---------------------------------------------------------------------------

class TestUpdateTaskNumGenerations:

    def test_updates_num_generations(self, db: DatabaseManager):
        task = make_task()
        db.register_task(task)
        db.update_task_num_generations(task.uuid, 500)
        assert db.get_task(task.uuid).num_generations == 500

    def test_update_overwrites_previous_value(self, db: DatabaseManager):
        task = make_task()
        db.register_task(task)
        db.update_task_num_generations(task.uuid, 200)
        db.update_task_num_generations(task.uuid, 999)
        assert db.get_task(task.uuid).num_generations == 999

    def test_other_task_unaffected(self, db: DatabaseManager):
        task1 = make_task(uuid="task-a")
        task2 = make_task(uuid="task-b")
        db.register_task(task1)
        db.register_task(task2)
        db.update_task_num_generations(task1.uuid, 42)
        assert db.get_task(task2.uuid).num_generations == task2.num_generations


# ---------------------------------------------------------------------------
# ImportedDatasetTask CRUD
# ---------------------------------------------------------------------------

def make_imported_task(uuid: str = "imp-task-1") -> ImportedDatasetTask:
    return ImportedDatasetTask(
        uuid=uuid,
        mode="base",
        dataset_name="bfcl",
        semantic_version="1.0.0",
        imported_dataset=ImportedDatasetConfig(
            name="bfcl",
            commit="abc123def456",
            args={"version": 4.0},
        ),
    )


class TestImportedDatasetTaskCRUD:

    def test_register_and_get_imported_task(self, db: DatabaseManager):
        task = make_imported_task()
        db.register_task(task)
        retrieved = db.get_task(task.uuid)
        assert isinstance(retrieved, ImportedDatasetTask)
        assert retrieved.uuid == task.uuid
        assert retrieved.dataset_name == "bfcl"

    def test_imported_dataset_config_round_trips(self, db: DatabaseManager):
        task = make_imported_task()
        db.register_task(task)
        retrieved = db.get_task(task.uuid)
        assert retrieved.imported_dataset.name == "bfcl"
        assert retrieved.imported_dataset.commit == "abc123def456"
        assert retrieved.imported_dataset.args == {"version": 4.0}

    def test_imported_task_in_get_all_tasks(self, db: DatabaseManager):
        db.register_task(make_imported_task())
        db.register_task(make_task())
        all_tasks = db.get_all_tasks()
        types = {type(t).__name__ for t in all_tasks}
        assert "ImportedDatasetTask" in types
        assert "AsyncGenerationTask" in types

    def test_register_imported_task_is_idempotent(self, db: DatabaseManager):
        task = make_imported_task()
        db.register_task(task)
        db.register_task(task)
        assert len(db.get_all_tasks()) == 1


# ---------------------------------------------------------------------------
# ImportedDatasetEventInstance CRUD
# ---------------------------------------------------------------------------

def make_imported_event(
    model_name: str = "test-model",
    task_uuid: str = "task-uuid-1",
    suffix: str = "",
) -> ImportedDatasetEventInstance:
    return ImportedDatasetEventInstance(
        uuid=f"imp-event{suffix}",
        parent_uuid=f"imp-parent{suffix}",
        model=model_name,
        task_uuid=task_uuid,
        path_to_scores=f"/output/test-model/bfcl_scores{suffix}.jsonl",
    )


class TestImportedDatasetEventCRUD:

    def _setup(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_imported_task())

    def test_register_imported_event_initial_phase_is_zero(self, db: DatabaseManager):
        self._setup(db)
        event = make_imported_event(task_uuid="imp-task-1")
        db.register_event(event)
        assert db.get_event_phase(event.uuid) == 0

    def test_get_event_by_model_and_task_returns_imported_instance(self, db: DatabaseManager):
        self._setup(db)
        event = make_imported_event(task_uuid="imp-task-1")
        db.register_event(event)
        results = db.get_event_by_model_and_task("test-model", "imp-task-1")
        assert len(results) == 1
        assert isinstance(results[0], ImportedDatasetEventInstance)
        assert results[0].uuid == event.uuid

    def test_imported_event_has_path_to_scores(self, db: DatabaseManager):
        self._setup(db)
        event = make_imported_event(task_uuid="imp-task-1")
        db.register_event(event)
        results = db.get_event_by_model_and_task("test-model", "imp-task-1")
        assert results[0].path_to_scores == event.path_to_scores

    def test_get_all_incomplete_events_includes_imported(self, db: DatabaseManager):
        self._setup(db)
        event = make_imported_event(task_uuid="imp-task-1")
        db.register_event(event)
        phase_0, _ = db.get_all_incomplete_events()
        uuids = {e.uuid for e in phase_0}
        assert event.uuid in uuids

    def test_incomplete_events_mixed_types(self, db: DatabaseManager):
        """Phase-0 list can contain both GradingEventInstance and ImportedDatasetEventInstance."""
        db.register_model(make_model())
        db.register_task(make_task())
        db.register_task(make_imported_task())
        grading_ev = make_event()
        imported_ev = make_imported_event(task_uuid="imp-task-1", suffix="-imp")
        db.register_event(grading_ev)
        db.register_event(imported_ev)
        phase_0, _ = db.get_all_incomplete_events()
        instance_types = {type(e).__name__ for e in phase_0}
        assert "GradingEventInstance" in instance_types
        assert "ImportedDatasetEventInstance" in instance_types

    def test_update_imported_event_phase(self, db: DatabaseManager):
        self._setup(db)
        event = make_imported_event(task_uuid="imp-task-1")
        db.register_event(event)
        db.update_event(event.uuid, 2)
        assert db.get_event_phase(event.uuid) == 2

    def test_grading_only_upsert_preserves_false(self, db: DatabaseManager):
        """Registering a model as grading_only=False after grading_only=True
        must keep grading_only=False (AND logic in the upsert: False AND True = False)."""
        model = make_model("upsert-model")
        db.register_model(model, grading_only=True)
        db.register_model(model, grading_only=False)
        all_models = db.get_all_models()
        assert "upsert-model" in all_models


# ---------------------------------------------------------------------------
# ModelFamily CRUD
# ---------------------------------------------------------------------------

class TestModelFamily:

    def test_register_and_get_model_family(self, db: DatabaseManager):
        spec = make_model_spec()
        db.register_model_family(spec)
        results = db.get_all_model_families()
        assert len(results) == 1

    def test_get_all_model_families_empty(self, db: DatabaseManager):
        assert db.get_all_model_families() == []

    def test_model_family_fields_round_trip(self, db: DatabaseManager):
        spec = make_model_spec(family_name="my-family")
        spec.cache_salt = CacheSaltConfig(mode="unique")
        spec.serving_slurm_resources = ServingSlurmResources(
            # Obviously arbitrary placeholders, distinct from the 12345 defaults.
            gpus_per_node=23456,
            cpus_per_task=34567,
            memory_gb=45678,
            time_limit="12:34:56",
        )
        db.register_model_family(spec)
        result = db.get_all_model_families()[0]
        assert result.local_model.model_family_name == "my-family"
        assert result.local_model.path_glob == "/checkpoints/test/**"
        assert result.local_model.version_level == -1
        assert result.local_model.enqueue_existing is True
        assert result.venv_path == spec.venv_path
        assert result.max_simultaneous_requests == 100
        assert result.max_time_to_deploy == 600
        assert result.allow_long_max_model_len is True
        assert result.vllm_cli_args == ["--tensor-parallel-size", "8"]
        assert result.openai_kwargs == {"temperature": 0.0}
        assert result.cache_salt.mode == "unique"
        assert result.serving_slurm_resources == spec.serving_slurm_resources
        assert result.model_type == ModelType.CHAT
        assert result.owner == "test.user"
        assert result.output_path == "/output/test-family"
        assert result.parser_type == "noop"
        assert result.tag == "any"
        assert result.prompt_prefix_instructions is None

    def test_legacy_null_model_family_cache_salt_hydrates_to_disabled(self, db: DatabaseManager):
        spec = make_model_spec(family_name="legacy-family")
        db.register_model_family(spec)
        db._cursor.execute(
            "UPDATE model_family SET cache_salt = NULL WHERE model_family_name = ?",
            (spec.local_model.model_family_name,),
        )
        db._connection.commit()
        result = db.get_all_model_families()[0]
        assert result.cache_salt == CacheSaltConfig(mode="disabled")

    def test_legacy_null_model_family_resources_use_defaults(
        self, db: DatabaseManager
    ):
        spec = make_model_spec(family_name="legacy-resources-family")
        db.register_model_family(spec)
        db._cursor.execute(
            "UPDATE model_family SET serving_slurm_resources = NULL "
            "WHERE model_family_name = ?",
            (spec.local_model.model_family_name,),
        )
        db._connection.commit()

        result = db.get_all_model_families()[0]
        assert result.serving_slurm_resources == ServingSlurmResources()

    def test_register_model_family_upserts(self, db: DatabaseManager):
        spec = make_model_spec(family_name="my-family")
        db.register_model_family(spec)

        updated_spec = make_model_spec(
            family_name="my-family",
            path_glob="/checkpoints/updated/**",
            parser_type="passthrough",
        )
        db.register_model_family(updated_spec)
        results = db.get_all_model_families()
        assert len(results) == 1
        assert results[0].local_model.path_glob == "/checkpoints/updated/**"
        assert results[0].parser_type == "passthrough"

    def test_model_family_tag_round_trips(self, db: DatabaseManager):
        spec = make_model_spec(tag="vision")
        db.register_model_family(spec)
        result = db.get_all_model_families()[0]
        assert result.tag == "vision"

    def test_model_family_prompt_prefix_instructions_round_trips(self, db: DatabaseManager):
        spec = make_model_spec(prompt_prefix_instructions=r"Answer in \boxed{}")
        db.register_model_family(spec)
        result = db.get_all_model_families()[0]
        assert result.prompt_prefix_instructions == r"Answer in \boxed{}"

    def test_multiple_model_families(self, db: DatabaseManager):
        for i in range(3):
            db.register_model_family(make_model_spec(
                family_name=f"family-{i}",
                path_glob=f"/checkpoints/family-{i}/**",
            ))
        results = db.get_all_model_families()
        assert len(results) == 3
        names = {r.local_model.model_family_name for r in results}
        assert names == {"family-0", "family-1", "family-2"}

    def test_model_family_returns_model_spec_instances(self, db: DatabaseManager):
        db.register_model_family(make_model_spec())
        results = db.get_all_model_families()
        assert all(isinstance(r, ModelSpec) for r in results)


# ---------------------------------------------------------------------------
# Gap 3: register_task() called twice with same UUID is idempotent
# ---------------------------------------------------------------------------

class TestRegisterTaskIdempotency:
    """register_task() with the same UUID must not raise and must not duplicate the row."""

    def test_same_uuid_twice_does_not_raise(self, db: DatabaseManager):
        task = make_task(uuid="idempotent-uuid")
        db.register_task(task)
        # Second call must not raise
        db.register_task(task)

    def test_same_uuid_twice_appears_once(self, db: DatabaseManager):
        task = make_task(uuid="idempotent-uuid-2")
        db.register_task(task)
        db.register_task(task)
        all_tasks = db.get_all_tasks()
        matching = [t for t in all_tasks if t.uuid == "idempotent-uuid-2"]
        assert len(matching) == 1, (
            f"Expected exactly 1 task with UUID idempotent-uuid-2, found {len(matching)}"
        )

    def test_second_call_preserves_original_data(self, db: DatabaseManager):
        task = make_task(uuid="idempotent-uuid-3")
        db.register_task(task)
        db.register_task(task)
        retrieved = db.get_task("idempotent-uuid-3")
        assert retrieved.dataset_name == task.dataset_name
        assert retrieved.data_path == task.data_path


# ---------------------------------------------------------------------------
# Gap 4: get_all_incomplete_events() with NULL phase row is handled gracefully
# ---------------------------------------------------------------------------

class TestGetAllIncompleteEventsUnexpectedPhase:
    """
    A row with an unexpected phase value (not 0, 1, 2, or -1) inserted directly
    via SQL must not appear in get_all_incomplete_events() results and must not
    cause the call to crash.

    Note: the schema enforces NOT NULL on phase, so NULL values cannot be inserted.
    Instead we test with an out-of-range phase value (e.g. 99) to verify the query's
    WHERE-clause filtering gracefully excludes unknown phases.
    """

    def test_unexpected_phase_row_is_skipped_gracefully(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task())

        # Insert a row with an unrecognised phase value (99) that bypasses register_event
        db._cursor.execute(
            """
            INSERT INTO eval_events (
                uuid, parent_uuid, model, task_uuid,
                path_to_generations, path_to_grades, path_to_scores,
                grader_type, parser_type, phase, event_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 99, 'grading')
            """,
            (
                "unknown-phase-uuid",
                "unknown-parent",
                "test-model",
                "task-uuid-1",
                "/output/unknown_gen.jsonl",
                "/output/unknown_grades.jsonl",
                "/output/unknown_scores.jsonl",
                "multiple_choice",
                "noop",
            ),
        )
        db._connection.commit()

        # Must not raise
        phase_0, phase_1 = db.get_all_incomplete_events()

        # The phase-99 row must NOT appear in either bucket
        all_uuids = {e.uuid for e in phase_0} | {e.uuid for e in phase_1}
        assert "unknown-phase-uuid" not in all_uuids

    def test_normal_events_still_returned_alongside_unexpected_phase_row(self, db: DatabaseManager):
        db.register_model(make_model())
        db.register_task(make_task())

        # A normal phase-0 event
        normal_event = make_event()
        db.register_event(normal_event)

        # Insert a row with phase=99 directly
        db._cursor.execute(
            """
            INSERT INTO eval_events (
                uuid, parent_uuid, model, task_uuid,
                path_to_generations, path_to_grades, path_to_scores,
                grader_type, parser_type, phase, event_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 99, 'grading')
            """,
            (
                "unknown-phase-uuid-2",
                "unknown-parent-2",
                "test-model",
                "task-uuid-1",
                "/output/unknown_gen2.jsonl",
                "/output/unknown_grades2.jsonl",
                "/output/unknown_scores2.jsonl",
                "multiple_choice",
                "noop",
            ),
        )
        db._connection.commit()

        phase_0, phase_1 = db.get_all_incomplete_events()
        p0_uuids = {e.uuid for e in phase_0}
        assert normal_event.uuid in p0_uuids
        assert "unknown-phase-uuid-2" not in p0_uuids


# ---------------------------------------------------------------------------
# Gap 5: concurrent register_task() calls with different UUIDs
# ---------------------------------------------------------------------------

class TestConcurrentRegisterTask:
    """
    Multiple async callers registering tasks with different UUIDs in rapid succession
    (interleaved on the same event loop) must all succeed without data loss.

    Note: DatabaseManager uses a single SQLite connection and is not thread-safe.
    These tests exercise the realistic usage pattern: multiple coroutines on the
    same asyncio event loop each calling register_task() before awaiting.
    """

    @pytest.mark.asyncio
    async def test_rapid_sequential_register_different_uuids_all_persist(self, db: DatabaseManager):
        """
        Register several tasks in rapid sequence from async coroutines.
        All must be persisted (no silent DROP due to INT constraints or
        commit ordering bugs).
        """
        import asyncio

        uuids = [f"rapid-task-{i}" for i in range(5)]
        tasks = [make_task(uuid=u) for u in uuids]

        async def register(task):
            # Yield control once (simulates interleaving) then call register_task
            await asyncio.sleep(0)
            db.register_task(task)

        await asyncio.gather(*(register(t) for t in tasks))

        all_tasks = db.get_all_tasks()
        stored_uuids = {t.uuid for t in all_tasks}
        for u in uuids:
            assert u in stored_uuids, f"task {u} was not persisted"

    @pytest.mark.asyncio
    async def test_rapid_sequential_register_same_uuid_idempotent(self, db: DatabaseManager):
        """
        Multiple coroutines registering the same UUID concurrently must not
        produce duplicate rows.
        """
        import asyncio

        task = make_task(uuid="rapid-same-uuid")

        async def register():
            await asyncio.sleep(0)
            db.register_task(task)

        await asyncio.gather(register(), register(), register())

        all_tasks = db.get_all_tasks()
        matching = [t for t in all_tasks if t.uuid == "rapid-same-uuid"]
        assert len(matching) == 1


def make_external_model(name: str = "gpt-4o") -> ModelInstance:
    return ModelInstance(
        name=name,
        path="https://api.openai.com/v1",
        revision=None,
        venv_path=None,
        vllm_cli_args=None,
        max_simultaneous_requests=8,
        max_time_to_deploy=None,
        allow_long_max_model_len=True,
        openai_kwargs={"temperature": 0.0},
        parser_type="noop",
        model_type=ModelType.CHAT,
        owner="test.user",
        output_path="/output/gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key",
        requests_per_minute=500,
        is_external=True,
    )


class TestExternalModelDatabase:
    def test_register_and_get_external_model(self, db: DatabaseManager):
        model = make_external_model()
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.name == "gpt-4o"
        assert retrieved.is_external is True
        assert retrieved.base_url == "https://api.openai.com/v1"
        assert retrieved.api_key == "sk-test-key"
        assert retrieved.requests_per_minute == 500

    def test_external_model_venv_path_is_none(self, db: DatabaseManager):
        model = make_external_model()
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.venv_path is None

    def test_external_model_vllm_cli_args_is_none(self, db: DatabaseManager):
        model = make_external_model()
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.vllm_cli_args is None

    def test_external_model_in_get_all_models(self, db: DatabaseManager):
        model = make_external_model()
        db.register_model(model)
        all_models = db.get_all_models()
        assert "gpt-4o" in all_models
        assert all_models["gpt-4o"].is_external is True

    def test_external_retry_policy_round_trips(self, db: DatabaseManager):
        model = make_external_model()
        model.external_retry_policy = ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=15,
            total_deadline_seconds=30,
            initial_backoff_seconds=0.5,
            max_backoff_seconds=4,
            jitter="equal",
        )

        db.register_model(model)

        assert db.get_model(model.name).external_retry_policy == (
            model.external_retry_policy
        )
        assert db.get_all_models()[model.name].external_retry_policy == (
            model.external_retry_policy
        )
        assert (
            db.get_model(model.name).external_retry_policy.jitter.value
            == "equal"
        )

    def test_external_retry_policy_cli_deadline_override_round_trips(
        self,
        db: DatabaseManager,
    ):
        model = make_external_model()
        model.external_retry_policy = ExternalRetryPolicy(
            request_timeout_seconds=7200,
            total_deadline_seconds=9000,
            initial_backoff_seconds=60,
            max_backoff_seconds=60,
        )

        db.register_model(model)

        assert db.get_model(model.name).external_retry_policy == (
            model.external_retry_policy
        )
        assert db.get_all_models()[model.name].external_retry_policy == (
            model.external_retry_policy
        )

    def test_legacy_null_external_retry_policy_uses_defaults(
        self,
        db: DatabaseManager,
    ):
        model = make_external_model()
        db.register_model(model)
        db._cursor.execute(
            "UPDATE models SET external_retry_policy = NULL "
            "WHERE name = ?",
            (model.name,),
        )
        db._connection.commit()

        assert (
            db.get_model(model.name).external_retry_policy
            == ExternalRetryPolicy()
        )

    def test_empty_external_retry_policy_fails_closed(
        self,
        db: DatabaseManager,
    ):
        model = make_model("external-model")
        model.is_external = True
        db.register_model(model)
        db._cursor.execute(
            "UPDATE models SET external_retry_policy = '' "
            "WHERE name = ?",
            (model.name,),
        )
        db._connection.commit()

        with pytest.raises(ValueError):
            db.get_model(model.name)

    @pytest.mark.parametrize(
        "stored_policy",
        [
            "{",
            "[]",
            '{"max_attempts": "1"}',
        ],
    )
    def test_corrupt_external_retry_policy_fails_closed(
        self,
        db: DatabaseManager,
        stored_policy: str,
    ):
        model = make_model("external-model")
        model.is_external = True
        db.register_model(model)
        db._cursor.execute(
            "UPDATE models SET external_retry_policy = ? "
            "WHERE name = ?",
            (stored_policy, model.name),
        )
        db._connection.commit()

        with pytest.raises(ValueError):
            db.get_model(model.name)

    def test_external_retry_policy_is_updated_on_upsert(
        self,
        db: DatabaseManager,
    ):
        model = make_external_model()
        db.register_model(model)
        updated_policy = ExternalRetryPolicy(
            max_attempts=2,
            request_timeout_seconds=11,
            total_deadline_seconds=29,
            initial_backoff_seconds=0.25,
            max_backoff_seconds=3,
        )
        model.external_retry_policy = updated_policy

        db.register_model(model)

        assert (
            db.get_model(model.name).external_retry_policy
            == updated_policy
        )

    def test_external_model_requests_per_minute_none(self, db: DatabaseManager):
        model = ModelInstance(
            name="ext-no-rpm",
            path="https://my-vllm.example.com/v1",
            venv_path=None,
            vllm_cli_args=None,
            max_simultaneous_requests=4,
            max_time_to_deploy=None,
            openai_kwargs={},
            parser_type="noop",
            model_type=ModelType.CHAT,
            owner="test",
            output_path="/out/ext",
            base_url="https://my-vllm.example.com/v1",
            api_key=None,
            requests_per_minute=None,
            is_external=True,
        )
        db.register_model(model)
        retrieved = db.get_model("ext-no-rpm")
        assert retrieved.requests_per_minute is None
        assert retrieved.api_key is None

    def test_non_external_model_is_external_false(self, db: DatabaseManager):
        model = make_model()
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.is_external is False


# ---------------------------------------------------------------------------
# Container / conda model DB round-trips
# ---------------------------------------------------------------------------

def make_container_model(name: str = "container-model") -> ModelInstance:
    return ModelInstance(
        name=name,
        path="org/container-model",
        revision=None,
        venv_path=None,
        container_image="nvcr.io/nvidia/pytorch:24.01-py3",
        container_mounts=["/data/models:/models:ro", "/data/output:/output:rw"],
        max_simultaneous_requests=100,
        max_time_to_deploy=600,
        allow_long_max_model_len=True,
        vllm_cli_args=["--tensor-parallel-size", "8"],
        openai_kwargs={"temperature": 0.0},
        parser_type="noop",
        model_type=ModelType.CHAT,
        owner="test.user",
        output_path="/output/container-model",
    )


def make_conda_model(name: str = "conda-model") -> ModelInstance:
    return ModelInstance(
        name=name,
        path="org/conda-model",
        revision=None,
        venv_path=None,
        conda_env="vllm-serving",
        max_simultaneous_requests=100,
        max_time_to_deploy=600,
        allow_long_max_model_len=True,
        vllm_cli_args=["--tensor-parallel-size", "8"],
        openai_kwargs={"temperature": 0.0},
        parser_type="noop",
        model_type=ModelType.CHAT,
        owner="test.user",
        output_path="/output/conda-model",
    )


def make_container_model_spec() -> ModelSpec:
    return ModelSpec(
        local_model=LocalModel(
            model_family_name="container-family",
            path_glob="/checkpoints/container/**",
            version_level=-1,
            enqueue_existing=True,
        ),
        container_image="nvcr.io/nvidia/pytorch:24.01-py3",
        container_mounts=["/data/models:/models:ro"],
        max_simultaneous_requests=100,
        max_time_to_deploy=600,
        vllm_cli_args=["--tensor-parallel-size", "8"],
        openai_kwargs={},
        model_type=ModelType.CHAT,
        owner="test.user",
        ready=True,
        output_path="/output/container-family",
        parser_type="noop",
    )


class TestContainerModelDatabase:
    def test_register_and_get_container_model(self, db: DatabaseManager):
        model = make_container_model()
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.container_image == "nvcr.io/nvidia/pytorch:24.01-py3"
        assert retrieved.container_mounts == ["/data/models:/models:ro", "/data/output:/output:rw"]
        assert retrieved.venv_path is None
        assert retrieved.conda_env is None

    def test_container_model_in_get_all_models(self, db: DatabaseManager):
        model = make_container_model()
        db.register_model(model)
        all_models = db.get_all_models()
        assert "container-model" in all_models
        assert all_models["container-model"].container_image == "nvcr.io/nvidia/pytorch:24.01-py3"

    def test_register_and_get_conda_model(self, db: DatabaseManager):
        model = make_conda_model()
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.conda_env == "vllm-serving"
        assert retrieved.venv_path is None
        assert retrieved.container_image is None

    def test_conda_model_in_get_all_models(self, db: DatabaseManager):
        model = make_conda_model()
        db.register_model(model)
        all_models = db.get_all_models()
        assert "conda-model" in all_models
        assert all_models["conda-model"].conda_env == "vllm-serving"

    def test_container_model_family_round_trip(self, db: DatabaseManager):
        spec = make_container_model_spec()
        db.register_model_family(spec)
        families = db.get_all_model_families()
        assert len(families) == 1
        assert families[0].container_image == "nvcr.io/nvidia/pytorch:24.01-py3"
        assert families[0].container_mounts == ["/data/models:/models:ro"]
        assert families[0].venv_path is None

    def test_venv_model_still_works(self, db: DatabaseManager):
        """Existing venv models are unaffected by the new columns."""
        model = make_model()
        db.register_model(model)
        retrieved = db.get_model(model.name)
        assert retrieved.venv_path == "/path/to/venv/bin/activate"
        assert retrieved.container_image is None
        assert retrieved.conda_env is None
        assert retrieved.container_mounts is None
