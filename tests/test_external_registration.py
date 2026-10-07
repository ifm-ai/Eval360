from __future__ import annotations

import asyncio
from contextlib import nullcontext
from unittest.mock import AsyncMock, patch

import pytest

import scheduler.openai_interface as openai_interface
from scheduler.database import DatabaseManager
from scheduler.event import EventManager
from scheduler.external_requests import RATE_LIMITERS
from scheduler.model import (
    CacheSaltConfig,
    ExternalRetryPolicy,
    ModelInstance,
    ModelType,
)
from scheduler.task import (
    AsyncGenerationTask,
    ImportedDatasetConfig,
    ImportedDatasetTask,
)
from tests.test_scheduler import _scheduler


def _external_model(
    *,
    name: str = "external",
    base_url: str = "https://api.example.test/v1",
    api_key: str = "sk-test",
    max_simultaneous_requests: int = 4,
    requests_per_minute: int | None = 60,
    api_model_name: str = "served-model",
    external_retry_policy: ExternalRetryPolicy | None = None,
    openai_kwargs: dict | None = None,
    prompt_prefix_instructions: str | None = None,
    cache_salt: CacheSaltConfig | None = None,
) -> ModelInstance:
    return ModelInstance(
        name=name,
        path=base_url,
        revision=None,
        venv_path=None,
        max_time_to_deploy=None,
        vllm_cli_args=None,
        output_path="/tmp/output",
        parser_type="noop",
        model_type="base",
        openai_kwargs=openai_kwargs or {},
        cache_salt=cache_salt or CacheSaltConfig(),
        max_simultaneous_requests=max_simultaneous_requests,
        owner="test",
        tag="any",
        prompt_prefix_instructions=prompt_prefix_instructions,
        base_url=base_url,
        api_key=api_key,
        requests_per_minute=requests_per_minute,
        is_external=True,
        api_model_name=api_model_name,
        external_retry_policy=(
            external_retry_policy or ExternalRetryPolicy()
        ),
    )


def _task(
    tmp_path,
    *,
    uuid: str = "task",
    judge: ModelInstance | None = None,
) -> AsyncGenerationTask:
    return AsyncGenerationTask(
        uuid=uuid,
        average_over=[1],
        pass_at=[1],
        mode="base",
        grader={
            "type": "multiple_choice",
            "llm_as_judge": judge,
        },
        data_path=str(tmp_path / "*.jsonl"),
        dataset_name=f"{uuid}_dataset",
        semantic_version="1.0.0",
        num_generations=1,
    )


def _imported_task(
    *,
    uuid: str = "imported-task",
) -> ImportedDatasetTask:
    return ImportedDatasetTask(
        uuid=uuid,
        mode="base",
        dataset_name=f"{uuid}_dataset",
        semantic_version="1.0.0",
        imported_dataset=ImportedDatasetConfig(
            name="bfcl",
            commit="abc123",
        ),
    )


def _persisted_model_or_none(database, name):
    getter = getattr(database, "get_model_if_exists", None)
    assert callable(getter), "non-throwing model lookup is missing"
    return getter(name)


@pytest.fixture(autouse=True)
def _clear_runtime_registries():
    openai_interface.LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()
    yield
    openai_interface.LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()


def test_database_exposes_rollback_transaction():
    transaction = getattr(DatabaseManager(), "transaction", None)

    assert callable(transaction), "database transaction boundary is missing"
    assert hasattr(transaction(), "__enter__")
    assert hasattr(transaction(), "__exit__")


def test_event_manager_exposes_deferred_enqueue_boundary():
    manager = EventManager(DatabaseManager())
    defer_enqueues = getattr(manager, "defer_enqueues", None)

    assert callable(
        defer_enqueues
    ), "event queue commit boundary is missing"
    context = defer_enqueues()
    assert not isinstance(context, nullcontext)
    assert hasattr(context, "__enter__")
    assert hasattr(context, "__exit__")


@pytest.mark.asyncio
async def test_same_name_endpoint_change_is_rejected_before_mutation(tmp_path):
    scheduler = _scheduler(tmp_path)
    original = _external_model(
        name="same",
        base_url="https://api-a.example.test/v1",
    )
    reconfigured = _external_model(
        name="same",
        base_url="https://api-b.example.test/v1",
        api_key="sk-other",
        requests_per_minute=120,
    )

    await scheduler.register_model_instance(original)
    persisted_before = scheduler.db_manager.get_model(original.name)
    pools_before = dict(openai_interface.LOCKED_CONNECTIONS)
    limiters_before = dict(RATE_LIMITERS)

    with pytest.raises(
        ValueError,
        match="immutable external registration.*same",
    ):
        await scheduler.register_model_instance(reconfigured)

    assert scheduler.db_manager.get_model(original.name) == persisted_before
    assert openai_interface.LOCKED_CONNECTIONS == pools_before
    assert RATE_LIMITERS == limiters_before


@pytest.mark.asyncio
async def test_restart_can_replace_persisted_external_registration(
    tmp_path,
    monkeypatch,
):
    """A new scheduler process owns, and may replace, a persisted endpoint."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVAL360_IN_MEMORY_DB", "false")
    original = _external_model(
        name="same",
        base_url="https://api-a.example.test/v1",
    )
    replacement = _external_model(
        name="same",
        base_url="https://api-b.example.test/v1",
        api_key="sk-replacement",
        max_simultaneous_requests=8,
        requests_per_minute=120,
    )

    first_scheduler = _scheduler(tmp_path)
    await first_scheduler.register_model_instance(original)

    # A scheduler restart creates new process-scoped pools and limiters.
    openai_interface.LOCKED_CONNECTIONS.clear()
    RATE_LIMITERS.clear()
    restarted_scheduler = _scheduler(tmp_path)
    await restarted_scheduler.register_model_instance(replacement)

    assert restarted_scheduler.db_manager.get_model("same") == replacement
    assert set(
        openai_interface.LOCKED_CONNECTIONS[replacement.serving_key].urls
    ) == {
        replacement.base_url
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family_name", "other-family"),
        ("path", "https://other-path.example.test/v1"),
        ("revision", "other-revision"),
        ("venv_path", "/tmp/other-venv"),
        ("conda_env", "other-conda"),
        ("container_image", "other-image.sqsh"),
        ("container_mounts", ["/host:/container"]),
        ("api_key", "sk-other"),
        ("max_simultaneous_requests", 8),
        ("max_time_to_deploy", 900),
        ("allow_long_max_model_len", False),
        ("vllm_cli_args", ["--dtype", "float16"]),
        ("vllm_logging_level", "DEBUG"),
        ("requests_per_minute", 120),
        ("api_model_name", "other-api-model"),
        ("external_retry_policy", ExternalRetryPolicy(max_attempts=2)),
        ("parser_type", "passthrough"),
        ("model_type", ModelType.CHAT),
        ("owner", "other-owner"),
        ("output_path", "/tmp/other-output"),
        ("tag", "other-tag"),
        ("openai_kwargs", {"temperature": 0.5}),
        ("prompt_prefix_instructions", "new prefix"),
        ("cache_salt", CacheSaltConfig(mode="static", salt="new-salt")),
    ],
)
async def test_same_name_request_policy_change_is_rejected(
    tmp_path,
    field,
    value,
):
    scheduler = _scheduler(tmp_path)
    original = _external_model(name="same")
    reconfigured = original.model_copy(update={field: value})

    await scheduler.register_model_instance(original)
    persisted_before = scheduler.db_manager.get_model(original.name)
    pools_before = dict(openai_interface.LOCKED_CONNECTIONS)
    limiters_before = dict(RATE_LIMITERS)

    with pytest.raises(
        ValueError,
        match="immutable external registration.*same",
    ):
        await scheduler.register_model_instance(reconfigured)

    assert scheduler.db_manager.get_model(original.name) == persisted_before
    assert openai_interface.LOCKED_CONNECTIONS == pools_before
    assert RATE_LIMITERS == limiters_before


def test_registration_snapshot_covers_every_model_field_and_hashes_secrets():
    model = _external_model(
        api_key="sk-do-not-persist-in-snapshot",
        openai_kwargs={
            "default_headers": {
                "authorization": "Bearer another-secret",
            },
        },
    )

    snapshot = model.registration_snapshot()

    assert set(snapshot) == set(ModelInstance.model_fields)
    assert snapshot["api_key"] != model.api_key
    assert "sk-do-not-persist-in-snapshot" not in str(snapshot)
    assert "Bearer another-secret" not in str(snapshot)


def test_registration_snapshot_hashes_supported_credential_aliases():
    credential_values = {
        "x-api-key": "snapshot-test-x-api-key",
        "service-key": "snapshot-test-service-key",
        "ingest-token": "snapshot-test-ingest-token",
        "credentials": "snapshot-test-credentials",
    }
    model = _external_model(
        openai_kwargs={"extra_body": credential_values},
    )

    snapshot = model.registration_snapshot()
    snapshot_headers = snapshot["openai_kwargs"]["extra_body"]

    assert all(value not in str(snapshot) for value in credential_values.values())
    assert set(snapshot_headers) == set(credential_values)
    assert all(
        value.startswith("sha256:")
        for value in snapshot_headers.values()
    )


@pytest.mark.asyncio
async def test_registration_cancellation_rolls_back_database_runtime_and_queue(
    tmp_path,
):
    scheduler = _scheduler(tmp_path)
    task = _task(tmp_path, uuid="preexisting-task")
    await scheduler.register_task(task)
    model = _external_model()

    with (
        patch.object(
            scheduler.event_manager,
            "enqueue",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await scheduler.register_model_instance(model)

    assert _persisted_model_or_none(scheduler.db_manager, model.name) is None
    assert scheduler.db_manager._cursor.execute(
        "SELECT uuid FROM eval_events"
    ).fetchall() == []
    assert scheduler.event_manager._launch_queue.empty()
    assert openai_interface.LOCKED_CONNECTIONS == {}
    assert RATE_LIMITERS == {}


@pytest.mark.asyncio
async def test_task_failure_rolls_back_judge_database_runtime_and_queue(
    tmp_path,
):
    scheduler = _scheduler(tmp_path)
    judge = _external_model(name="judge")
    task = _task(tmp_path, uuid="failing-task", judge=judge)

    with (
        patch.object(
            scheduler.db_manager,
            "register_task",
            side_effect=RuntimeError("task persistence failed"),
        ),
        pytest.raises(RuntimeError, match="task persistence failed"),
    ):
        await scheduler.register_task(task)

    assert _persisted_model_or_none(scheduler.db_manager, judge.name) is None
    assert scheduler.db_manager._cursor.execute(
        "SELECT uuid FROM task WHERE uuid = ?",
        (task.uuid,),
    ).fetchone() is None
    assert scheduler.event_manager._launch_queue.empty()
    assert openai_interface.LOCKED_CONNECTIONS == {}
    assert RATE_LIMITERS == {}


@pytest.mark.asyncio
async def test_imported_task_cancellation_rolls_back_database_and_queue(
    tmp_path,
):
    scheduler = _scheduler(tmp_path)
    model = _external_model(name="existing-model")
    scheduler.db_manager.register_model(model)
    task = _imported_task(uuid="cancelled-imported-task")

    with (
        patch.object(
            scheduler.event_manager,
            "enqueue",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await scheduler.register_task(task)

    assert scheduler.db_manager.get_task(task.uuid) is None
    assert scheduler.db_manager._cursor.execute(
        "SELECT uuid FROM eval_events WHERE task_uuid = ?",
        (task.uuid,),
    ).fetchall() == []
    assert scheduler.event_manager._launch_queue.empty()


@pytest.mark.asyncio
async def test_imported_task_enqueues_once_after_commit(tmp_path):
    scheduler = _scheduler(tmp_path)
    model = _external_model(name="existing-model")
    scheduler.db_manager.register_model(model)
    task = _imported_task(uuid="committed-imported-task")
    enqueue_states = []
    real_enqueue_nowait = scheduler.event_manager.enqueue_nowait

    def record_enqueue(event):
        enqueue_states.append(
            scheduler.db_manager._connection.in_transaction
        )
        real_enqueue_nowait(event)

    with patch.object(
        scheduler.event_manager,
        "enqueue_nowait",
        side_effect=record_enqueue,
    ) as enqueue_nowait:
        await scheduler.register_task(task)

    assert enqueue_nowait.call_count == 1
    assert enqueue_states == [False]
    assert scheduler.event_manager._launch_queue.qsize() == 1


@pytest.mark.asyncio
async def test_conflicting_judge_quota_leaves_no_partial_rows_or_resources(
    tmp_path,
):
    scheduler = _scheduler(tmp_path)
    generation = _external_model(
        name="generation",
        base_url="https://API.EXAMPLE.test:443/v1/",
        requests_per_minute=60,
    )
    judge = _external_model(
        name="judge",
        base_url="https://api.example.test/v1",
        requests_per_minute=120,
    )
    task = _task(tmp_path, uuid="conflicting-judge", judge=judge)

    await scheduler.register_model_instance(generation)
    pools_before = dict(openai_interface.LOCKED_CONNECTIONS)
    limiters_before = dict(RATE_LIMITERS)

    with pytest.raises(ValueError, match="conflicting requests_per_minute"):
        await scheduler.register_task(task)

    assert _persisted_model_or_none(scheduler.db_manager, judge.name) is None
    assert scheduler.db_manager._cursor.execute(
        "SELECT uuid FROM task WHERE uuid = ?",
        (task.uuid,),
    ).fetchone() is None
    assert openai_interface.LOCKED_CONNECTIONS == pools_before
    assert RATE_LIMITERS == limiters_before


@pytest.mark.asyncio
async def test_same_named_judge_registration_is_immutable_before_mutation(
    tmp_path,
):
    scheduler = _scheduler(tmp_path)
    original = _external_model(
        name="judge",
        base_url="https://judge-a.example.test/v1",
    )
    reconfigured = _external_model(
        name="judge",
        base_url="https://judge-b.example.test/v1",
        requests_per_minute=120,
    )

    await scheduler.register_task(
        _task(tmp_path, uuid="first-judge-task", judge=original)
    )
    persisted_before = scheduler.db_manager.get_model(original.name)
    pools_before = dict(openai_interface.LOCKED_CONNECTIONS)
    limiters_before = dict(RATE_LIMITERS)

    with pytest.raises(
        ValueError,
        match="immutable external registration.*judge",
    ):
        await scheduler.register_task(
            _task(tmp_path, uuid="changed-judge-task", judge=reconfigured)
        )

    assert scheduler.db_manager.get_model(original.name) == persisted_before
    assert scheduler.db_manager.get_task("changed-judge-task") is None
    assert openai_interface.LOCKED_CONNECTIONS == pools_before
    assert RATE_LIMITERS == limiters_before


@pytest.mark.asyncio
async def test_completed_pair_replay_does_not_create_or_enqueue_duplicate(
    tmp_path,
):
    scheduler = _scheduler(tmp_path)
    task = _task(tmp_path, uuid="completed-task")
    model = _external_model(name="completed-model")

    await scheduler.register_task(task)
    await scheduler.register_model_instance(model)
    event_row = scheduler.db_manager._cursor.execute(
        "SELECT uuid FROM eval_events WHERE model = ? AND task_uuid = ?",
        (model.name, task.uuid),
    ).fetchone()
    assert event_row is not None
    scheduler.db_manager.update_event(event_row[0], 2)
    while not scheduler.event_manager._launch_queue.empty():
        scheduler.event_manager._launch_queue.get_nowait()

    await scheduler.register_model_instance(model.model_copy(deep=True))

    assert scheduler.db_manager._cursor.execute(
        "SELECT COUNT(*) FROM eval_events WHERE model = ? AND task_uuid = ?",
        (model.name, task.uuid),
    ).fetchone() == (1,)
    assert scheduler.event_manager._launch_queue.empty()


@pytest.mark.asyncio
async def test_hf_resolution_failure_precedes_database_runtime_and_queue(
    tmp_path,
):
    scheduler = _scheduler(tmp_path)
    generation = _external_model(name="generation")
    await scheduler.register_model_instance(generation)
    task = _task(tmp_path, uuid="hf-failure").model_copy(
        update={
            "data_path": "hf://org/repo/data.jsonl",
            "num_generations": None,
        },
    )

    with (
        patch(
            "scheduler.scheduler.resolve_hf_path",
            side_effect=RuntimeError("download failed"),
        ),
        pytest.raises(RuntimeError, match="download failed"),
    ):
        await scheduler.register_task(task)

    assert scheduler.db_manager._cursor.execute(
        "SELECT uuid FROM task WHERE uuid = ?",
        (task.uuid,),
    ).fetchone() is None
    assert scheduler.db_manager._cursor.execute(
        "SELECT uuid FROM eval_events WHERE task_uuid = ?",
        (task.uuid,),
    ).fetchall() == []
    assert scheduler.event_manager._launch_queue.empty()


@pytest.mark.asyncio
async def test_exact_same_registration_is_idempotent(tmp_path):
    scheduler = _scheduler(tmp_path)
    model = _external_model(name="same")

    await scheduler.register_model_instance(model)
    pools_before = dict(openai_interface.LOCKED_CONNECTIONS)
    limiters_before = dict(RATE_LIMITERS)
    await scheduler.register_model_instance(model.model_copy(deep=True))

    assert scheduler.db_manager.get_model(model.name) == model
    assert openai_interface.LOCKED_CONNECTIONS == pools_before
    assert RATE_LIMITERS == limiters_before
