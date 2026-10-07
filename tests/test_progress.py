"""
Tests for ProgressManager.set_status and its integration with
EventManager.fail_all_events_with_models.
"""
import pytest
from unittest.mock import MagicMock, patch

from scheduler.progress import ProgressManager
from scheduler.event import EventManager, EventInstance, GradingEventInstance
from scheduler.task import AsyncGenerationTask, GraderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(uuid="task-1", dataset_name="mmlu", num_generations=10):
    return AsyncGenerationTask(
        uuid=uuid,
        average_over=[1],
        pass_at=[1],
        mode="base",
        grader=GraderConfig(type="multiple_choice"),
        data_path="/data/*.jsonl",
        dataset_name=dataset_name,
        semantic_version="1.0.0",
        meta=None,
        num_generations=num_generations,
    )


def make_event(model="my-model", task_uuid="task-1", suffix=""):
    return GradingEventInstance(
        uuid=f"ev{suffix}",
        parent_uuid=f"par{suffix}",
        model=model,
        task_uuid=task_uuid,
        path_to_generations=f"/out/gen{suffix}.jsonl",
        path_to_grades=f"/out/grd{suffix}.jsonl",
        path_to_scores=f"/out/scores{suffix}.yaml",
        grader_type="multiple_choice",
        parser_type="noop",
    )


def make_progress_manager(task=None):
    task = task or make_task()
    db = MagicMock()
    db.get_task.return_value = task
    return ProgressManager(db_manager=db)


# ---------------------------------------------------------------------------
# ProgressManager.set_status
# ---------------------------------------------------------------------------

class TestSetStatus:
    def test_initialises_bars_on_first_call(self):
        pm = make_progress_manager()
        event = make_event()
        assert event not in pm._events

        with patch("scheduler.progress.tqdm"):
            pm.set_status(event, "Queued")

        assert event in pm._events
        assert "Generation" in pm._events[event]
        assert "Grading" in pm._events[event]

    def test_status_appears_in_description(self):
        pm = make_progress_manager()
        event = make_event()

        with patch("scheduler.progress.tqdm"):
            pm.set_status(event, "Queued")
            bar = pm._events[event]["Generation"]
            pm.set_status(event, "Deploying")
            last_desc = bar.set_description_str.call_args[0][0]
            assert "Deploying" in last_desc

    def test_description_has_all_columns(self):
        pm = make_progress_manager()
        event = make_event()

        with patch("scheduler.progress.tqdm"):
            pm.set_status(event, "Queued")
            bar = pm._events[event]["Generation"]
            last_desc = bar.set_description_str.call_args[0][0]
            # All four columns present, separated by |
            assert "my-model" in last_desc
            assert "mmlu" in last_desc
            assert "Generation" in last_desc
            assert "Queued" in last_desc
            assert last_desc.count("|") == 3

    def test_subsequent_statuses_replace_in_description(self):
        pm = make_progress_manager()
        event = make_event()

        with patch("scheduler.progress.tqdm"):
            pm.set_status(event, "Queued")
            bar = pm._events[event]["Generation"]
            pm.set_status(event, "Deploying")
            last_desc = bar.set_description_str.call_args[0][0]
            assert "Deploying" in last_desc
            assert "Queued" not in last_desc

    def test_does_not_reinitialise_bars_on_second_call(self):
        pm = make_progress_manager()
        event = make_event()

        with patch("scheduler.progress.tqdm") as mock_tqdm:
            pm.set_status(event, "Queued")
            call_count_after_first = mock_tqdm.call_count
            pm.set_status(event, "Deploying")
            assert mock_tqdm.call_count == call_count_after_first

    def test_failed_status_appears_in_description(self):
        pm = make_progress_manager()
        event = make_event()

        with patch("scheduler.progress.tqdm"):
            pm.set_status(event, "Failed")
            bar = pm._events[event]["Generation"]
            last_desc = bar.set_description_str.call_args[0][0]
            assert "Failed" in last_desc


# ---------------------------------------------------------------------------
# EventManager.fail_all_events_with_models + progress_manager integration
# ---------------------------------------------------------------------------

class TestFailAllEventsWithModels:
    @pytest.mark.asyncio
    async def test_sets_failed_status_for_dead_model_events(self):
        db = MagicMock()
        em = EventManager(db_manager=db)
        event = make_event(model="dead-model")
        em.asyncio_task_dict[event] = []
        em.desired_models_dict["dead-model"] = MagicMock()

        pm = MagicMock(spec=ProgressManager)
        await em.fail_all_events_with_models(["dead-model"], progress_manager=pm)

        pm.set_status.assert_called_once_with(event, "Failed")

    @pytest.mark.asyncio
    async def test_does_not_update_live_model_events(self):
        db = MagicMock()
        em = EventManager(db_manager=db)
        dead_event = make_event(model="dead-model", suffix="-dead")
        live_event = make_event(model="live-model", suffix="-live")
        em.asyncio_task_dict[dead_event] = []
        em.asyncio_task_dict[live_event] = []
        em.desired_models_dict["dead-model"] = MagicMock()
        em.desired_models_dict["live-model"] = MagicMock()

        pm = MagicMock(spec=ProgressManager)
        await em.fail_all_events_with_models(["dead-model"], progress_manager=pm)

        pm.set_status.assert_called_once_with(dead_event, "Failed")

    @pytest.mark.asyncio
    async def test_works_without_progress_manager(self):
        db = MagicMock()
        em = EventManager(db_manager=db)
        event = make_event(model="dead-model")
        em.asyncio_task_dict[event] = []
        em.desired_models_dict["dead-model"] = MagicMock()

        # Should not raise when progress_manager is omitted
        await em.fail_all_events_with_models(["dead-model"])

    @pytest.mark.asyncio
    async def test_sets_failed_for_multiple_distinct_dead_models(self):
        # Each model has one event — multiple events per model is a known
        # limitation tracked in event.py (TODO comment on desired_models_dict).
        db = MagicMock()
        em = EventManager(db_manager=db)
        dead_names = ["dead-a", "dead-b", "dead-c"]
        events = [make_event(model=name, suffix=f"-{name}") for name in dead_names]
        for ev, name in zip(events, dead_names):
            em.asyncio_task_dict[ev] = []
            em.desired_models_dict[name] = MagicMock()

        pm = MagicMock(spec=ProgressManager)
        await em.fail_all_events_with_models(dead_names, progress_manager=pm)

        assert pm.set_status.call_count == 3
        for call in pm.set_status.call_args_list:
            assert call[0][1] == "Failed"


# ---------------------------------------------------------------------------
# ImportedDataset events have no num_generations — progress manager must
# handle them gracefully without crashing.
# ---------------------------------------------------------------------------

from scheduler.event import ImportedDatasetEventInstance
from scheduler.task import ImportedDatasetTask, ImportedDatasetConfig


def make_imported_task():
    return ImportedDatasetTask(
        uuid="imported-task",
        dataset_name="my_benchmark",
        semantic_version="1.0.0",
        meta=None,
        imported_dataset=ImportedDatasetConfig(name="test-runner", commit="abc123"),
    )


def make_imported_event():
    return ImportedDatasetEventInstance(
        uuid="ev-id",
        parent_uuid="par",
        model="my-model",
        task_uuid="imported-task",
        path_to_scores="/out/scores.yaml",
    )


# ---------------------------------------------------------------------------
# _to_desc — dynamic name width clamped to [10, 20]
# ---------------------------------------------------------------------------

class TestToDescNameWidth:
    def test_short_name_not_truncated(self):
        # Short names should not be truncated — they appear in full despite being < 10 chars
        pm = make_progress_manager(task=make_task(dataset_name="short"))
        event = make_event(model="m")
        desc = pm._to_desc(event, pm._db_manager.get_task(event.task_uuid), "Generation")
        assert "m" in desc
        assert "short" in desc

    def test_long_name_not_truncated(self):
        # No upper cap — long names appear in full so bars don't silently lose characters
        long_model = "a" * 25
        long_dataset = "b" * 25
        pm = make_progress_manager(task=make_task(dataset_name=long_dataset))
        event = make_event(model=long_model)
        with patch("scheduler.progress.tqdm"):
            pm.initialize(event)
        desc = pm._to_desc(event, pm._db_manager.get_task(event.task_uuid), "Generation")
        parts = desc.split("|")
        assert parts[0].strip() == "a" * 25
        assert parts[1].strip() == "b" * 25

    def test_widths_align_across_multiple_events(self):
        # All bars should use the same column width — the max across all active events
        short_task = make_task(uuid="t1", dataset_name="mmlu")
        long_task = make_task(uuid="t2", dataset_name="gpqa_diamond_extended")
        db = MagicMock()
        db.get_task.side_effect = lambda uuid: short_task if uuid == "t1" else long_task
        pm = ProgressManager(db_manager=db)
        short_event = make_event(model="k2-v2", task_uuid="t1", suffix="-s")
        long_event = make_event(model="k2-v2-instruct-test", task_uuid="t2", suffix="-l")
        with patch("scheduler.progress.tqdm"):
            pm.initialize(short_event)
            pm.initialize(long_event)
        # Both descs should use the same widths (max of all active events)
        desc_short = pm._to_desc(short_event, short_task, "Generation")
        desc_long = pm._to_desc(long_event, long_task, "Generation")
        # The model column in both descs should have the same width
        model_col_short = desc_short.split("|")[0]
        model_col_long = desc_long.split("|")[0]
        assert len(model_col_short) == len(model_col_long)

    def test_existing_bars_refreshed_when_wider_event_added(self):
        # When a second (wider) event is initialized, all existing bars should
        # have their descriptions updated to use the new wider column widths.
        short_task = make_task(uuid="t1", dataset_name="mmlu")
        long_task = make_task(uuid="t2", dataset_name="mmlu")
        db = MagicMock()
        db.get_task.side_effect = lambda uuid: short_task if uuid == "t1" else long_task
        pm = ProgressManager(db_manager=db)
        short_event = make_event(model="k2", task_uuid="t1", suffix="-s")
        long_event = make_event(model="k2-v2-instruct-test", task_uuid="t2", suffix="-l")
        with patch("scheduler.progress.tqdm"):
            pm.initialize(short_event)
            pm.set_status(short_event, "Queued")
            gen_bar = pm._events[short_event]["Generation"]
            desc_before = gen_bar.set_description_str.call_args[0][0]

            pm.initialize(long_event)
            # After second event: short_event's Grading bar should be refreshed to wider widths
            grading_bar = pm._events[short_event]["Grading"]
            desc_grading_after = grading_bar.set_description_str.call_args[0][0]

        # The Grading bar description should now use the wider model column
        model_col_before = desc_before.split("|")[0]
        model_col_grading_after = desc_grading_after.split("|")[0]
        assert len(model_col_grading_after) > len(model_col_before)

    def test_name_exactly_15_not_truncated(self):
        name = "x" * 15
        pm = make_progress_manager(task=make_task(dataset_name=name))
        event = make_event(model=name)
        with patch("scheduler.progress.tqdm"):
            pm.initialize(event)
        desc = pm._to_desc(event, pm._db_manager.get_task(event.task_uuid), "Generation")
        parts = desc.split("|")
        assert parts[0].strip() == name
        assert parts[1].strip() == name


# ---------------------------------------------------------------------------
# notify_downloading / notify_download_complete
# ---------------------------------------------------------------------------

class TestDownloadingStatus:
    def test_set_status_downloading_appears_in_description(self):
        pm = make_progress_manager(task=make_task(num_generations=None))
        event = make_event()
        with patch("scheduler.progress.tqdm"):
            pm.set_status(event, "Downloading")
            bar = pm._events[event]["Generation"]
            last_desc = bar.set_description_str.call_args[0][0]
            assert "Downloading" in last_desc

    def test_set_status_downloading_creates_zero_total_bar(self):
        pm = make_progress_manager(task=make_task(num_generations=None))
        event = make_event()
        with patch("scheduler.progress.tqdm") as mock_tqdm:
            pm.set_status(event, "Downloading")
        # bar created with total=0 since num_generations is None
        create_call = mock_tqdm.call_args_list[0]
        assert create_call[1]["total"] == 0

    def test_update_generation_total_updates_bar_total(self):
        pm = make_progress_manager(task=make_task(num_generations=None))
        event = make_event()
        with patch("scheduler.progress.tqdm"):
            pm.set_status(event, "Downloading")
            updated_task = make_task(num_generations=20)
            pm._db_manager.get_task.return_value = updated_task
            pm.update_generation_total(event, updated_task)
            bar = pm._events[event]["Generation"]
            expected_total = 20 * max(updated_task.average_over + updated_task.pass_at)
            assert bar.total == expected_total

    def test_update_generation_total_noop_if_event_not_initialized(self):
        pm = make_progress_manager()
        event = make_event()
        pm.update_generation_total(event, make_task(num_generations=10))  # should not raise

    def test_update_generation_total_resizes_chunks(self):
        pm = make_progress_manager(task=make_task(num_generations=None))
        event = make_event()
        with patch("scheduler.progress.tqdm"):
            pm.set_status(event, "Downloading")
            assert len(pm._events[event]["generation_chunks"]) == 0
            updated_task = make_task(num_generations=10)
            pm._db_manager.get_task.return_value = updated_task
            pm.update_generation_total(event, updated_task)
            expected_len = 10 * max(updated_task.average_over + updated_task.pass_at)
            assert len(pm._events[event]["generation_chunks"]) == expected_len


class TestProgressManagerImportedDataset:

    def test_initialize_does_not_crash(self):
        pm = make_progress_manager(task=make_imported_task())
        with patch("scheduler.progress.tqdm"):
            pm.initialize(make_imported_event())  # should not raise

    def test_initialize_adds_event(self):
        pm = make_progress_manager(task=make_imported_task())
        event = make_imported_event()
        with patch("scheduler.progress.tqdm"):
            pm.initialize(event)
        assert event in pm._events

    def test_set_status_does_not_crash(self):
        pm = make_progress_manager(task=make_imported_task())
        with patch("scheduler.progress.tqdm"):
            pm.set_status(make_imported_event(), "Running")  # should not raise

    def test_update_does_not_crash(self):
        pm = make_progress_manager(task=make_imported_task())
        with patch("scheduler.progress.tqdm"):
            pm.update(make_imported_event(), index=0, completed=1, new_elems=1, mode="Generation")
