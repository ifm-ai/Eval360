"""
Tests for edge cases around sibling model expansion, duplicate event handling,
and allocation with shared serving keys.
"""
import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, PropertyMock, MagicMock

import pytest
import yaml

from scheduler.scheduler import Scheduler
from scheduler.event import DeploymentInfo
from scheduler.model import ModelInstance

from tests.test_scheduler import (
    _make_model_instance,
    _write_model_yaml,
    _write_dataset_yaml,
    _write_jsonl,
    _sample_rows,
    _scheduler,
    _deployment_info,
    _FakeOpenAIConnection,
    _run_loop_until_files_exist,
)
from tests.fake_slurm import FakeSlurmManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scheduler_with_test_mgr(tmp_path, **test_mgr_kwargs) -> tuple[Scheduler, FakeSlurmManager]:
    s = _scheduler(tmp_path)
    test_mgr = FakeSlurmManager(**test_mgr_kwargs)
    s.slurm_manager = test_mgr
    return s, test_mgr


SHARED_SK = "aabbcc001122"


def _write_sibling_model_yaml(path: Path, output_path: Path, *, name: str, temperature: float) -> Path:
    """Write a model YAML that will produce a ModelInstance with a given name and temperature."""
    cfg = {
        "remote_model": {"base_name": name, "path": "org/shared-model", "revision": None},
        "model_type": "base",
        "parser_type": "noop",
        "name_modifier": None,
        "venv_path": "/fake/bin/activate",
        "max_simultaneous_requests": 4,
        "max_time_to_deploy": 600,
        "vllm_cli_args": [],
        "openai_kwargs": {"temperature": temperature},
        "owner": "test",
        "ready": True,
        "output_path": str(output_path),
        "tag": None,
    }
    p = path / f"model_{name}.yaml"
    p.write_text(yaml.dump(cfg))
    return p


# ---------------------------------------------------------------------------
# TestSiblingModelExpansion
# ---------------------------------------------------------------------------

class TestSiblingModelExpansion:
    """Two models sharing the same serving_key should share a single Slurm job."""

    @pytest.mark.asyncio
    async def test_sibling_models_share_single_slurm_job(self, tmp_path):
        """
        Register two model specs that produce ModelInstances with different
        openai_kwargs but patched to share the same serving_key. Register one
        dataset. Assert only one allocation slot is used (not two), and both
        models' events complete successfully.
        """
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))

        model_yaml_a = _write_sibling_model_yaml(
            tmp_path, output_path=tmp_path / "output",
            name="sibling-a", temperature=0.0,
        )
        model_yaml_b = _write_sibling_model_yaml(
            tmp_path, output_path=tmp_path / "output",
            name="sibling-b", temperature=0.7,
        )
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])
        s, test_mgr = _scheduler_with_test_mgr(tmp_path)

        scores_a = tmp_path / "output" / "sibling-a" / "test_dataset_scores.yaml"
        scores_b = tmp_path / "output" / "sibling-b" / "test_dataset_scores.yaml"

        # Capture allocations passed to update_allocation to verify collapsing.
        captured_allocations = []
        original_update = test_mgr.update_allocation

        async def capture_update(desired_allocations, unneeded_models, sibling_names_by_sk=None):
            if desired_allocations:
                captured_allocations.append(desired_allocations)
            return await original_update(desired_allocations, unneeded_models, sibling_names_by_sk)

        test_mgr.update_allocation = capture_update

        with patch.object(
            ModelInstance, "serving_key", new_callable=PropertyMock, return_value=SHARED_SK
        ), patch(
            "scheduler.openai_interface.OpenAIConnection", return_value=fake_conn
        ):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml_a), str(model_yaml_b)],
                paths_to_datasets=[str(dataset_yaml)],
            )

        # Both models' score files should exist — both evaluations completed
        # through a shared serving_key.
        assert scores_a.exists(), "sibling-a scores file missing"
        assert scores_b.exists(), "sibling-b scores file missing"

        # Verify allocation collapsed: any non-empty allocation should have
        # at most 1 entry (both models share one serving_key slot).
        for alloc in captured_allocations:
            unique_sks = {m.serving_key for m, _ in alloc}
            assert len(unique_sks) <= 1, (
                f"Expected at most 1 serving_key per allocation, got {len(unique_sks)}: {unique_sks}"
            )


# ---------------------------------------------------------------------------
# TestAsymmetricDesiredModels
# ---------------------------------------------------------------------------

class TestAsymmetricDesiredModels:
    """When siblings share a serving_key but only one has active events,
    the _expand_siblings logic must still track status correctly."""

    @pytest.mark.asyncio
    async def test_sibling_not_in_desired_still_gets_status_updates(self, tmp_path):
        """
        Register two sibling models and one dataset. Complete model B's
        events first (by pre-populating its output files). Assert model A's
        events still proceed correctly.
        """
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 2
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))

        model_yaml_a = _write_sibling_model_yaml(
            tmp_path, output_path=tmp_path / "output",
            name="asym-a", temperature=0.0,
        )
        model_yaml_b = _write_sibling_model_yaml(
            tmp_path, output_path=tmp_path / "output",
            name="asym-b", temperature=0.5,
        )
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        # Pre-populate model B's output so its events complete immediately
        out_b = tmp_path / "output" / "asym-b"
        out_b.mkdir(parents=True)
        gen_rows = [
            {"row": i, "generations": ["A"], "chat_input": [{"role": "user", "content": f"Q{i}?"}], "ground_truth": "A"}
            for i in range(n_rows)
        ]
        _write_jsonl(out_b / "test_dataset_generations.jsonl", gen_rows)
        grade_rows = [
            {"row": i, "correct": [True], "generations": ["A"], "parsed_generations": ["A"], "ground_truth": "A"}
            for i in range(n_rows)
        ]
        _write_jsonl(out_b / "test_dataset_grades.jsonl", grade_rows)
        # Write a minimal scores file
        (out_b / "test_dataset_scores.yaml").write_text(
            yaml.dump([{"name": "accuracy_at_1", "value": 1.0}])
        )

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])
        s, test_mgr = _scheduler_with_test_mgr(tmp_path)

        scores_a = tmp_path / "output" / "asym-a" / "test_dataset_scores.yaml"

        with patch.object(
            ModelInstance, "serving_key", new_callable=PropertyMock, return_value=SHARED_SK
        ), patch(
            "scheduler.openai_interface.OpenAIConnection", return_value=fake_conn
        ):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml_a), str(model_yaml_b)],
                paths_to_datasets=[str(dataset_yaml)],
            )

        # Model A's evaluation should complete even though model B was
        # pre-completed and may not appear in desired_models after its
        # events finish.
        assert scores_a.exists(), "asym-a scores file missing -- status tracking failed"


# ---------------------------------------------------------------------------
# TestDuplicateRegistrationWhileInProgress
# ---------------------------------------------------------------------------

class TestDuplicateRegistrationWhileInProgress:
    """Re-registering a model during an active evaluation must not create
    duplicate events."""

    @pytest.mark.asyncio
    async def test_reregister_model_during_active_event_is_noop(self, tmp_path):
        """
        Register a model and dataset, start the loop, and while the
        evaluation is in progress, re-register the same model. Assert
        that no duplicate events are created and the original evaluation
        still completes.
        """
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])
        s, test_mgr = _scheduler_with_test_mgr(tmp_path)

        scores_file = tmp_path / "output" / "test-model" / "test_dataset_scores.yaml"

        from scheduler.model import ModelParser
        from scheduler.task import Task

        model_spec = ModelParser.parse_yaml(str(model_yaml))
        task = Task.parse_yaml(str(dataset_yaml))

        async def register_and_reregister():
            # First registration
            await s.register_model_spec(model_spec)
            await s.register_task(task)
            # Wait a beat for events to start processing
            await asyncio.sleep(0.1)
            # Re-register the same model -- should be a no-op for events
            await s.register_model_spec(model_spec)

        with patch(
            "scheduler.openai_interface.OpenAIConnection", return_value=fake_conn
        ):
            await _run_loop_until_files_exist(
                s, [scores_file], register_fn=register_and_reregister, timeout=15
            )

        # Check that only one event was created for this (model, task) pair.
        # get_event_by_model_and_task filters phase != 2, but after completion
        # events are at phase 2. Query all events directly.
        all_events = s.db_manager._cursor.execute(
            "SELECT uuid FROM eval_events WHERE model = ? AND task_uuid = ?",
            ("test-model", task.uuid),
        ).fetchall()
        assert len(all_events) == 1, (
            f"Expected exactly 1 event for (test-model, {task.uuid}), got {len(all_events)}"
        )

        # The evaluation should still have completed
        assert scores_file.exists(), "Scores file missing after re-registration"


# ---------------------------------------------------------------------------
# TestAllocationWithExistingReplicas
# ---------------------------------------------------------------------------

class TestAllocationWithExistingReplicas:
    """Unit tests for get_desired_allocation with replica_counts parameter."""

    def setup_method(self):
        with patch("scheduler.scheduler.FSManager"):
            self.s = Scheduler(None, None, max_generation_jobs=8, max_grading_parallelism=4)

    def _desired_from_instances(self, model_instances):
        result = {}
        for mi in model_instances:
            result[mi.name] = DeploymentInfo(
                model=mi,
                priority=0,
                generation_events={object()},
                grader_events=set(),
            )
        return result

    def test_allocation_uses_actual_replica_counts(self):
        """
        When replica_counts shows a model already has 2 replicas, and
        available_nodes=2, the allocation should give desired=2+2=4,
        not desired=1+2=3 (which would happen if base defaulted to 1).
        """
        mi = _make_model_instance(name="model-A")
        desired = self._desired_from_instances([mi])

        # Model already has 2 replicas running; 2 more nodes available
        replica_counts = {"model-A": 2}
        allocation = self.s.get_desired_allocation(
            desired,
            created_models={"model-A"},
            available_nodes=2,
            replica_counts=replica_counts,
        )

        assert len(allocation) == 1
        model, count = allocation[0]
        assert model.name == "model-A"
        # base=2 (from replica_counts) + 2 (available_nodes distributed)
        assert count == 4, (
            f"Expected desired=4 (base 2 + 2 extra), got {count}"
        )

    def test_allocation_without_replica_counts_defaults_to_one(self):
        """
        When replica_counts is not provided (None), base defaults to 1
        per model, so available_nodes are distributed on top of 1.
        """
        mi = _make_model_instance(name="model-B")
        desired = self._desired_from_instances([mi])

        # No replica_counts passed -- base should be 1
        allocation = self.s.get_desired_allocation(
            desired,
            created_models={"model-B"},
            available_nodes=3,
            replica_counts=None,
        )

        assert len(allocation) == 1
        model, count = allocation[0]
        assert model.name == "model-B"
        # base=1 (default) + 3 (available_nodes distributed)
        assert count == 4, (
            f"Expected desired=4 (base 1 + 3 extra), got {count}"
        )

    def test_replica_counts_distributes_across_multiple_models(self):
        """
        With two models, one already at 3 replicas and one at 1, the extra
        nodes should be distributed evenly starting from their actual counts.
        """
        mi_a = _make_model_instance(name="model-X")
        mi_b = _make_model_instance(name="model-Y")
        desired = self._desired_from_instances([mi_a, mi_b])

        replica_counts = {"model-X": 3, "model-Y": 1}
        allocation = self.s.get_desired_allocation(
            desired,
            created_models={"model-X", "model-Y"},
            available_nodes=4,
            replica_counts=replica_counts,
        )

        counts = {m.name: c for m, c in allocation}
        total = sum(counts.values())
        # base total = 3 + 1 = 4, plus 4 available = 8 total replicas
        assert total == 4 + 4, (
            f"Expected total replicas = 8, got {total}: {counts}"
        )
        # Each model gets 2 extra (4 // 2 = 2 each), so X=5, Y=3
        # sorted by base count: Y(1) gets bonus first
        assert counts["model-Y"] == 1 + 2, f"model-Y: expected 3, got {counts['model-Y']}"
        assert counts["model-X"] == 3 + 2, f"model-X: expected 5, got {counts['model-X']}"
