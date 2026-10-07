"""
Edge-case tests for event failure isolation, --force flag, fail_all_events_with_models,
and grading resume from partial state.
"""
import asyncio
import json
import os
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest

from scheduler.scheduler import Scheduler
from scheduler.event import EventManager, DeploymentInfo, GradingEventInstance
from scheduler.model import ModelInstance
from scheduler.utils import Sentinel, ExceptionWrapper
from tests.fake_slurm import FakeSlurmManager
from tests.test_scheduler import (
    _make_model_instance,
    _write_model_yaml,
    _write_dataset_yaml,
    _write_jsonl,
    _sample_rows,
    _scheduler,
    _FakeOpenAIConnection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dataset_yaml_custom(path: Path, data_path: str, grader_type: str = "exact_match",
                                dataset_name: str = "test_dataset", uuid_str: str = "test-task-uuid") -> Path:
    cfg = {
        "uuid": uuid_str,
        "grader": {"type": grader_type},
        "average_over": [1],
        "pass_at": [1],
        "dataset_name": dataset_name,
        "data_path": data_path,
        "semantic_version": "1.0.0",
        "num_generations": None,
        "meta": {},
        "tag": None,
    }
    p = path / f"{dataset_name}.yaml"
    p.write_text(yaml.dump(cfg))
    return p


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestMultiDatasetPartialFailure:
    """One dataset's grading errors should not prevent the other from completing."""

    @pytest.mark.asyncio
    async def test_one_dataset_fails_other_completes(self, tmp_path):
        """
        Register one model with two datasets. For dataset_bad, the grader always
        raises (using a custom grader registered on the fly). For dataset_good,
        grading succeeds normally.

        With EVAL360_IGNORE_ERRORS=true, both events should complete: one with
        exception records in its grades file, the other with clean correct grades.
        """
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        from scheduler.grader import register
        from scheduler.grader.base import AccuracyGraderBase

        @register("partial-fail-grader")
        class PartialFailGrader(AccuracyGraderBase):
            async def grade_sample(self, sample):
                raise RuntimeError("grading exploded for bad dataset")

        n_rows = 3
        # Dataset "good" — uses exact_match grader (will succeed)
        data_dir_good = tmp_path / "data_good"
        data_dir_good.mkdir()
        _write_jsonl(data_dir_good / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        dataset_good_yaml = _write_dataset_yaml_custom(
            tmp_path, str(data_dir_good / "*.jsonl"),
            grader_type="exact_match",
            dataset_name="dataset_good",
            uuid_str="uuid-good",
        )

        # Dataset "bad" — uses always-failing grader
        data_dir_bad = tmp_path / "data_bad"
        data_dir_bad.mkdir()
        _write_jsonl(data_dir_bad / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        dataset_bad_yaml = _write_dataset_yaml_custom(
            tmp_path, str(data_dir_bad / "*.jsonl"),
            grader_type="partial-fail-grader",
            dataset_name="dataset_bad",
            uuid_str="uuid-bad",
        )

        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])

        s = _scheduler(tmp_path)
        s.slurm_manager = FakeSlurmManager()
        with (
            patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn),
            patch.dict(os.environ, {"EVAL360_IGNORE_ERRORS": "true"}),
        ):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml)],
                paths_to_datasets=[str(dataset_good_yaml), str(dataset_bad_yaml)],
            )

        out_dir = tmp_path / "output" / "test-model"

        # Good dataset should have clean grades with "correct" field
        good_scores = out_dir / "dataset_good_scores.yaml"
        assert good_scores.exists(), "good dataset scores file not created"
        good_grades = out_dir / "dataset_good_grades.jsonl"
        good_lines = [json.loads(l) for l in good_grades.read_text().strip().splitlines()]
        assert len(good_lines) == n_rows
        assert all("correct" in row for row in good_lines), "good grades missing 'correct' field"

        # Bad dataset should have exception records
        bad_scores = out_dir / "dataset_bad_scores.yaml"
        assert bad_scores.exists(), "bad dataset scores file not created"
        bad_grades = out_dir / "dataset_bad_grades.jsonl"
        bad_lines = [json.loads(l) for l in bad_grades.read_text().strip().splitlines()]
        assert len(bad_lines) == n_rows
        assert all("exception" in row for row in bad_lines), "bad grades missing exception records"


class TestForceFlag:
    """Tests for --force flag that clears and re-runs completed evaluations."""

    @pytest.mark.asyncio
    async def test_force_clears_and_reruns_completed_evaluation(self, tmp_path):
        """
        Pre-populate output files with OLD data. Run with force=True.
        The output files should contain NEW data (re-generated), not the old data.
        """
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 3
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        # Pre-populate output files with OLD data (answer "OLD_ANSWER")
        out_dir = tmp_path / "output" / "test-model"
        out_dir.mkdir(parents=True)
        old_gen_rows = _sample_rows(n_rows, ground_truth="A")
        for r in old_gen_rows:
            r["generations"] = ["OLD_ANSWER"]
        _write_jsonl(out_dir / "test_dataset_generations.jsonl", old_gen_rows)

        old_grade_rows = _sample_rows(n_rows, ground_truth="A")
        for r in old_grade_rows:
            r["generations"] = ["OLD_ANSWER"]
            r["parsed_generations"] = ["OLD_ANSWER"]
            r["correct"] = [False]
        _write_jsonl(out_dir / "test_dataset_grades.jsonl", old_grade_rows)

        scores_file = out_dir / "test_dataset_scores.yaml"
        scores_file.write_text('"pass@1_avg@1": 0.0\n')

        # Run with force=True; new canned answer is "A" (matches ground truth)
        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])

        s = _scheduler(tmp_path)
        s.slurm_manager = FakeSlurmManager()
        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml)],
                paths_to_datasets=[str(dataset_yaml)],
                force=True,
            )

        # Verify new generations replaced old ones
        gen_file = out_dir / "test_dataset_generations.jsonl"
        gen_lines = [json.loads(l) for l in gen_file.read_text().strip().splitlines()]
        assert len(gen_lines) == n_rows
        for row in gen_lines:
            assert row["generations"] == ["A"], f"Expected new answer 'A', got {row['generations']}"

        # Verify new grades replaced old ones
        grades_file = out_dir / "test_dataset_grades.jsonl"
        grade_lines = [json.loads(l) for l in grades_file.read_text().strip().splitlines()]
        assert len(grade_lines) == n_rows
        for row in grade_lines:
            assert row["correct"] == [True], f"Expected correct=[True], got {row['correct']}"

        # Verify scores reflect new (correct) answers
        content = scores_file.read_text()
        assert "1.0" in content or "1\n" in content, f"Expected perfect score, got: {content}"


class TestFailAllEventsSharedModel:
    """
    Tests a potential bug in EventManager.fail_all_events_with_models where
    `del self.desired_models_dict[event_instance.model]` inside a loop can
    raise KeyError if two events share the same model (second delete hits
    a key already removed by the first).
    """

    @pytest.mark.asyncio
    async def test_fail_all_events_two_events_same_model_no_crash(self, tmp_path):
        """
        Register one model with two datasets. Both events share the same model name.
        Call fail_all_events_with_models with that model. Should not raise KeyError.
        """
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        from scheduler.database import DatabaseManager

        db = DatabaseManager()
        em = EventManager(db)

        model = _make_model_instance(name="shared-model", output_path=tmp_path / "output")
        db.register_model(model)

        event1 = GradingEventInstance(
            uuid="ev-001",
            parent_uuid="",
            model="shared-model",
            task_uuid="task-1",
            path_to_generations=str(tmp_path / "gen1.jsonl"),
            path_to_grades=str(tmp_path / "grades1.jsonl"),
            path_to_scores=str(tmp_path / "scores1.yaml"),
            grader_type="exact_match",
            parser_type="noop",
        )
        event2 = GradingEventInstance(
            uuid="ev-002",
            parent_uuid="",
            model="shared-model",
            task_uuid="task-2",
            path_to_generations=str(tmp_path / "gen2.jsonl"),
            path_to_grades=str(tmp_path / "grades2.jsonl"),
            path_to_scores=str(tmp_path / "scores2.yaml"),
            grader_type="exact_match",
            parser_type="noop",
        )

        db.register_event(event1)
        db.register_event(event2)

        # Simulate both events being active (have asyncio tasks)
        dummy_task = asyncio.current_task() or asyncio.ensure_future(asyncio.sleep(0))
        em.asyncio_task_dict[event1] = [("generation", dummy_task)]
        em.asyncio_task_dict[event2] = [("generation", dummy_task)]

        # Add the model as desired (so it's in desired_models_dict)
        em.desired_models_dict["shared-model"] = DeploymentInfo(
            model=model,
            priority=0,
            generation_events={event1, event2},
            grader_events=set(),
        )

        # This should NOT raise KeyError — the bug is that the second iteration
        # tries to delete a key already removed by the first iteration.
        try:
            await em.fail_all_events_with_models(["shared-model"])
        except KeyError as exc:
            pytest.fail(
                f"fail_all_events_with_models raised KeyError on shared model: {exc}. "
                f"The method deletes the same key twice when two events share a model."
            )

        # Both events should be marked as failed (phase -1)
        ev1_phase = db._cursor.execute(
            "SELECT phase FROM eval_events WHERE uuid = ?", ("ev-001",)
        ).fetchone()[0]
        ev2_phase = db._cursor.execute(
            "SELECT phase FROM eval_events WHERE uuid = ?", ("ev-002",)
        ).fetchone()[0]
        assert ev1_phase == -1, f"Event 1 phase should be -1, got {ev1_phase}"
        assert ev2_phase == -1, f"Event 2 phase should be -1, got {ev2_phase}"


class TestGradingResumeFromPartialGrades:
    """Test that grading resumes from a partial grades file."""

    @pytest.mark.asyncio
    async def test_resume_grades_only_remaining_rows(self, tmp_path):
        """
        Pre-populate 2 of 4 rows in the grades file, with all 4 generations present.
        Run evaluate-now. The final grades file should have all 4 rows and scores
        should be computed correctly.
        """
        os.environ["EVAL360_IN_MEMORY_DB"] = "true"
        n_rows = 4
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_jsonl(data_dir / "test.jsonl", _sample_rows(n_rows, ground_truth="A"))
        model_yaml = _write_model_yaml(tmp_path, output_path=tmp_path / "output")
        dataset_yaml = _write_dataset_yaml(tmp_path, str(data_dir / "*.jsonl"))

        out_dir = tmp_path / "output" / "test-model"
        out_dir.mkdir(parents=True)

        # All 4 generations already present
        gen_rows = _sample_rows(n_rows, ground_truth="A")
        for r in gen_rows:
            r["generations"] = ["A"]
        _write_jsonl(out_dir / "test_dataset_generations.jsonl", gen_rows)

        # Only 2 of 4 grades present
        grade_rows = _sample_rows(2, ground_truth="A")
        for r in grade_rows:
            r["generations"] = ["A"]
            r["parsed_generations"] = ["A"]
            r["correct"] = [True]
        _write_jsonl(out_dir / "test_dataset_grades.jsonl", grade_rows)

        fake_conn = _FakeOpenAIConnection(canned_answers=["A"])

        s = _scheduler(tmp_path)
        s.slurm_manager = FakeSlurmManager()
        with patch("scheduler.openai_interface.OpenAIConnection", return_value=fake_conn):
            await s.run_evaluate_now(
                paths_to_model_specs=[str(model_yaml)],
                paths_to_datasets=[str(dataset_yaml)],
            )

        # All 4 rows should be graded
        grades_file = out_dir / "test_dataset_grades.jsonl"
        grade_lines = [json.loads(l) for l in grades_file.read_text().strip().splitlines()]
        assert len(grade_lines) == n_rows, f"Expected {n_rows} grades, got {len(grade_lines)}"

        # All correct, so score should be 1.0
        scores_file = out_dir / "test_dataset_scores.yaml"
        assert scores_file.exists(), "scores file not created"
        content = scores_file.read_text()
        assert "1.0" in content or "1\n" in content, f"Expected perfect score, got: {content}"
