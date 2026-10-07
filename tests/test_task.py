"""
Tests for Task.parse_yaml, focusing on the auto-count behaviour for num_generations.
"""
import json
import os
import textwrap
import pytest

from scheduler.task import Task, ImportedDatasetTask, ImportedDatasetConfig

_skip_if_root = pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permissions")


_IMPORTED_DATASET_YAML = textwrap.dedent("""\
    uuid: "test-imported-1"
    average_over:
      - 1
    pass_at:
      - 1
    dataset_name: bfcl
    semantic_version: "1.0.0"
    meta: null
    imported_dataset:
      name: bfcl
      commit: abcdef12345
      args:
        version: 4.0
        judge_model: gpt-4o
""")


def _write_yaml(path, extra=""):
    path.write_text(
        textwrap.dedent(f"""\
            uuid: "test-task-1"
            grader:
              type: multiple_choice
            average_over:
              - 1
            pass_at:
              - 1
            dataset_name: test_dataset
            data_path: {path.parent}/*.jsonl
            semantic_version: "1.0.0"
            meta: null
            {extra}
        """)
    )


def _write_jsonl(path, n_records):
    with open(path, "w") as f:
        for i in range(n_records):
            f.write(json.dumps({"row": i, "completion_input": "q", "chat_input": [], "ground_truth": "A"}) + "\n")


class TestNumGenerationsAutoCount:
    def test_auto_counts_when_omitted(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 5)
        _write_yaml(tmp_path / "task.yaml")
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.num_generations == 5

    def test_explicit_value_is_respected(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 5)
        _write_yaml(tmp_path / "task.yaml", extra="num_generations: 5")
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.num_generations == 5

    def test_counts_across_multiple_files(self, tmp_path):
        _write_jsonl(tmp_path / "a.jsonl", 3)
        _write_jsonl(tmp_path / "b.jsonl", 4)
        _write_yaml(tmp_path / "task.yaml")
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.num_generations == 7

    def test_skips_blank_lines(self, tmp_path):
        (tmp_path / "data.jsonl").write_text(
            json.dumps({"row": 0, "completion_input": "q", "chat_input": [], "ground_truth": "A"}) + "\n"
            "\n"
            + json.dumps({"row": 1, "completion_input": "q", "chat_input": [], "ground_truth": "B"}) + "\n"
        )
        _write_yaml(tmp_path / "task.yaml")
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.num_generations == 2

    def test_raises_when_no_files_match(self, tmp_path):
        _write_yaml(tmp_path / "task.yaml")
        # data_path glob matches nothing (no .jsonl files written)
        with pytest.raises(FileNotFoundError, match="No JSONL files found"):
            Task.parse_yaml(tmp_path / "task.yaml")

    def test_raises_when_explicit_value_mismatches_count(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 5)
        _write_yaml(tmp_path / "task.yaml", extra="num_generations: 99")
        with pytest.raises(ValueError, match="does not match the actual record count of 5"):
            Task.parse_yaml(tmp_path / "task.yaml")

    @_skip_if_root
    def test_raises_permission_error_when_auto_counting_with_unreadable_file(self, tmp_path):
        _write_jsonl(tmp_path / "a.jsonl", 3)
        unreadable = tmp_path / "b.jsonl"
        _write_jsonl(unreadable, 2)
        os.chmod(unreadable, 0o000)
        try:
            _write_yaml(tmp_path / "task.yaml")
            with pytest.raises(PermissionError, match="Cannot auto-count records"):
                Task.parse_yaml(tmp_path / "task.yaml")
        finally:
            os.chmod(unreadable, 0o644)

    @_skip_if_root
    def test_no_error_when_explicit_num_generations_with_partial_unreadable(self, tmp_path):
        _write_jsonl(tmp_path / "a.jsonl", 3)
        unreadable = tmp_path / "b.jsonl"
        _write_jsonl(unreadable, 2)
        os.chmod(unreadable, 0o000)
        try:
            # With explicit num_generations matching readable count, no error at parse time
            _write_yaml(tmp_path / "task.yaml", extra="num_generations: 3")
            task = Task.parse_yaml(tmp_path / "task.yaml")
            assert task.num_generations == 3
        finally:
            os.chmod(unreadable, 0o644)

    @_skip_if_root
    def test_raises_value_error_when_explicit_num_generations_mismatches_readable_count(self, tmp_path):
        _write_jsonl(tmp_path / "a.jsonl", 3)
        unreadable = tmp_path / "b.jsonl"
        _write_jsonl(unreadable, 2)
        os.chmod(unreadable, 0o000)
        try:
            # 5 total but only 3 readable; explicit=5 mismatches readable count of 3
            _write_yaml(tmp_path / "task.yaml", extra="num_generations: 5")
            with pytest.raises(ValueError, match="does not match the actual record count of 3"):
                Task.parse_yaml(tmp_path / "task.yaml")
        finally:
            os.chmod(unreadable, 0o644)


class TestImportedDatasetTask:
    def test_parse_yaml_imported_dataset(self, tmp_path):
        (tmp_path / "task.yaml").write_text(_IMPORTED_DATASET_YAML)
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert isinstance(task, ImportedDatasetTask)
        assert task.imported_dataset.name == "bfcl"
        assert task.imported_dataset.commit == "abcdef12345"
        assert task.imported_dataset.args == {"version": 4.0, "judge_model": "gpt-4o"}

    def test_no_jsonl_counting_for_imported_dataset(self, tmp_path):
        # No .jsonl files present — should not raise
        (tmp_path / "task.yaml").write_text(_IMPORTED_DATASET_YAML)
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert isinstance(task, ImportedDatasetTask)
        assert not hasattr(task, "num_generations")

    def test_imported_dataset_with_data_path_raises(self, tmp_path):
        yaml = _IMPORTED_DATASET_YAML + "data_path: /some/path/*.jsonl\n"
        (tmp_path / "task.yaml").write_text(yaml)
        with pytest.raises(ValueError, match="data_path must not be set"):
            Task.parse_yaml(tmp_path / "task.yaml")

    def test_imported_dataset_with_grader_raises(self, tmp_path):
        yaml = _IMPORTED_DATASET_YAML + "grader:\n  type: multiple_choice\n"
        (tmp_path / "task.yaml").write_text(yaml)
        with pytest.raises(ValueError, match="grader must not be set"):
            Task.parse_yaml(tmp_path / "task.yaml")

    def test_standard_task_without_grader_raises(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 3)
        yaml = textwrap.dedent(f"""\
            uuid: "test-task-2"
            average_over:
              - 1
            pass_at:
              - 1
            dataset_name: test_dataset
            data_path: {tmp_path}/*.jsonl
            semantic_version: "1.0.0"
            meta: null
        """)
        (tmp_path / "task.yaml").write_text(yaml)
        with pytest.raises(ValueError, match="grader"):
            Task.parse_yaml(tmp_path / "task.yaml")

    def test_imported_dataset_config_minimal(self):
        cfg = ImportedDatasetConfig(name="bfcl", commit="abc123")
        assert cfg.args == {}

    def test_imported_dataset_config_with_args(self):
        cfg = ImportedDatasetConfig(name="arena_hard", commit="def456", args={"judge": "gpt-4o"})
        assert cfg.args == {"judge": "gpt-4o"}


class TestTaskTags:
    def test_default_tag_is_any(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 1)
        _write_yaml(tmp_path / "task.yaml")
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.tag == "any"

    def test_custom_tag_parsed(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 1)
        extra = "tag: vision\n"
        _write_yaml(tmp_path / "task.yaml", extra=extra)
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.tag == "vision"

    def test_tag_normalizes_any(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 1)
        extra = "tag: Any\n"
        _write_yaml(tmp_path / "task.yaml", extra=extra)
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.tag == "any"

    def test_tag_yaml_null_normalizes_to_any(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 1)
        extra = "tag: null\n"
        _write_yaml(tmp_path / "task.yaml", extra=extra)
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.tag == "any"

    def test_tag_blank_string_normalizes_to_any(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", 1)
        extra = "tag: \"   \"\n"
        _write_yaml(tmp_path / "task.yaml", extra=extra)
        task = Task.parse_yaml(tmp_path / "task.yaml")
        assert task.tag == "any"
