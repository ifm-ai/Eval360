"""
Tests for Hugging Face URI support in data_path.

All HF network calls (HfApi.file_exists, hf_hub_download) are mocked so
no real HF credentials or network access are needed.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from scheduler.utils import (
    is_hf_uri,
    _parse_hf_uri,
    check_hf_file_exists,
    resolve_hf_path,
)


# ---------------------------------------------------------------------------
# is_hf_uri
# ---------------------------------------------------------------------------


class TestIsHfUri:
    def test_hf_uri_recognised(self):
        assert is_hf_uri("hf://example-org/example-repo/gpqa/gpqa.jsonl") is True

    def test_local_path_not_recognised(self):
        assert is_hf_uri("/data/foo.jsonl") is False

    def test_glob_not_recognised(self):
        assert is_hf_uri("/data/*.jsonl") is False

    def test_empty_string_not_recognised(self):
        assert is_hf_uri("") is False


# ---------------------------------------------------------------------------
# _parse_hf_uri
# ---------------------------------------------------------------------------


class TestParseHfUri:
    def test_with_subfolder_and_revision(self):
        repo_id, filename, subfolder, revision = _parse_hf_uri(
            "hf://example-org/example-repo/gpqa-diamond/gpqa_diamond.jsonl@main"
        )
        assert repo_id == "example-org/example-repo"
        assert filename == "gpqa_diamond.jsonl"
        assert subfolder == "gpqa-diamond"
        assert revision == "main"

    def test_without_revision_defaults_to_main(self):
        _, _, _, revision = _parse_hf_uri(
            "hf://example-org/example-repo/gpqa-diamond/gpqa_diamond.jsonl"
        )
        assert revision == "main"

    def test_without_subfolder(self):
        repo_id, filename, subfolder, revision = _parse_hf_uri(
            "hf://example-org/example-repo/gpqa_diamond.jsonl@v1"
        )
        assert repo_id == "example-org/example-repo"
        assert filename == "gpqa_diamond.jsonl"
        assert subfolder is None
        assert revision == "v1"

    def test_nested_subfolder(self):
        repo_id, filename, subfolder, revision = _parse_hf_uri(
            "hf://org/repo/a/b/c/file.jsonl@abc123"
        )
        assert repo_id == "org/repo"
        assert filename == "file.jsonl"
        assert subfolder == "a/b/c"
        assert revision == "abc123"

    def test_glob_pattern_in_filename(self):
        repo_id, filename, subfolder, revision = _parse_hf_uri(
            "hf://example-org/example-repo/bbh/*.jsonl@main"
        )
        assert repo_id == "example-org/example-repo"
        assert filename == "*.jsonl"
        assert subfolder == "bbh"
        assert revision == "main"

    def test_missing_path_raises(self):
        with pytest.raises(ValueError, match="Invalid HF URI"):
            _parse_hf_uri("hf://org/repo")


# ---------------------------------------------------------------------------
# check_hf_file_exists
# ---------------------------------------------------------------------------


class TestCheckHfFileExists:
    def test_raises_when_file_not_found(self):
        with patch("huggingface_hub.HfApi.file_exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="HF file not found"):
                check_hf_file_exists("hf://example-org/example-repo/gpqa/gpqa.jsonl")

    def test_passes_when_file_exists(self):
        with patch("huggingface_hub.HfApi.file_exists", return_value=True):
            check_hf_file_exists(
                "hf://example-org/example-repo/gpqa/gpqa.jsonl"
            )  # no exception

    def test_correct_args_passed_to_file_exists(self):
        with patch("huggingface_hub.HfApi.file_exists", return_value=True) as mock_fe:
            check_hf_file_exists(
                "hf://example-org/example-repo/gpqa-diamond/gpqa_diamond.jsonl@v2"
            )
        mock_fe.assert_called_once_with(
            repo_id="example-org/example-repo",
            filename="gpqa-diamond/gpqa_diamond.jsonl",
            repo_type="dataset",
            revision="v2",
        )

    def test_full_path_passed_when_no_subfolder(self):
        with patch("huggingface_hub.HfApi.file_exists", return_value=True) as mock_fe:
            check_hf_file_exists("hf://example-org/example-repo/flat.jsonl")
        mock_fe.assert_called_once_with(
            repo_id="example-org/example-repo",
            filename="flat.jsonl",
            repo_type="dataset",
            revision="main",
        )

    def test_glob_passes_when_matches_exist(self):
        repo_files = ["bbh/file1.jsonl", "bbh/file2.jsonl", "other/unrelated.txt"]
        with patch(
            "huggingface_hub.HfApi.list_repo_files", return_value=iter(repo_files)
        ):
            check_hf_file_exists("hf://org/repo/bbh/*.jsonl")  # no exception

    def test_glob_raises_when_no_matches(self):
        with patch("huggingface_hub.HfApi.list_repo_files", return_value=iter([])):
            with pytest.raises(FileNotFoundError, match="No HF files matched pattern"):
                check_hf_file_exists("hf://org/repo/bbh/*.jsonl")

    def test_glob_uses_list_repo_files_not_file_exists(self):
        repo_files = ["bbh/a.jsonl"]
        with (
            patch(
                "huggingface_hub.HfApi.list_repo_files", return_value=iter(repo_files)
            ) as mock_list,
            patch("huggingface_hub.HfApi.file_exists") as mock_fe,
        ):
            check_hf_file_exists("hf://org/repo/bbh/*.jsonl")
        mock_list.assert_called_once()
        mock_fe.assert_not_called()


# ---------------------------------------------------------------------------
# resolve_hf_path
# ---------------------------------------------------------------------------


class TestResolveHfPath:
    def test_returns_local_path_from_hf_hub_download(self):
        with patch("scheduler.utils.hf_hub_download", return_value="/cache/gpqa.jsonl"):
            result = resolve_hf_path(
                "hf://example-org/example-repo/gpqa-diamond/gpqa_diamond.jsonl@main"
            )
        assert result == "/cache/gpqa.jsonl"

    def test_cache_dir_forwarded(self):
        with patch(
            "scheduler.utils.hf_hub_download", return_value="/custom/gpqa.jsonl"
        ) as mock_dl:
            resolve_hf_path(
                "hf://example-org/example-repo/gpqa.jsonl", cache_dir="/custom/cache"
            )
        from scheduler.utils import _LoggingTqdm

        mock_dl.assert_called_once_with(
            repo_id="example-org/example-repo",
            filename="gpqa.jsonl",
            subfolder=None,
            repo_type="dataset",
            revision="main",
            cache_dir="/custom/cache",
            tqdm_class=_LoggingTqdm,
        )

    def test_none_cache_dir_forwarded(self):
        with patch(
            "scheduler.utils.hf_hub_download", return_value="/hf/cache/gpqa.jsonl"
        ) as mock_dl:
            resolve_hf_path(
                "hf://example-org/example-repo/gpqa/gpqa.jsonl", cache_dir=None
            )
        _, kwargs = mock_dl.call_args
        assert kwargs["cache_dir"] is None

    def test_single_file_download_works_without_tqdm_class_support(self):
        def strict_download(
            *,
            repo_id,
            filename,
            subfolder,
            repo_type,
            revision,
            cache_dir,
        ):
            assert repo_id == "example-org/example-repo"
            assert filename == "gpqa.jsonl"
            assert subfolder is None
            assert repo_type == "dataset"
            assert revision == "main"
            assert cache_dir is None
            return "/cache/gpqa.jsonl"

        with patch("scheduler.utils.hf_hub_download", side_effect=strict_download):
            result = resolve_hf_path("hf://example-org/example-repo/gpqa.jsonl")

        assert result == "/cache/gpqa.jsonl"

    def test_glob_downloads_all_matching_files(self):
        repo_files = ["bbh/a.jsonl", "bbh/b.jsonl", "other/unrelated.txt"]
        download_calls = []

        def fake_download(**kwargs):
            download_calls.append(kwargs["filename"])
            return f"/cache/bbh/{kwargs['filename']}"

        with (
            patch(
                "huggingface_hub.HfApi.list_repo_files", return_value=iter(repo_files)
            ),
            patch("scheduler.utils.hf_hub_download", side_effect=fake_download),
        ):
            result = resolve_hf_path("hf://org/repo/bbh/*.jsonl")

        assert sorted(download_calls) == ["a.jsonl", "b.jsonl"]
        assert result == "/cache/bbh/*.jsonl"

    def test_glob_raises_when_no_matches(self):
        with (
            patch("huggingface_hub.HfApi.list_repo_files", return_value=iter([])),
            patch("scheduler.utils.hf_hub_download"),
        ):
            with pytest.raises(FileNotFoundError, match="No HF files matched pattern"):
                resolve_hf_path("hf://org/repo/bbh/*.jsonl")

    def test_glob_returns_local_glob_pattern(self):
        repo_files = ["bbh/x.jsonl", "bbh/y.jsonl"]

        with (
            patch(
                "huggingface_hub.HfApi.list_repo_files", return_value=iter(repo_files)
            ),
            patch(
                "scheduler.utils.hf_hub_download",
                side_effect=lambda **kw: f"/snap/bbh/{kw['filename']}",
            ),
        ):
            result = resolve_hf_path("hf://org/repo/bbh/*.jsonl")

        assert result == "/snap/bbh/*.jsonl"

    def test_glob_download_works_without_tqdm_class_support(self):
        repo_files = ["bbh/x.jsonl", "bbh/y.jsonl"]
        download_calls = []

        def strict_download(
            *,
            repo_id,
            filename,
            subfolder,
            repo_type,
            revision,
            cache_dir,
        ):
            download_calls.append(
                (repo_id, filename, subfolder, repo_type, revision, cache_dir)
            )
            return f"/snap/bbh/{filename}"

        with (
            patch(
                "huggingface_hub.HfApi.list_repo_files", return_value=iter(repo_files)
            ),
            patch("scheduler.utils.hf_hub_download", side_effect=strict_download),
        ):
            result = resolve_hf_path("hf://org/repo/bbh/*.jsonl")

        assert download_calls == [
            ("org/repo", "x.jsonl", "bbh", "dataset", "main", None),
            ("org/repo", "y.jsonl", "bbh", "dataset", "main", None),
        ]
        assert result == "/snap/bbh/*.jsonl"


# ---------------------------------------------------------------------------
# Task.parse_yaml with HF URIs
# ---------------------------------------------------------------------------


def _write_task_yaml(path: Path, data_path: str, num_generations=None) -> Path:
    obj = {
        "uuid": "test-uuid",
        "dataset_name": "test_ds",
        "data_path": data_path,
        "semantic_version": "1.0.0",
        "average_over": [1],
        "pass_at": [1],
        "meta": None,
        "grader": {"type": "multiple_choice"},
    }
    if num_generations is not None:
        obj["num_generations"] = num_generations
    yaml_path = path / "task.yaml"
    yaml_path.write_text(yaml.dump(obj))
    return yaml_path


class TestTaskParseYamlHfUri:
    def test_calls_check_hf_file_exists(self, tmp_path):
        yaml_path = _write_task_yaml(tmp_path, "hf://org/repo/file.jsonl")
        with patch("scheduler.task.check_hf_file_exists") as mock_check:
            from scheduler.task import Task

            Task.parse_yaml(str(yaml_path))
        mock_check.assert_called_once_with("hf://org/repo/file.jsonl")

    def test_num_generations_left_as_none_for_hf(self, tmp_path):
        yaml_path = _write_task_yaml(tmp_path, "hf://org/repo/file.jsonl")
        with patch("scheduler.task.check_hf_file_exists"):
            from scheduler.task import Task

            task = Task.parse_yaml(str(yaml_path))
        assert task.num_generations is None

    def test_num_generations_preserved_if_set(self, tmp_path):
        yaml_path = _write_task_yaml(
            tmp_path, "hf://org/repo/file.jsonl", num_generations=100
        )
        with patch("scheduler.task.check_hf_file_exists"):
            from scheduler.task import Task

            task = Task.parse_yaml(str(yaml_path))
        assert task.num_generations == 100

    def test_count_jsonl_records_not_called_for_hf(self, tmp_path):
        yaml_path = _write_task_yaml(tmp_path, "hf://org/repo/file.jsonl")
        with (
            patch("scheduler.task.check_hf_file_exists"),
            patch("scheduler.task.count_jsonl_records") as mock_count,
        ):
            from scheduler.task import Task

            Task.parse_yaml(str(yaml_path))
        mock_count.assert_not_called()

    def test_file_not_found_raises(self, tmp_path):
        yaml_path = _write_task_yaml(tmp_path, "hf://org/repo/missing.jsonl")
        with patch(
            "scheduler.task.check_hf_file_exists",
            side_effect=FileNotFoundError("HF file not found"),
        ):
            from scheduler.task import Task

            with pytest.raises(FileNotFoundError):
                Task.parse_yaml(str(yaml_path))


# ---------------------------------------------------------------------------
# Scheduler.register_task with HF URIs
# ---------------------------------------------------------------------------


class TestSchedulerRegisterTaskHfUri:
    def _make_task(self, data_path, num_generations=None):
        from scheduler.task import AsyncGenerationTask

        # ensure multiple_choice grader is registered
        import scheduler.grader  # noqa: F401

        return AsyncGenerationTask(
            uuid="test-uuid",
            dataset_name="test_ds",
            data_path=data_path,
            semantic_version="1.0.0",
            average_over=[1],
            pass_at=[1],
            meta=None,
            num_generations=num_generations,
            grader={"type": "multiple_choice"},
        )

    @pytest.mark.asyncio
    async def test_downloads_and_autocounts_when_num_generations_none(self, tmp_path):
        task = self._make_task("hf://org/repo/file.jsonl")
        local_file = tmp_path / "file.jsonl"
        local_file.write_text('{"row": 0}\n{"row": 1}\n{"row": 2}\n')

        from scheduler.scheduler import Scheduler

        scheduler = Scheduler(None, None)
        scheduler.db_manager = MagicMock()
        scheduler.db_manager.get_all_models.return_value = {}
        # Simulate first-time registration with no previously resolved count.
        scheduler.db_manager.get_task.return_value = task

        async def fake_create_events(tasks):
            pass

        scheduler.event_manager = MagicMock()
        scheduler.event_manager.create_events_for_new_tasks = fake_create_events

        with patch("scheduler.scheduler.resolve_hf_path", return_value=str(local_file)):
            await scheduler.register_task(task)

        # Resolution is part of preparation, so the first durable task row
        # already contains the validated count.
        scheduler.db_manager.register_task.assert_called_once()
        assert scheduler.db_manager.register_task.call_args[0][0].num_generations == 3
        scheduler.db_manager.update_task_num_generations.assert_not_called()

    @pytest.mark.asyncio
    async def test_validates_num_generations_when_set(self, tmp_path):
        task = self._make_task("hf://org/repo/file.jsonl", num_generations=99)
        local_file = tmp_path / "file.jsonl"
        local_file.write_text('{"row": 0}\n{"row": 1}\n')

        from scheduler.scheduler import Scheduler

        scheduler = Scheduler(None, None)
        scheduler.db_manager = MagicMock()
        scheduler.db_manager.get_all_models.return_value = {}

        async def fake_create_events(tasks):
            pass

        scheduler.event_manager = MagicMock()
        scheduler.event_manager.create_events_for_new_tasks = fake_create_events

        with patch("scheduler.scheduler.resolve_hf_path", return_value=str(local_file)):
            with pytest.raises(ValueError, match="num_generations=99 does not match"):
                await scheduler.register_task(task)

    @pytest.mark.asyncio
    async def test_hf_cache_dir_passed_to_resolve(self, tmp_path):
        task = self._make_task("hf://org/repo/file.jsonl")
        local_file = tmp_path / "file.jsonl"
        local_file.write_text('{"row": 0}\n')

        from scheduler.scheduler import Scheduler

        scheduler = Scheduler(None, None, hf_cache_dir="/my/cache")
        scheduler.db_manager = MagicMock()
        # Simulate first-time registration: task was just inserted with num_generations=None
        scheduler.db_manager.get_task.return_value = task

        async def fake_create_events(tasks):
            pass

        scheduler.event_manager = MagicMock()
        scheduler.event_manager.create_events_for_new_tasks = fake_create_events

        with patch(
            "scheduler.scheduler.resolve_hf_path", return_value=str(local_file)
        ) as mock_resolve:
            await scheduler.register_task(task)

        mock_resolve.assert_called_once_with(
            "hf://org/repo/file.jsonl", cache_dir="/my/cache"
        )
