"""Tests for scheduler/filesystem_manager.py"""
import asyncio
import errno
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scheduler.filesystem_manager import (
    _glob_constant_prefix,
    _mkdir_shared_fs_safe,
    DirType,
    FSManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain(fs):
    """Drain all items currently in the queue and return them as a list."""
    items = []
    while not fs._queue.empty():
        items.append(fs._queue.get_nowait())
    return items


def _make_event(src_path):
    event = MagicMock()
    event.src_path = src_path
    return event


def _make_fs_and_handler(path, dirtype, info=None):
    """
    Create an FSManager with a real asyncio queue but a mocked Observer,
    register `path` with `dirtype`, and return (fs, handler).

    call_soon_threadsafe is replaced with a direct call so handler events
    are enqueued synchronously in tests.
    """
    captured = {}

    def capture_schedule(handler, watch_path, recursive=True):
        captured["handler"] = handler

    def immediate(fn, *args):
        fn(*args)

    with patch("scheduler.filesystem_manager.Observer") as mock_obs:
        mock_obs.return_value.schedule = MagicMock()
        mock_obs.return_value.start = MagicMock()
        fs = FSManager(model_directory=None, dataset_directory=None)

    fs._observer.schedule = capture_schedule
    fs._loop.call_soon_threadsafe = immediate
    fs.register_path(path, dirtype, info=info)
    return fs, captured["handler"]


# ---------------------------------------------------------------------------
# _glob_constant_prefix
# ---------------------------------------------------------------------------

class TestGlobConstantPrefix:

    def test_no_glob_returns_path_unchanged(self):
        assert _glob_constant_prefix("/foo/bar/baz") == "/foo/bar/baz"

    def test_star_glob_returns_parent(self):
        assert _glob_constant_prefix("/foo/bar/*.yaml") == "/foo/bar"

    def test_double_star_glob_returns_parent(self):
        assert _glob_constant_prefix("/foo/**/*.yaml") == "/foo"

    def test_question_mark_glob(self):
        assert _glob_constant_prefix("/foo/ba?.yaml") == "/foo"

    def test_bracket_glob(self):
        assert _glob_constant_prefix("/foo/[abc].yaml") == "/foo"

    def test_glob_in_middle_segment(self):
        assert _glob_constant_prefix("/foo/*/bar.yaml") == "/foo"

    def test_root_only(self):
        assert _glob_constant_prefix("/") == "/"

    def test_single_segment(self):
        assert _glob_constant_prefix("/foo") == "/foo"

    def test_relative_path_raises(self):
        with pytest.raises(ValueError, match="glob must be absolute"):
            _glob_constant_prefix("foo/bar/*.yaml")

    def test_deep_constant_prefix(self):
        assert _glob_constant_prefix("/a/b/c/d/e/*.yaml") == "/a/b/c/d/e"

    def test_glob_immediately_after_root(self):
        assert _glob_constant_prefix("/*.yaml") == "/"

    def test_path_with_spaces(self):
        assert _glob_constant_prefix("/foo/my dir/*.yaml") == "/foo/my dir"

    def test_star_only_segment(self):
        assert _glob_constant_prefix("/foo/*/baz") == "/foo"

    def test_multiple_glob_segments(self):
        # Only the first glob segment stops the prefix
        assert _glob_constant_prefix("/foo/*.yaml/*/bar") == "/foo"


# ---------------------------------------------------------------------------
# _mkdir_shared_fs_safe
# ---------------------------------------------------------------------------

class TestMkdirSharedFsSafe:

    @pytest.mark.asyncio
    async def test_creates_directory(self, tmp_path):
        target = tmp_path / "new_dir" / "nested"
        await _mkdir_shared_fs_safe(target)
        assert target.exists()

    @pytest.mark.asyncio
    async def test_existing_directory_is_ok(self, tmp_path):
        await _mkdir_shared_fs_safe(tmp_path)
        assert tmp_path.exists()

    @pytest.mark.asyncio
    async def test_retries_on_estale_then_succeeds(self, tmp_path):
        target = tmp_path / "stale_dir"
        call_count = 0
        original_mkdir = Path.mkdir

        def flaky_mkdir(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError(errno.ESTALE, "Stale file handle")
            original_mkdir(self, **kwargs)

        with patch.object(Path, "mkdir", flaky_mkdir):
            await _mkdir_shared_fs_safe(target, retries=5, delay=0)

        assert call_count == 3
        assert target.exists()

    @pytest.mark.asyncio
    async def test_raises_oserror_after_exhausting_retries(self, tmp_path):
        target = tmp_path / "always_stale"

        def always_stale(self, **kwargs):
            raise OSError(errno.ESTALE, "Stale file handle")

        with patch.object(Path, "mkdir", always_stale):
            with pytest.raises(OSError) as exc_info:
                await _mkdir_shared_fs_safe(target, retries=3, delay=0)

        assert exc_info.value.errno == errno.ESTALE

    @pytest.mark.asyncio
    async def test_non_estale_error_raises_immediately(self, tmp_path):
        call_count = 0

        def perm_error(self, **kwargs):
            nonlocal call_count
            call_count += 1
            raise OSError(errno.EACCES, "Permission denied")

        with patch.object(Path, "mkdir", perm_error):
            with pytest.raises(OSError) as exc_info:
                await _mkdir_shared_fs_safe(tmp_path / "x", retries=10, delay=0)

        assert call_count == 1
        assert exc_info.value.errno == errno.EACCES

    @pytest.mark.asyncio
    async def test_one_retry_exhausted_raises_oserror(self, tmp_path):
        # retries=1: try once, get ESTALE, raise immediately with no further attempts
        call_count = 0

        def always_stale(self, **kwargs):
            nonlocal call_count
            call_count += 1
            raise OSError(errno.ESTALE, "Stale file handle")

        with patch.object(Path, "mkdir", always_stale):
            with pytest.raises(OSError) as exc_info:
                await _mkdir_shared_fs_safe(tmp_path / "x", retries=1, delay=0)

        assert call_count == 1
        assert exc_info.value.errno == errno.ESTALE

    @pytest.mark.asyncio
    async def test_delay_is_awaited_between_retries(self, tmp_path):
        sleep_calls = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(delay):
            sleep_calls.append(delay)
            await original_sleep(0)

        original_mkdir = Path.mkdir
        call_count = 0

        def flaky_mkdir(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError(errno.ESTALE, "Stale file handle")
            original_mkdir(self, **kwargs)

        with patch("scheduler.filesystem_manager.asyncio.sleep", tracking_sleep), \
             patch.object(Path, "mkdir", flaky_mkdir):
            await _mkdir_shared_fs_safe(tmp_path / "x", retries=5, delay=0.5)

        assert sleep_calls == [0.5, 0.5]

    @pytest.mark.asyncio
    async def test_path_with_spaces(self, tmp_path):
        target = tmp_path / "dir with spaces" / "nested dir"
        await _mkdir_shared_fs_safe(target)
        assert target.exists()


# ---------------------------------------------------------------------------
# FSManager.register_path — glob handler filtering
# ---------------------------------------------------------------------------

class TestRegisterPathGlobFiltering:
    """
    Tests that the on_any_event handler in register_path correctly filters
    events by matching event.src_path (the variable), not the literal string
    'event.src_path', and that the queue contents are exactly correct.
    """

    @pytest.mark.asyncio
    async def test_single_matching_event_enqueued(self):
        fs, handler = _make_fs_and_handler("/foo/bar/*.yaml", DirType.DATA)
        e = _make_event("/foo/bar/model.yaml")
        handler.on_any_event(e)
        items = _drain(fs)
        assert len(items) == 1
        assert items[0][0] == DirType.DATA
        assert items[0][1] is e

    @pytest.mark.asyncio
    async def test_non_matching_extension_not_enqueued(self):
        fs, handler = _make_fs_and_handler("/foo/bar/*.yaml", DirType.DATA)
        handler.on_any_event(_make_event("/foo/bar/model.jsonl"))
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_wrong_directory_not_enqueued(self):
        fs, handler = _make_fs_and_handler("/foo/bar/*.yaml", DirType.DATA)
        handler.on_any_event(_make_event("/foo/other/model.yaml"))
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_non_glob_path_enqueues_any_event(self):
        fs, handler = _make_fs_and_handler("/foo/bar", DirType.DATA)
        e = _make_event("/completely/different/path.txt")
        handler.on_any_event(e)
        items = _drain(fs)
        assert len(items) == 1
        assert items[0][1] is e

    @pytest.mark.asyncio
    async def test_multiple_matching_events_all_enqueued_in_order(self):
        fs, handler = _make_fs_and_handler("/foo/*.yaml", DirType.DATA)
        events = [_make_event(f"/foo/model_{i}.yaml") for i in range(5)]
        for e in events:
            handler.on_any_event(e)
        items = _drain(fs)
        assert len(items) == 5
        for i, item in enumerate(items):
            assert item[0] == DirType.DATA
            assert item[1] is events[i]

    @pytest.mark.asyncio
    async def test_mix_of_matching_and_non_matching(self):
        fs, handler = _make_fs_and_handler("/foo/*.yaml", DirType.DATA)
        e_a = _make_event("/foo/a.yaml")       # match
        e_b = _make_event("/foo/b.json")       # no match
        e_c = _make_event("/foo/c.yaml")       # match
        e_d = _make_event("/bar/d.yaml")       # wrong dir
        e_e = _make_event("/foo/e.yaml")       # match
        for e in [e_a, e_b, e_c, e_d, e_e]:
            handler.on_any_event(e)
        items = _drain(fs)
        assert len(items) == 3
        assert [item[1] for item in items] == [e_a, e_c, e_e]

    @pytest.mark.asyncio
    async def test_dirtype_and_info_passed_through(self):
        info = {"extra": "data"}
        fs, handler = _make_fs_and_handler("/foo/*.yaml", DirType.MODELSPEC, info=info)
        e = _make_event("/foo/config.yaml")
        handler.on_any_event(e)
        items = _drain(fs)
        assert len(items) == 1
        assert items[0][0] == DirType.MODELSPEC
        assert items[0][1] is e
        assert items[0][2] is info

    @pytest.mark.asyncio
    async def test_double_star_glob_matches_nested_paths(self):
        fs, handler = _make_fs_and_handler("/foo/**/*.yaml", DirType.DATA)
        shallow = _make_event("/foo/model.yaml")
        deep = _make_event("/foo/a/b/c/model.yaml")
        no_match = _make_event("/foo/a/b/c/model.json")
        for e in [shallow, deep, no_match]:
            handler.on_any_event(e)
        items = _drain(fs)
        assert len(items) == 2
        assert items[0][1] is shallow
        assert items[1][1] is deep

    @pytest.mark.asyncio
    async def test_no_events_produces_empty_queue(self):
        fs, handler = _make_fs_and_handler("/foo/*.yaml", DirType.DATA)
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_all_non_matching_produces_empty_queue(self):
        fs, handler = _make_fs_and_handler("/foo/*.yaml", DirType.DATA)
        for path in ["/foo/a.json", "/bar/b.yaml", "/foo/sub/c.yaml"]:
            handler.on_any_event(_make_event(path))
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_same_event_fired_twice_enqueued_twice(self):
        # No deduplication — both occurrences should be enqueued
        fs, handler = _make_fs_and_handler("/foo/*.yaml", DirType.DATA)
        e = _make_event("/foo/model.yaml")
        handler.on_any_event(e)
        handler.on_any_event(e)
        items = _drain(fs)
        assert len(items) == 2
        assert items[0][1] is e
        assert items[1][1] is e

    @pytest.mark.asyncio
    async def test_path_that_is_prefix_of_glob_not_matched(self):
        # /foo/bar is a prefix of /foo/bar/*.yaml but is not itself a match
        fs, handler = _make_fs_and_handler("/foo/bar/*.yaml", DirType.DATA)
        handler.on_any_event(_make_event("/foo/bar"))
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_path_with_yaml_extension_in_subdir_not_matched(self):
        # *.yaml should not match files in subdirectories
        fs, handler = _make_fs_and_handler("/foo/*.yaml", DirType.DATA)
        handler.on_any_event(_make_event("/foo/sub/model.yaml"))
        assert _drain(fs) == []


# ---------------------------------------------------------------------------
# FSManager async iterator
# ---------------------------------------------------------------------------

class TestFSManagerAsyncIter:

    @pytest.mark.asyncio
    async def test_cancelled_error_raises_stop_async_iteration(self):
        with patch("scheduler.filesystem_manager.Observer"):
            fs = FSManager(model_directory=None, dataset_directory=None)

        async def cancel_after_get():
            task = asyncio.current_task()
            asyncio.get_event_loop().call_soon(task.cancel)
            return await fs.__anext__()

        with pytest.raises(StopAsyncIteration):
            await cancel_after_get()

    @pytest.mark.asyncio
    async def test_yields_items_in_order(self):
        with patch("scheduler.filesystem_manager.Observer"):
            fs = FSManager(model_directory=None, dataset_directory=None)

        items = [
            (DirType.DATA, "event_0", None),
            (DirType.MODELSPEC, "event_1", "info_1"),
            (DirType.MODELINSTANCE, "event_2", {"k": "v"}),
        ]
        for item in items:
            await fs._queue.put(item)

        for expected in items:
            result = await fs.__anext__()
            assert result == expected

    @pytest.mark.asyncio
    async def test_info_field_is_exact_object(self):
        with patch("scheduler.filesystem_manager.Observer"):
            fs = FSManager(model_directory=None, dataset_directory=None)

        info = {"key": "value"}
        await fs._queue.put((DirType.DATA, "event", info))
        result = await fs.__anext__()
        assert result[2] is info

    @pytest.mark.asyncio
    async def test_async_for_yields_all_items(self):
        with patch("scheduler.filesystem_manager.Observer"):
            fs = FSManager(model_directory=None, dataset_directory=None)

        expected = [(DirType.DATA, f"event_{i}", None) for i in range(5)]
        for item in expected:
            await fs._queue.put(item)

        collected = []
        # Cancel the task once we've collected everything so the async for ends
        async def collect():
            async for item in fs:
                collected.append(item)
                if len(collected) == len(expected):
                    raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await collect()

        assert collected == expected


# ---------------------------------------------------------------------------
# FSManager.ensures_exists_register_path
# ---------------------------------------------------------------------------

class TestEnsuresExistsRegisterPath:

    @pytest.mark.asyncio
    async def test_creates_watch_directory_if_missing(self, tmp_path):
        target = tmp_path / "new_dir"
        assert not target.exists()
        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(target), DirType.DATA, register_existing=False)
        assert target.exists()

    @pytest.mark.asyncio
    async def test_enqueues_all_existing_done_files(self, tmp_path):
        (tmp_path / "sub").mkdir()
        done_files = [
            tmp_path / "a_done.txt",
            tmp_path / "sub" / "b_done.txt",
            tmp_path / "sub" / "c_done.txt",
        ]
        for f in done_files:
            f.write_text("done")
        (tmp_path / "ignored.yaml").write_text("not a done file")

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(tmp_path), DirType.DATA, register_existing=True)

        items = _drain(fs)
        enqueued_paths = {item[1] for item in items}
        assert len(items) == 3
        for f in done_files:
            assert str(f) in enqueued_paths
        assert all(item[0] == DirType.DATA for item in items)

    @pytest.mark.asyncio
    async def test_non_done_files_not_enqueued(self, tmp_path):
        (tmp_path / "config.yaml").write_text("not done")
        (tmp_path / "results.jsonl").write_text("not done")
        (tmp_path / "job_done.txt").write_text("done")

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(tmp_path), DirType.DATA, register_existing=True)

        items = _drain(fs)
        assert len(items) == 1
        assert items[0][1].endswith("job_done.txt")

    @pytest.mark.asyncio
    async def test_register_existing_false_enqueues_nothing(self, tmp_path):
        for name in ["a_done.txt", "b_done.txt", "c_done.txt"]:
            (tmp_path / name).write_text("done")

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(tmp_path), DirType.DATA, register_existing=False)

        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_glob_path_enqueues_only_matching_done_files(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        match1 = models_dir / "v1_done.txt"
        match2 = models_dir / "v2_done.txt"
        no_match = tmp_path / "other_done.txt"
        for f in [match1, match2, no_match]:
            f.write_text("done")

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(models_dir / "*_done.txt"), DirType.MODELSPEC,
                register_existing=True)

        items = _drain(fs)
        paths = {item[1] for item in items}
        assert len(items) == 2
        assert str(match1) in paths
        assert str(match2) in paths
        assert str(no_match) not in paths
        assert all(item[0] == DirType.MODELSPEC for item in items)

    @pytest.mark.asyncio
    async def test_empty_directory_enqueues_nothing(self, tmp_path):
        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(tmp_path), DirType.DATA, register_existing=True)
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_glob_that_matches_nothing_enqueues_nothing(self, tmp_path):
        (tmp_path / "a_done.txt").write_text("done")

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(tmp_path / "nonexistent_prefix_*_done.txt"),
                DirType.DATA, register_existing=True)
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_deeply_nested_done_files_discovered(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        done = deep / "job_done.txt"
        done.write_text("done")

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(tmp_path), DirType.DATA, register_existing=True)

        items = _drain(fs)
        assert len(items) == 1
        assert items[0][1] == str(done)

    @pytest.mark.asyncio
    async def test_info_passed_through_to_enqueued_items(self, tmp_path):
        (tmp_path / "job_done.txt").write_text("done")
        info = {"model_name": "my-model"}

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
            fs._observer = mock_obs.return_value
            await fs.ensures_exists_register_path(
                str(tmp_path), DirType.MODELSPEC, register_existing=True, info=info)

        items = _drain(fs)
        assert len(items) == 1
        assert items[0][2] is info


# ---------------------------------------------------------------------------
# FSManager.register_path — scan_existing
# ---------------------------------------------------------------------------

class TestRegisterPathScanExisting:

    def _make_fs(self):
        """FSManager with mocked Observer."""
        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            mock_obs.return_value.start = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=None)
        fs._observer.schedule = MagicMock()
        fs._loop.call_soon_threadsafe = lambda fn, *args: fn(*args)
        return fs

    @pytest.mark.asyncio
    async def test_scan_existing_false_does_not_enqueue_files(self, tmp_path):
        (tmp_path / "model.yaml").write_text("owner: test")
        (tmp_path / "other.yaml").write_text("owner: test")
        fs = self._make_fs()
        fs.register_path(str(tmp_path), DirType.MODELSPEC, scan_existing=False)
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_scan_existing_default_does_not_enqueue_files(self, tmp_path):
        (tmp_path / "model.yaml").write_text("owner: test")
        fs = self._make_fs()
        fs.register_path(str(tmp_path), DirType.MODELSPEC)
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_scan_existing_enqueues_all_files_in_flat_dir(self, tmp_path):
        files = [(tmp_path / f"model_{i}.yaml") for i in range(3)]
        for f in files:
            f.write_text("owner: test")
        fs = self._make_fs()
        fs.register_path(str(tmp_path), DirType.MODELSPEC, scan_existing=True)
        items = _drain(fs)
        assert len(items) == 3
        assert all(item[0] == DirType.MODELSPEC for item in items)

    @pytest.mark.asyncio
    async def test_scan_existing_enqueues_nested_files_recursively(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.yaml").write_text("x")
        (sub / "nested.yaml").write_text("x")
        fs = self._make_fs()
        fs.register_path(str(tmp_path), DirType.DATA, scan_existing=True)
        items = _drain(fs)
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_scan_existing_emits_file_created_events(self, tmp_path):
        from watchdog.events import FileCreatedEvent
        (tmp_path / "model.yaml").write_text("x")
        fs = self._make_fs()
        fs.register_path(str(tmp_path), DirType.MODELSPEC, scan_existing=True)
        items = _drain(fs)
        assert len(items) == 1
        assert isinstance(items[0][1], FileCreatedEvent)
        assert items[0][1].src_path == str(tmp_path / "model.yaml")

    @pytest.mark.asyncio
    async def test_scan_existing_respects_glob_filter(self, tmp_path):
        (tmp_path / "config.yaml").write_text("x")
        (tmp_path / "config.json").write_text("x")
        (tmp_path / "notes.txt").write_text("x")
        fs = self._make_fs()
        fs.register_path(str(tmp_path / "*.yaml"), DirType.DATA, scan_existing=True)
        items = _drain(fs)
        assert len(items) == 1
        assert items[0][1].src_path == str(tmp_path / "config.yaml")

    @pytest.mark.asyncio
    async def test_scan_existing_double_star_glob(self, tmp_path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        (tmp_path / "top.yaml").write_text("x")
        (sub / "deep.yaml").write_text("x")
        (sub / "deep.json").write_text("x")
        fs = self._make_fs()
        fs.register_path(str(tmp_path / "**" / "*.yaml"), DirType.MODELSPEC, scan_existing=True)
        items = _drain(fs)
        paths = {item[1].src_path for item in items}
        assert str(tmp_path / "top.yaml") in paths
        assert str(sub / "deep.yaml") in paths
        assert str(sub / "deep.json") not in paths

    @pytest.mark.asyncio
    async def test_scan_existing_empty_directory_enqueues_nothing(self, tmp_path):
        fs = self._make_fs()
        fs.register_path(str(tmp_path), DirType.DATA, scan_existing=True)
        assert _drain(fs) == []

    @pytest.mark.asyncio
    async def test_scan_existing_skips_directories(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "file.yaml").write_text("x")
        fs = self._make_fs()
        fs.register_path(str(tmp_path), DirType.DATA, scan_existing=True)
        items = _drain(fs)
        # Only the file, not the directory
        assert len(items) == 1
        assert items[0][1].src_path == str(tmp_path / "file.yaml")

    @pytest.mark.asyncio
    async def test_scan_existing_info_passed_through(self, tmp_path):
        (tmp_path / "model.yaml").write_text("x")
        info = {"key": "val"}
        fs = self._make_fs()
        fs.register_path(str(tmp_path), DirType.MODELSPEC, info=info, scan_existing=True)
        items = _drain(fs)
        assert len(items) == 1
        assert items[0][2] is info

    @pytest.mark.asyncio
    async def test_fsmanager_init_scans_existing_model_dir(self, tmp_path):
        """FSManager.__init__ with model_directory scans existing files."""
        from watchdog.events import FileCreatedEvent
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "my_model.yaml").write_text("x")

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            mock_obs.return_value.start = MagicMock()
            fs = FSManager(model_directory=str(model_dir), dataset_directory=None)

        items = _drain(fs)
        assert len(items) == 1
        assert items[0][0] == DirType.MODELSPEC
        assert isinstance(items[0][1], FileCreatedEvent)
        assert items[0][1].src_path == str(model_dir / "my_model.yaml")

    @pytest.mark.asyncio
    async def test_fsmanager_init_scans_existing_dataset_dir(self, tmp_path):
        """FSManager.__init__ with dataset_directory scans existing files."""
        from watchdog.events import FileCreatedEvent
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "mmlu.yaml").write_text("x")

        with patch("scheduler.filesystem_manager.Observer") as mock_obs:
            mock_obs.return_value.schedule = MagicMock()
            mock_obs.return_value.start = MagicMock()
            fs = FSManager(model_directory=None, dataset_directory=str(data_dir))

        items = _drain(fs)
        assert len(items) == 1
        assert items[0][0] == DirType.DATA
        assert isinstance(items[0][1], FileCreatedEvent)
