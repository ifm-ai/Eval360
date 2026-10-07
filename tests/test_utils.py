import json
import os
import pytest
from scheduler.utils import (
    ExceptionWrapper,
    Sentinel,
    count_jsonl_records,
    dataset_iterator,
    expand_data_path,
    get_unreadable_paths,
    load_completed_row_ids,
    normalize_eval_input_record,
)
from scheduler.external_requests import ExternalRequestFailure

_skip_if_root = pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permissions")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLogger:
    def info(self, *a): pass
    def debug(self, *a): pass
    def warning(self, *a): pass


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_raw(path, text):
    path.write_text(text)


RECORDS = [{"id": i} for i in range(5)]


async def collect(gen):
    return [item async for item in gen]


def test_exception_wrapper_preserves_grouped_external_failure_evidence():
    failure = ExternalRequestFailure(
        RuntimeError("external endpoint unavailable"),
        error_code="endpoint_unreachable",
        attempts=2,
        elapsed_seconds=0.1,
        retriable=True,
        http_status=None,
    )

    factory = getattr(ExceptionWrapper, "from_exception", None)
    assert callable(factory)
    wrapper = factory(
        ExceptionGroup("request task failed", [failure]),
        {"row": 3},
    )

    assert isinstance(wrapper.exception, ExceptionGroup)
    assert "request task failed" in wrapper.trace
    assert "external endpoint unavailable" in wrapper.trace
    assert getattr(wrapper, "error_code", None) == "endpoint_unreachable"
    assert getattr(wrapper, "attempts", None) == 2
    assert getattr(wrapper, "elapsed_seconds", None) == 0.1
    assert getattr(wrapper, "retriable", None) is True
    assert getattr(wrapper, "http_status", None) is None


# ---------------------------------------------------------------------------
# expand_data_path
# ---------------------------------------------------------------------------

class TestExpandDataPath:
    def test_returns_sorted_matches(self, tmp_path):
        (tmp_path / "b.jsonl").touch()
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "c.jsonl").touch()
        result = expand_data_path(str(tmp_path / "*.jsonl"))
        assert result == sorted(result)
        assert len(result) == 3

    def test_returns_empty_list_when_no_match(self, tmp_path):
        result = expand_data_path(str(tmp_path / "*.jsonl"))
        assert result == []

    def test_exact_path_returns_single_file(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.touch()
        assert expand_data_path(str(f)) == [str(f)]


# ---------------------------------------------------------------------------
# get_unreadable_paths
# ---------------------------------------------------------------------------

class TestGetUnreadablePaths:
    def test_returns_empty_when_all_readable(self, tmp_path):
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "b.jsonl").touch()
        assert get_unreadable_paths(str(tmp_path / "*.jsonl")) == []

    def test_returns_empty_when_no_files_match(self, tmp_path):
        assert get_unreadable_paths(str(tmp_path / "*.jsonl")) == []

    @_skip_if_root
    def test_returns_unreadable_paths(self, tmp_path):
        readable = tmp_path / "a.jsonl"
        unreadable = tmp_path / "b.jsonl"
        readable.touch()
        unreadable.touch()
        os.chmod(unreadable, 0o000)
        try:
            result = get_unreadable_paths(str(tmp_path / "*.jsonl"))
            assert result == [str(unreadable)]
        finally:
            os.chmod(unreadable, 0o644)


# ---------------------------------------------------------------------------
# count_jsonl_records
# ---------------------------------------------------------------------------

class TestCountJsonlRecords:
    def test_counts_records_in_single_file(self, tmp_path):
        write_jsonl(tmp_path / "data.jsonl", RECORDS)
        assert count_jsonl_records(str(tmp_path / "*.jsonl")) == 5

    def test_counts_across_multiple_files(self, tmp_path):
        write_jsonl(tmp_path / "a.jsonl", RECORDS[:3])
        write_jsonl(tmp_path / "b.jsonl", RECORDS[3:])
        assert count_jsonl_records(str(tmp_path / "*.jsonl")) == 5

    def test_skips_blank_lines(self, tmp_path):
        write_raw(tmp_path / "data.jsonl",
                  json.dumps({"id": 0}) + "\n"
                  "\n"
                  + json.dumps({"id": 1}) + "\n")
        assert count_jsonl_records(str(tmp_path / "*.jsonl")) == 2

    def test_skips_multiple_consecutive_blank_lines(self, tmp_path):
        write_raw(tmp_path / "data.jsonl",
                  json.dumps({"id": 0}) + "\n\n\n"
                  + json.dumps({"id": 1}) + "\n")
        assert count_jsonl_records(str(tmp_path / "*.jsonl")) == 2

    def test_empty_file_counts_zero(self, tmp_path):
        (tmp_path / "data.jsonl").write_text("")
        assert count_jsonl_records(str(tmp_path / "*.jsonl")) == 0

    def test_raises_when_no_files_match(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No JSONL files found"):
            count_jsonl_records(str(tmp_path / "*.jsonl"))

    def test_no_trailing_newline(self, tmp_path):
        # A file with no trailing newline should still count the last record
        (tmp_path / "data.jsonl").write_bytes(
            json.dumps({"id": 0}).encode() + b"\n" +
            json.dumps({"id": 1}).encode()  # no trailing newline
        )
        assert count_jsonl_records(str(tmp_path / "*.jsonl")) == 2

    @_skip_if_root
    def test_raises_when_all_files_unreadable(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.touch()
        os.chmod(path, 0o000)
        try:
            with pytest.raises(PermissionError, match="No readable JSONL files"):
                count_jsonl_records(str(tmp_path / "*.jsonl"))
        finally:
            os.chmod(path, 0o644)

    @_skip_if_root
    def test_skips_unreadable_counts_only_readable(self, tmp_path):
        readable = tmp_path / "a.jsonl"
        unreadable = tmp_path / "b.jsonl"
        write_jsonl(readable, RECORDS[:3])
        write_jsonl(unreadable, RECORDS[3:])
        os.chmod(unreadable, 0o000)
        try:
            assert count_jsonl_records(str(tmp_path / "*.jsonl")) == 3
        finally:
            os.chmod(unreadable, 0o644)


# ---------------------------------------------------------------------------
# dataset_iterator (read_only=True)
# ---------------------------------------------------------------------------

class TestDatasetIteratorReadOnly:
    @pytest.mark.asyncio
    async def test_yields_all_records(self, tmp_path):
        write_jsonl(tmp_path / "data.jsonl", RECORDS)
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True))
        assert result == RECORDS

    @pytest.mark.asyncio
    async def test_skips_blank_lines(self, tmp_path):
        write_raw(tmp_path / "data.jsonl",
                  json.dumps({"id": 0}) + "\n"
                  "\n"
                  + json.dumps({"id": 1}) + "\n"
                  "\n"
                  + json.dumps({"id": 2}) + "\n")
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True))
        assert result == [{"id": 0}, {"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_yields_nothing_when_no_files(self, tmp_path):
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True))
        assert result == []

    @pytest.mark.asyncio
    async def test_stops_at_corrupt_line(self, tmp_path):
        write_raw(tmp_path / "data.jsonl",
                  json.dumps({"id": 0}) + "\n"
                  "not json\n"
                  + json.dumps({"id": 1}) + "\n")
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True))
        assert result == [{"id": 0}]

    @pytest.mark.asyncio
    async def test_resume_index_skips_leading_records(self, tmp_path):
        write_jsonl(tmp_path / "data.jsonl", RECORDS)
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True, resume_index=3))
        assert result == RECORDS[3:]

    @pytest.mark.asyncio
    async def test_sentinel_appended_when_requested(self, tmp_path):
        write_jsonl(tmp_path / "data.jsonl", [{"id": 0}])
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True, sentinel=True))
        assert result[-1] is Sentinel.COMPLETED
        assert result[:-1] == [{"id": 0}]

    @pytest.mark.asyncio
    async def test_no_sentinel_by_default(self, tmp_path):
        write_jsonl(tmp_path / "data.jsonl", [{"id": 0}])
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True))
        assert Sentinel.COMPLETED not in result

    @pytest.mark.asyncio
    async def test_reads_multiple_files_in_sorted_order(self, tmp_path):
        write_jsonl(tmp_path / "b.jsonl", [{"id": 1}])
        write_jsonl(tmp_path / "a.jsonl", [{"id": 0}])
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True))
        assert result == [{"id": 0}, {"id": 1}]

    @pytest.mark.asyncio
    async def test_empty_file_yields_nothing(self, tmp_path):
        (tmp_path / "data.jsonl").write_text("")
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True))
        assert result == []

    @_skip_if_root
    @pytest.mark.asyncio
    async def test_skips_unreadable_file(self, tmp_path):
        readable = tmp_path / "a.jsonl"
        unreadable = tmp_path / "b.jsonl"
        write_jsonl(readable, RECORDS[:3])
        write_jsonl(unreadable, RECORDS[3:])
        os.chmod(unreadable, 0o000)
        try:
            result = await collect(dataset_iterator(
                str(tmp_path / "*.jsonl"), MockLogger(), read_only=True))
            assert result == RECORDS[:3]
        finally:
            os.chmod(unreadable, 0o644)

    @pytest.mark.asyncio
    async def test_record_transform_can_normalize_legacy_prompt_label_rows(self, tmp_path):
        legacy_rows = [
            {
                "prompt": [{"role": "user", "content": "Problem 0"}],
                "label": "42",
            },
            {
                "prompt": [{"role": "user", "content": "Problem 1"}],
                "label": "43",
            },
        ]
        write_jsonl(tmp_path / "data.jsonl", legacy_rows)

        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"),
            MockLogger(),
            read_only=True,
            record_transform=normalize_eval_input_record,
        ))

        assert result == [
            {
                "row": 0,
                "prompt": [{"role": "user", "content": "Problem 0"}],
                "label": "42",
                "completion_input": "Problem 0",
                "chat_input": [{"role": "user", "content": "Problem 0"}],
                "ground_truth": "42",
            },
            {
                "row": 1,
                "prompt": [{"role": "user", "content": "Problem 1"}],
                "label": "43",
                "completion_input": "Problem 1",
                "chat_input": [{"role": "user", "content": "Problem 1"}],
                "ground_truth": "43",
            },
        ]

    @pytest.mark.asyncio
    async def test_record_transform_preserves_new_eval360_rows(self, tmp_path):
        rows = [
            {
                "row": 5,
                "completion_input": "Q?",
                "chat_input": [{"role": "user", "content": "Q?"}],
                "ground_truth": "A",
            }
        ]
        write_jsonl(tmp_path / "data.jsonl", rows)

        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"),
            MockLogger(),
            read_only=True,
            record_transform=normalize_eval_input_record,
        ))

        assert result == rows

    @pytest.mark.asyncio
    async def test_record_transform_supports_full_legacy_and_new_format_datasets(self, tmp_path):
        legacy_rows = [
            {
                "prompt": [{"role": "user", "content": f"Problem {i}"}],
                "label": str(i),
            }
            for i in range(30)
        ]
        new_rows = [
            {
                "row": i,
                "completion_input": f"Problem {i}",
                "chat_input": [{"role": "user", "content": f"Problem {i}"}],
                "ground_truth": str(i),
            }
            for i in range(30)
        ]

        for name, rows in {
            "legacy": legacy_rows,
            "eval360": new_rows,
        }.items():
            data_path = tmp_path / f"{name}.jsonl"
            write_jsonl(data_path, rows)

            result = await collect(dataset_iterator(
                str(data_path),
                MockLogger(),
                read_only=True,
                sentinel=True,
                record_transform=normalize_eval_input_record,
            ))

            assert result[-1] is Sentinel.COMPLETED
            records = result[:-1]
            assert len(records) == len(rows)
            assert [record["row"] for record in records] == list(range(len(rows)))
            assert all("completion_input" in record for record in records)
            assert all("chat_input" in record for record in records)
            assert all("ground_truth" in record for record in records)
            assert [record["ground_truth"] for record in records] == [
                str(i) for i in range(len(rows))
            ]


# ---------------------------------------------------------------------------
# dataset_iterator (read_only=False — truncation behaviour)
# ---------------------------------------------------------------------------

class TestDatasetIteratorReadWrite:
    @pytest.mark.asyncio
    async def test_yields_all_records(self, tmp_path):
        write_jsonl(tmp_path / "data.jsonl", RECORDS)
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=False))
        assert result == RECORDS

    @pytest.mark.asyncio
    async def test_truncates_file_at_corrupt_line(self, tmp_path):
        path = tmp_path / "data.jsonl"
        write_raw(path,
                  json.dumps({"id": 0}) + "\n"
                  "not json\n"
                  + json.dumps({"id": 1}) + "\n")
        await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=False))
        remaining = path.read_text()
        assert remaining == json.dumps({"id": 0}) + "\n"

    @pytest.mark.asyncio
    async def test_yields_error_record_without_truncating(self, tmp_path):
        """Error records (with 'exception' key) are yielded and count as incorrect; file is not truncated."""
        path = tmp_path / "data.jsonl"
        error_record = {"id": 1, "exception": "some error", "generations": []}
        original_content = (json.dumps({"id": 0}) + "\n"
                            + json.dumps(error_record) + "\n"
                            + json.dumps({"id": 2}) + "\n")
        write_raw(path, original_content)
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=False))
        assert result == [{"id": 0}, error_record, {"id": 2}]
        # File must NOT have been truncated
        assert path.read_text() == original_content

    @pytest.mark.asyncio
    async def test_skips_blank_lines_without_truncating(self, tmp_path):
        path = tmp_path / "data.jsonl"
        content = (json.dumps({"id": 0}) + "\n"
                   "\n"
                   + json.dumps({"id": 1}) + "\n")
        write_raw(path, content)
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=False))
        assert result == [{"id": 0}, {"id": 1}]
        # File must not have been truncated
        assert path.read_text() == content


# ---------------------------------------------------------------------------
# load_completed_row_ids
# ---------------------------------------------------------------------------

class TestLoadCompletedRowIds:
    def test_basic(self, tmp_path):
        path = tmp_path / "gen.jsonl"
        path.write_text(
            json.dumps({"row": 3, "generations": ["x"]}) + "\n"
            + json.dumps({"row": 1, "generations": ["y"]}) + "\n"
            + json.dumps({"row": 0, "generations": ["z"]}) + "\n"
        )
        assert load_completed_row_ids(str(path)) == {0, 1, 3}

    def test_empty_file(self, tmp_path):
        path = tmp_path / "gen.jsonl"
        path.write_text("")
        assert load_completed_row_ids(str(path)) == set()

    def test_file_not_found(self, tmp_path):
        assert load_completed_row_ids(str(tmp_path / "nonexistent.jsonl")) == set()

    def test_corrupt_last_line_ignored(self, tmp_path):
        path = tmp_path / "gen.jsonl"
        path.write_text(
            json.dumps({"row": 0, "generations": ["a"]}) + "\n"
            + json.dumps({"row": 1, "generations": ["b"]}) + "\n"
            + '{"row": 2, "gen'  # corrupt partial line
        )
        # row 2 still matched by regex even though JSON is incomplete
        # This is correct: the regex found "row": 2, so it counts as written
        result = load_completed_row_ids(str(path))
        assert 0 in result and 1 in result

    def test_exception_records_included(self, tmp_path):
        path = tmp_path / "gen.jsonl"
        path.write_text(
            json.dumps({"row": 0, "generations": ["a"]}) + "\n"
            + json.dumps({"row": 1, "exception": "timeout", "trace": "..."}) + "\n"
        )
        assert load_completed_row_ids(str(path)) == {0, 1}

    def test_large_row_numbers(self, tmp_path):
        path = tmp_path / "gen.jsonl"
        path.write_text(
            json.dumps({"row": 999999, "generations": ["x"]}) + "\n"
        )
        assert load_completed_row_ids(str(path)) == {999999}

    def test_duplicates_deduplicated(self, tmp_path):
        path = tmp_path / "gen.jsonl"
        path.write_text(
            json.dumps({"row": 0, "generations": ["a"]}) + "\n"
            + json.dumps({"row": 0, "generations": ["b"]}) + "\n"
            + json.dumps({"row": 1, "generations": ["c"]}) + "\n"
        )
        assert load_completed_row_ids(str(path)) == {0, 1}


# ---------------------------------------------------------------------------
# dataset_iterator with skip_rows
# ---------------------------------------------------------------------------

class TestDatasetIteratorSkipRows:
    @pytest.mark.asyncio
    async def test_skip_rows_filters_by_row_field(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(
            json.dumps({"row": i, "text": f"row{i}"}) for i in range(5)
        ) + "\n")
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True, skip_rows={1, 3}))
        assert [r["row"] for r in result] == [0, 2, 4]

    @pytest.mark.asyncio
    async def test_skip_rows_none_yields_all(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(
            json.dumps({"row": i, "text": f"row{i}"}) for i in range(3)
        ) + "\n")
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True, skip_rows=None))
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_skip_rows_with_sentinel(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(
            json.dumps({"row": i, "text": f"row{i}"}) for i in range(3)
        ) + "\n")
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True,
            skip_rows={0, 2}, sentinel=True))
        assert result[0]["row"] == 1
        assert result[-1] == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_skip_rows_empty_set_yields_all(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(
            json.dumps({"row": i, "text": f"row{i}"}) for i in range(3)
        ) + "\n")
        result = await collect(dataset_iterator(
            str(tmp_path / "*.jsonl"), MockLogger(), read_only=True, skip_rows=set()))
        assert len(result) == 3
