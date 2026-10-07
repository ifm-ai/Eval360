"""Integration tests: parsed_* metadata flows through the full grader pipeline.

Verifies that extract_reasoning_and_tools output (parsed_reasoning, parsed_answer,
parsed_tool_calls, etc.) is correctly populated in grade records when running
through grader.run(), and that parser_type selection works end-to-end.
"""
import pytest

from scheduler.grader.base import Grade, Score
from scheduler.grader.match import Match
from scheduler.utils import Sentinel


class MockEvent:
    def __init__(self, parser_type="noop"):
        self.parser_type = parser_type


async def make_samples_generator(*samples):
    for s in samples:
        yield s
    yield Sentinel.COMPLETED


async def empty_existing():
    if False:
        yield


async def collect_run(grader):
    """Run grader and return (grades, scores)."""
    grades = []
    scores = {}
    async for item in grader.run(existing=empty_existing(), average_over=[1], pass_at=[1]):
        if isinstance(item, Grade):
            grades.append(item.element)
        elif isinstance(item, Score):
            scores[item.name] = item.value
    return grades, scores


# ── 1. parsed_reasoning and parsed_answer flow through run() ──────────────────

@pytest.mark.asyncio
async def test_parsed_reasoning_in_grade_record():
    """Think-tagged generation produces parsed_reasoning/parsed_answer in the grade."""
    sample = {
        "completion_input": "Q",
        "generations": ["<think>my reasoning</think>the answer"],
        "ground_truth": "the answer",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("think_suffix"),
    )
    grades, _ = await collect_run(grader)
    assert grades, "Expected at least one grade"
    grade = grades[0]
    assert "parsed_reasoning" in grade
    assert grade["parsed_reasoning"] == ["my reasoning"]
    assert "parsed_answer" in grade
    assert grade["parsed_answer"] == ["the answer"]


@pytest.mark.asyncio
async def test_no_think_tag_produces_no_metadata():
    """Plain generation with no think tags must NOT add parsed_reasoning/answer."""
    sample = {
        "completion_input": "Q",
        "generations": ["plain answer"],
        "ground_truth": "plain answer",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("noop"),
    )
    grades, _ = await collect_run(grader)
    assert grades
    grade = grades[0]
    assert "parsed_reasoning" not in grade
    assert "parsed_answer" not in grade


# ── 2. Parser type selection end-to-end ───────────────────────────────────────

@pytest.mark.asyncio
async def test_noop_parser_does_not_strip_think_tags():
    """noop parser passes generation unchanged; grading uses the full raw text."""
    sample = {
        "completion_input": "Q",
        "generations": ["<think>hidden</think>expected"],
        "ground_truth": "<think>hidden</think>expected",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("noop"),
    )
    grades, _ = await collect_run(grader)
    assert grades[0]["correct"] == [True]


@pytest.mark.asyncio
async def test_think_suffix_parser_strips_think_block_for_grading():
    """think_suffix parser exposes only the text after </think> for grading."""
    sample = {
        "completion_input": "Q",
        "generations": ["<think>hidden reasoning</think>expected"],
        "ground_truth": "expected",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("think_suffix"),
    )
    grades, scores = await collect_run(grader)
    assert grades[0]["correct"] == [True]
    assert scores.get("accuracy (pass@1)") == 1.0


# ── 3. Edge cases in extract_reasoning_and_tools through the pipeline ─────────

@pytest.mark.asyncio
async def test_orphaned_closing_think_tag():
    """Orphaned </think> (opening tag was in prompt prefix) is extracted correctly."""
    sample = {
        "completion_input": "Q",
        "generations": ["partial reasoning</think>answer text"],
        "ground_truth": "answer text",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("think_suffix"),
    )
    grades, _ = await collect_run(grader)
    grade = grades[0]
    assert grade["parsed_reasoning"] == ["partial reasoning"]
    assert grade["parsed_answer"] == ["answer text"]


@pytest.mark.asyncio
async def test_unclosed_think_tag():
    """Unclosed <think> (truncated output) sets unclosed_think_tag and answer=None."""
    sample = {
        "completion_input": "Q",
        "generations": ["<think>truncated reasoning without close"],
        "ground_truth": "anything",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("think_suffix"),
    )
    grades, _ = await collect_run(grader)
    grade = grades[0]
    assert grade["parsed_reasoning"] == ["truncated reasoning without close"]
    assert grade["parsed_answer"] == [None]
    assert grade["parsed_unclosed_think_tag"] == [True]


@pytest.mark.asyncio
async def test_multiple_think_blocks_flag():
    """Multiple closed think blocks sets parsed_multiple_think_blocks in grade."""
    sample = {
        "completion_input": "Q",
        "generations": ["<think>first</think><think>second</think>answer"],
        "ground_truth": "answer",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("think_suffix"),
    )
    grades, _ = await collect_run(grader)
    grade = grades[0]
    assert grade["parsed_multiple_think_blocks"] == [True]
    assert grade["parsed_reasoning"] == ["first"]  # only first block captured


@pytest.mark.asyncio
async def test_tool_call_extraction():
    """Closed <tool_call> blocks are extracted into parsed_tool_calls."""
    sample = {
        "completion_input": "Q",
        "generations": ["<tool_call>call_args_here</tool_call>"],
        "ground_truth": "anything",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("noop"),
    )
    grades, _ = await collect_run(grader)
    grade = grades[0]
    assert grade["parsed_tool_calls"] == [["call_args_here"]]


@pytest.mark.asyncio
async def test_unclosed_tool_call():
    """Unclosed <tool_call> sets parsed_unclosed_tool_call flag."""
    sample = {
        "completion_input": "Q",
        "generations": ["<tool_call>partial args without close"],
        "ground_truth": "anything",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("noop"),
    )
    grades, _ = await collect_run(grader)
    grade = grades[0]
    assert grade["parsed_unclosed_tool_call"] == [True]
    assert grade["parsed_tool_calls"] == [["partial args without close"]]


@pytest.mark.asyncio
async def test_think_and_tool_call_combined():
    """Generation with both think block and tool call populates both metadata fields."""
    sample = {
        "completion_input": "Q",
        "generations": ["<think>reasoning</think>answer<tool_call>call</tool_call>"],
        "ground_truth": "anything",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("noop"),
    )
    grades, _ = await collect_run(grader)
    grade = grades[0]
    assert grade["parsed_reasoning"] == ["reasoning"]
    assert grade["parsed_tool_calls"] == [["call"]]


@pytest.mark.asyncio
async def test_multiple_generations_metadata_aligned():
    """Metadata lists are parallel to generations (one entry per generation, None for absent)."""
    sample = {
        "completion_input": "Q",
        "generations": [
            "<think>r1</think>answer",
            "plain",
            "<think>r3</think>answer",
        ],
        "ground_truth": "answer",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("think_suffix"),
    )
    grades, _ = await collect_run(grader)
    grade = grades[0]
    # 3 generations → 3-entry parallel lists
    assert len(grade["parsed_reasoning"]) == 3
    assert grade["parsed_reasoning"][0] == "r1"
    assert grade["parsed_reasoning"][1] is None  # no think block
    assert grade["parsed_reasoning"][2] == "r3"
    assert len(grade["parsed_answer"]) == 3
    assert grade["parsed_answer"][0] == "answer"
    assert grade["parsed_answer"][1] is None
    assert grade["parsed_answer"][2] == "answer"


@pytest.mark.asyncio
async def test_empty_generation_produces_no_metadata():
    """Empty string generation must not crash and adds no parsed_* keys."""
    sample = {
        "completion_input": "Q",
        "generations": [""],
        "ground_truth": "anything",
    }
    grader = Match(
        samples_generator=make_samples_generator(sample),
        event_manager=None,
        job_manager=None,
        event=MockEvent("noop"),
    )
    grades, _ = await collect_run(grader)
    grade = grades[0]
    assert "parsed_reasoning" not in grade
    assert "parsed_tool_calls" not in grade


@pytest.mark.asyncio
async def test_metadata_present_across_multiple_samples():
    """Metadata is correctly populated for every sample in a multi-sample run."""
    samples = [
        {
            "completion_input": "Q1",
            "generations": ["<think>r1</think>a1"],
            "ground_truth": "a1",
        },
        {
            "completion_input": "Q2",
            "generations": ["plain"],
            "ground_truth": "plain",
        },
        {
            "completion_input": "Q3",
            "generations": ["<think>r3</think>a3"],
            "ground_truth": "a3",
        },
    ]
    grader = Match(
        samples_generator=make_samples_generator(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent("think_suffix"),
    )
    grades, scores = await collect_run(grader)
    assert len(grades) == 3
    assert grades[0]["parsed_reasoning"] == ["r1"]
    # grade[1] has no think block; parsed_reasoning key absent for that sample
    assert "parsed_reasoning" not in grades[1]
    assert grades[2]["parsed_reasoning"] == ["r3"]
    assert scores.get("accuracy (pass@1)") == 1.0
