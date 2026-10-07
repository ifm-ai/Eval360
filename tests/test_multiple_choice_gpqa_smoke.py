import pytest

from scheduler.grader.base import Grade, Score
from scheduler.grader.multiple_choice import MultipleChoice, check_multiple_choice
from scheduler.grader.parser_registry import get_parser
from scheduler.utils import Sentinel


ONE_GENERATION = {
    "completion_input": "Q",
    "generations": [
        "Answer: A",
        "<think>hidden</think>\n( B )",
        "\\boxed{D}",
        "The answer is C",
    ],
    "ground_truth": "A",
}


class MockEvent:
    parser_type = "mc_answer"


async def samples_generator():
    yield ONE_GENERATION
    yield Sentinel.COMPLETED


async def empty_existing():
    if False:
        yield


async def existing_multiple_choice_row():
    yield {"row": 0, "correct": [True]}


async def row_samples_generator():
    yield {
        "row": 0,
        "completion_input": "Q",
        "generations": ["A"],
        "ground_truth": "A",
    }
    yield {
        "row": 1,
        "completion_input": "Q",
        "generations": ["B"],
        "ground_truth": "A",
    }
    yield Sentinel.COMPLETED


@pytest.mark.asyncio
async def test_multiple_choice_smoke_with_mc_parser():
    parser = get_parser("mc_answer")
    parsed = parser.parse_generations(ONE_GENERATION["generations"])

    grader = MultipleChoice(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )

    sample_with_parsed = {**ONE_GENERATION, "parsed_generations": parsed}
    result = await grader.grade_sample(sample_with_parsed)

    assert result["picked"] == ["A", "B", "D", "C"]
    assert result["correct"] == [True, False, False, False]
    assert result["accuracy"] == 0.25


@pytest.mark.asyncio
async def test_multiple_choice_run_emits_score_with_mc_parser():
    grader = MultipleChoice(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )

    scores = {}
    async for item in grader.run(existing=empty_existing(), average_over=[1], pass_at=[1]):
        if isinstance(item, Score):
            scores[item.name] = item.value

    assert scores.get("accuracy (pass@1)") == 0.25


def test_mc_answer_parser_handles_chat_style_arc_output_and_empty_generation():
    parser = get_parser("mc_answer")

    parsed = parser.parse_generations(
        [
            "<think>Reason through the options carefully.</think>\nAnswer: B",
            "",
        ]
    )

    assert parsed == ["B", None]


@pytest.mark.asyncio
async def test_multiple_choice_empty_generation_counts_as_incorrect():
    grader = MultipleChoice(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )

    sample = {
        "completion_input": "Q",
        "generations": ["   "],
        "parsed_generations": [None],
        "ground_truth": "A",
    }

    result = await grader.grade_sample(sample)

    assert result["picked"] == [None]
    assert result["correct"] == [False]
    assert result["accuracy"] == 0.0


@pytest.mark.asyncio
async def test_multiple_choice_grade_sample_passes_completion_sentinel_through():
    """What: verifies the completion sentinel is returned unchanged.
    Executes: `MultipleChoice.grade_sample` sentinel passthrough branch.
    Why: documents the newly covered stream-termination behavior without altering older tests.
    """
    grader = MultipleChoice(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )

    result = await grader.grade_sample(Sentinel.COMPLETED)

    assert result is Sentinel.COMPLETED


@pytest.mark.asyncio
async def test_multiple_choice_grade_sample_uses_raw_generation_when_parsed_is_none():
    """What: verifies raw generations are used when parser output is None.
    Executes: `MultipleChoice.grade_sample` fallback from parsed to raw generations.
    Why: protects parser-fallback behavior for the new coverage case.
    """
    grader = MultipleChoice(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )
    sample = {
        "completion_input": "Q",
        "generations": ["B", "ignored"],
        "parsed_generations": [None, "C"],
        "ground_truth": ["B", "C"],
    }

    result = await grader.grade_sample(sample)

    assert result["picked"] == ["B", "C"]
    assert result["correct"] == [True, True]
    assert result["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_multiple_choice_run_defaults_skip_existing_rows():
    """What: verifies default run settings skip existing rows and still score them.
    Executes: `MultipleChoice.run` defaults, existing-row resume, and score aggregation.
    Why: captures the newly added resume branch with deterministic in-memory streams.
    """
    grader = MultipleChoice(
        samples_generator=row_samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )
    grades = []
    scores = {}

    async for item in grader.run(existing=existing_multiple_choice_row(), average_over=[], pass_at=[]):
        if isinstance(item, Grade):
            grades.append(item.element)
        elif isinstance(item, Score):
            scores[item.name] = item.value

    assert [grade["row"] for grade in grades] == [1]
    assert scores["accuracy (avg over 1)"] == pytest.approx(0.5)
    assert scores["accuracy (pass@1)"] == pytest.approx(0.5)


def test_multiple_choice_helper_handles_empty_digit_and_symbol_fallbacks():
    """What: verifies choice extraction handles None, digits, and symbol fallbacks.
    Executes: `check_multiple_choice` helper branches for empty expected values and raw symbols.
    Why: documents the new edge-case coverage without changing older smoke tests.
    """
    result = check_multiple_choice(
        prompt="Q",
        generations=[None, "2", "?", "A"],
        expected=["A", "B", "", " "],
    )

    assert result["picked"] == [None, "B", "?", "A"]
    assert result["correct"] == [False, True, False, False]
