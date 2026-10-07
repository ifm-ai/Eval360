import pytest
from scheduler.grader.match import Match, check_match
from scheduler.grader.base import Score
from scheduler.utils import Sentinel


ONE_GENERATION = {
    "completion_input": "What is the capital of France?",
    "generations": [
        "Paris",
        "London",
        "Berlin",
        "Madrid",
    ],
    "ground_truth": "Paris",
}


class MockEvent:
    parser_type = "noop"


async def samples_generator():
    yield ONE_GENERATION
    yield Sentinel.COMPLETED


async def empty_existing():
    if False:
        yield


@pytest.mark.asyncio
async def test_match_grader(tmp_path):
    match_grader = Match(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )
    sample_with_parsed = {**ONE_GENERATION, "parsed_generations": ONE_GENERATION["generations"]}
    result = await match_grader.grade_sample(sample_with_parsed)
    assert result["completion_input"] == "What is the capital of France?"
    assert result["generations"] == ["Paris", "London", "Berlin", "Madrid"]
    assert result["picked"] == ["Paris", None, None, None]
    assert result["correct"] == [True, False, False, False]
    assert result["accuracy"] == 0.25

    match_grader2 = Match(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )
    scores = {}
    async for item in match_grader2.run(existing=empty_existing(), average_over=[1], pass_at=[1]):
        if isinstance(item, Score):
            scores[item.name] = item.value
    # 1 problem, 1/4 generations correct → pass@1 = 0.25
    assert scores.get("accuracy (pass@1)") == 0.25


def test_check_match_accepts_tuple_expected_options_and_separator():
    """What: verifies tuple expected values, options, and separators work together.
    Executes: `check_match` with tuple ground truth, option filtering, and separator matching.
    Why: documents the newly covered helper branch without changing older match tests.
    """
    result = check_match(
        prompt="Q",
        generations=["cat.", "caterpillar", "dog"],
        expected=("cat", "dog"),
        options=["cat", "dog"],
        separator=lambda char: not char.isalpha(),
    )

    assert result["picked"] == ["cat", None, "dog"]
    assert result["correct"] == [True, False, True]
    assert result["accuracy"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_match_grade_sample_uses_raw_generation_when_parsed_is_none():
    """What: verifies raw generations are used when parser output is None.
    Executes: `Match.grade_sample` fallback from `parsed_generations` to `generations`.
    Why: protects resumable parser behavior for newly added raw-fallback coverage.
    """
    match_grader = Match(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )
    sample = {
        "completion_input": "What is the capital of France?",
        "generations": ["Paris"],
        "parsed_generations": [None],
        "ground_truth": "Paris",
    }

    result = await match_grader.grade_sample(sample)

    assert result["picked"] == ["Paris"]
    assert result["correct"] == [True]
    assert result["accuracy"] == 1.0
