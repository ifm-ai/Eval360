import pytest
from scheduler.grader.math import MathVerify, parse_answer_with_verify, compare_answers
from scheduler.grader.base import Score
from scheduler.utils import Sentinel


ONE_GENERATION = {
    "completion_input": "A baker made 48 muffins in the morning, and then made half as many muffins in the afternoon. How many muffins did the baker make in total?",
    "generations": [
        "The baker made 48/2 = <<48/2=24>>24 muffins in the afternoon.\nThe baker made 48+24 = <<48+24=72>>72 muffins in total.\n#### 72",
        "The baker made 48/2 = <<48/2=24>>24 muffins in the afternoon.\nThe baker made 48+24 = <<48+24=72>>72 muffins in total.\nThe answer is 72",
        "The baker made 48/2 = <<48/2=24>>24 muffins in the afternoon.\nThe baker made 48+24 = <<48+24=72>>72 muffins in total.\n#### 73",
        "The baker made 48/2 = <<48/2=24>>24 muffins in the afternoon.\nThe baker made 48+24 = <<48+24=72>>72 muffins in total.\n</think> \\boxed{72}",
    ],
    "ground_truth": "72",
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
    math_verify_grader = MathVerify(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )
    sample_with_parsed = {**ONE_GENERATION, "parsed_generations": ONE_GENERATION["generations"]}
    result = await math_verify_grader.grade_sample(sample_with_parsed)
    assert result["prompt"] == "A baker made 48 muffins in the morning, and then made half as many muffins in the afternoon. How many muffins did the baker make in total?"
    assert result["generations"] == [
        "The baker made 48/2 = <<48/2=24>>24 muffins in the afternoon.\nThe baker made 48+24 = <<48+24=72>>72 muffins in total.\n#### 72",
        "The baker made 48/2 = <<48/2=24>>24 muffins in the afternoon.\nThe baker made 48+24 = <<48+24=72>>72 muffins in total.\nThe answer is 72",
        "The baker made 48/2 = <<48/2=24>>24 muffins in the afternoon.\nThe baker made 48+24 = <<48+24=72>>72 muffins in total.\n#### 73",
        "The baker made 48/2 = <<48/2=24>>24 muffins in the afternoon.\nThe baker made 48+24 = <<48+24=72>>72 muffins in total.\n</think> \\boxed{72}",
    ]
    assert result["expected"] == ["72"]
    assert result["correct"] == [True, True, False, True]
    assert result["accuracy"] == 0.75

    math_verify_grader2 = MathVerify(
        samples_generator=samples_generator(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )
    scores = {}
    async for item in math_verify_grader2.run(existing=empty_existing(), average_over=[1], pass_at=[1]):
        if isinstance(item, Score):
            scores[item.name] = item.value
    # 1 problem, 3/4 generations correct → pass@1 = 0.75
    assert scores.get("accuracy (pass@1)") == 0.75


class TestParseAndCompare:
    """Tests for MATH500-style answers that require $...$ wrapping to parse."""

    def test_sqrt_answer(self):
        pred = parse_answer_with_verify(r"\sqrt{51}")
        assert pred, "parse should succeed for \\sqrt{51}"
        assert compare_answers(pred, r"\sqrt{51}")

    def test_polar_coordinates_with_left_right(self):
        # boxed parser extracts (3,\frac{\pi}{2}), ground truth uses \left(...\right)
        pred = parse_answer_with_verify(r"(3, \frac{\pi}{2})")
        assert pred
        assert compare_answers(pred, r"\left( 3, \frac{\pi}{2} \right)")

    def test_polar_coordinates_reverse(self):
        # reverse: ground truth is bare tuple, model output uses \left(...\right)
        pred = parse_answer_with_verify(r"\left( 3, \frac{\pi}{2} \right)")
        assert pred
        assert compare_answers(pred, r"(3, \frac{\pi}{2})")

    def test_half_open_interval(self):
        pred = parse_answer_with_verify(r"[2,5)")
        assert pred
        assert compare_answers(pred, r"[2,5)")

    def test_fraction(self):
        pred = parse_answer_with_verify(r"\frac{3}{56}")
        assert pred
        assert compare_answers(pred, r"\frac{3}{56}")

    def test_wrong_answer_returns_false(self):
        pred = parse_answer_with_verify(r"1200")
        assert pred
        assert not compare_answers(pred, r"2220")
