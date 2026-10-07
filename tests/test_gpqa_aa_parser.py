import pytest

from scheduler.grader.multiple_choice import check_multiple_choice
from scheduler.grader.parser_registry import get_parser


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ("Explanation: final.\nAnswer: C", "C"),
        (r"Reasoning. \boxed{B}", "B"),
        ("Explanation mentions Answer: B, but revises it.\nAnswer: A", "A"),
        ("D. The fourth option is correct.", "D"),
        ("Reasoning without a final marker", None),
    ],
)
def test_gpqa_aa_visible_final_parser(generation, expected):
    parser = get_parser("gpqa_aa_visible_final")
    assert parser.parse_generations([generation]) == [expected]


@pytest.mark.parametrize(
    "close_tag",
    [
        "</ifm|think>",
        "</ifm|think_fast>",
        "</ifm|think_faster>",
        "</think>",
        "</think_fast>",
        "</think_faster>",
    ],
)
def test_gpqa_aa_parser_ignores_hidden_reasoning_choices(close_tag):
    parser = get_parser("gpqa_aa_visible_final")
    generation = f"analysis Answer: A\n{close_tag}\nExplanation: final.\nAnswer: D"
    assert parser.parse_generations([generation]) == ["D"]


def test_gpqa_aa_parser_does_not_fall_back_to_hidden_answer():
    parser = get_parser("gpqa_aa_visible_final")
    assert parser.parse_generations(["analysis Answer: A\n</ifm|think>"]) == [None]


def test_gpqa_aa_parser_and_multiple_choice_grader_contract():
    parser = get_parser("gpqa_aa_visible_final")
    parsed = parser.parse_generations(
        ["analysis Answer: A\n</ifm|think>\nExplanation: final.\nAnswer: D"]
    )
    result = check_multiple_choice(prompt="Q", generations=parsed, expected="D")
    assert result["picked"] == ["D"]
    assert result["correct"] == [True]
