import json

from scheduler.choice_scoring_schema import validate_choice_scoring_row

from data_prep_zoo import (
    prepare_mmlu_pro,
    prepare_mmlu_pro_5shot,
    prepare_mmlu_pro_reasoning,
)


def _doc(**overrides):
    doc = {
        "category": "computer_science",
        "question": "Which option is correct?",
        "options": ["wrong one", "wrong two", "right one"],
        "answer": "C",
        "cot_content": "A: Check the options. The answer is C.",
    }
    doc.update(overrides)
    return doc


def test_mmlu_pro_zero_shot_uses_direct_answer_scoring_prompt():
    record = prepare_mmlu_pro.build_record(_doc(), 0)

    assert record["completion_input"].endswith("Answer: Let's think step by step.")
    assert record["scoring_prompt_prefix"].endswith("Answer:")
    assert "Let's think step by step" not in record["scoring_prompt_prefix"]
    assert record["scoring_completions"] == ["A", "B", "C"]
    assert record["scoring_completion_labels"] == ["A", "B", "C"]
    validate_choice_scoring_row(record, phase="test")


def test_mmlu_pro_5shot_uses_direct_answer_scoring_prompt():
    shot = _doc(
        question="Which answer should be shown in the exemplar?",
        options=["alpha", "beta", "gamma"],
        answer="B",
        cot_content="A: The exemplar should reason here. The answer is B.",
    )
    record = prepare_mmlu_pro_5shot.build_record(_doc(), 0, [shot])

    assert "Answer: Let's think step by step." in record["completion_input"]
    assert "Answer: B" in record["scoring_prompt_prefix"]
    assert record["scoring_prompt_prefix"].endswith("Answer:")
    assert "Let's think step by step" not in record["scoring_prompt_prefix"]
    assert "The exemplar should reason here" not in record["scoring_prompt_prefix"]
    validate_choice_scoring_row(record, phase="test")


def test_mmlu_pro_reasoning_rows_stay_generative_only():
    record = prepare_mmlu_pro_reasoning.build_record(_doc(), 0)

    assert record["completion_input"].endswith("<reasoning>")
    assert "scoring_mode" not in record
    assert "scoring_completions" not in record
    assert record["ground_truth"] == "C"


def test_mmlu_pro_zero_shot_writer_outputs_valid_choice_scoring_jsonl(tmp_path):
    full_path = tmp_path / "mmlu_pro.jsonl"
    small_path = tmp_path / "mmlu_pro_small.jsonl"
    docs = [
        _doc(answer="C"),
        _doc(
            question="Which option is the tenth?",
            options=[
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            ],
            answer="J",
        ),
    ]

    row_count = prepare_mmlu_pro.write_records(
        docs,
        full_path,
        small_path,
        small_limit=1,
    )
    rows = [
        json.loads(line)
        for line in full_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    small_rows = [
        json.loads(line)
        for line in small_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert row_count == 2
    assert len(rows) == 2
    assert len(small_rows) == 1
    assert [row["row"] for row in rows] == [0, 1]
    assert rows[1]["scoring_completions"] == list("ABCDEFGHIJ")
    assert rows[1]["scoring_completion_labels"] == list("ABCDEFGHIJ")
    for row in rows:
        validate_choice_scoring_row(row, phase="serialized")
        assert row["scoring_prompt_prefix"].endswith("Answer:")
        assert "Let's think step by step" not in row["scoring_prompt_prefix"]


def test_mmlu_pro_5shot_writer_outputs_valid_choice_scoring_jsonl(tmp_path):
    full_path = tmp_path / "mmlu_pro_5shot.jsonl"
    small_path = tmp_path / "mmlu_pro_5shot_small.jsonl"
    docs = [_doc(answer="C")]
    exemplars_by_category = {
        "computer_science": [
            _doc(
                question="Which exemplar answer is correct?",
                answer="B",
                cot_content="A: Work through the choices. The answer is B.",
            )
        ]
    }

    row_count = prepare_mmlu_pro_5shot.write_records(
        docs,
        exemplars_by_category,
        full_path,
        small_path,
        small_limit=1,
    )
    rows = [
        json.loads(line)
        for line in full_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert row_count == 1
    assert len(rows) == 1
    row = rows[0]
    validate_choice_scoring_row(row, phase="serialized")
    assert "Answer: B" in row["scoring_prompt_prefix"]
    assert row["scoring_prompt_prefix"].endswith("Answer:")
    assert "Let's think step by step" not in row["scoring_prompt_prefix"]
    assert "Work through the choices" not in row["scoring_prompt_prefix"]
