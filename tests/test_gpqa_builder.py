import csv
import json
from pathlib import Path

import yaml

from data_layer.build_gpqa_diamond import (
    build_gpqa_dataset,
    build_gpqa_record,
    write_dataset_config,
    write_jsonl,
)
from scheduler.task import Task

FIXTURES = Path(__file__).parent / "fixtures"


def _write_gpqa_csv(path: Path) -> None:
    rows = [
        {
            "Question": "Q1?",
            "Correct Answer": "Correct 1",
            "Incorrect Answer 1": "Wrong 1a",
            "Incorrect Answer 2": "Wrong 1b",
            "Incorrect Answer 3": "Wrong 1c",
        },
        {
            "Question": "Q2?",
            "Correct Answer": "Correct 2",
            "Incorrect Answer 1": "Wrong 2a",
            "Incorrect Answer 2": "Wrong 2b",
            "Incorrect Answer 3": "Wrong 2c",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_gpqa_dataset_deterministic_for_same_seed(tmp_path):
    csv_path = tmp_path / "gpqa_diamond.csv"
    _write_gpqa_csv(csv_path)

    first = build_gpqa_dataset(csv_path, seed=11)
    second = build_gpqa_dataset(csv_path, seed=11)

    assert first == second


def test_build_gpqa_record_matches_canonical_prompt_contract():
    row = {
        "Question": "What?",
        "Correct Answer": "Correct",
        "Incorrect Answer 1": "Wrong A",
        "Incorrect Answer 2": "Wrong B",
        "Incorrect Answer 3": "Wrong C",
    }

    record = build_gpqa_record(row=row, row_index=0, seed=7)

    assert record["completion_input"].startswith(
        "You are a helpful assistant.\n\n"
        "The following are multiple choice questions (with answers). "
        "Choose the correct answer from the options.\n\n"
    )
    assert record["completion_input"].endswith("Answer:")
    assert record["chat_input"][0] == {
        "role": "system",
        "content": (
            "You are a helpful assistant.\n\n"
            "The following are multiple choice questions (with answers). "
            "Choose the correct answer from the options.\n\n"
        ),
    }
    assert record["chat_input"][1]["content"] == (
        "What?\n"
        "A. Correct\n"
        "B. Wrong B\n"
        "C. Wrong A\n"
        "D. Wrong C\n"
        "Answer:"
    )
    assert record["ground_truth"] == "A"
    assert "scoring_mode" not in record
    assert "scoring_completions" not in record
    assert "scoring_completion_labels" not in record


def test_builder_output_matches_golden_fixture():
    csv_path = FIXTURES / "gpqa_diamond_canonical.csv"
    expected_path = FIXTURES / "gpqa_diamond_expected.jsonl"

    records = build_gpqa_dataset(csv_path, seed=0)
    expected = [
        json.loads(line)
        for line in expected_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert records == expected


def test_generated_config_parses_with_task_and_count_matches(tmp_path):
    csv_path = tmp_path / "gpqa_diamond.csv"
    jsonl_path = tmp_path / "gpqa_diamond.jsonl"
    yaml_path = tmp_path / "gpqa_diamond.yaml"
    _write_gpqa_csv(csv_path)

    records = build_gpqa_dataset(csv_path, seed=3)
    write_jsonl(records, jsonl_path, overwrite=True)
    write_dataset_config(
        output_path=yaml_path,
        dataset_name="gpqa_diamond",
        data_path=str(jsonl_path),
        num_generations=len(records),
        semantic_version="1.0.0",
        average_over=[1],
        pass_at=[1],
        meta={"split": "test", "priority": "high", "fewshot": 0},
        dataset_uuid="gpqa_diamond",
    )

    task = Task.parse_yaml(yaml_path)
    assert task.num_generations == len(records)

    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(records)


def test_gpqa_task_config_uses_single_canonical_dataset_path():
    config_path = Path("data_zoo/gpqa-diamond/gpqa_diamond.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["data_path"] == "hf://<HF_ORG>/<EVAL_SOURCES_REPO>/gpqa-diamond/*.jsonl@main"
    assert config["dataset_name"] == "gpqa_diamond"
    assert config["meta"] == {"split": "test", "priority": "high", "fewshot": 0}
