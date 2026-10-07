import importlib.util
import json
import sys
from unittest.mock import patch
from pathlib import Path

import yaml

from scheduler.task import Task


def load_module():
    module_path = Path(__file__).resolve().parent.parent / "data_layer" / "build_arc_challenge.py"
    spec = importlib.util.spec_from_file_location("build_arc_challenge", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def make_row(
    question: str,
    *,
    labels: list[str] | None = None,
    texts: list[str] | None = None,
    answer_key: str = "A",
) -> dict:
    return {
        "question": question,
        "choices": {
            "label": labels or ["A", "B", "C", "D"],
            "text": texts or ["Choice A", "Choice B", "Choice C", "Choice D"],
        },
        "answerKey": answer_key,
    }


def test_parse_arc_row_normalizes_numeric_labels():
    module = load_module()

    question = module.parse_arc_row(
        make_row(
            "What color is the sky?",
            labels=["1", "2", "3", "4"],
            texts=["Red", "Green", "Blue", "Yellow"],
            answer_key="3",
        )
    )

    assert question.choices == ("Red", "Green", "Blue", "Yellow")
    assert question.answer_letter == "C"


def test_parse_arc_row_supports_five_choices_and_e_answer():
    module = load_module()

    question = module.parse_arc_row(
        make_row(
            "Which option is fifth?",
            labels=["1", "2", "3", "4", "5"],
            texts=["One", "Two", "Three", "Four", "Five"],
            answer_key="5",
        )
    )

    assert question.choices == ("One", "Two", "Three", "Four", "Five")
    assert question.answer_letter == "E"


def test_build_arc_record_zero_shot_shapes_prompt_and_chat():
    module = load_module()

    record = module.build_arc_record(
        make_row(
            "Which animal barks?",
            texts=["Cat", "Dog", "Fish", "Bird"],
            answer_key="B",
        ),
        row_index=7,
    )

    assert record["row"] == 7
    assert record["ground_truth"] == "B"
    assert record["scoring_mode"] == "choice_scoring"
    assert record["scoring_completions"] == ["A", "B", "C", "D"]
    assert record["scoring_completion_labels"] == ["A", "B", "C", "D"]
    assert record["completion_input"].startswith(module.SYSTEM_PROMPT)
    assert "Question:\nWhich animal barks?" in record["completion_input"]
    assert "A. Cat" in record["completion_input"]
    assert "B. Dog" in record["completion_input"]
    assert record["completion_input"].endswith("Answer:")
    assert record["chat_input"] == [
        {"role": "system", "content": module.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Question:\nWhich animal barks?\n"
                "Choices:\n"
                "A. Cat\n"
                "B. Dog\n"
                "C. Fish\n"
                "D. Bird\n"
                "Answer:"
            ),
        },
    ]


def test_build_arc_record_with_five_choices_emits_e_option():
    module = load_module()

    record = module.build_arc_record(
        make_row(
            "Which option is last?",
            labels=["A", "B", "C", "D", "E"],
            texts=["One", "Two", "Three", "Four", "Five"],
            answer_key="E",
        ),
        row_index=3,
    )

    assert record["ground_truth"] == "E"
    assert record["scoring_completions"] == ["A", "B", "C", "D", "E"]
    assert record["scoring_completion_labels"] == ["A", "B", "C", "D", "E"]
    assert "E. Five" in record["completion_input"]
    assert record["chat_input"][1]["content"].endswith(
        "A. One\nB. Two\nC. Three\nD. Four\nE. Five\nAnswer:"
    )


def test_select_fewshot_examples_is_deterministic():
    module = load_module()

    train_rows = [
        make_row(f"Question {index}?", answer_key="A")
        for index in range(30)
    ]

    first = module.select_fewshot_examples(train_rows, fewshot_count=25, fewshot_seed=0)
    second = module.select_fewshot_examples(train_rows, fewshot_count=25, fewshot_seed=0)

    assert [item.question for item in first] == [item.question for item in second]
    assert len(first) == 25


def test_build_arc_dataset_with_fewshot_includes_exemplar_answers():
    module = load_module()

    fewshot_examples = [
        module.parse_arc_row(make_row("Shot 1?", answer_key="A")),
        module.parse_arc_row(make_row("Shot 2?", answer_key="D")),
    ]

    records = module.build_arc_dataset(
        [make_row("Final question?", answer_key="C")],
        fewshot_examples=fewshot_examples,
    )

    assert len(records) == 1
    prompt = records[0]["chat_input"][1]["content"]
    assert "Question:\nShot 1?" in prompt
    assert "Answer: A" in prompt
    assert "Question:\nShot 2?" in prompt
    assert "Answer: D" in prompt
    assert prompt.endswith("Question:\nFinal question?\nChoices:\nA. Choice A\nB. Choice B\nC. Choice C\nD. Choice D\nAnswer:")


def test_parse_arc_row_rejects_duplicate_labels():
    module = load_module()

    row = make_row(
        "Bad labels?",
        labels=["A", "1", "C", "D"],
        texts=["A1", "A2", "A3", "A4"],
        answer_key="A",
    )

    try:
        module.parse_arc_row(row)
    except module.ArcBuildError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate labels to raise ArcBuildError")


def test_parse_arc_row_rejects_more_than_five_choices():
    module = load_module()

    row = make_row(
        "Too many options?",
        labels=["A", "B", "C", "D", "E", "F"],
        texts=["1", "2", "3", "4", "5", "6"],
        answer_key="A",
    )

    try:
        module.parse_arc_row(row)
    except module.ArcBuildError as exc:
        assert "unsupported number of choices" in str(exc).lower()
    else:
        raise AssertionError("Expected >4 choices to raise ArcBuildError")


def test_generated_config_parses_with_task_and_count_matches(tmp_path):
    module = load_module()

    jsonl_path = tmp_path / "arc_challenge.jsonl"
    yaml_path = tmp_path / "arc_challenge.yaml"
    records = module.build_arc_dataset(
        [
            make_row("Q1?", answer_key="A"),
            make_row("Q2?", answer_key="D"),
        ]
    )

    module.write_jsonl(records, jsonl_path, overwrite=True)
    module.write_dataset_config(
        output_path=yaml_path,
        dataset_name="arc_challenge",
        data_path=str(jsonl_path),
        num_generations=len(records),
        semantic_version="1.0.0",
        average_over=[1],
        pass_at=[1],
        meta={"split": "test", "priority": "high", "fewshot": 0, "subset": "ARC-Challenge"},
        dataset_uuid="arc_challenge",
    )

    task = Task.parse_yaml(yaml_path)
    assert task.num_generations == len(records)

    lines = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == len(records)


def test_write_jsonl_expands_home_relative_output_path(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setenv("HOME", str(tmp_path))
    output_path = Path("~/data/eval360/arc-challenge/arc_challenge.jsonl")

    module.write_jsonl(
        [module.build_arc_record(make_row("Q1?", answer_key="A"), row_index=0)],
        output_path,
        overwrite=True,
    )

    assert (tmp_path / "data/eval360/arc-challenge/arc_challenge.jsonl").exists()


def test_load_arc_split_missing_datasets_dependency_has_actionable_error():
    module = load_module()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        try:
            module._load_arc_split(split=module.TEST_SPLIT, revision=None)
        except ModuleNotFoundError as exc:
            assert "pip install -r data_layer/requirements.txt" in str(exc)
        else:
            raise AssertionError("Expected missing datasets import to raise ModuleNotFoundError")


def test_arc_task_configs_use_canonical_hf_paths():
    zero_shot = yaml.safe_load(Path("data_zoo/arc_challenge.yaml").read_text(encoding="utf-8"))
    fewshot = yaml.safe_load(
        Path("data_zoo/arc_challenge_25shot.yaml").read_text(encoding="utf-8")
    )

    assert zero_shot["data_path"] == (
        "hf://<HF_ORG>/<EVAL_SOURCES_REPO>/arc-challenge/arc_challenge.jsonl@main"
    )
    assert zero_shot["meta"] == {
        "split": "test",
        "priority": "high",
        "fewshot": 0,
        "subset": "ARC-Challenge",
    }
    assert zero_shot["num_generations"] == 1172

    assert fewshot["data_path"] == (
        "hf://<HF_ORG>/<EVAL_SOURCES_REPO>/arc-challenge/arc_challenge_25shot.jsonl@main"
    )
    assert fewshot["meta"] == {
        "split": "test",
        "priority": "high",
        "fewshot": 25,
        "subset": "ARC-Challenge",
    }
    assert fewshot["num_generations"] == 1172


def test_main_zero_shot_does_not_load_train_split(monkeypatch, tmp_path):
    module = load_module()
    calls = []

    def fake_load_arc_split(*, split, revision):
        calls.append(split)
        assert revision is None
        return [
            make_row(
                "Q1?",
                texts=["A1", "B1", "C1", "D1"],
                answer_key="A",
            )
        ]

    monkeypatch.setattr(module, "_load_arc_split", fake_load_arc_split)
    monkeypatch.setattr(module, "write_jsonl", lambda records, output_jsonl, overwrite: None)
    monkeypatch.setattr(module, "write_dataset_config", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "output_jsonl": tmp_path / "arc_challenge.jsonl",
                "overwrite": True,
                "config_output": None,
                "config_data_path": None,
                "dataset_name": None,
                "dataset_uuid": None,
                "semantic_version": "1.0.0",
                "average_over": [1],
                "pass_at": [1],
                "meta_json": None,
                "revision": None,
                "fewshot_count": 0,
                "fewshot_seed": 0,
            },
        )(),
    )

    module.main()

    assert calls == ["test"]
