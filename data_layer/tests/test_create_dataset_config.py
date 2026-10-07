import importlib.util
import sys
import uuid
from pathlib import Path


def load_module() -> object:
    """Load the create_dataset_config module directly from its file."""
    module_path = Path(__file__).resolve().parent.parent / "create_dataset_config.py"
    spec = importlib.util.spec_from_file_location("create_dataset_config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def test_create_dataset_config_generates_expected_yaml(tmp_path, monkeypatch):
    module = load_module()

    dataset_path = tmp_path / "sample_dataset.jsonl"
    dataset_path.write_text(
        (
            '{"row": 0, "choices": ["A", "B"], "ground_truth": 0}\n'
            '{"row": 1, "choices": ["C", "D"], "ground_truth": 1}\n'
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "generated_config.yaml"
    expected_yaml_path = tmp_path / "expected_config.yaml"

    expected_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(module.uuid, "uuid4", lambda: expected_uuid)

    expected_yaml_path.write_text(
        (
            'uuid: "00000000-0000-0000-0000-000000000000"\n'
            "grader:\n"
            '  type: "multiple_choice"\n'
            "average_over:\n"
            "  - 1\n"
            "  - 2\n"
            "pass_at:\n"
            "  - 1\n"
            "  - 4\n"
            'dataset_name: "sample_task"\n'
            f'data_path: "{dataset_path.resolve()}"\n'
            'semantic_version: "0.0.1"\n'
            "num_generations: 2\n"
            "meta:\n"
            '  split: "test"\n'
            '  priority: "high"\n'
            "  fewshot: 0\n"
        ),
        encoding="ascii",
    )

    argv = [
        "create_dataset_config.py",
        "--output",
        str(output_path),
        "--pass-at",
        "1",
        "4",
        "--avg-at",
        "1",
        "2",
        "--dataset-name",
        "sample_task",
        "--grader-type",
        "multiple_choice",
        "--data-path",
        str(dataset_path),
        "--semantic-version",
        "0.0.1",
        "--meta",
        '{"split": "test", "priority": "high", "fewshot": 0}',
    ]

    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    assert output_path.read_text(encoding="ascii") == expected_yaml_path.read_text(
        encoding="ascii"
    )


def test_create_dataset_config_accepts_choice_scoring_grader(tmp_path, monkeypatch):
    module = load_module()

    dataset_path = tmp_path / "choice_scoring.jsonl"
    dataset_path.write_text(
        (
            '{"row": 0, "scoring_mode": "choice_scoring", '
            '"scoring_completions": ["A", "B"], "ground_truth": "A"}\n'
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "generated_choice_scoring.yaml"

    monkeypatch.setattr(
        module.uuid,
        "uuid4",
        lambda: uuid.UUID("11111111-1111-1111-1111-111111111111"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_dataset_config.py",
            "--output",
            str(output_path),
            "--pass-at",
            "1",
            "--avg-at",
            "1",
            "--dataset-name",
            "choice_scoring_task",
            "--grader-type",
            "choice_scoring",
            "--data-path",
            str(dataset_path),
            "--semantic-version",
            "0.0.1",
        ],
    )

    module.main()

    assert output_path.read_text(encoding="ascii") == (
        'uuid: "11111111-1111-1111-1111-111111111111"\n'
        "grader:\n"
        '  type: "choice_scoring"\n'
        "average_over:\n"
        "  - 1\n"
        "pass_at:\n"
        "  - 1\n"
        'dataset_name: "choice_scoring_task"\n'
        f'data_path: "{dataset_path.resolve()}"\n'
        'semantic_version: "0.0.1"\n'
        "num_generations: 1\n"
    )
