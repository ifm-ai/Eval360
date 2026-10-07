"""Maintained YAML closure for the K2 Horizon 7B base full suite."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from scheduler.eval_config import (
    EvalConfigParser,
    LoadedEvalConfig,
    LoadedModelSpec,
    LoadedTask,
)
from scheduler.model import ModelParser, ModelType
from scheduler.task import Task


ROOT = Path(__file__).parents[1]
MODEL_TEMPLATE = ROOT / "model_zoo/k2_horizon_7b_base_full.yaml"
EVAL_TEMPLATE = ROOT / "group_zoo/k2_horizon_7b_base_full.yaml"
DATA_TEMPLATES = (
    ROOT / "data_zoo/k2_horizon_7b_base_full/aime_2026.yaml",
    ROOT / "data_zoo/k2_horizon_7b_base_full/bbh_zeroshot.yaml",
    ROOT / "data_zoo/k2_horizon_7b_base_full/gpqa_diamond.yaml",
    ROOT / "data_zoo/k2_horizon_7b_base_full/gsm8k_cot_zeroshot.yaml",
    ROOT / "data_zoo/k2_horizon_7b_base_full/ifeval.yaml",
    ROOT / "data_zoo/k2_horizon_7b_base_full/mbpp_zeroshot.yaml",
    ROOT / "data_zoo/k2_horizon_7b_base_full/mmlu_pro.yaml",
    ROOT / "data_zoo/k2_horizon_7b_base_full/mmlu_zeroshot.yaml",
)
CANONICAL_DATA_CONFIGS = (
    ROOT / "data_zoo/aime-2026.yaml",
    ROOT / "data_zoo/bbh_zeroshot.yaml",
    ROOT / "data_zoo/gpqa-diamond/gpqa_diamond.yaml",
    ROOT / "data_zoo/gsm8k_zeroshot.yaml",
    ROOT / "data_zoo/ifeval.yaml",
    ROOT / "data_zoo/mbpp_zeroshot.yaml",
    ROOT / "data_zoo/mmlu_pro.yaml",
    ROOT / "data_zoo/mmlu_zeroshot.yaml",
)
DATASET_REVISION = "2f926def94ae9516d311380678121cca173afd46"
EXPECTED_TASKS = (
    (
        "aime-2026",
        "math-verify",
        [32],
        [1, 2, 4, 8, 16, 32],
        30,
        "boxed",
        "data/aime-2026/aime-2026.jsonl",
        26998,
        "99ac5daa50f01c2e5dc048a64331269b44a8f7016292863f21764387b819785e",
    ),
    (
        "bbh_zeroshot",
        "bbh",
        [1],
        [1],
        6511,
        "passthrough",
        "data/bbh_zeroshot/bbh_zeroshot.jsonl",
        7673438,
        "2c3b87401a7b45ca21eff1da90d53e1a1ffa8482b64e94d10c89dacdffa29e72",
    ),
    (
        "gpqa_diamond",
        "multiple_choice",
        [16],
        [1],
        198,
        "mc_answer",
        "data/gpqa-diamond/gpqa_diamond.jsonl",
        314212,
        "66f387c172aabe10089de0d97dbe1d120fff4b55ab371746a633ca37ef088f59",
    ),
    (
        "gsm8k_zeroshot",
        "math-verify",
        [1],
        [1],
        1319,
        "gsm8k_base",
        "data/gsm8k_cot_zeroshot/gsm8k_cot_zeroshot.jsonl",
        863235,
        "642925848b6e3d3c3e94db0b9a35fcca1ac1946dcb845fee3475a4d32260111e",
    ),
    (
        "ifeval",
        "ifeval",
        [1],
        [1],
        541,
        "passthrough",
        "data/ifeval/ifeval.jsonl",
        414380,
        "693eeecb76e639d5ba91be690178abed0ec1dab9d532353c76ad213f0c40597d",
    ),
    (
        "mbpp_zeroshot",
        "mbpp-local",
        [1],
        [1],
        500,
        "passthrough",
        "data/mbpp_zeroshot/mbpp_zeroshot.jsonl",
        632909,
        "3347bb07cdf003935fdc5d4bb0feee104ed14f5e90a8ef7384752134106824f2",
    ),
    (
        "mmlu_pro",
        "multiple_choice",
        [1],
        [1],
        12032,
        "the_answer_is",
        "data/mmlu-pro/mmlu_pro.jsonl",
        22722936,
        "31749ae33e77378dfc064a99389cc32f35546d4c094ffe7f039e264b73a8b8ba",
    ),
    (
        "mmlu_zeroshot",
        "multiple_choice",
        [1],
        [1],
        14042,
        "the_answer_is",
        "data/mmlu_zeroshot/mmlu_zeroshot.jsonl",
        22244117,
        "76581f4e0fe24aa41d37fe7bffe90286065c017eef442271551e4c657a4ba9ed",
    ),
)
MULTISAMPLE_TEMPERATURES = {
    "aime-2026": 0.7,
    "gpqa_diamond": 1.0,
}


def _materialize(
    source: Path,
    destination: Path,
    replacements: dict[str, str],
) -> Path:
    text = source.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        assert placeholder in text
        text = text.replace(placeholder, value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def test_k2_horizon_full_suite_resolves_through_normal_yaml_parsers(
    tmp_path: Path,
) -> None:
    release = tmp_path / "hf-release"
    release.mkdir()
    serving = tmp_path / "serving-venv"
    (serving / "bin").mkdir(parents=True)
    (serving / "bin/activate").write_text("# test activation\n", encoding="utf-8")
    output = tmp_path / "output"
    dataset_root = tmp_path / "datasets"

    model_path = _materialize(
        MODEL_TEMPLATE,
        tmp_path / "configs/model.yaml",
        {
            "/__EVAL360_HF_RELEASE_PATH__": str(release),
            "/__EVAL360_SERVING_VENV_PATH__": str(serving / "bin/activate"),
            "/__EVAL360_OUTPUT_ROOT__": str(output),
        },
    )
    eval_path = _materialize(
        EVAL_TEMPLATE,
        tmp_path / "configs/eval.yaml",
        {"__EVAL360_OUTPUT_ROOT__": str(output)},
    )

    loaded_tasks = []
    raw_task_configs = []
    for template, expected in zip(DATA_TEMPLATES, EXPECTED_TASKS, strict=True):
        source_path = expected[6]
        count = expected[4]
        dataset = dataset_root / DATASET_REVISION / source_path
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text("{}\n" * count, encoding="utf-8")
        config_path = _materialize(
            template,
            tmp_path / "configs/data" / template.name,
            {"__EVAL360_DATASET_ROOT__": str(dataset_root)},
        )
        task = Task.parse_yaml(config_path)
        loaded_tasks.append(LoadedTask(path=str(config_path), task=task))
        raw_task_configs.append(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    model = ModelParser.parse_yaml(model_path, eval_mode=True)
    evaluation = EvalConfigParser.parse_yaml(str(eval_path))
    resolved = EvalConfigParser.build_resolved_pairs(
        loaded_models=[LoadedModelSpec(path=str(model_path), spec=model)],
        loaded_tasks=loaded_tasks,
        loaded_eval_configs=[LoadedEvalConfig(path=str(eval_path), config=evaluation)],
    )

    assert model.remote_model is not None
    assert model.remote_model.path == str(release)
    assert model.venv_path == str(serving / "bin/activate")
    assert model.model_type is ModelType.BASE
    assert model.tag == "k2_horizon_7b_base_full"
    assert model.max_simultaneous_requests == 300
    assert model.max_time_to_deploy == 900
    assert model.vllm_cli_args == [
        "--tensor-parallel-size",
        "1",
        "--trust-remote-code",
        "--hf-overrides",
        '{"max_position_embeddings":8192}',
        "--max-model-len",
        "8192",
        '--override-generation-config {"add_bos_token":true}',
    ]
    assert model.openai_kwargs == {"max_tokens": 2048, "temperature": 0.0}

    assert len(resolved) == len(EXPECTED_TASKS)
    assert [pair.eval_group_name for pair in resolved] == [
        "k2-horizon-base-aime-2026",
        "k2-horizon-base-bbh-zeroshot",
        "k2-horizon-base-gpqa-diamond",
        "k2-horizon-base-gsm8k-cot-zeroshot",
        "k2-horizon-base-ifeval",
        "k2-horizon-base-mbpp-zeroshot",
        "k2-horizon-base-mmlu-pro",
        "k2-horizon-base-mmlu-zeroshot",
    ]
    assert [pair.model.parser_type for pair in resolved] == [
        expected[5] for expected in EXPECTED_TASKS
    ]
    assert len({pair.model.serving_key for pair in resolved}) == 1

    for pair, raw, expected in zip(
        resolved, raw_task_configs, EXPECTED_TASKS, strict=True
    ):
        task_id, grader, average_over, pass_at, count, _, source_path, size, digest = (
            expected
        )
        assert pair.task.uuid.startswith(f"{task_id}-eval-")
        assert pair.task.grader.type == grader
        assert pair.task.average_over == average_over
        assert pair.task.pass_at == pass_at
        assert pair.task.num_generations == count
        expected_temperature = MULTISAMPLE_TEMPERATURES.get(task_id, 0.0)
        assert pair.task.openai_settings == {
            "max_tokens": 2048,
            "temperature": expected_temperature,
        }
        requested_samples = max(*average_over, *pass_at)
        assert (expected_temperature > 0.0) is (requested_samples > 1)
        assert raw["mode"] == "base"
        assert raw["enabled"] is True
        assert raw["meta"]["source_repository"] == "<HF_ORG>/<EVAL_SOURCES_REPO>"
        assert raw["meta"]["source_revision"] == DATASET_REVISION
        assert raw["meta"]["source_path"] == source_path
        assert raw["meta"]["source_size_bytes"] == size
        assert raw["meta"]["source_sha256"] == digest
        assert raw["data_path"] == str(dataset_root / DATASET_REVISION / source_path)


def test_k2_horizon_full_suite_templates_are_explicit_and_non_personal() -> None:
    paths = (MODEL_TEMPLATE, EVAL_TEMPLATE, *DATA_TEMPLATES)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert 'path: "/__EVAL360_HF_RELEASE_PATH__"' in combined
    assert 'venv_path: "/__EVAL360_SERVING_VENV_PATH__"' in combined
    assert combined.count('output_root: "__EVAL360_OUTPUT_ROOT__"') == 8
    assert combined.count('data_path: "__EVAL360_DATASET_ROOT__/') == 8
    # No absolute path may appear except the quoted /__EVAL360_*__ placeholders.
    # A "/" that continues a word, a path or a <PLACEHOLDER> does not start a
    # path; "Q:/A:" is the lm-eval prompt format named in a source field.
    scannable = combined.replace("(Q:/A:", "(")
    absolute_paths = re.findall(r"""(?<![\w./~>-])/(?!/)[^\s"',\])]*""", scannable)
    assert absolute_paths
    assert all(
        re.fullmatch(r"/__EVAL360_[A-Z0-9_]+__", path) for path in absolute_paths
    ), absolute_paths
    assert "max_generation_jobs" not in combined
    assert "max_grading_parallelism" not in combined


def test_k2_horizon_full_suite_preserves_canonical_task_semantics() -> None:
    semantic_fields = (
        "average_over",
        "dataset_name",
        "grader",
        "num_generations",
        "pass_at",
        "semantic_version",
        "uuid",
    )
    for suite_path, canonical_path in zip(
        DATA_TEMPLATES, CANONICAL_DATA_CONFIGS, strict=True
    ):
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        canonical = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))

        assert {field: suite[field] for field in semantic_fields} == {
            field: canonical[field] for field in semantic_fields
        }
        for field, value in canonical.get("meta", {}).items():
            assert suite["meta"][field] == value


def test_k2_horizon_model_reuses_base_settings_with_context_override() -> None:
    template = yaml.safe_load(MODEL_TEMPLATE.read_text(encoding="utf-8"))
    reviewed = yaml.safe_load(
        (ROOT / "model_zoo/k2_horizon_7b_final_base-mmlu-pro.yaml").read_text(encoding="utf-8")
    )

    for field in (
        "max_simultaneous_requests",
        "max_time_to_deploy",
        "model_type",
        "openai_kwargs",
    ):
        assert template[field] == reviewed[field]
    assert template["vllm_cli_args"] == [
        *reviewed["vllm_cli_args"][:-1],
        "--hf-overrides",
        '{"max_position_embeddings":8192}',
        "--max-model-len",
        "8192",
        reviewed["vllm_cli_args"][-1],
    ]
    assert template["allow_long_max_model_len"] is reviewed.get(
        "allow_long_max_model_len", True
    )
