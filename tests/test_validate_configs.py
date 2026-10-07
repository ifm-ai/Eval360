"""Tests for scripts/validate_configs.py."""
import sys
from pathlib import Path
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_configs import SEARCH_DIRS, scan_dirs, validate_file


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# YAML syntax
# ---------------------------------------------------------------------------

class TestYAMLSyntax:
    def test_invalid_yaml_is_hard_failure(self, tmp_path):
        p = _write(tmp_path, "bad.yaml", "key: [unclosed")
        status, msg = validate_file(p)
        assert status == "fail"
        assert "invalid YAML" in msg

    def test_non_dict_yaml_is_skipped(self, tmp_path):
        p = _write(tmp_path, "list.yaml", "- a\n- b\n")
        status, _ = validate_file(p)
        assert status == "ok"

    def test_score_output_file_is_skipped(self, tmp_path):
        p = _write(tmp_path, "scores.yaml", '"accuracy": 0.85\n"total": 100\n')
        status, _ = validate_file(p)
        assert status == "ok"


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

_VALID_MODEL = """\
remote_model:
  base_name: my-model
  path: org/my-model
  revision: null
model_type: instruct
parser_type: noop
name_modifier: null
venv_path: /venv/bin/activate
max_simultaneous_requests: 100
max_time_to_deploy: 600
vllm_cli_args: []
openai_kwargs: {}
owner: testuser
ready: true
output_path: /results
"""

class TestModelConfigs:
    def test_valid_model_config_passes(self, tmp_path):
        p = _write(tmp_path, "model.yaml", _VALID_MODEL)
        status, _ = validate_file(p)
        assert status == "ok"

    def test_model_missing_required_field_fails(self, tmp_path):
        # Remove parser_type → ValidationError
        bad = _VALID_MODEL.replace("parser_type: noop\n", "")
        p = _write(tmp_path, "model.yaml", bad)
        status, msg = validate_file(p)
        assert status == "fail"
        assert "schema error" in msg

    def test_model_wrong_type_fails(self, tmp_path):
        bad = _VALID_MODEL.replace("max_simultaneous_requests: 100", "max_simultaneous_requests: not_a_number")
        p = _write(tmp_path, "model.yaml", bad)
        status, msg = validate_file(p)
        assert status == "fail"
        assert "schema error" in msg


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

_VALID_DATASET = """\
uuid: "test_dataset"
grader:
  type: multiple_choice
average_over:
  - 1
pass_at:
  - 1
dataset_name: "test_dataset"
data_path: "hf://org/dataset/*.jsonl"
semantic_version: "1.0.0"
num_generations: 100
meta:
  fewshot: 0
"""

_VALID_EVAL = """\
version: 1
owner: testuser
groups:
  - name: reasoning-aime
    model_tag: reasoning
    data_tag: aime
    parser_type: noop
    output_root: /results
"""

class TestDatasetConfigs:
    def test_valid_dataset_config_passes(self, tmp_path):
        p = _write(tmp_path, "dataset.yaml", _VALID_DATASET)
        status, _ = validate_file(p)
        # May warn if HF is not reachable — either ok or warn is acceptable
        assert status in ("ok", "warn")

    def test_dataset_missing_grader_fails(self, tmp_path):
        obj = yaml.safe_load(_VALID_DATASET)
        del obj["grader"]
        p = _write(tmp_path, "dataset.yaml", yaml.dump(obj))
        status, msg = validate_file(p)
        assert status == "fail"
        assert "schema error" in msg

    def test_dataset_old_grader_type_field_fails(self, tmp_path):
        # Old schema used top-level grader_type instead of grader: {type: ...}
        bad = _VALID_DATASET.replace("grader:\n  type: multiple_choice", "grader_type: multiple_choice")
        p = _write(tmp_path, "dataset.yaml", bad)
        status, msg = validate_file(p)
        assert status == "fail"
        assert "schema error" in msg

    def test_dataset_missing_file_is_warning(self, tmp_path):
        bad = _VALID_DATASET.replace(
            'data_path: "hf://org/dataset/*.jsonl"',
            'data_path: "/nonexistent/path/*.jsonl"'
        )
        p = _write(tmp_path, "dataset.yaml", bad)
        status, _ = validate_file(p)
        assert status == "warn"

    def test_choice_scoring_requires_single_generation_contract(self, tmp_path):
        data_path = tmp_path / "choice_scoring.jsonl"
        data_path.write_text(
            '{"row": 0, "completion_input": "Q", "ground_truth": "A"}\n',
            encoding="utf-8",
        )
        obj = yaml.safe_load(_VALID_DATASET)
        obj["grader"] = {"type": "choice_scoring"}
        obj["average_over"] = [2]
        obj["pass_at"] = [1]
        obj["data_path"] = str(data_path)
        p = _write(tmp_path, "choice_scoring.yaml", yaml.dump(obj))

        status, msg = validate_file(p)

        assert status == "fail"
        assert "choice_scoring grader requires average_over=[1] and pass_at=[1]" in msg

    @pytest.mark.parametrize(
        "config_path",
        [
            "data_zoo/mmlu_pro_choice_scoring.yaml",
            "data_zoo/mmlu_pro_small_choice_scoring.yaml",
            "data_zoo/mmlu_pro_5shot_choice_scoring.yaml",
            "data_zoo/mmlu_pro_5shot_small_choice_scoring.yaml",
        ],
    )
    def test_mmlu_pro_choice_scoring_configs_parse(self, config_path):
        status, msg = validate_file(Path(config_path))
        assert status in ("ok", "warn"), msg

    @pytest.mark.parametrize(
        ("config_path", "expected_name", "expected_rows", "expected_fewshot"),
        [
            (
                "data_zoo/mmlu_pro_choice_scoring.yaml",
                "mmlu_pro",
                12032,
                0,
            ),
            (
                "data_zoo/mmlu_pro_small_choice_scoring.yaml",
                "mmlu_pro_small",
                100,
                0,
            ),
            (
                "data_zoo/mmlu_pro_5shot_choice_scoring.yaml",
                "mmlu_pro_5shot",
                12032,
                5,
            ),
            (
                "data_zoo/mmlu_pro_5shot_small_choice_scoring.yaml",
                "mmlu_pro_5shot_small",
                100,
                5,
            ),
        ],
    )
    def test_mmlu_pro_choice_scoring_configs_have_exact_runtime_contract(
        self,
        config_path,
        expected_name,
        expected_rows,
        expected_fewshot,
    ):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        assert config["grader"] == {"type": "choice_scoring"}
        assert config["average_over"] == [1]
        assert config["pass_at"] == [1]
        assert config["dataset_name"] == expected_name
        assert config["num_generations"] == expected_rows
        assert config["meta"]["fewshot"] == expected_fewshot
        assert config["meta"]["scoring"] == "base-model-nll"
        assert config["data_path"].endswith(".jsonl@main")


# ---------------------------------------------------------------------------
# Eval configs
# ---------------------------------------------------------------------------

class TestEvalConfigs:
    def test_valid_eval_config_passes(self, tmp_path):
        p = _write(tmp_path, "eval.yaml", _VALID_EVAL)
        status, _ = validate_file(p)
        assert status == "ok"

    def test_eval_config_missing_required_field_fails(self, tmp_path):
        obj = yaml.safe_load(_VALID_EVAL)
        del obj["groups"][0]["parser_type"]
        p = _write(tmp_path, "eval.yaml", yaml.dump(obj))
        status, msg = validate_file(p)
        assert status == "fail"
        assert "eval schema error" in msg

    def test_eval_config_unsupported_version_fails(self, tmp_path):
        obj = yaml.safe_load(_VALID_EVAL)
        obj["version"] = 2
        p = _write(tmp_path, "eval.yaml", yaml.dump(obj))
        status, msg = validate_file(p)
        assert status == "fail"
        assert "Unsupported eval config version" in msg

    def test_evals_dir_is_scanned_by_ci(self):
        # The README tells users to pass `--eval-paths evals/...`, so the CI
        # validator must scan that directory. If it isn't in SEARCH_DIRS, a
        # malformed eval config placed there silently passes CI and only fails
        # at run time.
        assert "evals" in SEARCH_DIRS

    def test_malformed_eval_config_in_evals_dir_fails_scan(self, tmp_path):
        # Drive the exact directory scan CI runs (scan_dirs over SEARCH_DIRS)
        # against a repo layout where the only config lives in evals/. A schema
        # error there must surface as a hard failure; if `evals` were dropped
        # from SEARCH_DIRS this file would be skipped and the scan would pass.
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()
        obj = yaml.safe_load(_VALID_EVAL)
        del obj["groups"][0]["parser_type"]
        (evals_dir / "reasoning.yaml").write_text(yaml.dump(obj))

        failures = scan_dirs(tmp_path, SEARCH_DIRS)

        assert failures == [Path("evals/reasoning.yaml")]

    def test_valid_eval_config_in_evals_dir_passes_scan(self, tmp_path):
        # Mirror of the negative test: a well-formed eval config in evals/ must
        # not be flagged, so the scan only blocks on genuine schema errors.
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()
        (evals_dir / "reasoning.yaml").write_text(_VALID_EVAL)

        failures = scan_dirs(tmp_path, SEARCH_DIRS)

        assert failures == []


# ---------------------------------------------------------------------------
# Unrecognized formats
# ---------------------------------------------------------------------------

class TestUnrecognized:
    def test_unrecognized_yaml_is_skipped(self, tmp_path):
        p = _write(tmp_path, "other.yaml", "foo: bar\nbaz: 123\n")
        status, _ = validate_file(p)
        assert status == "ok"
