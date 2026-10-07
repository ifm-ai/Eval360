import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_module() -> object:
    lm_eval = types.ModuleType("lm_eval")
    tasks = types.ModuleType("lm_eval.tasks")
    evaluator_utils = types.ModuleType("lm_eval.evaluator_utils")

    class TaskManager:
        pass

    tasks.TaskManager = TaskManager
    tasks.get_task_dict = lambda *_args, **_kwargs: {}
    evaluator_utils.get_task_list = lambda *_args, **_kwargs: []
    lm_eval.tasks = tasks
    lm_eval.evaluator_utils = evaluator_utils
    sys.modules["lm_eval"] = lm_eval
    sys.modules["lm_eval.tasks"] = tasks
    sys.modules["lm_eval.evaluator_utils"] = evaluator_utils

    module_path = Path(__file__).resolve().parent.parent / "persist_builtin_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "persist_builtin_dataset", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


class IndexLike:
    def __index__(self):
        return 1


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (1, "beta"),
        (IndexLike(), "beta"),
        ("1", "beta"),
        (" beta ", "beta"),
    ],
)
def test_resolve_choice_target_accepts_supported_target_forms(target, expected):
    module = load_module()

    assert module.resolve_choice_target(target, ["alpha", "beta"]) == expected


@pytest.mark.parametrize("target", [-1, "3", "missing", object()])
def test_resolve_choice_target_rejects_unsupported_target_forms(target):
    module = load_module()

    with pytest.raises(ValueError, match="Unable to resolve multiple-choice target"):
        module.resolve_choice_target(target, ["alpha", "beta"], context="doc_id=7")


def test_instances_to_record_accepts_numeric_string_choice_target():
    module = load_module()

    class Config:
        output_type = "multiple_choice"

    class FakeTask:
        config = Config()

        def doc_to_choice(self, doc):
            return ["alpha", "beta"]

        def doc_to_target(self, doc):
            return "1"

    class FakeInstance:
        doc = {"id": "row-1"}
        args = ["Question?\nAnswer:"]

    record = module.instances_to_record(FakeTask(), 7, [FakeInstance()], None, None)

    assert record["ground_truth"] == "beta"
    assert record["scoring_completions"] == ["alpha", "beta"]
    assert record["scoring_completion_labels"] == ["alpha", "beta"]


def test_instances_to_record_error_includes_doc_context_for_invalid_choice_target():
    module = load_module()

    class Config:
        output_type = "multiple_choice"

    class FakeTask:
        config = Config()

        def doc_to_choice(self, doc):
            return ["alpha", "beta"]

        def doc_to_target(self, doc):
            return "missing"

    class FakeInstance:
        doc = {"id": "row-1"}
        args = ["Question?\nAnswer:"]

    with pytest.raises(ValueError, match="doc_id=7"):
        module.instances_to_record(FakeTask(), 7, [FakeInstance()], None, None)
