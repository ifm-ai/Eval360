"""
Tests for the ImportedDataset runner registry and base class contract.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import scheduler.imported_dataset  # noqa: F401 — triggers auto-discovery
from scheduler.imported_dataset import (
    ImportedDatasetRunnerBase,
    ImportedDatasetResultError,
    get_runner,
    list_runners,
    register,
)
from scheduler.grader.base import Score


# ---------------------------------------------------------------------------
# A minimal concrete runner for testing the base class contract
# ---------------------------------------------------------------------------

@register("test-benchmark")
class _TestRunner(ImportedDatasetRunnerBase):
    def build_setup_script(self, repo_root: Path) -> str:
        return f"echo setup {repo_root}"

    def build_benchmark_script(self, model_instance, task_args, output_dir: Path) -> str:
        return f"echo benchmark {output_dir}"

    def parse_results(self, output_dir: Path, task) -> list[Score]:
        return [Score(name="accuracy", value=0.42)]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered_runner_is_retrievable(self):
        cls = get_runner("test-benchmark")
        assert cls is _TestRunner

    def test_list_runners_includes_registered(self):
        assert "test-benchmark" in list_runners()

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register("test-benchmark")(_TestRunner)

    def test_unknown_runner_raises(self):
        with pytest.raises(ValueError, match="Unknown imported dataset"):
            get_runner("nonexistent-benchmark")


# ---------------------------------------------------------------------------
# Base class contract tests
# ---------------------------------------------------------------------------

class TestRunnerContract:
    def setup_method(self):
        self.runner = _TestRunner()
        self.model = MagicMock()
        self.output_dir = Path("/tmp/test_output")

    def test_build_setup_script_returns_str(self):
        result = self.runner.build_setup_script(Path("/repo"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_benchmark_script_returns_str(self):
        result = self.runner.build_benchmark_script(self.model, {}, self.output_dir)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_parse_results_returns_scores(self):
        scores = self.runner.parse_results(self.output_dir, task=None)
        assert isinstance(scores, list)
        assert all(isinstance(s, Score) for s in scores)

    def test_imported_dataset_result_error_is_exception(self):
        with pytest.raises(ImportedDatasetResultError):
            raise ImportedDatasetResultError("results missing")


# ---------------------------------------------------------------------------
# Abstract method enforcement
# ---------------------------------------------------------------------------

class TestAbstractEnforcement:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            ImportedDatasetRunnerBase()

    def test_partial_implementation_raises(self):
        class _Partial(ImportedDatasetRunnerBase):
            def build_setup_script(self, repo_root):
                return ""
            def build_benchmark_script(self, model_instance, task_args, output_dir):
                return ""
            # parse_results not implemented

        with pytest.raises(TypeError):
            _Partial()
