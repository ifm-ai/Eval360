"""
Tests for grader registration:
  - All built-in graders are present after package import
  - get_grader returns the correct class
  - External graders can be registered via register_grader()
  - _load_grader_plugins() calls ep.load() for each eval360.graders entry point
"""
import pytest
from unittest.mock import MagicMock, patch

from scheduler.grader import get_grader, list_graders, _load_grader_plugins
from scheduler.grader.registry import register_grader, _REGISTRY, _UNAVAILABLE_GRADERS
from scheduler.grader.base import AccuracyGraderBase


# ---------------------------------------------------------------------------
# Fixture: restore registry state after each test that mutates it
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_snapshot():
    saved = dict(_REGISTRY)
    saved_unavailable = dict(_UNAVAILABLE_GRADERS)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)
    _UNAVAILABLE_GRADERS.clear()
    _UNAVAILABLE_GRADERS.update(saved_unavailable)


# ---------------------------------------------------------------------------
# Built-in grader registration
# ---------------------------------------------------------------------------

EXPECTED_BUILTIN_GRADERS = {
    "multiple_choice", "multiple-choice",
    "exact_match", "exact-match",
    "math_verify", "math-verify",
    "llm_as_judge",
    "sympy_llm_as_judge", "sympy-llm-as-judge",
    "kk", "knights-and-knaves",
    "countdown", "cd",
    "order", "order-puzzle",
    "sum", "sum-puzzle",
    "boxed-llm-as-judge",
    "humaneval", "human_eval", "human-eval",
}


class TestBuiltinRegistration:
    def test_all_expected_graders_registered(self):
        registered = set(list_graders())
        unavailable = set(_UNAVAILABLE_GRADERS)
        missing = EXPECTED_BUILTIN_GRADERS - registered - unavailable
        assert not missing, f"Expected grader types not registered: {missing}"

    def test_get_grader_multiple_choice(self):
        from scheduler.grader.multiple_choice import MultipleChoice
        assert get_grader("multiple_choice") is MultipleChoice
        assert get_grader("multiple-choice") is MultipleChoice

    def test_get_grader_exact_match(self):
        from scheduler.grader.match import Match
        assert get_grader("exact_match") is Match
        assert get_grader("exact-match") is Match

    def test_get_grader_math_verify(self):
        try:
            from scheduler.grader.math import MathVerify
        except ModuleNotFoundError:
            with pytest.raises(ImportError, match="unavailable because"):
                get_grader("math_verify")
            with pytest.raises(ImportError, match="unavailable because"):
                get_grader("math-verify")
        else:
            assert get_grader("math_verify") is MathVerify
            assert get_grader("math-verify") is MathVerify

    def test_get_grader_sympy_llm_as_judge(self):
        try:
            from scheduler.grader.sympy_llm_as_judge import SympyLLMasJudge
        except ModuleNotFoundError:
            with pytest.raises(ImportError, match="unavailable because"):
                get_grader("sympy_llm_as_judge")
            with pytest.raises(ImportError, match="unavailable because"):
                get_grader("sympy-llm-as-judge")
        else:
            assert get_grader("sympy_llm_as_judge") is SympyLLMasJudge
            assert get_grader("sympy-llm-as-judge") is SympyLLMasJudge

    def test_unknown_grader_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown grader type"):
            get_grader("does-not-exist")


# ---------------------------------------------------------------------------
# External / plugin grader registration
# ---------------------------------------------------------------------------

class TestExternalRegistration:
    def test_register_grader_directly(self, registry_snapshot):
        """register_grader() adds an external class to the registry."""
        class _ExternalGrader(AccuracyGraderBase):
            async def grade_sample(self, sample, *_):
                return sample

        register_grader("external-direct", _ExternalGrader)
        assert get_grader("external-direct") is _ExternalGrader

    def test_duplicate_registration_raises(self, registry_snapshot):
        class _G(AccuracyGraderBase):
            async def grade_sample(self, sample, *_):
                return sample

        register_grader("dup-grader", _G)
        with pytest.raises(ValueError, match="already registered"):
            register_grader("dup-grader", _G)

    def test_load_grader_plugins_calls_ep_load(self, registry_snapshot):
        """_load_grader_plugins() calls ep.load() for every entry point."""
        mock_ep = MagicMock()
        mock_ep.name = "fake-plugin"

        with patch("scheduler.grader.entry_points", return_value=[mock_ep]):
            _load_grader_plugins()

        mock_ep.load.assert_called_once()

    def test_load_grader_plugins_registers_via_load(self, registry_snapshot):
        """A plugin's ep.load() can call register_grader to add to the registry."""
        class _PluginGrader(AccuracyGraderBase):
            async def grade_sample(self, sample, *_):
                return sample

        def _fake_load():
            register_grader("plugin-via-entry-point", _PluginGrader)

        mock_ep = MagicMock()
        mock_ep.name = "my-plugin"
        mock_ep.load.side_effect = _fake_load

        with patch("scheduler.grader.entry_points", return_value=[mock_ep]):
            _load_grader_plugins()

        assert get_grader("plugin-via-entry-point") is _PluginGrader

    def test_load_grader_plugins_bad_ep_propagates(self, registry_snapshot):
        """A plugin that raises on load propagates the error immediately."""
        mock_ep = MagicMock()
        mock_ep.name = "broken-plugin"
        mock_ep.load.side_effect = ImportError("missing dep")

        with patch("scheduler.grader.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                _load_grader_plugins()
