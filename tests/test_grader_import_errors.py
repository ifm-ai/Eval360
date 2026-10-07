"""
Tests for grader auto-discovery error handling.

Syntax and runtime errors during discovery must still propagate immediately.
Missing optional third-party modules are treated differently: the grader is
recorded as unavailable so unrelated tasks can still start, and a clear error
is raised only if that grader is requested.
"""
import importlib
import pkgutil
import pytest
from unittest.mock import MagicMock, patch

import scheduler.grader as grader
from scheduler.grader import _discover_builtin_graders
from scheduler.grader.registry import _UNAVAILABLE_GRADERS


# Fake importer returned by the mocked pkgutil.iter_modules
def _fake_iter(modname):
    importer = MagicMock()
    return [(importer, modname, False)]


class TestGraderDiscoveryErrorHandling:
    @pytest.fixture(autouse=True)
    def unavailable_snapshot(self):
        saved = dict(_UNAVAILABLE_GRADERS)
        yield
        _UNAVAILABLE_GRADERS.clear()
        _UNAVAILABLE_GRADERS.update(saved)

    def test_syntax_error_propagates(self):
        """A SyntaxError in a grader file must not be silently swallowed."""
        with patch.object(pkgutil, "iter_modules", return_value=_fake_iter("grader.broken")):
            with patch.object(importlib, "import_module", side_effect=SyntaxError("invalid syntax")):
                with pytest.raises(SyntaxError):
                    _discover_builtin_graders()

    def test_import_error_propagates(self):
        """Generic ImportError still propagates so real regressions stay loud."""
        with patch.object(pkgutil, "iter_modules", return_value=_fake_iter("grader.broken")):
            with patch.object(importlib, "import_module", side_effect=ImportError("no module named 'something'")):
                with pytest.raises(ImportError):
                    _discover_builtin_graders()

    def test_missing_optional_module_marks_grader_unavailable(self):
        """ModuleNotFoundError for a third-party package should not abort startup."""
        err = ModuleNotFoundError("No module named 'absl'")
        err.name = "absl"
        with patch.object(pkgutil, "iter_modules", return_value=_fake_iter("grader.broken")):
            with patch.object(importlib, "import_module", side_effect=err):
                with patch.object(grader, "_extract_registered_aliases", return_value=["broken"]):
                    _discover_builtin_graders()
        with pytest.raises(ImportError, match="missing module 'absl'"):
            grader.get_grader("broken")

    def test_attribute_error_propagates(self):
        """Non-import exceptions also propagate."""
        with patch.object(pkgutil, "iter_modules", return_value=_fake_iter("grader.broken")):
            with patch.object(importlib, "import_module", side_effect=AttributeError("bad attribute")):
                with pytest.raises(AttributeError):
                    _discover_builtin_graders()
