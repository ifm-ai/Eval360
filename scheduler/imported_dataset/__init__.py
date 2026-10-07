"""
ImportedDataset runner package. Imports trigger self-registration of runner classes.

Discovery works the same way as scheduler/grader/__init__.py:

1. Built-in runners: every module in this package (except base/registry) is
   imported so its @register(...) decorator fires.

2. Plugin runners: any installed package that declares an entry point in the
   "eval360.imported_datasets" group is imported.  Example pyproject.toml::

       [project.entry-points."eval360.imported_datasets"]
       my_benchmark = "my_package.my_runner_module"

All exceptions from both discovery paths propagate immediately.
"""
import importlib
import logging
import pkgutil
from importlib.metadata import entry_points

from .base import ImportedDatasetRunnerBase, ImportedDatasetResultError
from .registry import get_runner, list_runners, register

_EXCLUDE = {"base", "registry"}
_logger = logging.getLogger(__name__)


def _discover_builtin_runners():
    for _importer, _modname, _ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
        _name = _modname.split(".")[-1]
        if _name not in _EXCLUDE:
            importlib.import_module(_modname)


def _load_runner_plugins():
    for ep in entry_points(group="eval360.imported_datasets"):
        ep.load()
        _logger.debug("Loaded imported dataset plugin: %s", ep.name)


_discover_builtin_runners()
_load_runner_plugins()

__all__ = [
    "ImportedDatasetRunnerBase",
    "ImportedDatasetResultError",
    "get_runner",
    "list_runners",
    "register",
    "_discover_builtin_runners",
    "_load_runner_plugins",
]
