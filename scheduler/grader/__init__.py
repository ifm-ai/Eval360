"""
Grader package. Imports trigger self-registration of grader classes.

Two discovery mechanisms run at import time:

1. Built-in graders: every module in this package (except base/registry) is
   imported so its @register(...) decorator fires.

2. Plugin graders: any installed package that declares an entry point in the
   "eval360.graders" group is imported.  Example pyproject.toml snippet:

       [project.entry-points."eval360.graders"]
       my_grader = "my_package.my_grader_module"

   The named module must call register() or register_grader() at import time.

Built-in grader imports are allowed to fail with ModuleNotFoundError for
optional third-party dependencies. In that case the grader aliases are recorded
as unavailable, startup continues, and a clear ImportError is raised only if a
task explicitly requests that grader. Other exceptions still propagate
immediately.
"""
import ast
import importlib
import logging
import pkgutil
from pathlib import Path
from importlib.metadata import entry_points

from .base import Grade, Score
from .registry import get_grader, list_graders, mark_grader_unavailable, register, register_grader

_EXCLUDE = {"base", "registry"}
_logger = logging.getLogger(__name__)


def _extract_registered_aliases(modname: str) -> list[str]:
    """Best-effort extraction of @register(...) aliases without importing the module."""
    spec = importlib.util.find_spec(modname)
    if spec is None or spec.origin is None:
        return [modname.rsplit(".", 1)[-1]]

    origin = Path(spec.origin)
    if origin.suffix != ".py":
        return [modname.rsplit(".", 1)[-1]]

    tree = ast.parse(origin.read_text(encoding="utf-8"), filename=str(origin))
    aliases: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            is_register = (
                isinstance(func, ast.Name) and func.id == "register"
            ) or (
                isinstance(func, ast.Attribute) and func.attr == "register"
            )
            if not is_register:
                continue
            for arg in decorator.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    aliases.append(arg.value)
    return aliases or [modname.rsplit(".", 1)[-1]]


def _discover_builtin_graders():
    """Import every module in this package (except base/registry) to trigger
    @register calls. Missing optional third-party dependencies are recorded as
    unavailable graders; other exceptions propagate immediately."""
    for _importer, _modname, _ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
        _name = _modname.split(".")[-1]
        if _name not in _EXCLUDE:
            try:
                importlib.import_module(_modname)
            except ModuleNotFoundError as exc:
                missing = getattr(exc, "name", None)
                if missing is None or missing == _modname or missing.startswith("scheduler."):
                    raise
                aliases = _extract_registered_aliases(_modname)
                mark_grader_unavailable(
                    *aliases,
                    reason=f"{_modname} failed to import due to missing module '{missing}'",
                )
                _logger.warning(
                    "Skipping grader module %s because optional dependency %s is missing",
                    _modname,
                    missing,
                )


def _load_grader_plugins():
    """Import every module registered under the ``eval360.graders`` entry-point
    group so that its ``@register`` call fires. All exceptions propagate
    immediately. Called once at package import time; can also be called in
    tests with a patched ``entry_points``."""
    for ep in entry_points(group="eval360.graders"):
        ep.load()
        _logger.debug("Loaded grader plugin: %s", ep.name)


# 1. Built-in graders
_discover_builtin_graders()

# 2. Plugin graders (installed via entry points)
_load_grader_plugins()

__all__ = ["Grade", "Score", "get_grader", "list_graders", "register", "register_grader",
           "_discover_builtin_graders", "_load_grader_plugins"]
