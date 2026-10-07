import os
import sys
import pytest

# Add repo root to sys.path so `import scheduler` resolves as the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure in-memory DB is used for all tests
os.environ.setdefault("EVAL360_IN_MEMORY_DB", "true")

# Trigger grader auto-discovery so grader types and parsers are registered before tests run
import scheduler.grader  # noqa: E402, F401


def pytest_addoption(parser):
    parser.addoption("--long", action="store_true", default=False,
                     help="Run long-running tests (disabled in CI, enabled locally via addopts)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--long"):
        skip = pytest.mark.skip(reason="skipped without --long")
        for item in items:
            if "long" in item.keywords:
                item.add_marker(skip)
