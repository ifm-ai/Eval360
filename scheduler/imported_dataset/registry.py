"""
Registry for ImportedDatasetRunner classes. Runners self-register via the register() decorator.
"""
from typing import Type

_REGISTRY: dict[str, Type] = {}


def register(*names: str):
    """Decorator to register a runner class by one or more benchmark name strings."""

    def _register(cls):
        for name in names:
            if name in _REGISTRY:
                raise ValueError(f"ImportedDataset runner '{name}' is already registered")
            _REGISTRY[name] = cls
        return cls

    return _register


def get_runner(name: str):
    """Get a runner class by its benchmark name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown imported dataset '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def list_runners() -> list[str]:
    """Return all registered benchmark names."""
    return list(_REGISTRY.keys())
