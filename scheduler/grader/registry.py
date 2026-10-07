"""
Registry for grader classes. Graders self-register via the register() decorator or function.
"""
from typing import Type

from .base import GraderBase

_REGISTRY: dict[str, Type[GraderBase]] = {}
_UNAVAILABLE_GRADERS: dict[str, str] = {}


def register(*grader_types: str):
    """Decorator to register a grader class by one or more type strings (aliases)."""

    def _register(cls: Type[GraderBase]):
        for grader_type in grader_types:
            if grader_type in _REGISTRY:
                raise ValueError(f"Grader type '{grader_type}' is already registered")
            _UNAVAILABLE_GRADERS.pop(grader_type, None)
            _REGISTRY[grader_type] = cls
        return cls

    return _register


def register_grader(grader_type: str, cls: Type[GraderBase]) -> None:
    """Explicitly register a grader class."""
    if grader_type in _REGISTRY:
        raise ValueError(f"Grader type '{grader_type}' is already registered")
    _UNAVAILABLE_GRADERS.pop(grader_type, None)
    _REGISTRY[grader_type] = cls


def mark_grader_unavailable(*grader_types: str, reason: str) -> None:
    """Remember grader aliases that could not be imported due to an optional dependency."""
    for grader_type in grader_types:
        if grader_type in _REGISTRY:
            continue
        _UNAVAILABLE_GRADERS[grader_type] = reason


def get_grader(grader_type: str) -> Type[GraderBase]:
    """Get a grader class by its type string."""
    if grader_type in _REGISTRY:
        return _REGISTRY[grader_type]
    if grader_type in _UNAVAILABLE_GRADERS:
        raise ImportError(
            f"Grader type '{grader_type}' is unavailable because {_UNAVAILABLE_GRADERS[grader_type]}"
        )
    available = ", ".join(sorted(_REGISTRY.keys()))
    raise ValueError(
        f"Unknown grader type '{grader_type}'. Available: {available}"
    )


def list_graders() -> list[str]:
    """Return all registered grader type names."""
    return list(_REGISTRY.keys())
