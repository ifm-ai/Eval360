from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scheduler import Scheduler

__all__ = ["Scheduler"]


def __getattr__(name):
    if name == "Scheduler":
        from .scheduler import Scheduler

        return Scheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
