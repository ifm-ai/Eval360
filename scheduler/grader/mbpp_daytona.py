"""MBPP grader using Daytona sandboxes for isolated code execution.

Each generated solution is concatenated with the assert-based test
statements from the ground truth and run inside a fresh Daytona sandbox.
Exit code 0 means all asserts passed.
"""

import logging

from .registry import register
from .daytona_base import DaytonaGraderBase

logger = logging.getLogger(__name__)


@register("mbpp-daytona")
class MBPPDaytonaGrader(DaytonaGraderBase):
    """Grade MBPP problems by executing solutions in Daytona sandboxes."""

    @staticmethod
    def ground_truth_to_test_list(ground_truth: str) -> list[str]:
        return [ground_truth]

    def build_test_harness(self, code: str, test_script: str) -> str:
        """Concatenate the generated function with the bundled assert statements."""
        return f"{code}\n{test_script}"
