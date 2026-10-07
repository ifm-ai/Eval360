"""
Base class for ImportedDataset runners.

Each external benchmark gets its own subclass that implements three methods:

  build_setup_script(repo_root)
      Returns a bash script containing only the benchmark's install commands
      (e.g. pip install ...).  The sbatch script owns the idempotency logic:
      it checks for a sentinel file, creates the venv, runs this script, and
      writes the sentinel on success — so partial installs caused by node
      preemption are automatically cleaned up and retried.  The venv exists
      but is not activated yet; $VENV points to its root.

  build_benchmark_script(model_instance, task, output_dir)
      Returns a self-contained bash script that runs the benchmark CLI,
      pointing it at the VLLM instance on localhost:8000 and writing all
      output to *output_dir*.  Both VLLM and this script run on the same
      Slurm node: VLLM is started in the background by the sbatch script
      before this is called.  The venv is already activated when this runs.

  parse_results(output_dir, task, model_instance)
      Reads the benchmark's output files from *output_dir*, validates that
      the run completed successfully, and returns a list of Score objects.
      Raises ImportedDatasetResultError if the results are missing or
      incomplete.
"""
import abc
from pathlib import Path

from ..grader.base import Score
from ..model import ModelInstance
from ..task import ImportedDatasetTask


class ImportedDatasetResultError(Exception):
    """Raised when benchmark results are missing or incomplete."""


class ImportedDatasetRunnerBase(abc.ABC):

    @abc.abstractmethod
    def build_setup_script(self, repo_root: Path) -> str:
        """
        Return a bash script containing only this benchmark's install commands.

        The sbatch script handles venv creation, idempotency (via a sentinel
        file), and activation — so this script should contain only the
        benchmark-specific install steps, e.g.::

            "$VENV/bin/pip" install -r /path/to/requirements.txt

        $VENV is set by the sbatch script and points to
        <repo_root>/.eval360/envs/<benchmark_name>.  The venv is already
        created but not yet activated when this script runs.
        """

    @abc.abstractmethod
    def build_benchmark_script(
        self,
        model_instance: ModelInstance,
        task: ImportedDatasetTask,
        output_dir: Path,
    ) -> str:
        """
        Return a bash script that runs the full benchmark against the VLLM
        instance running on localhost:8000.

        The script should:
        - Run all necessary benchmark commands (generation, evaluation, etc.)
        - Write all output to *output_dir*
        - Exit non-zero on failure

        Do not activate the venv — the sbatch script does this before running
        this script.  *model_instance* provides model path, etc.
        *task* provides benchmark-specific config via task.imported_dataset.args.
        *output_dir* is where parse_results will look for results.
        """

    @abc.abstractmethod
    def parse_results(self, output_dir: Path, task: ImportedDatasetTask, model_instance: ModelInstance) -> list[Score]:
        """
        Read benchmark output from *output_dir* and return Score objects.

        Raises ImportedDatasetResultError if the results are absent or
        incomplete (e.g. the benchmark crashed mid-run).
        """
