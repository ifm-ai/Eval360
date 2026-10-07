# TODO: combine with data cli and move to root
import argparse
import asyncio
import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path

from .evaluation_request import EvaluationRequest
from .model import validate_external_total_deadline_override
from .scheduler import Scheduler
from .terminal_result import (
    YamlTerminalInvocation,
    publish_terminal_result,
    publish_yaml_terminal_result,
)

_INSTANCE_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def _parse_instance_id(value):
    """Validate and return a scheduler instance ID supplied via the CLI."""
    if not _INSTANCE_ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "must be exactly 8 lowercase hexadecimal characters"
        )
    return value


def _parse_positive_finite_seconds(value):
    """Parse an explicit deadline beyond the model-configuration ceiling."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must exceed 7200 seconds"
        ) from exc
    try:
        return validate_external_total_deadline_override(seconds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must exceed 7200 seconds"
        ) from exc


def _add_global_args(p):
    """Add global flags to a parser so they may appear before or after the subcommand.

    Uses default=argparse.SUPPRESS so that flags not provided on a given parser
    leave any value already set by an earlier parser untouched in the namespace.
    """
    p.add_argument(
        "--max-generation-jobs",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum number of Slurm serving jobs. For evaluate-now, omission "
             "uses the number of resolved unique serving keys; required for "
             "long-running-scheduler."
    )
    p.add_argument(
        "--max-grading-parallelism",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum number of dataset x model pairs graded at once. For "
             "evaluate-now, omission permits every pair in the finite parsed "
             "request; required for long-running-scheduler."
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Directory for scheduler.log and slurm job output files (default: current directory)"
    )
    p.add_argument(
        "--hf-cache-dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Local directory for caching Hugging Face dataset files. "
             "Overrides HF_HUB_CACHE / HF_HOME. Defaults to the standard HF cache."
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable debug mode: capture raw templated prompts and structured "
             "generation metadata (reasoning, tool calls) in output files"
    )
    p.add_argument(
        "--instance-id",
        type=_parse_instance_id,
        default=argparse.SUPPRESS,
        help="Unique 8-char lowercase hex ID for this scheduler instance. "
             "Auto-generated if not provided. Use to run multiple schedulers "
             "on the same Slurm account without interference."
    )
    p.add_argument(
        "--external-total-deadline-seconds",
        type=_parse_positive_finite_seconds,
        default=argparse.SUPPRESS,
        help="Override total_deadline_seconds for every external candidate model. "
             "Must exceed 7200 seconds; omitted values preserve each "
             "model config or its default."
    )


def main():
    parser = argparse.ArgumentParser(
        prog="program.py",
        description="Program with multiple command groups",
        allow_abbrev=False,
    )

    # Global flags on the top-level parser (before the subcommand)
    _add_global_args(parser)

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # ---- long-running-scheduler ----
    scheduler_parser = subparsers.add_parser(
        "long-running-scheduler",
        help="Run the long-running scheduler",
        allow_abbrev=False,
    )
    # Global flags repeated here so they may also appear after the subcommand
    _add_global_args(scheduler_parser)
    scheduler_parser.add_argument(
        "--model-registration-path",
        required=True,
        type=str,
        help="Path to model registration"
    )
    scheduler_parser.add_argument(
        "--data-registration-path",
        required=True,
        type=str,
        help="Path to data registration"
    )
    scheduler_parser.add_argument(
        "--logprobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect per-token log probabilities for each generation (default: off)"
    )

    # ---- evaluate-now ----
    evaluate_parser = subparsers.add_parser(
        "evaluate-now",
        help="Run evaluation immediately",
        allow_abbrev=False,
    )
    # Global flags repeated here so they may also appear after the subcommand
    _add_global_args(evaluate_parser)
    evaluate_parser.add_argument(
        "--model-paths",
        "--model-path",  # deprecated alias kept for backwards compatibility
        dest="model_paths",
        nargs="+",
        type=str,
        help="One or more candidate model config paths. Without --eval-paths, produces the "
             "cartesian product with --data-paths; with --eval-paths, matched by tag."
    )
    evaluate_parser.add_argument(
        "--data-paths",
        nargs="+",
        type=str,
        help="One or more candidate data config paths. Without --eval-paths, produces the "
             "cartesian product with --model-paths; with --eval-paths, matched by tag."
    )
    evaluate_parser.add_argument(
        "--eval-paths",
        nargs="+",
        type=str,
        help="One or more eval config paths. When supplied, run only tagged pairs selected "
             "from the --model-paths and --data-paths candidate pools."
    )
    evaluate_parser.add_argument(
        "--ignore-grader-errors",
        action="store_true",
        help="Generation and grading errors are logged and written to their respective output files instead of causing an exception"
    )
    evaluate_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing output files and re-run all evaluations from scratch"
    )
    evaluate_parser.add_argument(
        "--logprobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect per-token log probabilities for each generation (default: off)"
    )
    evaluate_parser.add_argument(
        "--slurm-partition",
        type=str,
        default=None,
        help="Slurm partition to submit jobs to (default: the cluster's default partition)"
    )
    evaluate_parser.add_argument(
        "--no-slurm",
        action="store_true",
        default=False,
        help="Disable Slurm entirely; only external_model configs are allowed. "
             "Incompatible with --max-generation-jobs."
    )
    evaluate_parser.add_argument(
        "--salt-cache",
        action="store_true",
        default=False,
        help="Send a unique per-request cache salt to OpenAI-compatible providers "
             "to force cache misses or isolate cache validation runs."
    )
    evaluate_parser.add_argument(
        "--evaluation-request-path",
        type=str,
        help="Canonical request-native evaluation input; mutually exclusive with YAML inputs."
    )
    evaluate_parser.add_argument(
        "--terminal-result-path",
        type=str,
        help="Exclusive write-last result for request-native or ordinary "
             "eval-config YAML evaluation."
    )

    args = parser.parse_args()
    if args.command == "evaluate-now" and args.salt_cache and not args.force:
        parser.error(
            "--salt-cache requires --force to avoid mixing existing unsalted "
            "generations with salted requests"
        )
    if args.command == "evaluate-now":
        request_mode = args.evaluation_request_path is not None
        terminal_mode = args.terminal_result_path is not None
        if request_mode and not terminal_mode:
            parser.error(
                "--evaluation-request-path requires --terminal-result-path"
            )
        if request_mode and any(
            value is not None
            for value in (args.model_paths, args.data_paths, args.eval_paths)
        ):
            parser.error(
                "--evaluation-request-path is mutually exclusive with "
                "--model-paths, --data-paths, and --eval-paths"
            )
        if not request_mode and (
            args.model_paths is None or args.data_paths is None
        ):
            parser.error(
                "legacy evaluate-now requires --model-paths and --data-paths"
            )
        if terminal_mode and not request_mode and not args.eval_paths:
            parser.error(
                "ordinary-YAML --terminal-result-path requires --eval-paths"
            )

    # Global flags use SUPPRESS so unprovided flags are absent from the namespace.
    # getattr with a sentinel lets us distinguish "not provided" from any real value.
    max_generation_jobs = getattr(args, "max_generation_jobs", None)
    max_grading_parallelism = getattr(args, "max_grading_parallelism", None)
    log_dir = Path(getattr(args, "log_dir", None) or ".")
    hf_cache_dir = getattr(args, "hf_cache_dir", None)
    debug = getattr(args, "debug", False)
    instance_id = getattr(args, "instance_id", None)
    external_total_deadline_seconds = getattr(
        args,
        "external_total_deadline_seconds",
        None,
    )

    no_slurm = getattr(args, "no_slurm", False)

    if no_slurm and max_generation_jobs is not None:
        parser.error("--no-slurm is incompatible with --max-generation-jobs")
    if args.command == "long-running-scheduler":
        if max_generation_jobs is None:
            parser.error(
                "--max-generation-jobs is required for long-running-scheduler"
            )
        if max_grading_parallelism is None:
            parser.error(
                "--max-grading-parallelism is required for long-running-scheduler"
            )

    log_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch logic
    if args.command == "long-running-scheduler":
        run_scheduler(
            model_registration_path=args.model_registration_path,
            data_registration_path=args.data_registration_path,
            max_generation_jobs=max_generation_jobs,
            max_grading_parallelism=max_grading_parallelism,
            log_dir=log_dir,
            hf_cache_dir=hf_cache_dir,
            force_logprobs=args.logprobs,
            debug=debug,
            instance_id=instance_id,
            external_total_deadline_seconds=external_total_deadline_seconds,
        )
    elif args.command == "evaluate-now":
        request_kwargs = {}
        if args.terminal_result_path is not None:
            request_kwargs = {
                "actual_runner_entrypoint": sys.argv[0],
                "terminal_result_path": args.terminal_result_path,
            }
            if args.evaluation_request_path is not None:
                request_kwargs["evaluation_request_path"] = (
                    args.evaluation_request_path
                )
        run_evaluation(
            model_paths=args.model_paths,
            data_paths=args.data_paths,
            eval_paths=getattr(args, "eval_paths", None),
            max_generation_jobs=max_generation_jobs,
            max_grading_parallelism=max_grading_parallelism,
            ignore_errors=args.ignore_grader_errors,
            force=args.force,
            log_dir=log_dir,
            hf_cache_dir=hf_cache_dir,
            force_logprobs=args.logprobs,
            debug=debug,
            slurm_partition=args.slurm_partition,
            instance_id=instance_id,
            no_slurm=no_slurm,
            salt_cache=args.salt_cache,
            external_total_deadline_seconds=external_total_deadline_seconds,
            **request_kwargs,
        )


def _configure_logging(log_dir):
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if os.environ.get("STDOUT_LOGGING", "false") == "false":
        handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_dir / "scheduler.log",
            when="midnight",
            encoding="utf-8",
        )
        # Rotate to scheduler_YYYYMMDD.log instead of scheduler.log.YYYY-MM-DD
        def _namer(name):
            p = Path(name)
            return str(p.parent / f"scheduler_{p.suffix[1:]}.log")
        handler.namer = _namer
        handler.suffix = "%Y%m%d"
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(handler)


def run_scheduler(model_registration_path, data_registration_path, max_generation_jobs, max_grading_parallelism, log_dir, hf_cache_dir=None, force_logprobs=False, debug=False, instance_id=None, external_total_deadline_seconds=None):
    import signal
    _configure_logging(log_dir)
    _scheduler_logger = logging.getLogger("cli")

    def _sigterm_handler(signum, frame):
        _scheduler_logger.warning(
            "Received SIGTERM (signal %d) — process is being killed. "
            "If exit code is 137, the process was OOM-killed after SIGTERM.",
            signum,
        )
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    async def loop():
        scheduler = Scheduler(model_registration_path, data_registration_path, max_generation_jobs, max_grading_parallelism, log_dir=log_dir, hf_cache_dir=hf_cache_dir, force_logprobs=force_logprobs, debug=debug, instance_id=instance_id, external_total_deadline_seconds=external_total_deadline_seconds)
        await scheduler.loop()

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        pass
    except Exception:
        _scheduler_logger.exception("Scheduler exited with unhandled exception")
        raise


def run_evaluation(model_paths, data_paths, max_generation_jobs, max_grading_parallelism, ignore_errors, force, log_dir, eval_paths=None, hf_cache_dir=None, force_logprobs=True, debug=False, slurm_partition=None, instance_id=None, no_slurm=False, salt_cache=False, external_total_deadline_seconds=None, evaluation_request_path=None, terminal_result_path=None, actual_runner_entrypoint=None):
    if evaluation_request_path is not None and terminal_result_path is None:
        raise ValueError(
            "evaluation_request_path requires terminal_result_path"
        )
    if evaluation_request_path is not None and any(
        value for value in (model_paths, data_paths, eval_paths)
    ):
        raise ValueError(
            "evaluation_request_path is mutually exclusive with legacy YAML inputs"
        )
    if evaluation_request_path is not None and no_slurm:
        raise ValueError(
            "evaluation-request schema 1.0 requires its bound Slurm serving runtime"
        )
    if evaluation_request_path is None and (not model_paths or not data_paths):
        raise ValueError("legacy evaluate-now requires model_paths and data_paths")
    if (
        terminal_result_path is not None
        and evaluation_request_path is None
        and not eval_paths
    ):
        raise ValueError(
            "ordinary-YAML terminal_result_path requires eval_paths"
        )
    runner_entrypoint = (
        actual_runner_entrypoint
        if actual_runner_entrypoint is not None
        else sys.argv[0]
    )
    evaluation_request = (
        EvaluationRequest.load(
            request_path=evaluation_request_path,
            terminal_result_path=terminal_result_path,
            actual_runner_entrypoint=runner_entrypoint,
        )
        if evaluation_request_path is not None
        else None
    )
    yaml_terminal_invocation = (
        YamlTerminalInvocation.bind(
            model_paths=model_paths,
            data_paths=data_paths,
            eval_paths=eval_paths,
            terminal_result_path=terminal_result_path,
            actual_runner_entrypoint=runner_entrypoint,
        )
        if terminal_result_path is not None and evaluation_request is None
        else None
    )
    previous_ignore_errors = os.environ.get("EVAL360_IGNORE_ERRORS")
    os.environ["EVAL360_IN_MEMORY_DB"] = "true"
    os.environ["STDOUT_LOGGING"] = "false"
    os.environ["EVAL360_IGNORE_ERRORS"] = "true" if ignore_errors else "false"
    _configure_logging(log_dir)

    async def loop():
        scheduler = Scheduler(None, None, max_generation_jobs, max_grading_parallelism, log_dir=log_dir, hf_cache_dir=hf_cache_dir, force_logprobs=force_logprobs, debug=debug, slurm_partition=slurm_partition, instance_id=instance_id, no_slurm=no_slurm, salt_cache=salt_cache, external_total_deadline_seconds=external_total_deadline_seconds)
        if evaluation_request is None:
            return await scheduler.run_evaluate_now(
                model_paths,
                data_paths,
                force=force,
                eval_paths=eval_paths,
                yaml_terminal_invocation=yaml_terminal_invocation,
            )
        return await scheduler.run_evaluate_now(
            None,
            None,
            force=force,
            evaluation_request=evaluation_request,
        )
    try:
        scheduler_result = asyncio.run(loop())
    except BaseException:
        if previous_ignore_errors is None:
            os.environ.pop("EVAL360_IGNORE_ERRORS", None)
        else:
            os.environ["EVAL360_IGNORE_ERRORS"] = previous_ignore_errors
        raise
    if evaluation_request is not None:
        if scheduler_result is None:
            raise RuntimeError("scheduler returned no terminal-result evidence")
        return publish_terminal_result(evaluation_request, scheduler_result)
    if yaml_terminal_invocation is not None:
        if scheduler_result is None:
            raise RuntimeError("scheduler returned no YAML terminal-result evidence")
        return publish_yaml_terminal_result(
            yaml_terminal_invocation,
            scheduler_result,
        )
    return None


if __name__ == "__main__":
    main()
