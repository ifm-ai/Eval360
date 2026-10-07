"""
Tests for _configure_logging and --no-slurm flag in scheduler/cli.py.

Specifically guards against the stdout-leak regression: module-level
logging.basicConfig() calls (one per scheduler module) run at import
time and attach a StreamHandler to the root logger before
_configure_logging() is called.  If _configure_logging() doesn't clear
existing handlers first, logs go to BOTH stdout and the log file.
"""
import logging
import logging.handlers
import os
import signal
from unittest.mock import patch, MagicMock

import pytest

from scheduler.cli import _configure_logging


@pytest.fixture(autouse=True)
def restore_root_handlers():
    """Save and restore root logger state around every test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)


class TestConfigureLogging:
    def test_no_stdout_leak_after_basicconfig(self, tmp_path):
        """Regression test: if root.handlers.clear() is removed from
        _configure_logging(), this test fails because a StreamHandler
        added by basicConfig() would remain alongside the file handler.
        """
        root = logging.getLogger()

        # Simulate what every scheduler module does at import time
        logging.basicConfig(level=logging.INFO)
        assert any(type(h) is logging.StreamHandler for h in root.handlers), \
            "precondition: basicConfig() should have added a StreamHandler"

        with patch.dict(os.environ, {"STDOUT_LOGGING": "false"}):
            _configure_logging(tmp_path)

        stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
        assert stream_handlers == [], \
            "StreamHandler leaked to root logger — logs would go to stdout AND the file"

    def test_single_file_handler(self, tmp_path):
        with patch.dict(os.environ, {"STDOUT_LOGGING": "false"}):
            _configure_logging(tmp_path)

        assert len(logging.getLogger().handlers) == 1
        assert isinstance(logging.getLogger().handlers[0],
                          logging.handlers.TimedRotatingFileHandler)

    def test_stdout_logging_env_var(self, tmp_path):
        with patch.dict(os.environ, {"STDOUT_LOGGING": "true"}):
            _configure_logging(tmp_path)

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert type(root.handlers[0]) is logging.StreamHandler

    def test_log_file_created(self, tmp_path):
        with patch.dict(os.environ, {"STDOUT_LOGGING": "false"}):
            _configure_logging(tmp_path)

        logging.getLogger("test").info("hello")
        assert (tmp_path / "scheduler.log").exists()

    def test_rotated_log_namer_uses_scheduler_date_format(self, tmp_path):
        """What: verifies rotated scheduler logs are named with the scheduler_YYYYMMDD.log pattern.
        Executes: _configure_logging() and the TimedRotatingFileHandler.namer callback it installs.
        Why: operators and log collection jobs depend on stable dated filenames after rotation.
        """
        with patch.dict(os.environ, {"STDOUT_LOGGING": "false"}):
            _configure_logging(tmp_path)

        handler = logging.getLogger().handlers[0]
        rotated_name = handler.namer(str(tmp_path / "scheduler.log.20260529"))

        assert rotated_name == str(tmp_path / "scheduler_20260529.log")


class TestMainDispatch:
    """What: groups tests for argument parsing and dispatch in scheduler.cli.main.
    Executes: main() through the long-running-scheduler and evaluate-now subcommands.
    Why: protects the CLI contract for global flag placement and dispatch kwargs.
    """

    @pytest.mark.parametrize(
        "instance_id",
        [
            "d5v3c016",  # contains a non-hex character
            "ABCD1234",  # uppercase is not valid in Slurm job discovery
            "",
            "abc1234",
            "abc123456",
            "abc-1234",
            " abc1234",
        ],
    )
    def test_rejects_invalid_instance_id(self, instance_id, monkeypatch, capsys):
        import sys
        from scheduler.cli import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval360",
                "--max-generation-jobs",
                "1",
                "--max-grading-parallelism",
                "1",
                "--instance-id",
                instance_id,
                "evaluate-now",
                "--model-paths",
                "model.yaml",
                "--data-paths",
                "data.yaml",
            ],
        )

        with pytest.raises(SystemExit, match="2"):
            main()

        assert (
            "must be exactly 8 lowercase hexadecimal characters"
            in capsys.readouterr().err
        )

    def test_long_running_scheduler_forwards_global_args_after_subcommand(self, tmp_path, monkeypatch):
        """What: verifies global flags placed after the subcommand should reach run_scheduler.
        Executes: main() argument parsing through the long-running-scheduler subcommand.
        Why: covers the supported ordering where users place shared scheduler flags after the action name.
        """
        import sys
        from scheduler.cli import main

        log_dir = tmp_path / "logs"
        hf_cache_dir = tmp_path / "hf"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval360",
                "long-running-scheduler",
                "--model-registration-path",
                "models.yaml",
                "--data-registration-path",
                "data.yaml",
                "--max-generation-jobs",
                "3",
                "--max-grading-parallelism",
                "4",
                "--log-dir",
                str(log_dir),
                "--hf-cache-dir",
                str(hf_cache_dir),
                "--debug",
                "--instance-id",
                "cafebabe",
                "--external-total-deadline-seconds",
                "9001",
                "--logprobs",
            ],
        )

        with patch("scheduler.cli.run_scheduler") as run_scheduler:
            main()

        run_scheduler.assert_called_once()
        assert run_scheduler.call_args.kwargs == {
            "model_registration_path": "models.yaml",
            "data_registration_path": "data.yaml",
            "max_generation_jobs": 3,
            "max_grading_parallelism": 4,
            "log_dir": log_dir,
            "hf_cache_dir": str(hf_cache_dir),
            "force_logprobs": True,
            "debug": True,
            "instance_id": "cafebabe",
            "external_total_deadline_seconds": 9001,
        }

    def test_evaluate_now_forwards_global_args_before_subcommand(self, tmp_path, monkeypatch):
        """What: verifies global flags placed before the subcommand should reach run_evaluation.
        Executes: main() argument parsing through the evaluate-now subcommand.
        Why: covers the top-level flag ordering commonly used by shell aliases and wrapper scripts.
        """
        import sys
        from scheduler.cli import main

        log_dir = tmp_path / "eval-logs"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval360",
                "--max-generation-jobs",
                "2",
                "--max-grading-parallelism",
                "5",
                "--log-dir",
                str(log_dir),
                "--hf-cache-dir",
                "/tmp/hf-cache",
                "--debug",
                "--instance-id",
                "1234abcd",
                "evaluate-now",
                "--model-paths",
                "m1.yaml",
                "m2.yaml",
                "--data-paths",
                "d1.yaml",
                "--ignore-grader-errors",
                "--force",
                "--no-logprobs",
                "--slurm-partition",
                "custom-partition",
            ],
        )

        with patch("scheduler.cli.run_evaluation") as run_evaluation:
            main()

        run_evaluation.assert_called_once()
        assert run_evaluation.call_args.kwargs == {
            "model_paths": ["m1.yaml", "m2.yaml"],
            "data_paths": ["d1.yaml"],
            "eval_paths": None,
            "max_generation_jobs": 2,
            "max_grading_parallelism": 5,
            "ignore_errors": True,
            "force": True,
            "log_dir": log_dir,
            "hf_cache_dir": "/tmp/hf-cache",
            "force_logprobs": False,
            "debug": True,
            "slurm_partition": "custom-partition",
            "instance_id": "1234abcd",
            "no_slurm": False,
            "salt_cache": False,
            "external_total_deadline_seconds": None,
        }

    def test_evaluate_now_forwards_external_total_deadline_override(
        self,
        tmp_path,
        monkeypatch,
    ):
        import sys
        from scheduler.cli import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval360",
                "--max-grading-parallelism",
                "1",
                "--external-total-deadline-seconds",
                "9000",
                "evaluate-now",
                "--no-slurm",
                "--model-paths",
                "model.yaml",
                "--data-paths",
                "data.yaml",
            ],
        )

        with patch("scheduler.cli.run_evaluation") as run_evaluation:
            main()

        assert (
            run_evaluation.call_args.kwargs["external_total_deadline_seconds"]
            == 9000
        )

    @pytest.mark.parametrize(
        "value",
        ["0", "-1", "7200", "inf", "nan", "invalid"],
    )
    def test_external_total_deadline_rejects_invalid_values(
        self,
        value,
        monkeypatch,
        capsys,
    ):
        import sys
        from scheduler.cli import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval360",
                "--max-grading-parallelism",
                "1",
                "--external-total-deadline-seconds",
                value,
                "evaluate-now",
                "--no-slurm",
                "--model-paths",
                "model.yaml",
                "--data-paths",
                "data.yaml",
            ],
        )

        with pytest.raises(SystemExit, match="2"):
            main()

        assert "must exceed 7200 seconds" in capsys.readouterr().err

    def test_evaluate_now_with_eval_paths_forwards_candidate_pools(self, tmp_path, monkeypatch):
        """What: verifies eval-config mode still forwards model/data candidate pools.
        Executes: main() argument parsing through the evaluate-now subcommand with --eval-paths.
        Why: --eval-paths filters tagged pairs from --model-paths/--data-paths; it must not
        make data_paths disappear from the scheduler call.
        """
        import sys
        from scheduler.cli import main

        log_dir = tmp_path / "eval-logs"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval360",
                "--max-generation-jobs",
                "2",
                "--max-grading-parallelism",
                "5",
                "--log-dir",
                str(log_dir),
                "evaluate-now",
                "--model-paths",
                "m1.yaml",
                "m2.yaml",
                "--data-paths",
                "d1.yaml",
                "d2.yaml",
                "--eval-paths",
                "eval.yaml",
            ],
        )

        with patch("scheduler.cli.run_evaluation") as run_evaluation:
            main()

        run_evaluation.assert_called_once()
        assert run_evaluation.call_args.kwargs["model_paths"] == ["m1.yaml", "m2.yaml"]
        assert run_evaluation.call_args.kwargs["data_paths"] == ["d1.yaml", "d2.yaml"]
        assert run_evaluation.call_args.kwargs["eval_paths"] == ["eval.yaml"]

    def test_evaluate_now_allows_omitted_parallelism_flags(self, monkeypatch):
        """One-shot limits are resolved only after its finite inputs parse."""
        import sys

        from scheduler.cli import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval360",
                "evaluate-now",
                "--model-paths",
                "model.yaml",
                "--data-paths",
                "data.yaml",
            ],
        )

        with patch("scheduler.cli.run_evaluation") as run_evaluation:
            main()

        assert run_evaluation.call_args.kwargs["max_generation_jobs"] is None
        assert run_evaluation.call_args.kwargs["max_grading_parallelism"] is None

    @pytest.mark.parametrize(
        "provided_limit",
        [
            ["--max-generation-jobs", "1"],
            ["--max-grading-parallelism", "1"],
        ],
    )
    def test_long_running_scheduler_still_requires_both_parallelism_flags(
        self, monkeypatch, provided_limit
    ):
        """Only finite one-shot requests can derive omitted limits."""
        import sys

        from scheduler.cli import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval360",
                "long-running-scheduler",
                "--model-registration-path",
                "/models",
                "--data-registration-path",
                "/data",
                *provided_limit,
            ],
        )

        with (
            patch("scheduler.cli.run_scheduler") as run_scheduler,
            pytest.raises(SystemExit),
        ):
            main()

        run_scheduler.assert_not_called()


class TestRunSchedulerErrorLogging:
    """Tests that run_scheduler logs errors instead of dying silently."""

    def test_sigterm_handler_is_registered(self, tmp_path):
        """run_scheduler must register a SIGTERM signal handler."""
        from scheduler.cli import run_scheduler

        registered_signals = {}

        def fake_signal(sig, handler):
            registered_signals[sig] = handler

        fake_scheduler = MagicMock()
        fake_scheduler.return_value.loop = MagicMock(side_effect=SystemExit(0))

        with (
            patch("scheduler.cli.Scheduler", fake_scheduler),
            patch("signal.signal", side_effect=fake_signal),
            patch.dict(os.environ, {"STDOUT_LOGGING": "true"}),
        ):
            try:
                run_scheduler(
                    model_registration_path="/fake/models",
                    data_registration_path="/fake/data",
                    max_generation_jobs=1,
                    max_grading_parallelism=1,
                    log_dir=str(tmp_path),
                    external_total_deadline_seconds=9001,
                )
            except SystemExit:
                pass

        assert signal.SIGTERM in registered_signals
        assert (
            fake_scheduler.call_args.kwargs[
                "external_total_deadline_seconds"
            ]
            == 9001
        )

    def test_unhandled_exception_is_logged(self, tmp_path):
        """An unhandled exception from the scheduler loop must be logged before propagating."""
        from scheduler.cli import run_scheduler

        boom = RuntimeError("something went wrong")

        async def bad_loop():
            raise boom

        fake_scheduler = MagicMock()
        fake_scheduler.return_value.loop = bad_loop

        # Patch _configure_logging to a no-op so our capturing handler isn't cleared,
        # and patch the cli logger's exception method directly to record calls.
        logged_exceptions = []
        cli_logger = logging.getLogger("cli")
        original_exception = cli_logger.exception

        def capture_exception(msg, *args, **kwargs):
            logged_exceptions.append(msg)
            original_exception(msg, *args, **kwargs)

        with (
            patch("scheduler.cli.Scheduler", fake_scheduler),
            patch("scheduler.cli._configure_logging"),
            patch.object(cli_logger, "exception", side_effect=capture_exception),
            patch.dict(os.environ, {"STDOUT_LOGGING": "true"}),
        ):
            with pytest.raises(RuntimeError, match="something went wrong"):
                run_scheduler(
                    model_registration_path="/fake/models",
                    data_registration_path="/fake/data",
                    max_generation_jobs=1,
                    max_grading_parallelism=1,
                    log_dir=str(tmp_path),
                )

        assert logged_exceptions, "expected logger.exception() to be called for the unhandled exception"

    def test_keyboard_interrupt_is_swallowed(self, tmp_path):
        """What: verifies KeyboardInterrupt from asyncio.run stops the scheduler cleanly.
        Executes: run_scheduler() with asyncio.run patched to raise KeyboardInterrupt.
        Why: covers the interactive shutdown path so Ctrl+C does not surface as an unhandled scheduler error.
        """
        from scheduler.cli import run_scheduler

        def interrupt_and_close(coro):
            coro.close()
            raise KeyboardInterrupt

        with (
            patch("scheduler.cli.asyncio.run", side_effect=interrupt_and_close),
            patch("scheduler.cli._configure_logging"),
            patch("signal.signal"),
        ):
            run_scheduler(
                model_registration_path="/fake/models",
                data_registration_path="/fake/data",
                max_generation_jobs=1,
                max_grading_parallelism=1,
                log_dir=tmp_path,
            )


class TestRunEvaluation:
    """What: groups tests for immediate evaluation environment setup and shutdown behavior.
    Executes: scheduler.cli.run_evaluation() and the Scheduler.run_evaluate_now path.
    Why: protects the immediate-evaluation entry point used by evaluate-now CLI runs.
    """

    def test_run_evaluation_sets_env_and_forwards_scheduler_arguments(self, tmp_path, monkeypatch):
        """What: verifies run_evaluation should configure env vars and call Scheduler.run_evaluate_now.
        Executes: run_evaluation(), _configure_logging(), Scheduler construction, and run_evaluate_now().
        Why: evaluate-now relies on this no-database path to preserve parsed CLI options through Scheduler startup.
        """
        from scheduler.cli import run_evaluation

        created = []

        class FakeScheduler:
            """Minimal scheduler double that records constructor and run arguments."""

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.run_args = None
                created.append(self)

            async def run_evaluate_now(
                self,
                model_paths,
                data_paths,
                force=False,
                eval_paths=None,
                yaml_terminal_invocation=None,
            ):
                self.run_args = (model_paths, data_paths, force)
                assert yaml_terminal_invocation is None

        for key in ("EVAL360_IN_MEMORY_DB", "STDOUT_LOGGING", "EVAL360_IGNORE_ERRORS"):
            monkeypatch.delenv(key, raising=False)

        with (
            patch("scheduler.cli.Scheduler", FakeScheduler),
            patch("scheduler.cli._configure_logging") as configure_logging,
        ):
            run_evaluation(
                model_paths=["model.yaml"],
                data_paths=["data.yaml"],
                max_generation_jobs=2,
                max_grading_parallelism=3,
                ignore_errors=True,
                force=True,
                log_dir=tmp_path,
                hf_cache_dir="/tmp/hf",
                force_logprobs=False,
                debug=True,
                slurm_partition="custom-partition",
                instance_id="abc12345",
                no_slurm=True,
                external_total_deadline_seconds=9002,
            )

        assert os.environ["EVAL360_IN_MEMORY_DB"] == "true"
        assert os.environ["STDOUT_LOGGING"] == "false"
        assert os.environ["EVAL360_IGNORE_ERRORS"] == "true"
        configure_logging.assert_called_once_with(tmp_path)
        assert len(created) == 1
        assert created[0].args == (None, None, 2, 3)
        assert created[0].kwargs == {
            "log_dir": tmp_path,
            "hf_cache_dir": "/tmp/hf",
            "force_logprobs": False,
            "debug": True,
            "slurm_partition": "custom-partition",
            "instance_id": "abc12345",
            "no_slurm": True,
            "salt_cache": False,
            "external_total_deadline_seconds": 9002,
        }
        assert created[0].run_args == (["model.yaml"], ["data.yaml"], True)

    def test_run_evaluation_propagates_keyboard_interrupt(self, tmp_path):
        """What: verifies KeyboardInterrupt from immediate evaluation propagates.
        Executes: run_evaluation() with asyncio.run patched to raise KeyboardInterrupt.
        Why: an interrupted evaluate-now invocation must not report a successful process exit.
        """
        from scheduler.cli import run_evaluation

        def interrupt_and_close(coro):
            coro.close()
            raise KeyboardInterrupt

        with (
            patch("scheduler.cli.asyncio.run", side_effect=interrupt_and_close),
            patch("scheduler.cli._configure_logging"),
        ):
            with pytest.raises(KeyboardInterrupt):
                run_evaluation(
                    model_paths=["model.yaml"],
                    data_paths=["data.yaml"],
                    max_generation_jobs=1,
                    max_grading_parallelism=1,
                    ignore_errors=False,
                    force=False,
                    log_dir=tmp_path,
                )


class TestTerminalResultCLI:
    """Terminal evidence has strict request-native and ordinary-YAML routes."""

    def test_request_and_terminal_result_are_forwarded_together(self, monkeypatch):
        from scheduler.cli import main

        monkeypatch.setattr(
            "sys.argv",
            [
                "eval360",
                "--max-generation-jobs",
                "1",
                "--max-grading-parallelism",
                "1",
                "evaluate-now",
                "--evaluation-request-path",
                "request.json",
                "--terminal-result-path",
                "result.json",
            ],
        )
        with patch("scheduler.cli.run_evaluation") as run_evaluation:
            main()

        kwargs = run_evaluation.call_args.kwargs
        assert kwargs["evaluation_request_path"] == "request.json"
        assert kwargs["terminal_result_path"] == "result.json"
        assert kwargs["actual_runner_entrypoint"] == "eval360"
        assert kwargs["model_paths"] is None
        assert kwargs["data_paths"] is None

    def test_yaml_terminal_result_is_forwarded_without_input_manifest(
        self,
        monkeypatch,
    ):
        from scheduler.cli import main

        monkeypatch.setattr(
            "sys.argv",
            [
                "/runtime/bin/eval360",
                "evaluate-now",
                "--model-paths",
                "model.yaml",
                "--data-paths",
                "task.yaml",
                "--eval-paths",
                "eval.yaml",
                "--terminal-result-path",
                "result.json",
            ],
        )
        with patch("scheduler.cli.run_evaluation") as run_evaluation:
            main()

        kwargs = run_evaluation.call_args.kwargs
        assert kwargs["actual_runner_entrypoint"] == "/runtime/bin/eval360"
        assert kwargs["terminal_result_path"] == "result.json"
        assert "evaluation_request_path" not in kwargs
        assert kwargs["model_paths"] == ["model.yaml"]
        assert kwargs["data_paths"] == ["task.yaml"]
        assert kwargs["eval_paths"] == ["eval.yaml"]

    def test_yaml_terminal_result_requires_eval_config_before_dispatch(
        self,
        monkeypatch,
    ):
        from scheduler.cli import main

        monkeypatch.setattr(
            "sys.argv",
            [
                "eval360",
                "evaluate-now",
                "--model-paths",
                "model.yaml",
                "--data-paths",
                "task.yaml",
                "--terminal-result-path",
                "result.json",
            ],
        )
        with (
            patch("scheduler.cli.run_evaluation") as run_evaluation,
            pytest.raises(SystemExit),
        ):
            main()
        run_evaluation.assert_not_called()

    def test_partial_terminal_result_arguments_fail_before_dispatch(
        self,
        monkeypatch,
    ):
        from scheduler.cli import main

        monkeypatch.setattr(
            "sys.argv",
            [
                "eval360",
                "--max-grading-parallelism",
                "1",
                "evaluate-now",
                "--terminal-result-path",
                "result.json",
            ],
        )
        with (
            patch("scheduler.cli.run_evaluation") as run_evaluation,
            pytest.raises(SystemExit),
        ):
            main()
        run_evaluation.assert_not_called()

    def test_request_is_mutually_exclusive_with_legacy_yaml(
        self,
        monkeypatch,
    ):
        from scheduler.cli import main

        monkeypatch.setattr(
            "sys.argv",
            [
                "eval360",
                "--max-generation-jobs",
                "1",
                "--max-grading-parallelism",
                "1",
                "evaluate-now",
                "--evaluation-request-path",
                "request.json",
                "--terminal-result-path",
                "result.json",
                "--model-paths",
                "model.yaml",
                "--data-paths",
                "task.yaml",
            ],
        )
        with (
            patch("scheduler.cli.run_evaluation") as run_evaluation,
            pytest.raises(SystemExit),
        ):
            main()
        run_evaluation.assert_not_called()


# ---------------------------------------------------------------------------
# --no-slurm flag CLI tests
# ---------------------------------------------------------------------------

class TestNoSlurmCLIFlag:
    def _parse(self, args):
        """Parse CLI args and return the namespace (may raise SystemExit on error)."""
        import sys
        from io import StringIO
        from scheduler.cli import main
        # Patch sys.argv and capture SystemExit
        old_argv = sys.argv
        try:
            sys.argv = ["eval360"] + args
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

    def test_no_slurm_without_max_generation_jobs_is_allowed(self, tmp_path):
        """--no-slurm should not require --max-generation-jobs."""
        import sys
        from io import StringIO
        from unittest.mock import patch as _patch
        from scheduler.cli import run_evaluation

        with _patch("scheduler.cli.run_evaluation") as mock_run:
            sys.argv = [
                "eval360",
                "--max-grading-parallelism", "4",
                "evaluate-now",
                "--no-slurm",
                "--model-paths", "/fake/model.yaml",
                "--data-paths", "/fake/data.yaml",
            ]
            try:
                from scheduler.cli import main
                main()
            except SystemExit:
                pass

        assert mock_run.called

    def test_no_slurm_with_max_generation_jobs_raises(self, tmp_path):
        """--no-slurm and --max-generation-jobs together should error."""
        import sys
        with pytest.raises(SystemExit) as exc:
            sys.argv = [
                "eval360",
                "--max-generation-jobs", "2",
                "--max-grading-parallelism", "4",
                "evaluate-now",
                "--no-slurm",
                "--model-paths", "/fake/model.yaml",
                "--data-paths", "/fake/data.yaml",
            ]
            from scheduler.cli import main
            main()
        # Should have exited with an error
        assert exc.value.code != 0

    def test_no_slurm_passed_to_run_evaluation(self, tmp_path):
        """When --no-slurm is set, run_evaluation should receive no_slurm=True."""
        import sys
        from unittest.mock import patch as _patch

        with _patch("scheduler.cli.run_evaluation") as mock_run:
            sys.argv = [
                "eval360",
                "--max-grading-parallelism", "4",
                "evaluate-now",
                "--no-slurm",
                "--model-paths", "/fake/model.yaml",
                "--data-paths", "/fake/data.yaml",
            ]
            try:
                from scheduler.cli import main
                main()
            except SystemExit:
                pass

        call_kwargs = mock_run.call_args[1] if mock_run.call_args else {}
        call_args = mock_run.call_args[0] if mock_run.call_args else ()
        # no_slurm=True should be passed
        assert call_kwargs.get("no_slurm") is True or (len(call_args) > 0 and True in call_args)

    def test_evaluate_now_without_explicit_limits_is_dispatched(self, tmp_path):
        """Slurm one-shot mode derives both limits after parsing inputs."""
        import sys
        with patch("scheduler.cli.run_evaluation") as run_evaluation:
            sys.argv = [
                "eval360",
                "evaluate-now",
                "--model-paths", "/fake/model.yaml",
                "--data-paths", "/fake/data.yaml",
            ]
            from scheduler.cli import main
            main()

        assert run_evaluation.call_args.kwargs["max_generation_jobs"] is None
        assert run_evaluation.call_args.kwargs["max_grading_parallelism"] is None
