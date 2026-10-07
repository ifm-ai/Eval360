"""
Integration tests for scheduler/slurm/imported_dataset_script.sh.

The script is run as a real bash subprocess with fake PATH commands substituted
for vllm (python), curl, and python3, so no cluster or GPU is needed.
"""
import base64
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scheduler/slurm/imported_dataset_script.sh"
CONTAINER_SCRIPT = Path(__file__).parent.parent / "scheduler/slurm/imported_dataset_script_container.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _write_exe(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _setup_fake_bin(bin_dir: Path, curl_exit: int = 0) -> None:
    """Write minimal fake executables into bin_dir."""
    # Fake python: used for the background VLLM heredoc.
    _write_exe(bin_dir / "python", textwrap.dedent("""\
        #!/bin/bash
        if [[ -n "${VLLM_ENV_CAPTURE_FILE:-}" ]]; then
            {
                echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-}"
                echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-}"
                echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-}"
            } > "$VLLM_ENV_CAPTURE_FILE"
        fi
        exit 0
    """))

    # Fake bootstrap python: handles `-m venv PATH` by creating the dir
    # structure the script expects (bin/activate). Ignores other invocations.
    _write_exe(bin_dir / "python3", textwrap.dedent("""\
        #!/bin/bash
        if [[ "$1" == "-m" && "$2" == "venv" ]]; then
            mkdir -p "$3/bin"
            printf 'export VIRTUAL_ENV="%s"\\n' "$3" > "$3/bin/activate"
        fi
    """))

    # Fake curl: controls whether VLLM appears healthy.
    _write_exe(bin_dir / "curl", f"#!/bin/bash\nexit {curl_exit}\n")


def _make_env(
    tmp_path: Path,
    curl_exit: int = 0,
    setup_script: str = "echo setup_ran",
    benchmark_script: str = "echo benchmark_ran",
    max_time_to_deploy: int = 30,
) -> tuple[dict, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _setup_fake_bin(bin_dir, curl_exit=curl_exit)

    # Minimal activate for the VLLM serving venv (just sets VIRTUAL_ENV).
    venv_activate = tmp_path / "vllm-activate"
    venv_activate.write_text(f"export VIRTUAL_ENV='{tmp_path}/vllm-venv'\n")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["venv_path"] = str(venv_activate)
    env["bootstrap_python"] = str(bin_dir / "python3")
    env["model_path"] = "org/model"
    env["vllm_args"] = _b64(json.dumps([]))
    env["runner_name"] = "test-runner"
    env["repo_root"] = str(repo_root)
    env["max_time_to_deploy"] = str(max_time_to_deploy)
    env["setup_script_b64"] = _b64(setup_script)
    env["benchmark_script_b64"] = _b64(benchmark_script)
    env["output_dir"] = str(output_dir)
    return env, repo_root


def _run(env: dict, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_script(script: Path, env: dict, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _sentinel(repo_root: Path) -> Path:
    return repo_root / ".eval360/envs/test-runner/.setup_complete"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImportedDatasetScript:

    def test_successful_run_exits_zero(self, tmp_path):
        env, _ = _make_env(tmp_path)
        assert _run(env).returncode == 0

    def test_accepts_venv_root_path(self, tmp_path):
        env, repo_root = _make_env(tmp_path)
        serving_venv = tmp_path / "serving-venv"
        (serving_venv / "bin").mkdir(parents=True)
        (serving_venv / "bin" / "activate").write_text(f"export VIRTUAL_ENV='{serving_venv}'\n")
        env["venv_path"] = str(serving_venv)

        result = _run(env)
        assert result.returncode == 0

    def test_missing_venv_activation_script_fails_fast(self, tmp_path):
        env, _ = _make_env(tmp_path)
        env["venv_path"] = str(tmp_path / "missing-venv")

        result = _run(env)
        assert result.returncode != 0
        assert "Missing venv activation script" in result.stderr

    def test_sentinel_created_on_first_run(self, tmp_path):
        env, repo_root = _make_env(tmp_path)
        _run(env)
        assert _sentinel(repo_root).exists()

    def test_setup_script_executes(self, tmp_path):
        env, _ = _make_env(tmp_path, setup_script="echo setup_was_here")
        result = _run(env)
        assert "setup_was_here" in result.stdout

    def test_benchmark_script_executes(self, tmp_path):
        env, _ = _make_env(tmp_path, benchmark_script="echo benchmark_was_here")
        result = _run(env)
        assert "benchmark_was_here" in result.stdout

    def test_setup_skipped_when_sentinel_exists(self, tmp_path):
        env, repo_root = _make_env(tmp_path, setup_script="echo setup_was_here")
        # Pre-create sentinel and a valid venv activate.
        venv = repo_root / ".eval360/envs/test-runner"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "activate").write_text(f"export VIRTUAL_ENV='{venv}'\n")
        (venv / ".setup_complete").touch()

        result = _run(env)
        assert result.returncode == 0
        assert "setup_was_here" not in result.stdout

    def test_broken_venv_deleted_when_sentinel_missing(self, tmp_path):
        env, repo_root = _make_env(tmp_path)
        # Create a partial venv dir with no sentinel.
        broken_marker = repo_root / ".eval360/envs/test-runner/BROKEN"
        broken_marker.parent.mkdir(parents=True)
        broken_marker.touch()

        _run(env)
        assert not broken_marker.exists(), "broken venv dir should have been deleted"
        assert _sentinel(repo_root).exists()

    def test_vllm_health_timeout_exits_nonzero(self, tmp_path):
        env, _ = _make_env(tmp_path, curl_exit=1, max_time_to_deploy=0)
        result = _run(env)
        assert result.returncode != 0
        assert "did not become healthy" in result.stderr

    def test_vllm_compile_caches_are_isolated_per_job(self, tmp_path):
        env, _ = _make_env(tmp_path)
        slurm_tmpdir = tmp_path / "slurm-tmp"
        capture_file = tmp_path / "vllm-env.txt"
        env["SLURM_JOB_ID"] = "12345"
        env["SLURM_TMPDIR"] = str(slurm_tmpdir)
        env["VLLM_ENV_CAPTURE_FILE"] = str(capture_file)

        result = _run(env)
        assert result.returncode == 0

        expected_root = slurm_tmpdir / f"eval360-vllm-{os.getuid()}-12345"
        captured = capture_file.read_text()
        assert f"VLLM_CACHE_ROOT={expected_root / 'vllm'}" in captured
        assert f"TORCHINDUCTOR_CACHE_DIR={expected_root / 'torchinductor'}" in captured
        assert f"TRITON_CACHE_DIR={expected_root / 'triton'}" in captured

    def test_stale_shared_tmp_cache_parent_is_ignored(self, tmp_path):
        env, _ = _make_env(tmp_path)
        slurm_tmpdir = tmp_path / "slurm-tmp"
        stale_parent = slurm_tmpdir / "eval360-vllm"
        stale_parent.mkdir(parents=True)
        stale_parent.chmod(0o555)
        env["SLURM_JOB_ID"] = "12345"
        env["SLURM_TMPDIR"] = str(slurm_tmpdir)

        try:
            result = _run(env)
        finally:
            stale_parent.chmod(0o755)

        assert result.returncode == 0

    def test_setup_complete_job_sentinel_written_after_setup(self, tmp_path):
        env, _ = _make_env(tmp_path)
        output_dir = Path(env["output_dir"])
        _run(env)
        assert (output_dir / ".setup_complete_job").exists()

    def test_setup_complete_job_sentinel_not_written_on_setup_failure(self, tmp_path):
        env, _ = _make_env(tmp_path, setup_script="exit 1")
        output_dir = Path(env["output_dir"])
        result = _run(env)
        assert result.returncode != 0
        assert not (output_dir / ".setup_complete_job").exists()

    def test_benchmark_failure_exits_nonzero(self, tmp_path):
        env, _ = _make_env(tmp_path, benchmark_script="exit 42")
        result = _run(env)
        assert result.returncode != 0

    def test_container_script_does_not_wait_on_setup_pid(self):
        script = CONTAINER_SCRIPT.read_text()
        assert "SETUP_PID" not in script
        assert "wait $SETUP_PID" not in script

    def test_container_script_runs_benchmark(self, tmp_path):
        env, _ = _make_env(
            tmp_path,
            setup_script="echo container_setup_was_here",
            benchmark_script="echo container_benchmark_was_here",
        )

        result = _run_script(CONTAINER_SCRIPT, env)

        assert result.returncode == 0
        assert "container_setup_was_here" in result.stdout
        assert "container_benchmark_was_here" in result.stdout
