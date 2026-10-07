"""Tests for serving environment support: venv, conda, and container (enroot/pyxis)."""

import base64
import json
import os
import pytest
import subprocess
import sys
import textwrap
from pathlib import Path
from pydantic import ValidationError
from unittest.mock import patch, MagicMock, AsyncMock

from scheduler.model import (
    ModelInstance,
    ModelParser,
    ModelSpec,
    ServingSlurmResources,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTAINER_SCRIPT = Path(__file__).parent.parent / "scheduler/slurm/sbatch_script_container.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _base_spec_dict(**overrides):
    d = {
        "remote_model": {"path": "org/model", "base_name": "mymodel"},
        "venv_path": "/venv",
        "max_simultaneous_requests": 4,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/tmp/out",
        "parser_type": "noop",
    }
    d.update(overrides)
    return d


def _container_spec_dict(**overrides):
    d = _base_spec_dict()
    del d["venv_path"]
    d["container_image"] = "nvcr.io/nvidia/pytorch:24.01-py3"
    d.update(overrides)
    return d


def _conda_spec_dict(**overrides):
    d = _base_spec_dict()
    del d["venv_path"]
    d["conda_env"] = "vllm-serving"
    d.update(overrides)
    return d


def _fake_sbatch_exec(captured):
    """Return an async subprocess mock that captures sbatch args."""
    async def fake_exec_async(*args, **kwargs):
        if args[0] == "squeue":
            proc = AsyncMock()
            proc.communicate.return_value = (b"", b"")
            proc.returncode = 0
            return proc
        captured.extend(args)
        proc = AsyncMock()
        proc.communicate.return_value = (b"Submitted batch job 1", b"")
        proc.returncode = 0
        return proc
    return fake_exec_async


def _make_mock_model(*, venv_path=None, conda_env=None,
                     container_image=None, container_mounts=None, **kw):
    """Build a MagicMock model instance with serving env fields set correctly."""
    m = MagicMock()
    m.name = kw.get("name", "m")
    m.api_model_name = kw.get("api_model_name", None)
    m.revision = kw.get("revision", None)
    m.vllm_cli_args = kw.get("vllm_cli_args", [])
    m.path = kw.get("path", "org/model")
    m.allow_long_max_model_len = kw.get("allow_long_max_model_len", True)
    m.vllm_logging_level = kw.get("vllm_logging_level", "WARNING")
    m.venv_path = venv_path
    m.conda_env = conda_env
    m.container_image = container_image
    m.container_mounts = container_mounts
    m.serving_slurm_resources = kw.get(
        "serving_slurm_resources", ServingSlurmResources()
    )
    m.uses_container = container_image is not None
    m.uses_conda = conda_env is not None

    import hashlib
    payload = json.dumps({
        "path": m.path, "revision": m.revision,
        "vllm_cli_args": sorted(m.vllm_cli_args),
        "venv_path": venv_path, "conda_env": conda_env,
        "container_image": container_image,
        "serving_slurm_resources": m.serving_slurm_resources.model_dump(
            mode="json"
        ),
    }, sort_keys=True)
    m.serving_key = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return m


# ---------------------------------------------------------------------------
# ModelSpec: serving environment XOR validation
# ---------------------------------------------------------------------------

class TestModelSpecServingEnv:

    def test_venv_path_only_accepted(self):
        spec = ModelSpec.model_validate(_base_spec_dict())
        assert spec.venv_path == "/venv"
        assert spec.conda_env is None
        assert spec.container_image is None

    def test_conda_env_only_accepted(self):
        spec = ModelSpec.model_validate(_conda_spec_dict())
        assert spec.conda_env == "vllm-serving"
        assert spec.venv_path is None
        assert spec.container_image is None

    def test_container_image_only_accepted(self):
        spec = ModelSpec.model_validate(_container_spec_dict())
        assert spec.container_image == "nvcr.io/nvidia/pytorch:24.01-py3"
        assert spec.venv_path is None
        assert spec.conda_env is None

    def test_no_serving_env_raises(self):
        d = _base_spec_dict()
        del d["venv_path"]
        with pytest.raises(ValidationError, match='Exactly one of'):
            ModelSpec.model_validate(d)

    def test_venv_and_container_raises(self):
        d = _base_spec_dict(container_image="nvcr.io/test:latest")
        with pytest.raises(ValidationError, match='Only one serving environment'):
            ModelSpec.model_validate(d)

    def test_venv_and_conda_raises(self):
        d = _base_spec_dict(conda_env="myenv")
        with pytest.raises(ValidationError, match='Only one serving environment'):
            ModelSpec.model_validate(d)

    def test_conda_and_container_raises(self):
        d = _conda_spec_dict(container_image="nvcr.io/test:latest")
        with pytest.raises(ValidationError, match='Only one serving environment'):
            ModelSpec.model_validate(d)

    def test_all_three_raises(self):
        d = _base_spec_dict(conda_env="myenv", container_image="nvcr.io/test:latest")
        with pytest.raises(ValidationError, match='Only one serving environment'):
            ModelSpec.model_validate(d)

    def test_container_mounts_without_container_image_raises(self):
        d = _base_spec_dict(container_mounts=["/data:/data:ro"])
        with pytest.raises(ValidationError, match='"container_mounts" requires "container_image"'):
            ModelSpec.model_validate(d)

    def test_container_mounts_with_container_image_accepted(self):
        d = _container_spec_dict(container_mounts=["/data:/data:ro", "/models:/models:rw"])
        spec = ModelSpec.model_validate(d)
        assert spec.container_mounts == ["/data:/data:ro", "/models:/models:rw"]

    def test_container_mount_matching_remote_model_path_rejected(self):
        d = _container_spec_dict(container_mounts=["org/model:/model"])
        with pytest.raises(ValidationError, match="conflicts with the model path"):
            ModelSpec.model_validate(d)

    def test_container_mount_matching_local_model_glob_rejected(self):
        d = {
            "local_model": {
                "path_glob": "/checkpoints/mymodel/checkpoint_*",
                "model_family_name": "test",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
            "container_mounts": ["/checkpoints/mymodel/checkpoint_050000:/model"],
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        }
        with pytest.raises(ValidationError, match="conflicts with the model path"):
            ModelSpec.model_validate(d)

    def test_container_mount_child_of_local_model_glob_rejected(self):
        d = {
            "local_model": {
                "path_glob": "/checkpoints/mymodel/checkpoint_*",
                "model_family_name": "test",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
            "container_mounts": ["/checkpoints/mymodel/checkpoint_050000/foo:/model"],
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        }
        with pytest.raises(ValidationError, match="conflicts with the model path"):
            ModelSpec.model_validate(d)

    def test_container_mount_parent_of_local_model_glob_rejected(self):
        d = {
            "local_model": {
                "path_glob": "/checkpoints/mymodel/checkpoint_*",
                "model_family_name": "test",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
            "container_mounts": ["/checkpoints/mymodel:/mnt/model"],
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        }
        with pytest.raises(ValidationError, match="conflicts with the model path"):
            ModelSpec.model_validate(d)

    def test_container_mount_unrelated_path_accepted(self):
        d = {
            "local_model": {
                "path_glob": "/checkpoints/mymodel/checkpoint_*",
                "model_family_name": "test",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
            "container_mounts": ["/scratch:/scratch:rw"],
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        }
        spec = ModelSpec.model_validate(d)
        assert spec.container_mounts == ["/scratch:/scratch:rw"]

    def test_container_mount_similar_prefix_not_rejected(self):
        """A mount whose path shares a prefix but is not under the glob should pass."""
        d = {
            "local_model": {
                "path_glob": "/checkpoints/mymodel/checkpoint_*",
                "model_family_name": "test",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
            "container_mounts": ["/checkpoints/mymodel_other:/mnt/other"],
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        }
        spec = ModelSpec.model_validate(d)
        assert spec.container_mounts == ["/checkpoints/mymodel_other:/mnt/other"]

    def test_container_mount_identity_mount_of_glob_match_rejected(self):
        """Even an identity mount of the model path is rejected — scheduler handles it."""
        d = {
            "local_model": {
                "path_glob": "/checkpoints/mymodel/checkpoint_*",
                "model_family_name": "test",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
            "container_mounts": ["/checkpoints/mymodel/checkpoint_050000:/checkpoints/mymodel/checkpoint_050000"],
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        }
        with pytest.raises(ValidationError, match="conflicts with the model path"):
            ModelSpec.model_validate(d)

    def test_container_no_mounts_accepted(self):
        """Container with no user mounts should be fine."""
        d = _container_spec_dict()
        assert "container_mounts" not in d or d.get("container_mounts") is None
        spec = ModelSpec.model_validate(d)
        assert spec.container_image is not None

    def test_container_multiple_mounts_one_bad_rejected(self):
        """If one of several mounts conflicts, the whole config is rejected."""
        d = {
            "local_model": {
                "path_glob": "/checkpoints/mymodel/checkpoint_*",
                "model_family_name": "test",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
            "container_mounts": [
                "/scratch:/scratch:rw",
                "/checkpoints/mymodel/checkpoint_050000:/model",
                "/data:/data:ro",
            ],
            "max_simultaneous_requests": 4,
            "vllm_cli_args": [],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        }
        with pytest.raises(ValidationError, match="conflicts with the model path"):
            ModelSpec.model_validate(d)


# ---------------------------------------------------------------------------
# ModelInstance: serving_key and properties
# ---------------------------------------------------------------------------

class TestModelInstanceServingKey:

    def test_serving_key_differs_venv_vs_container(self):
        venv_inst = ModelInstance(
            name="m", path="org/model", venv_path="/venv",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        container_inst = ModelInstance(
            name="m", path="org/model", container_image="nvcr.io/test:latest",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        assert venv_inst.serving_key != container_inst.serving_key

    def test_serving_key_differs_venv_vs_conda(self):
        venv_inst = ModelInstance(
            name="m", path="org/model", venv_path="/venv",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        conda_inst = ModelInstance(
            name="m", path="org/model", conda_env="vllm-serving",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        assert venv_inst.serving_key != conda_inst.serving_key

    def test_serving_key_same_for_identical_container(self):
        a = ModelInstance(
            name="m", path="org/model", container_image="nvcr.io/test:latest",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        b = ModelInstance(
            name="m", path="org/model", container_image="nvcr.io/test:latest",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        assert a.serving_key == b.serving_key

    def test_uses_container_property(self):
        inst = ModelInstance(
            name="m", path="org/model", container_image="nvcr.io/test:latest",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        assert inst.uses_container is True
        assert inst.uses_conda is False

    def test_uses_conda_property(self):
        inst = ModelInstance(
            name="m", path="org/model", conda_env="vllm-serving",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        assert inst.uses_conda is True
        assert inst.uses_container is False

    def test_venv_instance_neither_conda_nor_container(self):
        inst = ModelInstance(
            name="m", path="org/model", venv_path="/venv",
            max_simultaneous_requests=4, vllm_cli_args=[], openai_kwargs={},
            parser_type="noop", model_type="instruct", owner="test", output_path="/tmp",
        )
        assert inst.uses_container is False
        assert inst.uses_conda is False


# ---------------------------------------------------------------------------
# ModelParser: fields passed through to ModelInstance
# ---------------------------------------------------------------------------

class TestModelParserPassthrough:

    def test_container_fields_propagated(self):
        spec = ModelSpec.model_validate(_container_spec_dict(
            container_mounts=["/data:/data:ro"]
        ))
        from pathlib import Path
        instance = ModelParser.model_instance_from_path(Path("org/model"), spec)
        assert instance.container_image == "nvcr.io/nvidia/pytorch:24.01-py3"
        assert instance.container_mounts == ["/data:/data:ro"]
        assert instance.venv_path is None
        assert instance.conda_env is None

    def test_conda_env_propagated(self):
        spec = ModelSpec.model_validate(_conda_spec_dict())
        from pathlib import Path
        instance = ModelParser.model_instance_from_path(Path("org/model"), spec)
        assert instance.conda_env == "vllm-serving"
        assert instance.venv_path is None
        assert instance.container_image is None

    def test_venv_fields_still_propagated(self):
        spec = ModelSpec.model_validate(_base_spec_dict())
        from pathlib import Path
        instance = ModelParser.model_instance_from_path(Path("org/model"), spec)
        assert instance.venv_path == "/venv"
        assert instance.conda_env is None
        assert instance.container_image is None

    def test_local_model_container_fields_propagated(self):
        """local_model with container_image must propagate to ModelInstance."""
        from pathlib import Path
        spec = ModelSpec.model_validate({
            "local_model": {
                "path_glob": "/checkpoints/model/checkpoint_*",
                "model_family_name": "test-family",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
            "container_mounts": ["/models:/models:ro"],
            "max_simultaneous_requests": 4,
            "vllm_cli_args": ["--tensor-parallel-size 8"],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        })
        instance = ModelParser.model_instance_from_path(
            Path("/checkpoints/model/checkpoint_0050000/config.json"), spec
        )
        assert instance.container_image == "nvcr.io/nvidia/pytorch:24.01-py3"
        assert instance.container_mounts == ["/models:/models:ro"]
        assert instance.venv_path is None

    def test_local_model_conda_env_propagated(self):
        """local_model with conda_env must propagate to ModelInstance."""
        from pathlib import Path
        spec = ModelSpec.model_validate({
            "local_model": {
                "path_glob": "/checkpoints/model/checkpoint_*",
                "model_family_name": "test-family",
                "version_level": -1,
                "enqueue_existing": True,
            },
            "conda_env": "vllm-serving",
            "max_simultaneous_requests": 4,
            "vllm_cli_args": ["--tensor-parallel-size 8"],
            "openai_kwargs": {},
            "model_type": "instruct",
            "owner": "test",
            "ready": True,
            "output_path": "/tmp/out",
            "parser_type": "noop",
        })
        instance = ModelParser.model_instance_from_path(
            Path("/checkpoints/model/checkpoint_0050000/config.json"), spec
        )
        assert instance.conda_env == "vllm-serving"
        assert instance.venv_path is None
        assert instance.container_image is None


# ---------------------------------------------------------------------------
# SlurmManager: sbatch command construction
# ---------------------------------------------------------------------------

class TestSlurmManagerCommand:

    @pytest.mark.asyncio
    async def test_container_sbatch_includes_container_flags(self, tmp_path):
        from scheduler.slurm_manager import SlurmManager
        jm = SlurmManager(log_dir=str(tmp_path))
        model = _make_mock_model(
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
            container_mounts=["/models:/models:ro"],
            vllm_cli_args=["--tensor-parallel-size 8"],
        )

        captured = []
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_sbatch_exec(captured)):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert any("--container-image=nvcr.io/nvidia/pytorch:24.01-py3" in str(a) for a in captured)
        assert any("--container-writable" in str(a) for a in captured)
        mount_args = [a for a in captured if isinstance(a, str) and a.startswith("--container-mounts=")]
        assert len(mount_args) == 1
        assert "org/model:org/model" in mount_args[0]
        assert "/models:/models:ro" in mount_args[0]
        assert any("sbatch_script_container.sh" in str(a) for a in captured)
        export_arg = [a for a in captured if isinstance(a, str) and a.startswith("--export=")]
        assert len(export_arg) == 1
        assert "venv_path" not in export_arg[0]
        assert "conda_env" not in export_arg[0]

    @pytest.mark.asyncio
    async def test_conda_sbatch_uses_conda_script(self, tmp_path):
        from scheduler.slurm_manager import SlurmManager
        jm = SlurmManager(log_dir=str(tmp_path))
        model = _make_mock_model(conda_env="vllm-serving")

        captured = []
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_sbatch_exec(captured)):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert any("sbatch_script_conda.sh" in str(a) for a in captured)
        assert not any("--container-image" in str(a) for a in captured)
        export_arg = [a for a in captured if isinstance(a, str) and a.startswith("--export=")]
        assert len(export_arg) == 1
        assert "conda_env=vllm-serving" in export_arg[0]
        assert "venv_path" not in export_arg[0]

    @pytest.mark.asyncio
    async def test_container_auto_injects_model_path_mount(self, tmp_path):
        """Container mode must auto-inject an identity mount for model_instance.path."""
        from scheduler.slurm_manager import SlurmManager
        jm = SlurmManager(log_dir=str(tmp_path))
        model = _make_mock_model(
            container_image="/images/vllm.sqsh",
            path="/data/checkpoints/checkpoint_050000",
            vllm_cli_args=["--tensor-parallel-size 8"],
        )

        captured = []
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_sbatch_exec(captured)):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        mount_args = [a for a in captured if isinstance(a, str) and "--container-mounts=" in a]
        assert len(mount_args) == 1
        model_path_mounted = any(
            "/data/checkpoints/checkpoint_050000:/data/checkpoints/checkpoint_050000" in a
            for a in mount_args
        )
        assert model_path_mounted, f"Model path identity mount not found in {mount_args}"

    @pytest.mark.asyncio
    async def test_container_auto_mount_coexists_with_user_mounts(self, tmp_path):
        """User mounts and auto-injected model path mount should both appear."""
        from scheduler.slurm_manager import SlurmManager
        jm = SlurmManager(log_dir=str(tmp_path))
        model = _make_mock_model(
            container_image="/images/vllm.sqsh",
            path="/data/checkpoints/checkpoint_050000",
            container_mounts=["/scratch:/scratch:rw"],
            vllm_cli_args=["--tensor-parallel-size 8"],
        )

        captured = []
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_sbatch_exec(captured)):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        mount_args = [a for a in captured if isinstance(a, str) and "--container-mounts=" in a]
        assert len(mount_args) == 1
        assert "," in mount_args[0]
        has_user_mount = any("/scratch:/scratch:rw" in a for a in mount_args)
        has_model_mount = any(
            "/data/checkpoints/checkpoint_050000:/data/checkpoints/checkpoint_050000" in a
            for a in mount_args
        )
        assert has_user_mount, f"User mount not found in {mount_args}"
        assert has_model_mount, f"Model path mount not found in {mount_args}"

    @pytest.mark.asyncio
    async def test_venv_sbatch_no_container_or_conda_flags(self, tmp_path):
        from scheduler.slurm_manager import SlurmManager
        jm = SlurmManager(log_dir=str(tmp_path))
        model = _make_mock_model(venv_path="/venv/bin/activate")

        captured = []
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_sbatch_exec(captured)):
            await jm.update_allocation([(model, 1)], unneeded_models=[])

        assert not any("--container-image" in str(a) for a in captured)
        assert any("sbatch_script.sh" in str(a) for a in captured)
        assert not any("sbatch_script_conda.sh" in str(a) for a in captured)
        export_arg = [a for a in captured if isinstance(a, str) and a.startswith("--export=")]
        assert len(export_arg) == 1
        assert "venv_path=/venv/bin/activate" in export_arg[0]


# ---------------------------------------------------------------------------
# Container sbatch script runtime guards
# ---------------------------------------------------------------------------

class TestContainerSbatchScript:

    def _env(self, tmp_path: Path) -> dict:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        env["SLURM_TMPDIR"] = str(tmp_path / "slurm-tmp")
        env["model_path"] = "org/model"
        env["vllm_args"] = base64.b64encode(json.dumps([]).encode()).decode()
        env.pop("VLLM_BIN", None)
        env.pop("bootstrap_python", None)
        return env

    def _run(self, env: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(CONTAINER_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=2,
        )

    @pytest.mark.parametrize("case", ["missing", "not_executable", "not_python"])
    def test_invalid_bootstrap_python_fails_before_background_sleep(self, tmp_path, case):
        env = self._env(tmp_path)
        bootstrap_python = tmp_path / "bootstrap-python"
        if case == "not_executable":
            bootstrap_python.write_text("#!/bin/bash\nexit 0\n")
        elif case == "not_python":
            _write_executable(bootstrap_python, "#!/bin/bash\nexit 42\n")
        env["bootstrap_python"] = str(bootstrap_python)

        result = self._run(env)

        assert result.returncode == 127
        assert "Invalid bootstrap_python" in result.stderr

    def test_explicit_invalid_vllm_bin_fails_without_path_fallback(self, tmp_path):
        env = self._env(tmp_path)
        env["bootstrap_python"] = sys.executable
        fallback_marker = tmp_path / "fallback-vllm-ran"
        _write_executable(
            tmp_path / "bin" / "vllm",
            textwrap.dedent(f"""\
                #!/bin/bash
                touch "{fallback_marker}"
                exit 0
            """),
        )
        env["VLLM_BIN"] = str(tmp_path / "missing-vllm")

        result = self._run(env)

        assert result.returncode == 127
        assert "Invalid VLLM_BIN" in result.stderr
        assert not fallback_marker.exists()


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

class TestYamlParsing:

    def test_parse_container_yaml(self, tmp_path):
        yaml_content = """
remote_model:
  base_name: mymodel
  path: org/model
  revision: null
model_type: instruct
parser_type: noop
container_image: "nvcr.io/nvidia/pytorch:24.01-py3"
container_mounts:
  - "/models:/models:ro"
max_simultaneous_requests: 4
vllm_cli_args: []
openai_kwargs: {}
owner: test
ready: true
output_path: /tmp/out
"""
        yaml_path = tmp_path / "model.yaml"
        yaml_path.write_text(yaml_content)
        spec = ModelParser.parse_yaml(yaml_path)
        assert spec.container_image == "nvcr.io/nvidia/pytorch:24.01-py3"
        assert spec.venv_path is None
        assert spec.conda_env is None

    def test_parse_conda_yaml(self, tmp_path):
        yaml_content = """
remote_model:
  base_name: mymodel
  path: org/model
  revision: null
model_type: instruct
parser_type: noop
conda_env: vllm-serving
max_simultaneous_requests: 4
vllm_cli_args: []
openai_kwargs: {}
owner: test
ready: true
output_path: /tmp/out
"""
        yaml_path = tmp_path / "model.yaml"
        yaml_path.write_text(yaml_content)
        spec = ModelParser.parse_yaml(yaml_path)
        assert spec.conda_env == "vllm-serving"
        assert spec.venv_path is None
        assert spec.container_image is None

    def test_parse_venv_yaml_still_works(self, tmp_path):
        yaml_content = """
remote_model:
  base_name: mymodel
  path: org/model
  revision: null
model_type: instruct
parser_type: noop
venv_path: /venv
max_simultaneous_requests: 4
vllm_cli_args: []
openai_kwargs: {}
owner: test
ready: true
output_path: /tmp/out
"""
        yaml_path = tmp_path / "model.yaml"
        yaml_path.write_text(yaml_content)
        spec = ModelParser.parse_yaml(yaml_path)
        assert spec.venv_path == "/venv"
        assert spec.conda_env is None
        assert spec.container_image is None
