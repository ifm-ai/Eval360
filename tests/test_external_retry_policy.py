import asyncio
from collections import UserDict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from scheduler.model import ModelInstance, ModelParser, ModelSpec
from scheduler.scheduler import Scheduler


def _external_spec(**external_overrides):
    external_model = {
        "base_name": "served-model",
        "base_url": "https://example.test/v1",
    }
    external_model.update(external_overrides)
    return {
        "external_model": external_model,
        "max_simultaneous_requests": 4,
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/tmp/out",
        "parser_type": "noop",
    }


def _non_external_spec(source, max_simultaneous_requests):
    spec = {
        "venv_path": "/tmp/venv",
        "max_simultaneous_requests": max_simultaneous_requests,
        "vllm_cli_args": [],
        "openai_kwargs": {},
        "model_type": "instruct",
        "owner": "test",
        "ready": True,
        "output_path": "/tmp/out",
        "parser_type": "noop",
    }
    if source == "remote_model":
        spec[source] = {
            "base_name": "served-model",
            "path": "org/model",
            "revision": None,
        }
    else:
        spec[source] = {
            "path_glob": "/tmp/checkpoints/*",
            "model_family_name": "served-model",
        }
    return spec


TIMING_SAFETY_LIMITS = [
    ("request_timeout_seconds", 7200),
    ("total_deadline_seconds", 7200),
    ("initial_backoff_seconds", 60),
    ("max_backoff_seconds", 60),
]


def test_retry_policy_defaults_are_bounded():
    spec = ModelSpec.model_validate(_external_spec())

    assert spec.external_model.retry_policy.max_attempts == 4
    assert spec.external_model.retry_policy.request_timeout_seconds == 7200
    assert spec.external_model.retry_policy.total_deadline_seconds == 7200
    assert spec.external_model.retry_policy.initial_backoff_seconds == 1
    assert spec.external_model.retry_policy.max_backoff_seconds == 60


def test_retry_policy_defaults_to_full_jitter_and_reaches_instance():
    spec = ModelSpec.model_validate(_external_spec())

    assert spec.external_model.retry_policy.model_dump().get("jitter") == "full"

    instance = ModelParser.model_instance_from_path(
        Path(spec.external_model.base_url),
        spec,
    )
    assert instance.external_retry_policy.model_dump().get("jitter") == "full"


@pytest.mark.parametrize("jitter", ["equal", "decorrelated"])
def test_retry_policy_accepts_configurable_jitter_strategies(jitter):
    spec = ModelSpec.model_validate(
        _external_spec(retry_policy={"jitter": jitter})
    )

    instance = ModelParser.model_instance_from_path(
        Path(spec.external_model.base_url),
        spec,
    )
    assert (
        instance.external_retry_policy.model_dump(mode="json")["jitter"]
        == jitter
    )


@pytest.mark.parametrize("jitter", ["none", "random", 1])
def test_retry_policy_rejects_unknown_jitter_strategies(jitter):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _external_spec(retry_policy={"jitter": jitter})
        )


def test_retry_policy_rejects_mutation_after_validation():
    policy = ModelSpec.model_validate(
        _external_spec()
    ).external_model.retry_policy

    with pytest.raises(ValidationError):
        policy.max_attempts = 101

    assert policy.max_attempts == 4


def test_external_model_rejects_invalid_retry_policy_replacement():
    external_model = ModelSpec.model_validate(
        _external_spec()
    ).external_model

    with pytest.raises(ValidationError):
        external_model.retry_policy = {"max_attempts": "1"}


def test_external_model_rejects_retry_policy_replacement_above_config_limit():
    external_model = ModelSpec.model_validate(
        _external_spec()
    ).external_model

    with pytest.raises(ValidationError):
        external_model.retry_policy = {"total_deadline_seconds": 9000}


def test_model_instance_rejects_invalid_retry_policy_replacement():
    spec = ModelSpec.model_validate(_external_spec())
    instance = ModelParser.model_instance_from_path(
        Path(spec.external_model.base_url),
        spec,
    )

    with pytest.raises(ValidationError):
        instance.external_retry_policy = {"max_attempts": "1"}


def test_retry_policy_accepts_native_numeric_values_and_reaches_instance():
    spec = ModelSpec.model_validate(
        _external_spec(
            retry_policy={
                "max_attempts": 2,
                "request_timeout_seconds": 15,
                "total_deadline_seconds": 20.5,
                "initial_backoff_seconds": 0.25,
                "max_backoff_seconds": 2,
            }
        )
    )

    instance = ModelParser.model_instance_from_path(
        Path(spec.external_model.base_url),
        spec,
    )

    assert instance.external_retry_policy == spec.external_model.retry_policy
    assert instance.external_retry_policy.total_deadline_seconds == 20.5


def test_model_parser_cli_override_replaces_configured_total_deadline(tmp_path):
    config_path = tmp_path / "external-model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _external_spec(
                retry_policy={"total_deadline_seconds": 30},
            )
        )
    )

    spec = ModelParser.parse_yaml(
        config_path,
        external_total_deadline_seconds=9000,
    )

    assert spec.external_model.retry_policy.total_deadline_seconds == 9000


def test_model_parser_cli_override_injects_missing_retry_policy(tmp_path):
    config_path = tmp_path / "external-model.yaml"
    config_path.write_text(yaml.safe_dump(_external_spec()))

    spec = ModelParser.parse_yaml(
        config_path,
        external_total_deadline_seconds=9000,
    )

    assert spec.external_model.retry_policy.total_deadline_seconds == 9000


def test_model_parser_cli_override_must_exceed_configuration_ceiling(tmp_path):
    config_path = tmp_path / "external-model.yaml"
    config_path.write_text(yaml.safe_dump(_external_spec()))

    with pytest.raises(ValueError, match="exceed.*7200"):
        ModelParser.parse_yaml(
            config_path,
            external_total_deadline_seconds=7200,
        )


def test_model_parser_without_cli_override_preserves_configured_deadline(
    tmp_path,
):
    config_path = tmp_path / "external-model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _external_spec(
                retry_policy={"total_deadline_seconds": 30},
            )
        )
    )

    spec = ModelParser.parse_yaml(config_path)

    assert spec.external_model.retry_policy.total_deadline_seconds == 30


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_attempts", 0),
        ("request_timeout_seconds", 0),
        ("total_deadline_seconds", 0),
        ("initial_backoff_seconds", -1),
        ("max_backoff_seconds", 0),
    ],
)
def test_retry_policy_rejects_invalid_bounds(field, value):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _external_spec(retry_policy={field: value})
        )


@pytest.mark.parametrize(("field", "limit"), TIMING_SAFETY_LIMITS)
def test_retry_policy_accepts_timing_value_at_safety_limit(field, limit):
    policy = {field: limit}
    if field == "initial_backoff_seconds":
        policy["max_backoff_seconds"] = limit

    spec = ModelSpec.model_validate(
        _external_spec(retry_policy=policy)
    )

    assert getattr(spec.external_model.retry_policy, field) == limit


@pytest.mark.parametrize(("field", "limit"), TIMING_SAFETY_LIMITS)
def test_retry_policy_rejects_timing_value_above_safety_limit(field, limit):
    policy = {field: limit + 1}
    if field == "initial_backoff_seconds":
        policy["max_backoff_seconds"] = limit + 1

    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _external_spec(retry_policy=policy)
        )


@pytest.mark.parametrize(
    "field",
    [
        "max_attempts",
        "request_timeout_seconds",
        "total_deadline_seconds",
        "initial_backoff_seconds",
        "max_backoff_seconds",
    ],
)
@pytest.mark.parametrize("value", [True, False, "1"])
def test_retry_policy_rejects_booleans_and_numeric_strings(field, value):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _external_spec(retry_policy={field: value})
        )


@pytest.mark.parametrize(
    "field",
    [
        "request_timeout_seconds",
        "total_deadline_seconds",
        "initial_backoff_seconds",
        "max_backoff_seconds",
    ],
)
@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), float("nan")],
)
def test_retry_policy_rejects_non_finite_values(field, value):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _external_spec(retry_policy={field: value})
        )


def test_retry_policy_rejects_excessive_attempt_count():
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _external_spec(retry_policy={"max_attempts": 101})
        )


def test_retry_policy_rejects_backoff_cap_below_initial_delay():
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _external_spec(
                retry_policy={
                    "initial_backoff_seconds": 2,
                    "max_backoff_seconds": 1,
                }
            )
        )


@pytest.mark.parametrize(
    "requests_per_minute",
    [True, False, 0, -1, 1.0, "1"],
)
def test_requests_per_minute_is_a_strict_positive_integer(
    requests_per_minute,
):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _external_spec(
                requests_per_minute=requests_per_minute
            )
        )


@pytest.mark.parametrize("value", [True, "4", 4.0])
def test_model_spec_rejects_coercible_request_concurrency(value):
    spec = _external_spec()
    spec["max_simultaneous_requests"] = value

    with pytest.raises(ValidationError):
        ModelSpec.model_validate(spec)


@pytest.mark.parametrize("value", [True, "4", 4.0, 0])
def test_model_spec_rejects_invalid_request_concurrency_from_mapping(value):
    spec = _external_spec()
    spec["max_simultaneous_requests"] = value

    with pytest.raises(ValidationError):
        ModelSpec.model_validate(UserDict(spec))


@pytest.mark.parametrize("value", [True, "4", 4.0])
def test_model_instance_rejects_coercible_request_concurrency(value):
    spec = ModelSpec.model_validate(_external_spec())
    instance = ModelParser.model_instance_from_path(
        Path(spec.external_model.base_url),
        spec,
    )
    instance_data = instance.model_dump()
    instance_data["max_simultaneous_requests"] = value

    with pytest.raises(ValidationError):
        ModelInstance.model_validate(instance_data)


@pytest.mark.parametrize("is_external", [1, "true"])
@pytest.mark.parametrize("value", [True, "4", 4.0, 0, -1])
def test_model_instance_rejects_invalid_concurrency_with_external_marker(
    is_external,
    value,
):
    spec = ModelSpec.model_validate(_external_spec())
    instance = ModelParser.model_instance_from_path(
        Path(spec.external_model.base_url),
        spec,
    )
    instance_data = instance.model_dump()
    instance_data["is_external"] = is_external
    instance_data["max_simultaneous_requests"] = value

    with pytest.raises(ValidationError):
        ModelInstance.model_validate(instance_data)


def test_external_model_rejects_zero_request_concurrency():
    spec = _external_spec()
    spec["max_simultaneous_requests"] = 0

    with pytest.raises(ValidationError):
        ModelSpec.model_validate(spec)


@pytest.mark.parametrize("source", ["local_model", "remote_model"])
@pytest.mark.parametrize(
    "value",
    ["4", 4.0],
    ids=["numeric-string", "integral-float"],
)
def test_non_external_model_spec_preserves_concurrency_coercion(
    source,
    value,
):
    spec = ModelSpec.model_validate(
        _non_external_spec(source, value)
    )

    assert spec.max_simultaneous_requests == 4


@pytest.mark.parametrize("source", ["local_model", "remote_model"])
@pytest.mark.parametrize("value", [0, -1])
def test_non_external_model_spec_rejects_non_positive_concurrency(
    source,
    value,
):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_non_external_spec(source, value))


@pytest.mark.parametrize(
    "value",
    ["4", 4.0],
    ids=["numeric-string", "integral-float"],
)
@pytest.mark.parametrize(
    "is_external",
    [False, 0, "false"],
    ids=["literal-false", "integer-zero", "false-string"],
)
def test_non_external_model_instance_preserves_concurrency_coercion(
    value,
    is_external,
):
    instance = ModelInstance.model_validate(
        {
            "name": "served-model",
            "path": "org/model",
            "max_simultaneous_requests": value,
            "openai_kwargs": {},
            "parser_type": "noop",
            "model_type": "instruct",
            "owner": "test",
            "output_path": "/tmp/out",
            "is_external": is_external,
        }
    )

    assert instance.max_simultaneous_requests == 4


@pytest.mark.parametrize("value", [0, -1])
def test_non_external_model_instance_rejects_non_positive_concurrency(value):
    with pytest.raises(ValidationError):
        ModelInstance.model_validate(
            {
                "name": "served-model",
                "path": "org/model",
                "max_simultaneous_requests": value,
                "openai_kwargs": {},
                "parser_type": "noop",
                "model_type": "instruct",
                "owner": "test",
                "output_path": "/tmp/out",
                "is_external": False,
            }
        )


@pytest.mark.asyncio
async def test_register_restored_external_model_applies_launch_deadline():
    spec = ModelSpec.model_validate(
        _external_spec(retry_policy={"total_deadline_seconds": 30})
    )
    restored = ModelParser.model_instance_from_path(
        Path(spec.external_model.base_url),
        spec,
    )
    scheduler = object.__new__(Scheduler)
    scheduler._external_total_deadline_seconds = 9000
    scheduler._registration_lock = asyncio.Lock()
    scheduler._owned_external_registrations = set()
    scheduler.db_manager = MagicMock()
    scheduler.event_manager = MagicMock()
    scheduler.event_manager.create_events_for_new_model = AsyncMock()
    pool = MagicMock()
    pool.add_url = AsyncMock()

    with (
        patch("scheduler.scheduler.openai_interface.LOCKED_CONNECTIONS", {}),
        patch(
            "scheduler.scheduler.openai_interface.ModelConnectionPool",
            return_value=pool,
        ),
    ):
        await scheduler.register_model_instance(restored)

    registered = scheduler.db_manager.register_model.call_args.args[0]
    assert registered.external_retry_policy.total_deadline_seconds == 9000
