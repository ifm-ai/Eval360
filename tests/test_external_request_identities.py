from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

import scheduler.openai_interface as openai_interface
from scheduler.model import ModelParser, ModelSpec
from scheduler.rate_limiter import RateLimiter


def _external_spec(**external_overrides):
    external_model = {
        "base_name": "served-model",
        "base_url": "https://api.example.test/v1",
        "api_key_env": "OPENAI_API_KEY",
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


def _external_instance(**external_overrides):
    spec = ModelSpec.model_validate(_external_spec(**external_overrides))
    return ModelParser.model_instance_from_path(
        Path(spec.external_model.base_url),
        spec,
    )


def _external_requests_module():
    spec = importlib.util.find_spec("scheduler.external_requests")
    assert spec is not None, "shared external request primitives are missing"
    return importlib.import_module("scheduler.external_requests")


@pytest.fixture(autouse=True)
def _clear_runtime_registries():
    openai_interface.LOCKED_CONNECTIONS.clear()
    yield
    openai_interface.LOCKED_CONNECTIONS.clear()
    spec = importlib.util.find_spec("scheduler.external_requests")
    if spec is not None:
        module = importlib.import_module("scheduler.external_requests")
        module.RATE_LIMITERS.clear()


def test_external_endpoint_is_canonicalized_during_validation():
    instance = _external_instance(
        base_url=" https://API.EXAMPLE.test:443/v1/ "
    )

    assert instance.base_url == "https://api.example.test/v1"
    assert instance.path == "https://api.example.test/v1"


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://api.example.test/v1",
        "api.example.test/v1",
        "https:///v1",
        "https://api.example.test:bad/v1",
        "https://user:secret@api.example.test/v1",
        "https://api.example.test/v1?api_key=secret",
        "https://api.example.test/v1#credential",
    ],
)
def test_external_endpoint_rejects_invalid_http_urls(base_url):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_external_spec(base_url=base_url))


def test_equivalent_endpoint_urls_share_one_serving_key():
    first = _external_instance(
        base_url="https://API.EXAMPLE.test:443/v1/"
    )
    second = _external_instance(
        base_url="https://api.example.test/v1"
    )

    assert first.serving_key == second.serving_key


def test_meaningful_endpoint_path_keeps_a_distinct_serving_key():
    first = _external_instance(base_url="https://api.example.test/v1")
    second = _external_instance(base_url="https://api.example.test/v2")

    assert first.serving_key != second.serving_key


@pytest.mark.asyncio
async def test_connection_pool_normalizes_equivalent_url_aliases():
    pool = openai_interface.ModelConnectionPool(capacity_per_url=2)

    await pool.add_url("https://API.EXAMPLE.test:443/v1/")
    await pool.add_url("https://api.example.test/v1")

    assert pool.urls == ["https://api.example.test/v1"]


def test_connection_pool_rejects_live_capacity_drift():
    get_pool = getattr(
        openai_interface,
        "get_or_create_connection_pool",
        None,
    )
    assert callable(get_pool), "connection-pool registration guard is missing"

    first = get_pool("serving-key", 2)
    assert get_pool("serving-key", 2) is first
    with pytest.raises(ValueError, match="conflicting"):
        get_pool("serving-key", 3)


def test_rate_limiter_exposes_immutable_registration_value():
    limiter = RateLimiter(42)

    assert getattr(limiter, "requests_per_minute", None) == 42


def test_equivalent_endpoint_and_credential_share_quota_identity():
    external_requests = _external_requests_module()
    first = _external_instance(
        base_url="https://API.EXAMPLE.test:443/v1/",
        requests_per_minute=60,
    )
    second = _external_instance(
        base_url="https://api.example.test/v1",
        requests_per_minute=60,
    )
    first.api_key = "same-secret"
    second.api_key = "same-secret"

    first_limiter = external_requests.get_external_rate_limiter(first)
    second_limiter = external_requests.get_external_rate_limiter(second)

    assert first_limiter is second_limiter
    assert len(external_requests.RATE_LIMITERS) == 1


def test_quota_registry_identity_does_not_retain_raw_credential():
    external_requests = _external_requests_module()
    instance = _external_instance(requests_per_minute=60)
    instance.api_key = "credential-must-not-appear"

    external_requests.get_external_rate_limiter(instance)

    assert external_requests.RATE_LIMITERS
    assert all(
        instance.api_key not in identity
        for identity in external_requests.RATE_LIMITERS
    )


def test_equivalent_quota_rejects_conflicting_live_rpm():
    external_requests = _external_requests_module()
    first = _external_instance(requests_per_minute=60)
    second = _external_instance(requests_per_minute=120)
    first.api_key = "same-secret"
    second.api_key = "same-secret"

    external_requests.get_external_rate_limiter(first)

    with pytest.raises(ValueError, match="conflicting"):
        external_requests.get_external_rate_limiter(second)
