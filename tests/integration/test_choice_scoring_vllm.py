"""Live choice-scoring coverage for OpenAI-compatible base/completions servers.

What this tests:
    A real endpoint can run the same choice-scoring path used by
    OpenAIConnection.launch_requests(), and the raw echo-logprob row can be
    consumed by ChoiceScoring.run().

Why this exists:
    The regular unit tests mock completions responses, so they cannot catch
    backend compatibility regressions in echo=True logprobs, batched prompt
    handling, SDK serialization shape, or grader consumption of a real payload.

Corner cases covered:
    The test is skipped unless explicitly configured, requires a /v1 base URL so
    it exercises the external-model client path exactly, asserts that choice
    scoring does not write legacy generations, validates both full-prompt and
    completion-only logprob arrays, and verifies the resulting grade/score
    shape instead of only checking that the HTTP requests succeeded.
"""

import os
from unittest.mock import MagicMock

import pytest

from scheduler.choice_scoring_schema import validate_choice_scoring_row
from scheduler.event import EventInstance
from scheduler.grader.base import Grade, Score
from scheduler.grader.choice_scoring import ChoiceScoring
from scheduler.job import JobManager
from scheduler.model import CacheSaltConfig, ModelInstance, ModelType
from scheduler.openai_interface import (
    LOCKED_CONNECTIONS,
    ModelConnectionPool,
    OpenAIConnection,
)
from scheduler.progress import ProgressManager
from scheduler.task import AsyncGenerationTask
from scheduler.utils import ExceptionWrapper, Sentinel


BASE_URL_ENV = "EVAL360_CHOICE_SCORING_BASE_URL"
MODEL_ENV = "EVAL360_CHOICE_SCORING_MODEL"
API_KEY_ENV = "EVAL360_CHOICE_SCORING_API_KEY"


@pytest.fixture(autouse=True)
def clear_locked_connections():
    LOCKED_CONNECTIONS.clear()
    yield
    LOCKED_CONNECTIONS.clear()


async def _requests(*items):
    for item in items:
        yield item
    yield Sentinel.COMPLETED


async def _empty_existing():
    if False:
        yield


async def _collect(generator):
    return [item async for item in generator]


def _live_endpoint_config() -> tuple[str, str, str]:
    base_url = os.environ.get(BASE_URL_ENV, "").rstrip("/")
    model_name = os.environ.get(MODEL_ENV, "")
    api_key = os.environ.get(API_KEY_ENV, "no-key")
    if not base_url or not model_name:
        pytest.skip(
            f"set {BASE_URL_ENV} and {MODEL_ENV} to run live choice-scoring coverage"
        )
    if not base_url.endswith("/v1"):
        pytest.skip(f"{BASE_URL_ENV} must include the OpenAI-compatible /v1 base path")
    return base_url, model_name, api_key


def _make_live_connection(
    base_url: str, model_name: str, api_key: str
) -> OpenAIConnection:
    model = ModelInstance.model_construct(
        name="choice-scoring-live-test",
        family_name="choice-scoring-live-test",
        path=model_name,
        revision=None,
        venv_path=None,
        conda_env=None,
        container_image=None,
        container_mounts=None,
        max_simultaneous_requests=1,
        max_time_to_deploy=60,
        allow_long_max_model_len=True,
        vllm_cli_args=None,
        vllm_logging_level="WARNING",
        openai_kwargs={},
        cache_salt=CacheSaltConfig(),
        parser_type="noop",
        model_type=ModelType.BASE,
        owner="test",
        output_path="/tmp/eval360-choice-scoring-live-test",
        tag="any",
        prompt_prefix_instructions=None,
        base_url=base_url,
        api_key=api_key,
        requests_per_minute=None,
        is_external=True,
        api_model_name=model_name,
    )
    task = MagicMock(spec=AsyncGenerationTask)
    task.average_over = [1]
    task.pass_at = [1]
    task.openai_settings = None
    task.grader = MagicMock()
    task.grader.type = "choice_scoring"

    LOCKED_CONNECTIONS[model.serving_key] = ModelConnectionPool(
        model.max_simultaneous_requests
    )
    LOCKED_CONNECTIONS[model.serving_key]._slots[base_url] = [
        model.max_simultaneous_requests
    ]

    job_manager = MagicMock(spec=JobManager)
    job_manager.is_url_live.return_value = True
    return OpenAIConnection(
        model=model,
        task=task,
        event_instance=MagicMock(spec=EventInstance),
        job_manager=job_manager,
        progress_manager=MagicMock(spec=ProgressManager),
        new_field_name="generations",
    )


async def _grade_choice_scoring_row(row: dict):
    class MockEvent:
        parser_type = "noop"

    async def generated_samples():
        yield row
        yield Sentinel.COMPLETED

    grader = ChoiceScoring(
        samples_generator=generated_samples(),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )
    return await _collect(
        grader.run(existing=_empty_existing(), average_over=[1], pass_at=[1])
    )


@pytest.mark.asyncio
async def test_live_choice_scoring_echo_logprobs_flow_to_grader():
    base_url, model_name, api_key = _live_endpoint_config()
    conn = _make_live_connection(base_url, model_name, api_key)
    row = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "completion_input": "Question: 1+1?\nA. 2\nB. 3\nAnswer:",
        "scoring_completions": ["A", "B"],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth": "A",
    }
    validate_choice_scoring_row(row, phase="integration")
    hook_calls = 0

    async def completion_hook():
        nonlocal hook_calls
        hook_calls += 1

    request_results = await _collect(
        conn.launch_requests(_requests(row), offset=0, completion_hook=completion_hook)
    )

    generated_rows = [
        item for item in request_results if item is not Sentinel.COMPLETED
    ]
    assert hook_calls == 1
    assert len(generated_rows) == 1
    generated = generated_rows[0]
    assert not isinstance(generated, ExceptionWrapper), getattr(generated, "trace", "")
    assert "generations" not in generated
    assert len(generated["choice_scoring_full_logprobs"]) == 2
    assert len(generated["choice_scoring_completion_logprobs"]) == 2
    assert generated["choice_scoring_metadata"] == {
        "full_prompts": [
            "Question: 1+1?\nA. 2\nB. 3\nAnswer: A",
            "Question: 1+1?\nA. 2\nB. 3\nAnswer: B",
        ],
        "completion_prompts": ["Answer: A", "Answer: B"],
    }

    graded_items = await _grade_choice_scoring_row(generated)
    grades = [item for item in graded_items if isinstance(item, Grade)]
    scores = [item for item in graded_items if isinstance(item, Score)]

    assert len(grades) == 1
    assert {score.name for score in scores} == {
        "acc",
        "acc_char",
        "acc_token",
        "acc_compl",
        "nll",
        "nll_char",
        "nll_token",
        "nll_compl",
    }
    graded = grades[0].element
    assert graded["ground_truth_index"] == 0
    assert len(graded["choice_nll"]) == 2
    assert len(graded["choice_nll_completion"]) == 2
    assert len(graded["scoring_completion_n_tokens"]) == 2
    assert graded["picked"][0] in {"A", "B"}
    assert isinstance(graded["correct"][0], bool)
