from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scheduler.cache_salt import CacheSaltConfig
from scheduler.eval_config import (
    EvalConfigParser,
    LoadedEvalConfig,
    LoadedModelSpec,
    LoadedTask,
)
from scheduler.grader.lcr_aa import LCRAAJudge, OFFICIAL_JUDGE_TEMPLATE
from scheduler.model import ModelParser
from scheduler.task import Task


ROOT = Path(__file__).resolve().parents[1]


def test_requested_model_data_group_configs_resolve_together(monkeypatch):
    model_path = ROOT / "model_zoo/lcr_aa_375b_vllm.yaml"
    data_path = ROOT / "data_zoo/lcr_aa.yaml"
    group_path = ROOT / "group_zoo/lcr_aa.yaml"
    # The dataset repo is a placeholder in this copy, so record the HF
    # existence check instead of calling the Hub.
    hf_checks = []
    monkeypatch.setattr("scheduler.task.check_hf_file_exists", hf_checks.append)

    model = ModelParser.parse_yaml(model_path)
    task = Task.parse_yaml(data_path)
    assert hf_checks == ["hf://<HF_ORG>/<EVAL_SOURCES_REPO>/lcr_aa/lcr_aa.jsonl@main"]
    eval_config = EvalConfigParser.parse_yaml(group_path)
    resolved = EvalConfigParser.build_resolved_pairs(
        loaded_models=[LoadedModelSpec(path=str(model_path), spec=model)],
        loaded_tasks=[LoadedTask(path=str(data_path), task=task)],
        loaded_eval_configs=[
            LoadedEvalConfig(path=str(group_path), config=eval_config)
        ],
    )

    assert len(resolved) == 1
    pair = resolved[0]
    assert pair.task.average_over == [3]
    assert pair.task.pass_at == [1]
    assert pair.task.data_path == (
        "hf://<HF_ORG>/<EVAL_SOURCES_REPO>/lcr_aa/lcr_aa.jsonl@main"
    )
    assert pair.model.parser_type == "think_suffix"
    assert pair.task.openai_settings == {
        "temperature": 1.0,
        "max_tokens": 400000,
        "extra_body": {"chat_template_kwargs": {"reasoning_effort": "high"}},
    }
    for omitted in ("top_p", "top_k", "min_p", "seed", "stop"):
        assert omitted not in pair.task.openai_settings


def test_official_judge_prompt_and_strict_parser():
    messages = LCRAAJudge.create_judge_messages(
        "Question?", ["answer one", "answer two"], "candidate"
    )
    assert messages == [
        {
            "role": "user",
            "content": OFFICIAL_JUDGE_TEMPLATE.format(
                question="Question?",
                official_answer="['answer one', 'answer two']",
                candidate_answer="candidate",
            ),
        }
    ]
    assert LCRAAJudge.parse_judge_response(" CORRECT\n") == (True, "CORRECT")
    assert LCRAAJudge.parse_judge_response("incorrect") == (False, "INCORRECT")
    assert LCRAAJudge.parse_judge_response("CORRECT because...") == (False, None)
    assert LCRAAJudge.parse_judge_response(None) == (False, None)


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.responses)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeConnection:
    def __init__(self, client):
        self.client = client

    async def get_client(self):
        return self.client


@pytest.mark.asyncio
async def test_empty_and_reasoning_fallback_answers_skip_the_judge():
    completions = _FakeCompletions(["CORRECT"])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    grader = object.__new__(LCRAAJudge)
    grader.model = SimpleNamespace(
        name="judge",
        api_model_name="judge-api-name",
        openai_kwargs={"temperature": 0.0, "max_tokens": 16384, "n": 1},
        cache_salt=CacheSaltConfig(),
    )
    grader.openai_connection = _FakeConnection(client)

    sample = {
        "question": "Question?",
        "ground_truth": ["official"],
        "generations": ["final answer", "", "reasoning only"],
        "parsed_generations": ["final answer", None, "reasoning only"],
        "reasoning": ["reasoning only"],
    }
    result = await grader.grade_sample(sample)

    assert result["correct"] == [True, False, False]
    assert result["lcr_aa_grading_source"] == [
        "judge",
        "empty_final",
        "reasoning_substituted_for_empty_final",
    ]
    assert len(completions.calls) == 1
    assert completions.calls[0]["model"] == "judge-api-name"
