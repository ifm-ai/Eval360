from unittest.mock import AsyncMock, MagicMock
import importlib
import sys
import types

import pytest

from scheduler.grader.llm_as_judge import LLMasJudgeBoxedMatch
from scheduler.model import CacheSaltConfig


def _completion(content='</think>{"GRADE": "CORRECT"}'):
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


def _make_grader():
    grader = LLMasJudgeBoxedMatch.__new__(LLMasJudgeBoxedMatch)
    grader.name = "llm_as_judge"
    grader.model = MagicMock()
    grader.model.name = "judge-model"
    grader.model.api_model_name = None
    grader.model.openai_kwargs = {}
    grader.model.cache_salt = CacheSaltConfig()
    grader.client = MagicMock()
    grader.openai_connection = AsyncMock()
    grader.openai_connection.get_client = AsyncMock(return_value=grader.client)
    return grader


@pytest.mark.asyncio
async def test_llm_as_judge_sends_static_cache_salt():
    grader = _make_grader()
    grader.model.cache_salt = CacheSaltConfig(mode="static", salt="judge-partition")
    grader.client.chat.completions.create = AsyncMock(return_value=_completion())
    sample = {
        "completion_input": "Question?",
        "ground_truth": "A",
        "generations": ["A"],
        "parsed_generations": ["A"],
    }

    result = await grader.grade_sample(sample)

    assert result["correct"] == [1]
    kwargs = grader.client.chat.completions.create.await_args.kwargs
    assert kwargs["extra_body"]["cache_salt"] == "judge-partition"


@pytest.mark.asyncio
async def test_llm_as_judge_sends_unique_cache_salt_per_generation():
    grader = _make_grader()
    grader.model.cache_salt = CacheSaltConfig(mode="unique")
    grader.client.chat.completions.create = AsyncMock(return_value=_completion())
    sample = {
        "completion_input": "Question?",
        "ground_truth": "A",
        "generations": ["A", "A"],
        "parsed_generations": ["A", "A"],
    }

    result = await grader.grade_sample(sample)

    assert result["correct"] == [1, 1]
    salts = [
        call.kwargs["extra_body"]["cache_salt"]
        for call in grader.client.chat.completions.create.await_args_list
    ]
    assert len(salts) == 2
    assert salts[0] != salts[1]


@pytest.mark.asyncio
async def test_llm_as_judge_rejects_raw_cache_salt():
    grader = _make_grader()
    grader.model.openai_kwargs = {"extra_body": {"cache_salt": "raw"}}
    grader.client.chat.completions.create = AsyncMock(return_value=_completion())
    sample = {
        "completion_input": "Question?",
        "ground_truth": "A",
        "generations": ["A"],
        "parsed_generations": ["A"],
    }

    with pytest.raises(ValueError, match="extra_body.cache_salt"):
        await grader.grade_sample(sample)


@pytest.mark.asyncio
async def test_llm_as_judge_does_not_mutate_model_openai_kwargs():
    grader = _make_grader()
    grader.model.openai_kwargs = {"extra_body": {"guided_choice": ["A", "B"]}}
    grader.model.cache_salt = CacheSaltConfig(mode="static", salt="judge-partition")
    grader.client.chat.completions.create = AsyncMock(return_value=_completion())
    sample = {
        "completion_input": "Question?",
        "ground_truth": "A",
        "generations": ["A"],
        "parsed_generations": ["A"],
    }

    await grader.grade_sample(sample)

    assert grader.model.openai_kwargs == {"extra_body": {"guided_choice": ["A", "B"]}}


def _install_optional_judge_dependency_stubs(monkeypatch):
    math_verify = types.ModuleType("math_verify")
    math_verify.parse = lambda value: value
    math_verify.verify = lambda left, right: left == right
    monkeypatch.setitem(sys.modules, "math_verify", math_verify)

    pylatexenc = types.ModuleType("pylatexenc")
    latex2text = types.ModuleType("pylatexenc.latex2text")

    class LatexNodes2Text:
        def latex_to_text(self, value):
            return value

    latex2text.LatexNodes2Text = LatexNodes2Text
    pylatexenc.latex2text = latex2text
    monkeypatch.setitem(sys.modules, "pylatexenc", pylatexenc)
    monkeypatch.setitem(sys.modules, "pylatexenc.latex2text", latex2text)


def _make_direct_llm_judge(grader_cls):
    grader = grader_cls.__new__(grader_cls)
    grader.client = MagicMock()
    grader.openai_connection = AsyncMock()
    grader.openai_connection.get_client = AsyncMock(return_value=grader.client)
    grader.model = MagicMock()
    grader.model.name = "judge-model"
    grader.model.api_model_name = None
    grader.model.openai_kwargs = {}
    grader.model.cache_salt = CacheSaltConfig(mode="static", salt="judge-partition")
    return grader


@pytest.mark.asyncio
async def test_math_verify_llm_as_judge_sends_cache_salt(monkeypatch):
    _install_optional_judge_dependency_stubs(monkeypatch)
    module = importlib.import_module("scheduler.grader.math_verify_llm_as_judge")
    grader = _make_direct_llm_judge(module.MathVerifyLLMasJudge)
    completion = MagicMock()
    completion.choices[0].message.content = "Yes"
    grader.client.chat.completions.create = AsyncMock(return_value=completion)

    result = await grader._check_equality_with_llm("42", "42")

    assert result is True
    kwargs = grader.client.chat.completions.create.await_args.kwargs
    assert kwargs["extra_body"]["cache_salt"] == "judge-partition"


@pytest.mark.asyncio
async def test_math_verify_llm_as_judge_rejects_raw_cache_salt(monkeypatch):
    _install_optional_judge_dependency_stubs(monkeypatch)
    module = importlib.import_module("scheduler.grader.math_verify_llm_as_judge")
    grader = _make_direct_llm_judge(module.MathVerifyLLMasJudge)
    grader.model.openai_kwargs = {"extra_body": {"cache_salt": "raw"}}
    grader.client.chat.completions.create = AsyncMock(return_value=MagicMock())

    with pytest.raises(ValueError, match="extra_body.cache_salt"):
        await grader._check_equality_with_llm("42", "42")


@pytest.mark.asyncio
async def test_sympy_llm_as_judge_sends_cache_salt(monkeypatch):
    _install_optional_judge_dependency_stubs(monkeypatch)
    module = importlib.import_module("scheduler.grader.sympy_llm_as_judge")
    grader = _make_direct_llm_judge(module.SympyLLMasJudge)
    completion = MagicMock()
    completion.choices[0].message.content = "Yes"
    grader.client.chat.completions.create = AsyncMock(return_value=completion)

    result = await grader._check_equality_with_llm("42", "42")

    assert result is True
    kwargs = grader.client.chat.completions.create.await_args.kwargs
    assert kwargs["extra_body"]["cache_salt"] == "judge-partition"


@pytest.mark.asyncio
async def test_sympy_llm_as_judge_rejects_raw_cache_salt(monkeypatch):
    _install_optional_judge_dependency_stubs(monkeypatch)
    module = importlib.import_module("scheduler.grader.sympy_llm_as_judge")
    grader = _make_direct_llm_judge(module.SympyLLMasJudge)
    grader.model.openai_kwargs = {"extra_body": {"cache_salt": "raw"}}
    grader.client.chat.completions.create = AsyncMock(return_value=MagicMock())

    with pytest.raises(ValueError, match="extra_body.cache_salt"):
        await grader._check_equality_with_llm("42", "42")
