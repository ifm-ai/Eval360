"""
ImportedDataset runner for the Berkeley Function-Calling Leaderboard (BFCL).

Usage in dataset YAML::

    imported_dataset:
      name: bfcl
      commit: f7cf7359b7ac615a0b294831c5ba2bc95ee4a000
      args:
        test_category: "all_scoring"            # full BFCL v4 scoring suite
        # test_category: ["non_live", "live", "multi_turn", "agentic"]  # v4 segments

model_instance.name (the VLLM served name) is used as the BFCL model name for
API calls.  The script defines a K2 Horizon vLLM OpenAI-compatible BFCL handler
inline, adapted from BFCL's OpenAICompletionsHandler (Apache-2.0; see
THIRD_PARTY_NOTICES.md), so that BFCL's full FC pipeline is used: chat.completions.create(tools=[...]),
tool_calls parsing, and multi-turn FC support.

chat_template_kwargs (e.g. reasoning_effort) are injected into every
chat.completions.create call via extra_body, patched in __init__.

VLLM is already running on localhost:8000 when the benchmark script executes.
"""
import csv
from pathlib import Path
from textwrap import dedent

from .base import ImportedDatasetResultError, ImportedDatasetRunnerBase
from .registry import register
from ..grader.base import Score
from ..model import ModelInstance
from ..task import ImportedDatasetTask

# Official BFCL v4 leaderboard reproduction package for commit f7cf735.
BFCL_EVAL_VERSION = "2025.12.17"

# Python snippet that registers the model in BFCL's MODEL_CONFIG_MAPPING using
# a K2 Horizon vLLM OpenAI-compatible handler defined inline. The PyPI bfcl-eval
# package used in the job venv has no handler for this model, so the script
# carries its own. Parts of it are adapted from BFCL at commit f7cf735
# (ShishirPatil/gorilla, Apache-2.0); see THIRD_PARTY_NOTICES.md.
# QuickTestingOSSHandler is not used because OSSHandler.inference() hardcodes
# prompting mode regardless of is_fc_model.
#
# chat_template_kwargs (e.g. reasoning_effort) are injected into every
# chat.completions.create call via extra_body by patching the client in __init__.
# The OpenAI-compatible handler reads its base_url/api_key from OPENAI_BASE_URL /
# OPENAI_API_KEY, so those are set from the REMOTE_* env vars at script start.
_BENCHMARK_SCRIPT_TEMPLATE = dedent("""\
    import os as _os
    import re as _re
    import json as _json
    import threading as _threading
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    # The OpenAI-compatible handler reads these env vars in __init__.
    _os.environ.setdefault("OPENAI_BASE_URL", _os.environ.get("REMOTE_OPENAI_BASE_URL", "http://localhost:8000/v1"))
    _os.environ.setdefault("OPENAI_API_KEY",  _os.environ.get("REMOTE_OPENAI_API_KEY", "EMPTY"))

    _model_name = {model_name!r}
    _chat_template_kwargs = {chat_template_kwargs!r}
    _max_tokens = {max_tokens!r}
    _num_threads = {num_threads!r}
    _save_raw_inference = {save_raw_inference!r}
    _raw_log_dir = {raw_log_dir!r}
    _atif_log_dir = {atif_log_dir!r}
    _web_search_backend = {web_search_backend!r}
    _web_search_api_keys_file = {web_search_api_keys_file!r}
    _web_search_api_keys_file_rel = {web_search_api_keys_file_rel!r}
    _web_fetch_force_mode = {web_fetch_force_mode!r}
    _web_fetch_max_chars = {web_fetch_max_chars!r}

    _think_tag_counts = {{"</think>": 0, "</think_fast>": 0, "</think_faster>": 0, "none": 0}}
    _raw_log_lock = _threading.Lock()
    _current_category = None

    # Per-thread state for accurate sample ID and turn tracking.
    # _pre_query_processing_FC is called once per test case (before any API calls),
    # so we stash the test_entry["id"] there.  Within the patched create() we detect
    # turn boundaries by watching for user-message changes on the same thread.
    _thread_local = _threading.local()

    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {{k: _to_serializable(v) for k, v in obj.items()}}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        if hasattr(obj, "model_dump"):
            return _to_serializable(obj.model_dump())
        if hasattr(obj, "__dict__"):
            return _to_serializable(obj.__dict__)
        return obj

    def _inline_reasoning(msg):
        # Inline reasoning_content into content as <think>...</think> for logging.
        # Check for key presence (not truthiness) — K2 Horizon sends <think></think> even
        # for empty reasoning, so the log must reflect that.
        if not isinstance(msg, dict):
            return msg
        if "reasoning_content" not in msg:
            return msg
        msg = dict(msg)
        content = msg.pop("reasoning_content", "") or ""
        msg["content"] = f"<think>{{content}}</think>" + (msg.get("content") or "")
        return msg

    def _extract_qwen_json_tool_calls(content):
        # Qwen's HF chat template asks for JSON inside <tool_call> tags, while
        # some vLLM parsers expect nested XML and leave the JSON in content.
        tool_calls = []
        for idx, raw in enumerate(_re.findall(r"<tool_call>\\s*(.*?)\\s*</tool_call>", content, _re.DOTALL)):
            try:
                payload = _json.loads(raw.strip())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            arguments = payload.get("arguments", {{}})
            if not isinstance(name, str) or not name:
                continue
            if isinstance(arguments, str):
                arguments_str = arguments
            else:
                arguments_str = _json.dumps(arguments, ensure_ascii=False)
            tool_calls.append({{
                "id": f"call_qwen_xml_{{idx}}",
                "type": "function",
                "function": {{"name": name, "arguments": arguments_str}},
            }})
        for idx, raw in enumerate(_re.findall(r"<ifm\\|tool_call>\\s*(.*?)\\s*</ifm\\|tool_call>", content, _re.DOTALL)):
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("{{"):
                try:
                    payload = _json.loads(raw)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    name = payload.get("name")
                    arguments = payload.get("arguments", {{}})
                    if isinstance(name, str) and name:
                        if isinstance(arguments, str):
                            arguments_str = arguments
                        else:
                            arguments_str = _json.dumps(arguments, ensure_ascii=False)
                        tool_calls.append({{
                            "id": f"call_ifm_json_{{idx}}",
                            "type": "function",
                            "function": {{"name": name, "arguments": arguments_str}},
                        }})
                continue
            name = raw.split("<ifm|arg_key>", 1)[0].strip()
            if not name:
                continue
            arguments = {{}}
            for key, _arg_type, value in _re.findall(
                r"<ifm\\|arg_key>\\s*(.*?)\\s*</ifm\\|arg_key>\\s*"
                r"(?:<ifm\\|arg_type>\\s*(.*?)\\s*</ifm\\|arg_type>\\s*)?"
                r"<ifm\\|arg_value>\\s*(.*?)\\s*</ifm\\|arg_value>",
                raw,
                _re.DOTALL,
            ):
                key = (key or "").strip()
                value = (value or "").strip()
                if not key:
                    continue
                try:
                    parsed_value = _json.loads(value)
                except Exception:
                    parsed_value = value
                arguments[key] = parsed_value
            tool_calls.append({{
                "id": f"call_ifm_xml_{{idx}}",
                "type": "function",
                "function": {{"name": name, "arguments": _json.dumps(arguments, ensure_ascii=False)}},
            }})
        return tool_calls

    def _split_k2_horizon_reasoning(content):
        text = content or ""
        for tag in (
            "ifm|think",
            "ifm|think_fast",
            "ifm|think_faster",
            "think",
            "think_fast",
            "think_faster",
        ):
            close_tag = f"</{{tag}}>"
            if close_tag not in text:
                continue
            before, after = text.split(close_tag, 1)
            open_tag = f"<{{tag}}>"
            if open_tag in before:
                before = before.rsplit(open_tag, 1)[-1]
            reasoning = _re.sub(
                r"^<think(?:_fast|_faster)?>\\s*</think(?:_fast|_faster)?>\\s*",
                "",
                before.strip("\\n"),
            ).strip("\\n")
            return reasoning, after.lstrip("\\n")
        return "", text

    def _normalise_k2_horizon_response_content(content, reasoning_content=""):
        inline_reasoning, cleaned_content = _split_k2_horizon_reasoning(content or "")
        return (reasoning_content or inline_reasoning or ""), cleaned_content

    def _response_usage_tokens(api_response):
        usage = getattr(api_response, "usage", None)
        return (
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    def _empty_response_data(api_response):
        input_token, output_token = _response_usage_tokens(api_response)
        return {{
            "model_responses": [],
            "model_responses_message_for_chat_history": {{
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
            }},
            "tool_call_ids": [],
            "input_token": input_token,
            "output_token": output_token,
        }}

    def _choice_message_parts(api_response):
        choice = api_response.choices[0]
        message = getattr(choice, "message", None)
        text = getattr(choice, "text", None) or ""
        field_reasoning = getattr(choice, "reasoning_content", None) or getattr(
            choice, "reasoning", None
        ) or ""
        if message is None:
            return None, text, field_reasoning, []
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None) or []
        field_reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        ) or field_reasoning
        if content is None and not tool_calls and text:
            content = text
        return message, (content or ""), field_reasoning, tool_calls

    def _search_api_key_file_candidates(path, rel_path):
        candidates = []

        def _add(candidate):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        repo_root = _os.environ.get("repo_root")
        if rel_path and repo_root:
            _add(_os.path.join(repo_root, rel_path))
        if rel_path:
            _add(_os.path.abspath(rel_path))
        _add(path)
        return candidates

    def _load_search_api_keys(path, rel_path=None):
        if not path and not rel_path:
            return [], None
        candidates = _search_api_key_file_candidates(path, rel_path)
        resolved_path = next((candidate for candidate in candidates if _os.path.exists(candidate)), None)
        if not resolved_path:
            raise FileNotFoundError("Search API key file not found. Tried: " + ", ".join(candidates))
        loaded = []
        with open(resolved_path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\\\"")
                if not key:
                    continue
                _os.environ[key] = value
                loaded.append(key)
        return loaded, resolved_path

    def _patch_web_search_tooling():
        if not (_web_search_backend or _web_fetch_force_mode or _web_fetch_max_chars):
            return

        try:
            from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI
        except Exception as exc:
            raise RuntimeError(f"Failed to import BFCL WebSearchAPI for patching: {{exc}}") from exc

        loaded_keys, resolved_key_path = _load_search_api_keys(
            _web_search_api_keys_file,
            _web_search_api_keys_file_rel,
        )
        if loaded_keys:
            print(
                f"[WebSearchAPI] Loaded search API keys from {{resolved_key_path}}: "
                + ", ".join(sorted(loaded_keys))
            )

        backend = (_web_search_backend or "").strip().lower()
        if backend:
            if backend != "serper":
                raise ValueError(f"Unsupported web_search_backend={{_web_search_backend!r}}; expected 'serper'")

            # Adapted from BFCL's WebSearchAPI.search_engine_query (Apache-2.0),
            # modified to call Serper instead of SerpAPI.
            def _serper_search_engine_query(self, keywords, max_results=10, region="wt-wt"):
                import random as _random
                import time as _time
                import requests as _requests

                api_key = _os.getenv("SERPER_API_KEY")
                if not api_key:
                    return {{"error": "SERPER_API_KEY is not set"}}

                max_results = 10 if max_results is None else int(max_results)
                payload = {{"q": keywords, "num": max_results}}
                headers = {{"X-API-KEY": api_key, "Content-Type": "application/json"}}
                backoff = 2
                max_429_retries = 5
                retry_count = 0

                while True:
                    try:
                        response = _requests.post(
                            "https://google.serper.dev/search",
                            headers=headers,
                            json=payload,
                            timeout=20,
                        )
                        if response.status_code == 429:
                            retry_count += 1
                            if retry_count > max_429_retries:
                                return {{
                                    "error": (
                                        "Serper search failed: received HTTP 429 "
                                        f"after {{max_429_retries}} retries"
                                    )
                                }}
                            wait_time = backoff + _random.uniform(0, backoff)
                            print(
                                "*" * 100
                                + f"\\n[WebSearchAPI] Received 429 from Serper. Retrying in {{wait_time:.1f}} seconds..."
                                + "*" * 100
                            )
                            _time.sleep(wait_time)
                            backoff = min(backoff * 2, 120)
                            continue
                        response.raise_for_status()
                        search_results = response.json()
                        break
                    except Exception as exc:
                        return {{"error": f"Serper search failed: {{exc}}"}}

                organic = search_results.get("organic") or []
                if not organic:
                    return {{"error": "Failed to retrieve the search results from Serper."}}

                results = []
                for result in organic[:max_results]:
                    item = {{
                        "title": result.get("title", ""),
                        "href": result.get("link", ""),
                    }}
                    if getattr(self, "show_snippet", True):
                        item["body"] = result.get("snippet", "")
                    results.append(item)
                return results

            WebSearchAPI.search_engine_query = _serper_search_engine_query
            print("[WebSearchAPI] Patched search_engine_query backend=serper")

        if _web_fetch_force_mode or _web_fetch_max_chars:
            _original_fetch_url_content = WebSearchAPI.fetch_url_content

            def _patched_fetch_url_content(self, url, mode="raw"):
                fetch_mode = _web_fetch_force_mode or mode
                result = _original_fetch_url_content(self, url, mode=fetch_mode)
                if (
                    _web_fetch_max_chars
                    and isinstance(result, dict)
                    and isinstance(result.get("content"), str)
                    and len(result["content"]) > int(_web_fetch_max_chars)
                ):
                    result = dict(result)
                    result["content"] = result["content"][: int(_web_fetch_max_chars)]
                return result

            WebSearchAPI.fetch_url_content = _patched_fetch_url_content
            print(
                f"[WebSearchAPI] Patched fetch_url_content: force_mode={{_web_fetch_force_mode}}, "
                f"max_chars={{_web_fetch_max_chars}}"
            )

    def _build_atif_steps(turns):
        # Convert accumulated (messages, tools, response) turns to ATIF steps.
        # Tool results from a turn appear as new messages in the *next* turn's
        # message list, so we look ahead to populate each agent step's observation.
        steps = []
        step_id = 1
        total_pt = 0
        total_ct = 0
        prev_len = 0
        for turn_idx, (messages, tools, response) in enumerate(turns):
            new_msgs = messages[prev_len:]
            prev_len = len(messages)
            # Emit system / user messages that are new this turn.
            # Tool-result messages (role "tool") are wired into the previous
            # agent step's observation, not emitted as standalone steps.
            for msg in new_msgs:
                role = msg.get("role") if isinstance(msg, dict) else None
                content = (msg.get("content") if isinstance(msg, dict) else None) or ""
                if role == "system":
                    steps.append({{"step_id": step_id, "source": "system", "message": content}})
                    step_id += 1
                elif role == "user":
                    steps.append({{"step_id": step_id, "source": "user", "message": content}})
                    step_id += 1
            # Extract assistant response fields.
            choice = ((response.get("choices") or [{{}}])[0])
            asst = choice.get("message") or {{}}
            content = asst.get("content") or ""
            reasoning = asst.get("reasoning_content")
            tc_raw = asst.get("tool_calls") or []
            usage = response.get("usage") or {{}}
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            total_pt += pt
            total_ct += ct
            # Build structured tool_calls list.
            tool_calls = []
            for tc in tc_raw:
                fn = tc.get("function") or {{}}
                args = fn.get("arguments") or "{{}}"
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except Exception:
                        pass
                tool_calls.append({{
                    "tool_call_id": tc.get("id") or f"call_{{step_id}}_{{len(tool_calls)}}",
                    "function_name": fn.get("name") or "",
                    "arguments": args,
                }})
            # Look ahead to next turn's messages to find tool results (observations).
            observation = None
            if tool_calls and turn_idx + 1 < len(turns):
                next_messages = turns[turn_idx + 1][0]
                next_new = next_messages[len(messages):]
                tool_results = [
                    {{"source_call_id": m.get("tool_call_id"), "content": m.get("content") or ""}}
                    for m in next_new
                    if isinstance(m, dict) and m.get("role") == "tool"
                ]
                if tool_results:
                    observation = {{"results": tool_results}}
            agent_step = {{
                "step_id": step_id,
                "source": "agent",
                "message": content,
                "metrics": {{"prompt_tokens": pt, "completion_tokens": ct}},
            }}
            if reasoning:
                agent_step["reasoning_content"] = reasoning
            if tool_calls:
                agent_step["tool_calls"] = tool_calls
            if observation:
                agent_step["observation"] = observation
            steps.append(agent_step)
            step_id += 1
        return steps, total_pt, total_ct

    def _write_atif(sample_id, category):
        if not _save_raw_inference:
            return
        try:
            turns = getattr(_thread_local, "atif_turns", [])
            if not turns:
                return
            steps, total_pt, total_ct = _build_atif_steps(turns)
            trajectory = {{
                "schema_version": "ATIF-v1.4",
                "session_id": f"{{category or 'unknown'}}/{{sample_id}}",
                "agent": {{"name": "eval360-bfcl", "model_name": _model_name}},
                "steps": steps,
                "final_metrics": {{
                    "total_prompt_tokens": total_pt,
                    "total_completion_tokens": total_ct,
                    "total_steps": len(steps),
                }},
            }}
            sample_dir = _os.path.join(_atif_log_dir, category or "unknown")
            _os.makedirs(sample_dir, exist_ok=True)
            path = _os.path.join(sample_dir, f"{{sample_id}}.jsonl")
            with open(path, "w") as _f:
                _f.write(_json.dumps(trajectory) + "\\n")
        except Exception:
            pass

    def _append_raw_log(messages, response, kwargs=None):
        if not _save_raw_inference:
            return
        try:
            # Find the last user message to detect turn boundaries
            last_user_msg = next(
                (m for m in reversed(messages) if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "user"),
                None,
            )
            user_content = (last_user_msg.get("content", "") if isinstance(last_user_msg, dict) else getattr(last_user_msg, "content", "")) or ""

            # Detect turn boundary: user message changed since last call on this thread
            last_user = getattr(_thread_local, "last_user", None)
            if last_user is None:
                _thread_local.turn = 0
            elif user_content != last_user:
                _thread_local.turn = getattr(_thread_local, "turn", 0) + 1
            _thread_local.last_user = user_content

            sample_id = getattr(_thread_local, "test_case_id", "unknown")
            turn = getattr(_thread_local, "turn", 0)
            sample_dir = _os.path.join(_raw_log_dir, _current_category or "unknown", str(sample_id))
            _os.makedirs(sample_dir, exist_ok=True)
            path = _os.path.join(sample_dir, f"turn_{{turn}}.jsonl")
            serialized_messages = [_inline_reasoning(m) for m in _to_serializable(messages)]
            tools = _to_serializable(kwargs.get("tools")) if kwargs else None
            entry = {{"messages": serialized_messages, "tools": tools, "response": response}}
            with _raw_log_lock:
                with open(path, "a") as _f:
                    _f.write(_json.dumps(entry) + "\\n")
        except Exception:
            pass

    class _LocalK2HorizonVLLMOpenAICompletionsHandler(OpenAICompletionsHandler):
        # Adapted from BFCL's OpenAICompletionsHandler (Apache-2.0); _query_FC and
        # _add_vllm_reasoning_content_if_available are modified copies of its
        # _query_FC and _add_reasoning_content_if_available_FC. The bfcl-eval
        # package installed in the job venv has no handler for this model.
        TOOL_FORMAT_ALIASES = {{
            "desv32": "dsv32",
        }}
        SUPPORTED_TOOL_FORMATS = {{
            "default",
            "qwen3",
            "minimax",
            "glm",
            "dsv32",
            "gptoss",
            "python",
        }}
        SUPPORTED_TOOL_PRESENTATION_FORMATS = {{"json", "xml", "markdown"}}
        SUPPORTED_TOOL_CALL_FORMATS = {{"json", "xml", "xml_typed"}}
        SUPPORTED_REASONING_EFFORTS = {{"high", "medium", "low"}}
        SUPPORTED_TOOL_CHOICES = {{"auto", "required"}}

        def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
            super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

            served_model_name = kwargs.get("served_model_name") or _os.getenv(
                "BFCL_SERVED_MODEL_NAME"
            )
            if served_model_name:
                self.model_name = served_model_name

            requested_tool_format = (
                kwargs.get("tool_format")
                or _os.getenv("BFCL_TOOL_FORMAT")
                or "default"
            ).strip().lower()
            self.tool_format = self.TOOL_FORMAT_ALIASES.get(
                requested_tool_format, requested_tool_format
            )

            self.tool_presentation_format = (
                kwargs.get("tool_presentation_format")
                or _os.getenv("BFCL_TOOL_PRESENTATION_FORMAT")
            )
            if self.tool_presentation_format:
                self.tool_presentation_format = self.tool_presentation_format.strip().lower()
                if self.tool_presentation_format not in self.SUPPORTED_TOOL_PRESENTATION_FORMATS:
                    raise ValueError(
                        "Unsupported K2 Horizon tool_presentation_format "
                        + repr(self.tool_presentation_format)
                        + ". Supported values: "
                        + repr(sorted(self.SUPPORTED_TOOL_PRESENTATION_FORMATS))
                    )

            self.tool_call_format = (
                kwargs.get("tool_call_format")
                or _os.getenv("BFCL_TOOL_CALL_FORMAT")
            )
            if self.tool_call_format:
                self.tool_call_format = self.tool_call_format.strip().lower()
                if self.tool_call_format not in self.SUPPORTED_TOOL_CALL_FORMATS:
                    raise ValueError(
                        "Unsupported K2 Horizon tool_call_format "
                        + repr(self.tool_call_format)
                        + ". Supported values: "
                        + repr(sorted(self.SUPPORTED_TOOL_CALL_FORMATS))
                    )

            self.reasoning_effort = (
                kwargs.get("reasoning_effort")
                or _os.getenv("BFCL_REASONING_EFFORT")
                or "high"
            ).strip().lower()
            self.tool_choice = (
                kwargs.get("tool_choice") or _os.getenv("BFCL_TOOL_CHOICE") or "auto"
            ).strip().lower()

            if self.tool_format not in self.SUPPORTED_TOOL_FORMATS:
                supported_tool_formats = sorted(
                    self.SUPPORTED_TOOL_FORMATS | self.TOOL_FORMAT_ALIASES.keys()
                )
                raise ValueError(
                    "Unsupported K2 Horizon tool_format "
                    + repr(self.tool_format)
                    + ". Supported values: "
                    + repr(supported_tool_formats)
                )
            if self.reasoning_effort not in self.SUPPORTED_REASONING_EFFORTS:
                raise ValueError(
                    "Unsupported K2 Horizon reasoning_effort "
                    + repr(self.reasoning_effort)
                    + ". Supported values: "
                    + repr(sorted(self.SUPPORTED_REASONING_EFFORTS))
                )
            if self.tool_choice not in self.SUPPORTED_TOOL_CHOICES:
                raise ValueError(
                    "Unsupported K2 Horizon tool_choice "
                    + repr(self.tool_choice)
                    + ". Supported values: "
                    + repr(sorted(self.SUPPORTED_TOOL_CHOICES))
                )

        def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
            # Pre-existing assistant messages in test data lack reasoning_content.
            # Inject a placeholder so the strict jinja template doesn't raise.
            for msg in first_turn_message:
                if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                    msg["reasoning_content"] = ""
            return super().add_first_turn_message_FC(inference_data, first_turn_message)

        def _query_FC(self, inference_data: dict):
            message: list[dict] = inference_data["message"]
            tools = inference_data["tools"]
            chat_template_kwargs = {{"reasoning_effort": self.reasoning_effort}}
            if self.tool_presentation_format or self.tool_call_format:
                chat_template_kwargs["tool_presentation_format"] = self.tool_presentation_format
                chat_template_kwargs["tool_call_format"] = self.tool_call_format
            else:
                chat_template_kwargs["tool_format"] = self.tool_format
            extra_body = {{"chat_template_kwargs": chat_template_kwargs}}
            inference_data["inference_input_log"] = {{
                "message": repr(message),
                "tools": tools,
                "tool_choice": self.tool_choice,
                "extra_body": extra_body,
            }}

            kwargs = {{
                "messages": message,
                "model": self.model_name,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
                "store": False,
                "extra_body": extra_body,
            }}
            if len(tools) > 0:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = self.tool_choice
            return self.generate_with_backoff(**kwargs)

        def _parse_query_response_FC(self, api_response) -> dict:
            try:
                response_data = super()._parse_query_response_FC(api_response)
            except Exception:
                response_data = _empty_response_data(api_response)
            self._add_vllm_reasoning_content_if_available(api_response, response_data)
            return response_data

        @staticmethod
        def _add_vllm_reasoning_content_if_available(api_response, response_data: dict) -> None:
            _message, content, field_reasoning, tool_calls = _choice_message_parts(api_response)
            reasoning_content, content = _normalise_k2_horizon_response_content(
                content,
                field_reasoning,
            )
            response_data["reasoning_content"] = reasoning_content

            if tool_calls:
                response_data["model_responses_message_for_chat_history"] = {{
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": reasoning_content,
                    "tool_calls": [
                        {{
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {{
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            }},
                        }}
                        for tool_call in tool_calls
                    ],
                }}
            else:
                response_data["model_responses_message_for_chat_history"] = {{
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning_content,
                }}

    class _Eval360K2HorizonHandler(_LocalK2HorizonVLLMOpenAICompletionsHandler):
        # Injects chat_template_kwargs into every chat.completions.create call,
        # saves raw inputs/outputs (including thinking tokens) to raw_inference_log.jsonl,
        # and tracks the distribution of thinking-token tags in model responses.

        def __init__(self, *a, **kw):
            if _chat_template_kwargs:
                for attr in (
                    "tool_format",
                    "tool_presentation_format",
                    "tool_call_format",
                    "reasoning_effort",
                    "tool_choice",
                ):
                    value = _chat_template_kwargs.get(attr)
                    if value is not None and attr not in kw:
                        kw[attr] = value
            super().__init__(*a, **kw)
            if _chat_template_kwargs:
                print(
                    f"[Eval360K2HorizonHandler] Using chat_template_kwargs {{_chat_template_kwargs}}"
                )

        def generate_with_backoff(self, **kwargs):
            if _chat_template_kwargs:
                eb = kwargs.setdefault("extra_body", {{}})
                merged_kwargs = dict(eb.get("chat_template_kwargs") or {{}})
                merged_kwargs.update({{
                    key: value
                    for key, value in _chat_template_kwargs.items()
                    if value is not None
                }})
                normalized_kwargs = {{}}
                if getattr(self, "reasoning_effort", None):
                    normalized_kwargs["reasoning_effort"] = self.reasoning_effort
                if getattr(self, "tool_presentation_format", None) or getattr(self, "tool_call_format", None):
                    merged_kwargs.pop("tool_format", None)
                    if getattr(self, "tool_presentation_format", None):
                        normalized_kwargs["tool_presentation_format"] = self.tool_presentation_format
                    if getattr(self, "tool_call_format", None):
                        normalized_kwargs["tool_call_format"] = self.tool_call_format
                elif getattr(self, "tool_format", None):
                    normalized_kwargs["tool_format"] = self.tool_format
                if getattr(self, "tool_choice", None) and "tool_choice" in merged_kwargs:
                    normalized_kwargs["tool_choice"] = self.tool_choice
                if normalized_kwargs:
                    merged_kwargs.update(normalized_kwargs)
                eb["chat_template_kwargs"] = merged_kwargs
            if _max_tokens is not None:
                kwargs.setdefault("max_tokens", _max_tokens)
            response, latency = super().generate_with_backoff(**kwargs)
            try:
                raw_response = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else response.to_dict()
                )
                _append_raw_log(kwargs.get("messages", []), raw_response, kwargs)
                if _save_raw_inference:
                    if not hasattr(_thread_local, "atif_turns"):
                        _thread_local.atif_turns = []
                    _thread_local.atif_turns.append((
                        _to_serializable(kwargs.get("messages", [])),
                        _to_serializable(kwargs.get("tools")),
                        raw_response,
                    ))
                    _write_atif(getattr(_thread_local, "test_case_id", "unknown"), _current_category)
            except Exception:
                pass
            return response, latency

        def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
            # Called once per test case before any API calls — stash the ID and
            # reset turn tracking so turns are numbered correctly within each sample.
            _thread_local.test_case_id = test_entry["id"]
            _thread_local.last_user = None
            _thread_local.turn = 0
            _thread_local.atif_turns = []
            return super()._pre_query_processing_FC(inference_data, test_entry)

        def _parse_query_response_FC(self, api_response) -> dict:
            result = super()._parse_query_response_FC(api_response)
            try:
                _message, content, field_reasoning, tool_calls = _choice_message_parts(api_response)
                reasoning, cleaned_content = _normalise_k2_horizon_response_content(
                    content,
                    field_reasoning,
                )
                parsed_tool_calls = _extract_qwen_json_tool_calls(cleaned_content)
                if not tool_calls and parsed_tool_calls:
                    result["model_responses"] = [
                        {{tc["function"]["name"]: tc["function"]["arguments"]}}
                        for tc in parsed_tool_calls
                    ]
                    result["tool_call_ids"] = [tc["id"] for tc in parsed_tool_calls]
                    result["model_responses_message_for_chat_history"] = {{
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": reasoning,
                        "tool_calls": parsed_tool_calls,
                    }}
                elif not tool_calls and cleaned_content and result.get("model_responses") == []:
                    # BFCL v4 agentic tasks (memory / web_search) finish with a
                    # natural-language answer after tool use.  The upstream
                    # OpenAI handler currently treats an empty tool_calls list as
                    # an empty function-call response and drops message.content,
                    # which makes the agentic scorer report no_last_message.
                    result["model_responses"] = cleaned_content
                    result["tool_call_ids"] = []
                    result["model_responses_message_for_chat_history"] = {{
                        "role": "assistant",
                        "content": cleaned_content,
                        "reasoning_content": reasoning,
                    }}
                elif not tool_calls and result.get("model_responses") == content:
                    result["model_responses"] = cleaned_content
                    result["model_responses_message_for_chat_history"] = {{
                        "role": "assistant",
                        "content": cleaned_content,
                        "reasoning_content": reasoning,
                    }}
            except Exception:
                pass
            try:
                _message, content, _field_reasoning, _tool_calls = _choice_message_parts(api_response)
                m = _re.search(r'</(?:ifm\\|)?think(_fast|_faster)?>', content, _re.DOTALL)
                if m:
                    prefix = "ifm|" if "ifm|" in m.group(0) else ""
                    tag = "</" + prefix + "think" + (m.group(1) or "") + ">"
                    _think_tag_counts[tag] = _think_tag_counts.get(tag, 0) + 1
                else:
                    _think_tag_counts["none"] += 1
            except Exception:
                _think_tag_counts["none"] += 1
            return result

    # Register under the exact served name AND the slash form (e.g. "k2_horizon_25000"
    # and "k2/horizon/25000") because BFCL's evaluate step reverses the path-safe
    # underscore substitution when looking up the handler config.
    _config = ModelConfig(
        model_name=_model_name,
        display_name=_model_name,
        url="",
        org="",
        license="",
        model_handler=_Eval360K2HorizonHandler,
        is_fc_model=True,
        underscore_to_dot=True,
    )
    MODEL_CONFIG_MAPPING[_model_name] = _config
    _model_name_slash = _model_name.replace("_", "/")
    if _model_name_slash != _model_name:
        MODEL_CONFIG_MAPPING[_model_name_slash] = _config

    from bfcl_eval.__main__ import cli
    _patch_web_search_tooling()

    # Typer raises SystemExit(0) on success; catch it so the script doesn't
    # abort before the next step runs. Non-zero exits propagate to fail the job.
    for _category in {test_categories!r}:
        _current_category = _category
        try:
            cli(["generate",
                "--model", _model_name,
                "--test-category", _category,
                "--temperature", str({temperature!r}),
                "--allow-overwrite",
                "--num-threads", str(_num_threads),
            ])
        except SystemExit as e:
            if e.code:
                raise

        try:
            cli(["evaluate",
                "--model", _model_name,
                "--test-category", _category,
            ])
        except SystemExit as e:
            if e.code:
                raise

    total = sum(_think_tag_counts.values())
    print("Eval360K2HorizonHandler think tag summary:")
    for tag, count in sorted(_think_tag_counts.items()):
        pct = 100 * count / total if total else 0
        print(f"  {{tag}}: {{count}} ({{pct:.1f}}%)")

    with open({think_tag_file!r}, "w") as _f:
        _json.dump(_think_tag_counts, _f)
""")


@register("bfcl")
class BFCLRunner(ImportedDatasetRunnerBase):

    def build_setup_script(self, repo_root: Path) -> str:
        return f'"$VENV/bin/pip" install "bfcl-eval=={BFCL_EVAL_VERSION}" soundfile'

    def build_benchmark_script(
        self,
        model_instance: ModelInstance,
        task: ImportedDatasetTask,
        output_dir: Path,
    ) -> str:
        model_name = model_instance.name  # must match --served-model-name set by VLLM
        raw = task.imported_dataset.args.get("test_category", "non_live")
        test_categories = raw if isinstance(raw, list) else [raw]
        save_raw_inference = bool(task.imported_dataset.args.get("save_raw_inference", False))
        chat_template_kwargs = (
            (model_instance.openai_kwargs or {}).get("extra_body", {}).get("chat_template_kwargs") or {}
        )
        temperature = (model_instance.openai_kwargs or {}).get("temperature", 0.001)
        max_tokens = (model_instance.openai_kwargs or {}).get("max_tokens")
        num_threads = int(task.imported_dataset.args.get("num_threads", 50))
        web_search_backend = task.imported_dataset.args.get("web_search_backend")
        web_search_api_keys_file = task.imported_dataset.args.get("web_search_api_keys_file")
        web_search_api_keys_file_rel = self._repo_relative_path(web_search_api_keys_file)
        web_fetch_force_mode = task.imported_dataset.args.get("web_fetch_force_mode")
        web_fetch_max_chars = task.imported_dataset.args.get("web_fetch_max_chars")
        think_tag_file = output_dir / "think_tag_counts.json"
        raw_log_dir = output_dir / "raw_inference"
        atif_log_dir = output_dir / "atif_trajectories"
        script = _BENCHMARK_SCRIPT_TEMPLATE.format(
            model_name=model_name, test_categories=test_categories,
            chat_template_kwargs=chat_template_kwargs,
            max_tokens=max_tokens,
            num_threads=num_threads,
            temperature=temperature,
            think_tag_file=str(think_tag_file),
            save_raw_inference=save_raw_inference,
            raw_log_dir=str(raw_log_dir),
            atif_log_dir=str(atif_log_dir),
            web_search_backend=web_search_backend,
            web_search_api_keys_file=web_search_api_keys_file,
            web_search_api_keys_file_rel=web_search_api_keys_file_rel,
            web_fetch_force_mode=web_fetch_force_mode,
            web_fetch_max_chars=web_fetch_max_chars,
        )

        return f"""\
export BFCL_PROJECT_ROOT="{output_dir}"
export REMOTE_OPENAI_BASE_URL="http://localhost:8000/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"

"$VENV/bin/python" - <<'BFCL_EOF'
{script}
BFCL_EOF
"""

    @staticmethod
    def _repo_relative_path(path_value: str | None) -> str | None:
        if not path_value:
            return None
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            return str(path)

        roots = [Path.cwd()]
        try:
            roots.append(Path(__file__).resolve().parents[2])
        except IndexError:
            pass

        for root in roots:
            try:
                return str(path.resolve().relative_to(root.resolve()))
            except (OSError, ValueError):
                continue
        return None

    def parse_results(self, output_dir: Path, task: ImportedDatasetTask, model_instance: ModelInstance) -> list[Score]:
        # BFCL aggregates all category results into data_overall.csv after evaluate runs.
        # Using this as the source of truth is simpler and handles all categories uniformly,
        # since some categories (web_search, memory) are stored under agentic/ rather than
        # their own subdirectory.
        csv_path = output_dir / "score" / "data_overall.csv"
        if not csv_path.exists():
            raise ImportedDatasetResultError(
                f"BFCL overall results not found: {csv_path}. "
                f"Run 'bfcl evaluate' to generate scores."
            )

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ImportedDatasetResultError(f"BFCL overall results CSV is empty: {csv_path}")

        row = rows[0]
        scores = []
        for col, val in row.items():
            if not val or val == "N/A" or not val.endswith("%"):
                continue
            try:
                name = "bfcl_" + col.lower().replace(" ", "_").replace("-", "_")
                scores.append(Score(name=name, value=float(val.rstrip("%")) / 100))
            except ValueError:
                pass

        if not scores:
            raise ImportedDatasetResultError(f"No accuracy scores found in {csv_path}")

        think_tag_file = output_dir / "think_tag_counts.json"
        if think_tag_file.exists():
            import json
            counts = json.loads(think_tag_file.read_text())
            total = sum(counts.values())
            if total:
                for tag, count in counts.items():
                    safe = tag.lstrip("<").rstrip(">").replace("/", "").replace("_", "_")
                    scores.append(Score(name=f"bfcl_think_tag_{safe}_frac", value=count / total))

        return scores
