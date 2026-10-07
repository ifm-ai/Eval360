"""Parser utilities and registry."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, Sequence


class Parser:
    """Parser wrapper that normalizes generation text through a callable.

    In addition to parsing (stripping reasoning tags, extracting answers, etc.),
    a Parser is responsible for extracting structured metadata from each raw
    generation via :meth:`extract_reasoning_and_tools`.  The metadata is written
    into ``parsed_*`` fields of the grade record by
    ``GraderBase.parse_generations``.

    **Custom parsers** registered with :func:`register_parser` wrap a plain
    callable and inherit the default ``extract_reasoning_and_tools``
    implementation.  If you need to customise metadata extraction, subclass
    ``Parser`` directly::

        from scheduler.grader.parser_registry import Parser, register_parser

        class MyParser(Parser):
            def extract_reasoning_and_tools(self, generation):
                # extend: also capture a custom <scratchpad> tag
                result = super().extract_reasoning_and_tools(generation)
                m = re.search(r"<scratchpad>(.*?)</scratchpad>",
                              generation or "", re.DOTALL)
                if m:
                    result["scratchpad"] = m.group(1).strip()
                return result

        # Register an instance of your subclass directly:
        _PARSER_REGISTRY["my_parser"] = MyParser(lambda g: g)
    """

    def __init__(self, processor: Callable[[str], str]):
        self._processor = processor

    def parse_generations(self, generations: Sequence[Any]) -> list[str]:
        return [self._processor(generation) for generation in generations]

    def extract_reasoning_and_tools(self, generation: str) -> dict:
        """Extract reasoning and tool_call blocks from a raw generation string.

        Called once per generation by ``GraderBase.parse_generations``.  The
        returned dict is merged into ``parsed_*`` fields of the grade record
        (e.g. ``parsed_reasoning``, ``parsed_answer``, ``parsed_tool_calls``).

        **Default behaviour** handles:

        * ``<think>…</think>`` and all ``<think_*>`` variants
          (``<think_fast>``, ``<think_faster>``, etc., case-insensitive)
        * Orphaned closing tag — when the opening ``<think*>`` tag is in the
          prompt prefix so the generation starts mid-block
        * Unclosed opening tag — generation was truncated before ``</think*>``
        * ``<tool_call>…</tool_call>`` blocks (closed and unclosed)

        **Keys present only when content is found:**

        * ``"reasoning"`` — text inside the think block (tags stripped)
        * ``"answer"`` — text after the closing think tag; ``None`` for
          unclosed think tags
        * ``"tool_calls"`` — list of strings, one per ``<tool_call>`` block
        * ``"unclosed_think_tag"`` — ``True`` when no closing tag was found
        * ``"unclosed_tool_call"`` — ``True`` when no closing tag was found
        * ``"multiple_think_blocks"`` — ``True`` when more than one closed
          think block is present (only the first is captured)

        Override this in a ``Parser`` subclass to customise metadata extraction
        for a specific model or output format (see class docstring for example).
        """
        if not generation:
            return {}
        fields = {}

        # Reasoning: try closed <think*>...</think*> first
        think_match = re.search(
            r"<think[^>]*>(.*?)</think[^>]*>",
            generation,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if think_match:
            all_think = re.findall(
                r"<think[^>]*>.*?</think[^>]*>",
                generation,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if len(all_think) > 1:
                fields["multiple_think_blocks"] = True
            fields["reasoning"] = think_match.group(1).strip()
            after_think = generation[think_match.end():].strip()
            fields["answer"] = after_think if after_think else None
        else:
            # Fall back: orphaned closing </think*> tag (opening tag was in the prompt prefix)
            orphan_close = re.search(
                r"^(.*?)</think[^>]*>(.*)",
                generation,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if orphan_close:
                fields["reasoning"] = orphan_close.group(1).strip()
                after_think = orphan_close.group(2).strip()
                fields["answer"] = after_think if after_think else None
            else:
                # Fall back to unclosed <think*> tag (no closing tag at all)
                open_think = re.search(
                    r"<think[^>]*>(.*)",
                    generation,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if open_think:
                    fields["reasoning"] = open_think.group(1).strip()
                    fields["answer"] = None
                    fields["unclosed_think_tag"] = True

        # Tool calls: try closed <tool_call>...</tool_call> first
        tool_matches = re.findall(
            r"<tool_call>(.*?)</tool_call>",
            generation,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if tool_matches:
            fields["tool_calls"] = [tc.strip() for tc in tool_matches]
            if "answer" not in fields:
                stripped = re.sub(
                    r"<tool_call>.*?</tool_call>",
                    "",
                    generation,
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip()
                fields["answer"] = stripped if stripped else None
        else:
            open_tool = re.search(
                r"<tool_call>(.*)",
                generation,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if open_tool:
                fields["tool_calls"] = [open_tool.group(1).strip()]
                fields["unclosed_tool_call"] = True
                if "answer" not in fields:
                    fields["answer"] = None

        return fields


_PARSER_REGISTRY: Dict[str, Parser] = {}
DEFAULT_PARSER = "passthrough"


def register_parser(*names: str):
    """Decorator for registering a parser callable under the given name."""

    def decorator(func: Callable[[str], str]):
        for name in names:
            if name in _PARSER_REGISTRY:
                raise ValueError(
                    f"Duplicate parser registration: '{name}' is already registered"
                )
            _PARSER_REGISTRY[name] = Parser(func)
        return func

    return decorator


def list_parsers() -> Iterable[str]:
    return iter(_PARSER_REGISTRY.keys())


def get_parser(name: str | None = None) -> Parser:
    try:
        return _PARSER_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(list_parsers()))
        raise KeyError(f"Unknown parser '{name}'. Available: {available}") from exc


__all__ = ["get_parser", "list_parsers", "register_parser", "DEFAULT_PARSER"]
