"""Built-in parser registrations."""
from __future__ import annotations

from collections import deque
import re
from typing import Callable

from .parser_registry import register_parser


def _strip_example_code(code: str) -> str:
    """Keep only imports, function/class defs — strip trailing example usage.

    Models often append example usage like ``print(func(...))`` after the
    function definition.  When this runs in the test harness it can crash
    before the real asserts execute.

    Uses a simple line-based heuristic instead of ast.parse to avoid
    segfaults on Python 3.14 pre-release with malformed generated code.
    """
    lines = code.split("\n")
    kept: list[str] = []
    in_block = False
    indent = 0

    for line in lines:
        stripped = line.lstrip()

        # Start of a new top-level def/class/import
        if re.match(r"^(def |async def |class |import |from \S+ import )", stripped) and not line[0:1].isspace():
            in_block = True
            kept.append(line)
            continue

        # Continuation of an indented block
        if in_block:
            if stripped == "" or line[0:1].isspace():
                kept.append(line)
            else:
                # Non-indented non-def line → end of block, start of example code
                in_block = False
                # Don't append — this is example usage
                continue
        elif not stripped:
            # Blank line between top-level statements — keep if we have content
            if kept:
                kept.append(line)
        elif re.match(r"^(import |from \S+ import )", stripped):
            # Top-level import outside a block
            kept.append(line)

    # Remove trailing blank lines
    while kept and not kept[-1].strip():
        kept.pop()

    return "\n".join(kept) if kept else code

_ANSWER_IS_PATTERN = r"the (?:correct )?answer is "
_THE_ANSWER_IS_LETTER_PATTERN = r"\bthe answer is\b[\s:：-]*[\(\[]?\s*([A-Z])\s*[\)\]]?"

@register_parser("so_the_answer_is")
def _find_after_prefix_parser(generation: str) -> str:
    m = re.search(_ANSWER_IS_PATTERN, generation, flags=re.IGNORECASE)
    if not m:
        return None
    tail = generation[m.end():]
    match = re.match(r"(.*)(?=.)", tail)
    return match.group(1) if match else None


@register_parser("so_the_answer_is_last")
def _find_after_prefix_last_parser(generation: str) -> str:
    """Like so_the_answer_is but takes the LAST occurrence of 'the (correct) answer is'.

    Use for base models that may loop or re-evaluate, producing multiple
    'the answer is' phrases before settling on a final answer.
    """
    positions = list(re.finditer(_ANSWER_IS_PATTERN, generation, flags=re.IGNORECASE))
    if not positions:
        return None
    tail = generation[positions[-1].end():]
    match = re.match(r"(.*)(?=.)", tail)
    return match.group(1) if match else None


@register_parser("passthrough", "noop", "identity")
def _passthrough_parser(generation: str) -> str:
    return generation


@register_parser("answer_tag")
def _answer_tag_parser(generation: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", generation, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None 
    extracted = match.group(1).strip()
    return extracted if extracted else None


@register_parser("think_suffix", "think_tag")
def _think_suffix_parser(generation: str) -> str:
    if generation is None:
        return None

    # Split on any </think...> closing tag variant and keep the tail.
    parts = re.split(r"</think[^>]*>", generation, flags=re.IGNORECASE | re.DOTALL)
    tail = parts[-1].strip() if parts else None
    return tail if tail else None


@register_parser("the_answer_is")
def _the_answer_is_parser(generation: str) -> str:
    match = re.search(_THE_ANSWER_IS_LETTER_PATTERN, generation, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


@register_parser("the_answer_is_last")
def _the_answer_is_last_parser(generation: str) -> str:
    matches = re.findall(_THE_ANSWER_IS_LETTER_PATTERN, generation, flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1]


@register_parser("boxed")
def _boxed_parser(generation: str) -> str:
    if generation is None:
        return generation

    pattern = r'\\(?:boxed|fbox)\s*\{'
    matches = list(re.finditer(pattern, generation, flags=re.IGNORECASE))
    if not matches:
        return None 

    # Start from the last match so we capture the final boxed/fbox expression.
    last_match = matches[-1]
    start = last_match.end()
    brace_count = 1
    i = start
    while i < len(generation) and brace_count > 0:
        if generation[i] == '{':
            brace_count += 1
        elif generation[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return generation[start:i-1].strip()
    return generation

@register_parser("gsm8k_base")
def _gsm8k_parser(generation: str) -> str:
    ans = generation
    if "### " in ans:
        ans = ans.split("### ")[-1].strip(" \n")
    if "**Answer:" in ans:
        ans = ans.split("**Answer:")[-1].strip(" \n")
    if "**Answer**" in ans:
        ans = ans.split("**Answer**")[-1].strip(" \n")
    if "**Final Answer:**" in ans:
        ans = ans.split("**Final Answer:**")[-1].strip(" \n")
    if "Answer:" in ans:
        ans = ans.split("Answer:")[-1].strip(" \n")
    if "boxed{" in ans:
        ans = ans.split("boxed")[-1].strip(" \n")
    if "The answer is " in ans:
        ans = ans.split("The answer is ")[-1].strip(" \n")
    return ans if ans else None

@register_parser("code_completion", "humaneval")
def _code_completion_parser(generation: str) -> str:
    if generation is None:
        return None
    text = generation
    # Strip <think>...</think> blocks (thinking models like Qwen3)
    # Case A: Closed </think> tag — strip everything up to and including it.
    # Use \n* (not \s*) to preserve leading indentation on the code line.
    think_match = re.search(r"</think>\n*", text)
    if think_match:
        text = text[think_match.end():]
    # Case B: Unclosed <think> (truncated by max_tokens) — try to extract code
    elif re.match(r"\s*<think>", text):
        fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1)
        else:
            # Last resort: grab everything after the last blank line
            # (models tend to put code at the end of thinking)
            parts = re.split(r"\n\n", text)
            text = parts[-1] if parts else text
    # Extract code from markdown fences if present
    fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return text if text.strip() else None

def _normalize_mc_match(value: str) -> str:
    candidate = value.strip().upper()
    if candidate in {"1", "2", "3", "4", "5"}:
        return "ABCDE"[int(candidate) - 1]
    return candidate


_MC_EMPHASIS = r"(?:\*\*|__|`)?"
_MC_CHOICE = r"([A-E1-5])"
_MC_LINE_SCAN_LIMIT = 96
_MC_TAIL_SCAN_CHARS = 4096
_THINK_CLOSE_RE = re.compile(r"</think[^>]*>", flags=re.IGNORECASE)

_MC_TAG_PATTERNS = (
    re.compile(
        rf"<(?:final\s*)?answer>\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*</(?:final\s*)?answer>",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"<answer>\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*</answer>",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\\(?:boxed|fbox)\s*\{{\s*{_MC_CHOICE}\s*\}}",
        flags=re.IGNORECASE,
    ),
)

_MC_STRONG_LINE_PATTERNS = (
    re.compile(
        rf"^\s*(?:final[.:]?\s*)?answer\b\s*[:：-]?\s*(?:is\s*)?{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*(?:the\s+)?choice\s+is\s+{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
)

_MC_WEAK_LINE_PATTERNS = (
    re.compile(
        rf"\b(?:final[.:]?\s*)?answer\b\s*[:：-]?\s*(?:is\s*)?{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:the\s+)?choice\s+is\s+{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:match(?:es|ed)?|coincides?\s+with|corresponds?\s+to|gives?|yields?|confirms?)\s+(?:option|choice)\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\bonly\s+(?:option\s*)?{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:option|choice)\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s+(?:is|would\s+be|remains?|works?|fits?|correct|best|right|compatible|consistent|larger)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s+is\s+(?:correct|best|right|compatible|consistent)\b",
        flags=re.IGNORECASE,
    ),
)


def _mc_match_last(pattern: re.Pattern[str], text: str) -> str | None:
    last = None
    for match in pattern.finditer(text):
        last = match
    if last is None:
        return None
    return _normalize_mc_match(last.group(1))


def _collect_mc_scan_lines(text: str) -> tuple[str | None, list[str]]:
    first = None
    trailing = deque(maxlen=_MC_LINE_SCAN_LIMIT)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if first is None:
            first = line
        trailing.append(line)
    return first, list(trailing)


def _extract_mc_from_line(line: str, *, allow_weak: bool) -> str | None:
    for pattern in _MC_TAG_PATTERNS:
        match = pattern.search(line)
        if match:
            return _normalize_mc_match(match.group(1))

    for pattern in _MC_STRONG_LINE_PATTERNS:
        match = pattern.search(line)
        if match:
            return _normalize_mc_match(match.group(1))

    if not allow_weak:
        return None

    for pattern in _MC_WEAK_LINE_PATTERNS:
        match = pattern.search(line)
        if match:
            return _normalize_mc_match(match.group(1))

    return None


def _extract_mc_letter(text: str) -> str | None:
    first_line, trailing_lines = _collect_mc_scan_lines(text)

    if first_line is not None:
        first_line_match = _extract_mc_from_line(first_line, allow_weak=False)
        if first_line_match is not None:
            return first_line_match

    for line in reversed(trailing_lines):
        line_match = _extract_mc_from_line(line, allow_weak=True)
        if line_match is not None:
            return line_match

    # Some models inline the answer tag or boxed answer inside a paragraph. Keep
    # this search bounded to the tail, where the final answer typically appears.
    tail = text[-_MC_TAIL_SCAN_CHARS:]
    for pattern in _MC_TAG_PATTERNS:
        tail_match = _mc_match_last(pattern, tail)
        if tail_match is not None:
            return tail_match

    for pattern in _MC_WEAK_LINE_PATTERNS:
        tail_match = _mc_match_last(pattern, tail)
        if tail_match is not None:
            return tail_match

    return None


@register_parser("mc_answer", "multiple_choice_answer", "think_tag_mc")
def _mc_answer_parser(generation: str) -> str:
    if generation is None:
        return None

    text = generation.strip()
    if not text:
        return None

    # In thinking-mode outputs, prefer the portion after the final </think> tag.
    candidate = text
    last_close = None
    for match in _THINK_CLOSE_RE.finditer(text):
        last_close = match
    if last_close is not None:
        candidate = text[last_close.end():].strip()

    extracted = _extract_mc_letter(candidate)
    if extracted is not None:
        return extracted

    # Fallback to the full text if extraction after </think> fails.
    return _extract_mc_letter(text)


@register_parser("code-think")
def _code_parser(generation: str) -> str:
    if generation is None:
        return generation

    match = re.search(r"</think>(.*)$", generation, flags=re.IGNORECASE | re.DOTALL)
    if match:
        generation = match.group(1).strip()

    matches = re.findall(r"```python3?\n(.*?)```", generation, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1]



def _best_code_block(matches: list[str]) -> str:
    """Pick the best code block from a list of ````` ```python ````` matches.

    Prefers the *last* block that contains a function or class definition.
    When models output the solution in one block and example usage (e.g.
    ``print(func(...))``) in a separate block, this avoids picking the
    example block.  Falls back to the last block if none contain a def.
    """
    for block in reversed(matches):
        if re.search(r"^(def |async def |class )", block, re.MULTILINE):
            return _strip_example_code(block)
    return _strip_example_code(matches[-1])


@register_parser("code")
def _code_block_parser(generation: str) -> str:
    if generation is None:
        return generation
    matches = re.findall(r"```python3?\n(.*?)```", generation, flags=re.DOTALL)
    if not matches:
        return None
    return _best_code_block(matches)


@register_parser("longbench-v2", "the-correct-answer-is")
def _longbench_v2_parser(generation: str) -> str:
    if generation is None:
        return generation
    PATTERNS = [
        re.compile(r"the\s+correct\s+answer\s+is\s*\(?\s*([ABCD])", re.I),
        re.compile(r"\(([ABCD])\)", re.I),
    ]
    for pat in PATTERNS:
        m = pat.search(generation)
        if m:
            return m.group(1).upper()
    return None


__all__: list[str] = []
