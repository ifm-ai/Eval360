"""Parser for the GPQA AA prompt contract."""

from __future__ import annotations

import re

from .parser_registry import register_parser


GPQA_AA_REASONING_CLOSE_TAGS = (
    "</ifm|think>",
    "</ifm|think_fast>",
    "</ifm|think_faster>",
    "</think>",
    "</think_fast>",
    "</think_faster>",
)


@register_parser("gpqa_aa_visible_final")
def gpqa_aa_visible_final(generation: str | None) -> str | None:
    """Extract the GPQA AA choice from the visible final response.

    This intentionally mirrors the GPQA AA scorer. In particular,
    once a K2 reasoning-close tag is present, choices in hidden reasoning are
    ignored and a missing visible answer is scored as missing.
    """
    if not generation:
        return None
    text = str(generation).strip()

    closing_tag_positions = [(text.rfind(tag), tag) for tag in GPQA_AA_REASONING_CLOSE_TAGS]
    position, tag = max(closing_tag_positions)
    if position >= 0:
        text = text[position + len(tag):].strip()
        if not text:
            return None

    patterns = (
        r"\\boxed\s*\{\s*([A-Z])\s*\}",
        r"\b(?:the\s+)?(?:final\s+|correct\s+)?answer[*_`]*\s*(?:is|:|-)"
        r"[*_`]*\s*(?:option\s+|choice\s+)?[*_`]*[\(\[]?([A-Z])\b",
        r"\b(?:the\s+)?(?:final\s+|correct\s+)?(?:answer\s+)?(?:option|choice)"
        r"[*_`]*\s*(?:(?:is|:|-)[*_`]*\s*)?[*_`]*[\(\[]?([A-Z])\b",
        r"\b(?:I\s+)?(?:choose|select)\s+(?:option\s+|choice\s+)?"
        r"[*_`]*[\(\[]?([A-Z])\b",
    )
    candidates: list[tuple[int, str]] = []
    for pattern in patterns:
        candidates.extend(
            (match.start(), match.group(1).upper())
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )
    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1]

    leading_choice = re.match(
        r"^\s*[*_`]*[\(\[]?([A-Z])[\)\]]?[*_`]*"
        r"(?:[.:\-\u2013\u2014][*_`]*|\s)",
        text,
        flags=re.IGNORECASE,
    )
    if leading_choice:
        return leading_choice.group(1).upper()

    line_choices = {
        match.group(1).upper()
        for match in re.finditer(
            r"(?:^|\n)\s*[*_`]*[\(\[]?([A-Z])[\)\].:][*_`]*(?:\s|$)",
            text,
            flags=re.IGNORECASE,
        )
    }
    if len(line_choices) == 1:
        return line_choices.pop()

    final_line_choice = re.search(
        r"(?:^|\n)\s*[*_`]*[\(\[]?([A-Z])[\)\].:][*_`]*[^\n]*\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if final_line_choice:
        return final_line_choice.group(1).upper()

    standalone = re.search(
        r"(?:^|\n)\s*[*_`]*[\(\[]?([A-Z])[\)\].]?[*_`]*\s*$",
        text,
        flags=re.IGNORECASE,
    )
    return standalone.group(1).upper() if standalone else None
