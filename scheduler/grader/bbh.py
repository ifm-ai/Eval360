"""Grader for BIG-Bench Hard (BBH), CoT zero-shot.

The stock ``exact_match`` grader checks ``generation.startswith(answer)``,
which scores ~0 for a chain-of-thought model that reasons before answering.
This grader instead uses lm-evaluation-harness's canonical per-subtask
answer-extraction filters (vendored in ``bbh_lib/``) — a MultiChoiceRegexFilter
for choice subtasks, NumberParseRegexFilter for arithmetic, WordSortFilter for
word_sorting, etc. — then compares the extracted answer to the ground truth.

Each sample must carry ``meta.subtask`` (present in the 0-shot ``bbh_zeroshot``
data) so the right filter can be dispatched.
"""
import copy
import re
from typing import Any

from ..metrics import get_accuracy, get_bootstrap_accuracy_std
from ..utils import Sentinel
from .base import AccuracyGraderBase
from .registry import register
from .bbh_lib import (
    RegexFilter,
    MapRegexFilter,
    NumberParseRegexFilter,
    WordSortFilter,
    MultiChoiceRegexFilter,
)


# ---------------------------------------------------------------------------
# Per-subtask filter table. Each subtask uses exactly the filter config its
# lm-eval task YAML uses, so extracted answers match lm-eval's canonical
# scoring.
# ---------------------------------------------------------------------------
def _build_filters() -> dict:
    F: dict = {}

    # Plain regex filters (lm-eval `regex` flexible-extract).
    F["boolean_expressions"] = RegexFilter(
        r"\b(True|False)\b", group_select=-1, fallback="[invalid]"
    )
    F["causal_judgement"] = RegexFilter(
        r"\b(Yes|No|yes|no)\b", group_select=-1, fallback="[invalid]"
    )
    F["dyck_languages"] = RegexFilter(
        r'(?<= )([" \[\(<{}>\)\]]+)|([" \[\(<{}>\)\]]+)',
        group_select=-1, fallback="[invalid]",
    )
    F["formal_fallacies"] = RegexFilter(
        r"\b(valid|invalid)\b", group_select=-1, fallback="[invalid]"
    )
    F["navigate"] = RegexFilter(
        r"\b(Yes|No|yes|no)\b", group_select=-1, fallback="[invalid]"
    )

    # Multiple-choice subtasks: all share the same "(A)"-style config.
    mc_kwargs = dict(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
        fallback="[invalid]",
    )
    for s in (
        "date_understanding", "disambiguation_qa", "geometric_shapes",
        "hyperbaton", "logical_deduction_three_objects",
        "logical_deduction_five_objects", "logical_deduction_seven_objects",
        "movie_recommendation", "penguins_in_a_table",
        "reasoning_about_colored_objects", "ruin_names",
        "salient_translation_error_detection", "snarks", "temporal_sequences",
        "tracking_shuffled_objects_three_objects",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
    ):
        F[s] = MultiChoiceRegexFilter(**mc_kwargs)

    # Number-parse subtasks. The regex is digits/hyphen only -- it must NOT
    # include '.' or ',' or a trailing isolated "." in prose ("...total.")
    # would be picked up as the (last) match.
    num_kwargs = dict(
        regex_pattern=r"([-0-9]+)",
        group_select=-1,
        fallback="[invalid]",
    )
    F["multistep_arithmetic_two"] = NumberParseRegexFilter(**num_kwargs)
    F["object_counting"] = NumberParseRegexFilter(**num_kwargs)

    # Map-regex subtasks.
    F["sports_understanding"] = MapRegexFilter(
        regex_pattern_to_value={
            r"\b(no|not plausible)\b": "no",
            r"\b(yes|plausible)\b": "yes",
        },
        group_select=-1,
        ignore_case=True,
        fallback="[invalid]",
    )
    F["web_of_lies"] = MapRegexFilter(
        regex_pattern_to_value={
            r"\b(no|does not tell the truth|is not telling the truth)\b": "no",
            r"\b(yes|tells the truth|is telling the truth)\b": "yes",
        },
        group_select=-1,
        ignore_case=True,
        fallback="[invalid]",
    )

    # Word-sorting.
    F["word_sorting"] = WordSortFilter()
    return F


_FILTERS = _build_filters()


# ---------------------------------------------------------------------------
# doc["input"] reconstruction. MultiChoiceRegexFilter / WordSortFilter need
# the raw question to parse the answer choices / word list. Eval360's
# completion_input for the 0-shot data is "<desc>\n\nQ: <input>\nA: Let's
# think step by step." — take the text after the last "Q:" marker.
# ---------------------------------------------------------------------------
_Q_SPLIT_RE = re.compile(r"\nQ:\s*|\bQ:\s*", re.MULTILINE)


def _extract_doc_input(completion_input: str) -> str:
    if not completion_input:
        return ""
    parts = _Q_SPLIT_RE.split(completion_input)
    if len(parts) < 2:
        return completion_input
    target_q = parts[-1]
    trailing = target_q.find("\nA:")
    if trailing >= 0:
        target_q = target_q[:trailing]
    return target_q.strip()


# lm-evaluation-harness's bbh exact_match config (see the cot_zeroshot
# template yaml): ignore_case=True plus these regexes_to_ignore. Note that
# `ignore_punctuation` is commented out upstream, so brackets must NOT be
# stripped -- they are the actual answer for the dyck_languages subtask.
_BBH_IGNORE_REGEXES = [re.compile(r) for r in (r"\.$", r",", r"\\", r"\n", r'"')]


def _normalize_for_match(s: Any) -> str:
    """Replicate lm-evaluation-harness's bbh exact_match normalization."""
    if s is None:
        return ""
    s = str(s)
    for rgx in _BBH_IGNORE_REGEXES:
        s = rgx.sub("", s)
    return s.lower().strip()


# Closing-bracket characters. A Dyck-word *completion* (the dyck_languages
# answer) consists only of these.
_DYCK_CLOSERS = set(")]}>")
_DYCK_RUN_RE = re.compile(r'[\s"\[\](){}<>]+')


def _dyck_completion(generation: str) -> str | None:
    """Extract the Dyck-word completion from a generation, whitespace-removed.

    Reasoning models print the bare completion (e.g. ``]]>``) and often also
    the full balanced word (e.g. ``<[[]]>``). The completion is the part the
    task grades and contains *only* closing brackets, so we keep the last
    bracket run that is all-closing -- the full word, having opening
    brackets, is skipped. Returns None if no such run is found.
    """
    # LaTeX spacing commands (\, \; \: \! \quad ...) are whitespace; drop them
    # so "[ \; ( \; ) \; ]" is seen as one bracket run, not fragments.
    text = re.sub(r"\\(?:[,;:! ]|q?quad)", " ", generation or "")
    cand = None
    for run in _DYCK_RUN_RE.findall(text):
        r = re.sub(r"\s+", "", run)
        if r and all(c in _DYCK_CLOSERS for c in r):
            cand = r
    return cand


def check_bbh(
    generations: list[str],
    subtask: str | None,
    doc_input: str,
    expected: Any,
):
    """Extract per-generation answers with the subtask filter and compare."""
    picked: list = [None] * len(generations)
    correct: list = [False] * len(generations)

    # dyck_languages: lm-eval's filter + exact_match is format-broken here.
    # The ground truth is space-separated ("] ] >"), models write it without
    # spaces, and group_select=-1 grabs the full balanced word instead of the
    # completion. Grade the completion (a run of only closing brackets)
    # directly and whitespace-insensitively.
    if subtask == "dyck_languages":
        gt_d = re.sub(r"\s+", "", str(expected))
        for i, gen in enumerate(generations):
            comp = _dyck_completion(gen)
            picked[i] = comp if comp is not None else "[invalid]"
            correct[i] = bool(gt_d) and comp == gt_d
        return {
            "picked": picked,
            "correct": correct,
            "accuracy": get_accuracy(correct),
            "bootstrap_std": get_bootstrap_accuracy_std(correct),
        }

    flt = _FILTERS.get(subtask) if subtask else None
    if flt is not None:
        docs = [{"input": doc_input, "target": expected}]
        try:
            filtered = flt.apply([list(generations)], docs)
            extracted = filtered[0] if filtered else []
        except Exception:
            extracted = []
        gt_norm = _normalize_for_match(expected)
        for i in range(len(generations)):
            ex = extracted[i] if i < len(extracted) else ""
            picked[i] = ex
            correct[i] = bool(gt_norm) and _normalize_for_match(ex) == gt_norm

    return {
        "picked": picked,
        "correct": correct,
        "accuracy": get_accuracy(correct),
        "bootstrap_std": get_bootstrap_accuracy_std(correct),
    }


@register("bbh")
class BBH(AccuracyGraderBase):
    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample
        assert isinstance(sample, dict), "sample must be a dict"
        assert "completion_input" in sample, "sample must have a 'completion_input' key"
        assert "parsed_generations" in sample, "sample must have a 'parsed_generations' key"
        assert "ground_truth" in sample, "sample must have a 'ground_truth' key"

        generations = []
        for index, generation in enumerate(sample["parsed_generations"]):
            if generation is None:
                generations.append(sample["generations"][index])
            else:
                generations.append(generation)

        meta = sample.get("meta") or {}
        subtask = meta.get("subtask")
        doc_input = _extract_doc_input(sample.get("completion_input", ""))
        expected = sample["ground_truth"]
        if isinstance(expected, list):
            expected = expected[0] if expected else ""

        new_fields = check_bbh(
            generations=generations,
            subtask=subtask,
            doc_input=doc_input,
            expected=expected,
        )

        result = copy.deepcopy(sample)
        result.update(new_fields)
        return result
