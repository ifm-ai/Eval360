from typing import Any, Union, Callable, Optional
import copy
import logging
import re
from ..utils import Sentinel
from ..metrics import get_accuracy, get_bootstrap_accuracy_std
from .base import AccuracyGraderBase
from .registry import register
from math_verify import parse, verify


logger = logging.getLogger(__name__)


def _strip_markdown_latex(text: str) -> str:
    """Pre-clean a generation so the answer regexes can see raw numbers.

    Reasoning models tend to wrap answers in markdown bold (``**42**``),
    LaTeX-escaped dollars (``\\$42``), or LaTeX thousands separators
    (``70{,}000``); none of that carries semantic value but it defeats the
    plain-number regexes below.
    """
    if not text:
        return text
    t = text
    t = t.replace("\\$", "$")                 # LaTeX-escaped dollar -> plain
    t = re.sub(r"\{\s*,\s*\}", ",", t)        # LaTeX thousands sep 70{,}000 -> 70,000
    t = t.replace("**", "").replace("__", "")  # markdown bold/italic markers
    t = re.sub(r"\\[(\[\])]", "", t)          # inline-math delimiters \( \) \[ \]
    t = t.replace("\\,", "")                  # LaTeX thin space
    t = re.sub("[\u00a0\u202f]", " ", t)    # nbsp / narrow-nbsp -> space
    return t


def parse_answer_with_verify(text):
    """Parse answer using math_verify library with fallback to regex patterns."""
    text = _strip_markdown_latex(text)
    try:
        # Prefer $...$ wrapping: raw LaTeX like \sqrt{51} or \left(...\right)
        # parses incorrectly or not at all without math delimiters
        parsed = parse(f"${text}$")
        if parsed:
            return parsed
        parsed = parse(text)
        if parsed:
            return parsed
    except Exception:
        pass

    # Fallback to regex patterns for extraction
    answer_patterns = [
        r"\\boxed\s*\{\s*\$?([\-0-9\.,]+)\s*\}",
        r"The answer is:?\s*\$?([\-0-9\.,]+)",
        r"#### ?\$?([\-0-9\.,]+)",
        r"Therefore,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"So,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Thus,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Hence,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Final answer:?\s*\$?([\-0-9\.,]+)",
        r"The final answer is:?\s*\$?([\-0-9\.,]+)",
        r"The answer is:?\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?",
        r"=\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?\.?\s*(?:The answer|$)",
    ]
    for pat in answer_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            # take the last match (usually the final answer)
            ans = matches[-1].replace(",", "").strip().rstrip(".")
            if ans:
                try:
                    return parse(ans)
                except Exception:
                    return ans

    sentence_end_pattern = r"(?:is|are|equals?|makes?|has|have|gets?|arrives?|covers?|travels?)\s+\$?([\-0-9\.,]+)(?:\s*(?:miles?|minutes?|hours?|dollars?|GB))?\.?\s*$"
    m = re.search(sentence_end_pattern, text, re.MULTILINE | re.IGNORECASE)
    if m:
        ans = m.group(1).replace(",", "").strip().rstrip(".")
        if ans:
            try:
                return parse(ans)
            except:
                return ans

    # last fallback: find the last number in the last complete sentence
    sentences = text.split('.')
    for sent in reversed(sentences):
        # skip sentences containing Human/Assistant (possibly irrelevant content)
        if 'Human:' in sent or 'Assistant:' in sent:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sent)
        if numbers:
            num = numbers[-1].lstrip('0') or '0'
            try:
                return parse(num)
            except Exception:
                return num
    return None


def compare_answers(answer, gold_raw):
    """Compare parsed prediction against raw ground truth string."""
    if not answer or gold_raw is None:
        return False
    try:
        # Try with gold as raw string (verify can parse simple expressions internally)
        if verify(answer, gold_raw):
            return True
        # For complex LaTeX (tuples, intervals, radicals with \left/\right),
        # parse the ground truth explicitly and call verify(gold, pred)
        gold_parsed = parse_answer_with_verify(gold_raw)
        if gold_parsed:
            return verify(gold_parsed, answer)
    except Exception:
        pass
    return False


_CAND_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _candidate_numbers(s: str) -> list[str]:
    """Pull plain numbers out of a short candidate-answer fragment.

    A fragment containing a fraction (``\\frac{..}{..}`` / ``\\dfrac`` /
    ``\\tfrac`` or a bare ``a/b``) yields NO candidates: pulling a bare
    integer out of e.g. ``2\\frac{2}{3}`` would wrongly match an integer
    ground truth. Such answers are left to math_verify, which evaluates
    the fraction properly.
    """
    s = re.sub("[\u00a0\u202f]", " ", s)
    s = re.sub(r"\{\s*,\s*\}", ",", s)  # LaTeX thousands sep: 120{,}000 -> 120,000
    if re.search(r"\\[dt]?frac|\d\s*/\s*\d", s):
        return []
    return [m.group(0).replace(",", "") for m in _CAND_NUM.finditer(s)]


def _answer_candidates(text: str) -> list[str]:
    """Disciplined "primary answer" extractions, used only as a recovery
    layer when math_verify's whole-text parse missed.

    math_verify.parse() is fed the entire generation and sometimes locks
    onto a trailing number (a unit, a sanity-check, a parenthetical aside)
    instead of the model's stated final answer. Each candidate here is a
    place the *emphasised* final answer lives -- a ``\\boxed{}`` body, the
    last bold ``**...**`` span, or the first number after an answer marker.
    It is deliberately NOT a loose "last number in the text", so a wrong
    primary answer is never rescued by a coincidental number elsewhere.

    Must run on the raw generation (before markdown stripping) so the
    ``**...**`` bold markers are still intact.
    """
    if not text:
        return []
    cands: list[str] = []

    # 1. \boxed{...} body (brace-matched). Prefer a $-prefixed number.
    if "boxed" in text:
        body = text.split("boxed")[-1]
        if body.startswith("{"):
            depth, buf = 1, []
            for ch in body[1:]:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if depth == 0:
                    break
                buf.append(ch)
            body = "".join(buf)
            dollar = re.findall(r"\$\s*(-?\d[\d,]*(?:\.\d+)?)", body)
            cands += [d.replace(",", "") for d in dollar] or _candidate_numbers(body)

    # 2. Last bold **...** span that contains a number.
    for span in reversed(re.findall(r"\*\*(.+?)\*\*", text, re.DOTALL)):
        nums = _candidate_numbers(span)
        if nums:
            cands.append(nums[-1])
            break

    # 3. First number after an explicit answer marker.
    markers = list(
        re.finditer(r"(?:final\s+answer|the\s+answer\s+is|\banswer\s*[:=])", text, re.I)
    )
    if markers:
        tail = text[markers[-1].end(): markers[-1].end() + 80]
        nums = _candidate_numbers(tail)
        if nums:
            cands.append(nums[0])

    return cands


def check_math_verify(
    prompt: Any,
    generations: list[str],
    expected: Union[str, list[str], tuple[str]],
    separator: Callable[[str], bool] = None,
    options: Optional[list[str]] = None,
):
    """
    checks if generated responses match the expected result using math_verify.

    Args:
        prompt: The input prompt.
        generations: The generated responses from the model.
        expected: The expected response or list of responses.
        separator: Optional function to check if a character is a separator.
        options: Optional list of options to match against the sampled response.

    Returns:
        The matched option or None if no match found.
    """
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]
    if options is None:
        options = expected

    correct = [False] * len(generations)
    
    for i, sampled in enumerate(generations):
        answer = parse_answer_with_verify(sampled)
        if answer is not None:
            for gold in expected:
                if compare_answers(answer, gold):
                    correct[i] = True
                    break
        if correct[i]:
            continue
        # Recovery layer: math_verify's whole-text parse can lock onto a
        # trailing number (a unit, a sanity-check, a parenthetical aside)
        # instead of the model's stated final answer. Re-check against the
        # disciplined "primary answer" candidates before giving up.
        for cand in _answer_candidates(sampled or ""):
            parsed = parse_answer_with_verify(cand)
            if parsed is None:
                continue
            if any(compare_answers(parsed, gold) for gold in expected):
                correct[i] = True
                break
    result = {
        "prompt": prompt,
        "generations": generations,
        "expected": expected,
        "correct": correct,
        "accuracy": get_accuracy(correct),
        "boostrap_std": get_bootstrap_accuracy_std(correct),
    }
    return result


@register("math-verify", "math_verify")
class MathVerify(AccuracyGraderBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    async def grade_sample(self, sample: Any, *_):
        # if the sample is the sentinel, we have reached the end of the data
        if sample == Sentinel.COMPLETED:
            return sample
        assert isinstance(sample, dict), "sample must be a dict"
        assert "completion_input" in sample, "sample must have an 'completion_input' key"
        assert "parsed_generations" in sample, "sample must have an 'parsed_generations' key" 
        assert "ground_truth" in sample, "sample must have an 'ground_truth' key"
        assert isinstance(sample["ground_truth"], str) or isinstance(
            sample["ground_truth"], list
        ), "sample['ground_truth'] must be a string or list of strings"
        
        generations = []
        for index, generation in enumerate(sample["parsed_generations"]):
            if generation is None:
                generations.append(sample["generations"][index])
            else:
                generations.append(generation)

        new_fields = check_math_verify(
            prompt=sample["completion_input"],
            generations=generations,
            expected=sample["ground_truth"],
        )
        result = copy.deepcopy(sample)
        result.update(new_fields)
        return result
