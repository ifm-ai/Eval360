# Sections 1a and 1b are ported from openai/prm800k (grading/math_normalize.py and
# grading/grader.py; MIT, partly derived from hendrycks/math, MIT). Section 2
# (EQUALITY_TEMPLATE) is copied from openai/simple-evals (MIT).
# See THIRD_PARTY_NOTICES.md.
import copy
import re
import asyncio
import logging
from typing import Any, Optional

from ..utils import Sentinel
from .base import AccuracyGraderBase
from ..cache_salt import request_kwargs_with_cache_salt
from ..external_requests import ExternalRequestFailure
from ..metrics import get_accuracy, get_bootstrap_accuracy_std
from .registry import register

logger = logging.getLogger(__name__)

import sympy
from sympy.parsing import sympy_parser
from pylatexenc import latex2text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# sympy might hang on these patterns
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"

# ---------------------------------------------------------------------------
# Section 1a: Hendrycks MATH normalization (ported from math_normalize.py)
# ---------------------------------------------------------------------------

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def normalize_answer_hendrycks(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


# ---------------------------------------------------------------------------
# Section 1b: Sympy grading utilities (ported from grader.py)
# ---------------------------------------------------------------------------

def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    # TODO from @nightlessbaron. This is known to be high on memory usage.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    if expr is None:
        return None

    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer, OR
    (b) sympy can simplify the difference between the expressions to 0.
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer_hendrycks(ground_truth)
    given_answer_normalized_mathd = normalize_answer_hendrycks(given_answer)

    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


# ---------------------------------------------------------------------------
# Section 2: LLM Equality Template
# ---------------------------------------------------------------------------

# Copied from openai/simple-evals (common.py; MIT, Copyright (c) 2024 OpenAI).
# See THIRD_PARTY_NOTICES.md.
EQUALITY_TEMPLATE = r"""Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s"""

SYMPY_TIMEOUT_SECONDS = 5


# ---------------------------------------------------------------------------
# Section 3: Grader Class
# ---------------------------------------------------------------------------

@register("sympy-llm-as-judge", "sympy_llm_as_judge")
class SympyLLMasJudge(AccuracyGraderBase):
    """
    Hybrid math grader: tries deterministic sympy-based comparison first,
    falls back to LLM-as-judge when sympy cannot confirm equality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.task.grader.llm_as_judge
        self.openai_connection = self.request_openai_connection(self.model)

    async def _check_equality_with_llm(self, expr1: str, expr2: str) -> bool:
        prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
        messages = [
            {"role": "user", "content": prompt},
        ]
        client = await self.openai_connection.get_client()
        for _ in range(3):
            try:
                request_kwargs = request_kwargs_with_cache_salt(
                    self.model.openai_kwargs,
                    self.model.cache_salt,
                )
                completion = await client.chat.completions.create(
                    model=self.model.api_model_name or self.model.name,
                    messages=messages,
                    **request_kwargs,
                )
                response = completion.choices[0].message.content
                if response and len(response.strip()) > 0:
                    return "yes" in response.lower().strip()
            except ValueError:
                raise
            except ExternalRequestFailure:
                # Retrying here would multiply the already-exhausted shared
                # external policy by this historical three-call loop.
                raise
            except Exception as e:
                logger.warning("LLM equality check failed: %s", e)
                continue
        return False

    async def _grade_single_generation(
        self, generation: str, ground_truths: list[str]
    ) -> tuple[int, dict]:
        """
        Returns (score, diagnostics) where diagnostics captures
        the full decision flow for debugging.
        """
        diag = {
            "method": "none",
            "sympy_result": None,
            "sympy_error": None,
            "llm_called": False,
            "llm_result": None,
            "score": 0,
        }

        if not generation:
            generation = "No answer"

        # Step 1: Try sympy-based grading
        sympy_attempted = False
        for gt in ground_truths:
            sympy_attempted = True
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(grade_answer, generation, gt),
                    timeout=SYMPY_TIMEOUT_SECONDS,
                )
                if result:
                    logger.debug("Sympy confirmed equality for gt=%s", gt)
                    diag.update(method="sympy", sympy_result="match", score=1)
                    return 1, diag
            except asyncio.TimeoutError:
                logger.warning("Sympy timed out comparing generation to gt=%s", gt)
                diag.update(sympy_result="timeout")
            except Exception as e:
                logger.warning("Sympy error: %s", e)
                diag.update(sympy_result="error", sympy_error=str(e))

        # If sympy was attempted but never matched or errored, mark as no_match
        if sympy_attempted and diag["sympy_result"] is None:
            diag["sympy_result"] = "no_match"

        # Step 2: LLM fallback
        logger.debug("Falling back to LLM for equality check")
        diag["llm_called"] = True
        for gt in ground_truths:
            if await self._check_equality_with_llm(gt, generation):
                diag.update(method="llm", llm_result="match", score=1)
                return 1, diag

        diag["llm_result"] = "no_match"
        return 0, diag

    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict), "sample must be a dict"
        assert "completion_input" in sample, "sample must have 'completion_input'"
        assert "parsed_generations" in sample, "sample must have 'parsed_generations'"
        assert "ground_truth" in sample, "sample must have 'ground_truth'"

        ground_truth = sample["ground_truth"]
        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        elif isinstance(ground_truth, (list, tuple)):
            ground_truths = list(ground_truth)
        else:
            ground_truths = [str(ground_truth)]

        generations = []
        for i, gen in enumerate(sample["parsed_generations"]):
            if gen is None:
                generations.append(sample["generations"][i])
            else:
                generations.append(gen)

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for gen in generations:
                tasks.append(
                    tg.create_task(self._grade_single_generation(gen, ground_truths))
                )

        result = copy.deepcopy(sample)
        scores = [t.result()[0] for t in tasks]
        per_gen_diags = [t.result()[1] for t in tasks]

        result["correct"] = scores
        result["partial_accuracy"] = sum(scores) / len(scores)
        result["accuracy"] = 1 if any(scores) else 0
        result["grading_diagnostics"] = {
            "per_generation": per_gen_diags,
            "sympy_match_count": sum(1 for d in per_gen_diags if d["sympy_result"] == "match"),
            "sympy_no_match_count": sum(1 for d in per_gen_diags if d["sympy_result"] == "no_match"),
            "sympy_timeout_count": sum(1 for d in per_gen_diags if d["sympy_result"] == "timeout"),
            "sympy_error_count": sum(1 for d in per_gen_diags if d["sympy_result"] == "error"),
            "llm_called_count": sum(1 for d in per_gen_diags if d["llm_called"]),
            "llm_match_count": sum(1 for d in per_gen_diags if d["llm_result"] == "match"),
            "llm_no_match_count": sum(1 for d in per_gen_diags if d["llm_result"] == "no_match"),
        }
        return result
