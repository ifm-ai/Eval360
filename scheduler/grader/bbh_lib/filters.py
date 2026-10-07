"""Answer-extraction filters for BIG-Bench Hard (BBH), vendored from
lm-evaluation-harness (`lm_eval/filters/extraction.py` and
`lm_eval/tasks/bbh/cot_zeroshot/utils.py`).

These are copied verbatim (minus lm-eval's `@register_filter` decorators and
registry imports) so the BBH grader is self-contained and needs no
lm-evaluation-harness dependency. The only behavioural change is in
`NumberParseRegexFilter`: the optional spelled-out-number fallback degrades
gracefully when the `regex` / `word2number` packages are not installed.
"""
import collections
import re
import sys
import unicodedata


class Filter:
    """Minimal stand-in for ``lm_eval.api.filter.Filter``.

    A filter maps a batch of model responses (list-of-lists, one inner list
    per input) plus the matching docs to extracted answers of the same shape.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        return resps


class RegexFilter(Filter):
    """Extract a value from text via a regex, with a fallback when no match."""

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        match = match[0] if match else self.fallback
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        return [filter_set(x) for x in resps]


class ExtendedRegexFilter(RegexFilter):
    punct_tbl = dict.fromkeys(
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
    )

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def filter_ignores(self, st):
        if self.regexes_to_ignore is not None:
            for s in self.regexes_to_ignore:
                st = re.sub(s, "", st)
        if self.ignore_case:
            st = st.lower()
        if self.ignore_punctuation:
            st = st.translate(self.punct_tbl)
        return st

    def find_match(self, regex, resp, convert_dict={}):
        match = regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict:
                match = convert_dict[match]
        return match


class MapRegexFilter(ExtendedRegexFilter):
    def __init__(
        self,
        regex_pattern_to_value: dict = {},
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        super().__init__(
            "|".join(list(regex_pattern_to_value.keys())),
            group_select,
            fallback,
            ignore_case,
            ignore_punctuation,
            regexes_to_ignore,
        )
        self.regex_to_value = {
            re.compile(r): v for r, v in regex_pattern_to_value.items()
        }

    def apply(self, resps, docs):
        filtered_resps = []
        for r in resps:
            filtered = []
            for resp in r:
                whole = self.find_match(self.regex, self.filter_ignores(resp))
                match = None
                if whole:
                    for regex, mapped_value in self.regex_to_value.items():
                        if self.find_match(regex, self.filter_ignores(whole)):
                            match = mapped_value
                            break
                if not whole or not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)
        return filtered_resps


class NumberParseRegexFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        # Optional spelled-out-number fallback. Requires the third-party
        # `regex` and `word2number` packages; if absent, only the numeric
        # regex is used (BBH number answers are overwhelmingly digits).
        english_number_regex = None
        w2n = None
        try:
            import regex as _regex
            from word2number import w2n as _w2n

            english_number_regex = _regex.compile(
                "((?:(?:zero|one|two|three|four|five|(?:twen|thir|for|fif|six|seven|nine)(?:|teen|ty)|eight(?:|een|y)|ten|eleven|twelve|fourteen|hundred|thousand|(?:m|b|tr)illion)(?:zero|one|two|three|four|five|(?:twen|thir|for|fif|six|seven|nine)(?:|teen|ty)|eight(?:|een|y)|ten|eleven|twelve|fourteen|hundred|thousand|(?:m|b|tr)illion|[^\\S\r\n]|,|and|&)+)?(?:zero|one|two|three|four|five|(?:twen|thir|for|fif|six|seven|nine)(?:|teen|ty)|eight(?:|een|y)|ten|eleven|twelve|fourteen|hundred|thousand|(?:m|b|tr)illion))"
            )
            w2n = _w2n
        except Exception:
            pass

        filtered_resps = []
        for r in resps:
            filtered = []
            for resp in r:
                match = self.find_match(self.regex, resp)
                if not match and english_number_regex is not None:
                    spelled = self.find_match(english_number_regex, resp.lower())
                    if spelled:
                        try:
                            match = str(w2n.word_to_num(spelled))
                        except Exception:
                            match = None
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)
        return filtered_resps


class WordSortFilter(Filter):
    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            words = doc["input"].split("List:")[1].strip().split()
            regex = re.compile("|".join([f"\\b{w}\\b" for w in words]))
            filtered = []
            for resp in r:
                match = regex.findall(resp)
                match.reverse()
                ordered_words = reversed(
                    collections.OrderedDict(zip(match, [None] * len(match)))
                )
                filtered.append(" ".join(ordered_words))
            filtered_resps.append(filtered)
        return filtered_resps


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"
            without_paren_fallback_regexes = []
            without_paren_to_target = {}
            multiple_choices_regex = re.compile(r"\([A-Z]\)([^\n^(]*)")
            match = multiple_choices_regex.findall(doc["input"])
            for m in match:
                m = self.filter_ignores(m.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"
                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"
                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = self.find_match(self.regex, resp)
                if not match:
                    match = self.find_match(
                        fallback_regex, self.filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = self.find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)
        return filtered_resps
