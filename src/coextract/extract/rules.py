"""Rule based extraction.

For every field in the registry we run two families of patterns:

* label anchored patterns generated from the field's aliases, and
* hand written phrase patterns for sentence style wording such as
  "the Contract Sum will be increased by $12,480.00".

A match only becomes a candidate if its captured text parses into the field's
type. That single constraint throws away most of what a naive regex sweep
would produce.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Optional

from ..fields import (
    DECREASE_CUES,
    FIELD_SPECS,
    INCREASE_CUES,
    NO_CHANGE,
    FieldSpec,
    compiled_extra_patterns,
    compiled_label_patterns,
)
from ..ingest import Document
from ..normalize.parse import (
    clean_string_value,
    detect_currency,
    parse_currency,
    parse_date_value,
    parse_days,
    parse_money,
    parse_percent,
)
from ..normalize.text import snippet_around
from .base import Candidate, apply_note_penalties

# Fields normally printed in the header block. A match in the first part of the
# document is slightly more believable for these.
HEADER_BIASED = {
    "change_order_number",
    "change_order_title",
    "project_name",
    "project_number",
    "contract_number",
    "owner_name",
    "contractor_name",
    "architect_name",
    "date_issued",
}

CONTEXT_WINDOW = 90


class RuleExtractor:
    name = "rules"

    def __init__(self, dayfirst: bool = False) -> None:
        self.dayfirst = dayfirst
        self._label_cache: dict[str, Any] = {}
        self._extra_cache: dict[str, Any] = {}

    # -- pattern access -----------------------------------------------------
    def _labels(self, spec: FieldSpec):
        if spec.name not in self._label_cache:
            self._label_cache[spec.name] = compiled_label_patterns(spec)
        return self._label_cache[spec.name]

    def _extras(self, spec: FieldSpec):
        if spec.name not in self._extra_cache:
            self._extra_cache[spec.name] = compiled_extra_patterns(spec)
        return self._extra_cache[spec.name]

    # -- value parsing ------------------------------------------------------
    def _parse(self, spec: FieldSpec, raw: str) -> tuple[Any, list[str]]:
        if spec.kind == "money":
            return parse_money(raw)
        if spec.kind == "date":
            return parse_date_value(raw, dayfirst=self.dayfirst)
        if spec.kind == "days":
            return parse_days(raw)
        if spec.kind == "percent":
            return parse_percent(raw)
        if spec.kind == "currency":
            return parse_currency(raw)
        return clean_string_value(raw, max_chars=spec.max_chars)

    # -- scoring helpers ----------------------------------------------------
    def _position_factor(self, spec: FieldSpec, start: int, total: int) -> float:
        if spec.name not in HEADER_BIASED or total == 0:
            return 1.0
        relative = start / total
        if relative <= 0.25:
            return 1.0
        if relative <= 0.6:
            return 0.94
        return 0.86

    def _sign_adjust(
        self, spec: FieldSpec, value: Any, context: str, forced_sign: Optional[str]
    ) -> tuple[Any, list[str]]:
        if not isinstance(value, (Decimal, int)) or spec.kind not in ("money", "days"):
            return value, []
        if spec.name in ("original_contract_sum", "contract_sum_before", "contract_sum_after"):
            return value, []
        if forced_sign == "negative" and value > 0:
            return -value, []
        if value > 0 and DECREASE_CUES.search(context) and not INCREASE_CUES.search(context):
            return -value, ["sign_from_context"]
        return value, []

    # -- main ---------------------------------------------------------------
    def extract(self, doc: Document) -> list[Candidate]:
        text = doc.text
        total = len(text)
        candidates: list[Candidate] = []

        for spec in FIELD_SPECS:
            candidates.extend(self._extract_field(spec, doc, text, total))

        candidates.extend(self._currency_candidate(doc))
        return _dedupe(candidates)

    def _extract_field(self, spec: FieldSpec, doc: Document, text: str, total: int) -> list[Candidate]:
        found: list[Candidate] = []

        for matcher, pattern, strength in self._labels(spec):
            below_label = matcher.startswith("label_below")
            for match in pattern.finditer(text):
                raw = match.group("value") or ""
                value, notes = self._parse(spec, raw)
                if value is None:
                    continue
                start, end = match.span("value")
                context = text[max(0, match.start() - CONTEXT_WINDOW) : end + CONTEXT_WINDOW]
                value, sign_notes = self._sign_adjust(spec, value, context, None)
                notes = notes + sign_notes + (["value_below_label"] if below_label else [])
                score = strength * self._position_factor(spec, match.start(), total)
                found.append(
                    Candidate(
                        field_name=spec.name,
                        value=value,
                        raw_text=raw.strip(),
                        score=apply_note_penalties(score, notes),
                        extractor=self.name,
                        matcher=matcher,
                        start=start,
                        end=end,
                        page=doc.page_of(start),
                        notes=notes,
                    )
                )

        for matcher, pattern, strength, sign in self._extras(spec):
            for match in pattern.finditer(text):
                raw = match.group("value") or ""
                value, notes = self._parse(spec, raw)
                if value is None:
                    continue
                start, end = match.span("value")
                context = text[max(0, match.start() - CONTEXT_WINDOW) : end + CONTEXT_WINDOW]
                value, sign_notes = self._sign_adjust(spec, value, context, sign)
                notes = notes + sign_notes
                found.append(
                    Candidate(
                        field_name=spec.name,
                        value=value,
                        raw_text=raw.strip(),
                        score=apply_note_penalties(strength, notes),
                        extractor=self.name,
                        matcher=matcher,
                        start=start,
                        end=end,
                        page=doc.page_of(start),
                        notes=notes,
                    )
                )

        if spec.kind == "enum":
            found.extend(self._enum_candidates(spec, doc, text))

        if spec.zero_on_no_change:
            found.extend(self._no_change_candidates(spec, doc, text))

        return found

    def _enum_candidates(self, spec: FieldSpec, doc: Document, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        exclusions = [re.compile(rx, re.IGNORECASE) for rx in spec.cue_exclusions]
        for regex, enum_value, strength in spec.enum_cues:
            for match in re.finditer(regex, text, re.IGNORECASE):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                line = text[line_start : line_end if line_end != -1 else len(text)]
                # Both the exclusion check and the label boost are scoped to
                # the cue's own line. Scoping matters in both directions: a
                # neighbouring "Date Executed:" caption must not veto a real
                # "Status: Executed", and a log table's "Status" column header
                # must not endorse every row beneath it.
                if any(rx.search(line) for rx in exclusions):
                    continue
                window = text[line_start : match.start()]
                boosted = strength
                for alias, alias_strength in spec.aliases:
                    if re.search(alias, window, re.IGNORECASE):
                        boosted = min(0.98, strength + 0.08 * alias_strength)
                        break
                out.append(
                    Candidate(
                        field_name=spec.name,
                        value=enum_value,
                        raw_text=match.group(0),
                        score=boosted,
                        extractor=self.name,
                        matcher=f"enum_cue:{enum_value}",
                        start=match.start(),
                        end=match.end(),
                        page=doc.page_of(match.start()),
                    )
                )
        return out

    def _no_change_candidates(self, spec: FieldSpec, doc: Document, text: str) -> list[Candidate]:
        """Turn "no change to the Contract Time" into an explicit zero.

        An explicit zero is very different from a missing value, and treating
        them the same is one of the easiest ways to produce a wrong total.
        """
        out: list[Candidate] = []
        zero: Any = Decimal("0.00") if spec.kind == "money" else 0
        for alias, strength in spec.aliases:
            pattern = re.compile(
                rf"(?:^|\n)[^\S\n]{{0,12}}{alias}[^\n]{{0,60}}", re.IGNORECASE
            )
            for match in pattern.finditer(text):
                line = match.group(0)
                if NO_CHANGE.search(line):
                    out.append(
                        Candidate(
                            field_name=spec.name,
                            value=zero,
                            raw_text=line.strip(),
                            score=min(0.93, strength * 0.95),
                            extractor=self.name,
                            matcher="no_change_phrase",
                            start=match.start(),
                            end=match.end(),
                            page=doc.page_of(match.start()),
                            notes=["explicit_no_change"],
                        )
                    )
        # Standalone sentences such as "There is no change to the Contract Time."
        for match in NO_CHANGE.finditer(text):
            window = text[max(0, match.start() - 70) : match.end() + 70].lower()
            if spec.name == "schedule_impact_days" and re.search(r"time|schedule|days|completion", window):
                anchor = True
            elif spec.name in ("change_amount", "net_previous_changes") and re.search(
                r"cost|sum|price|amount|dollar", window
            ):
                anchor = True
            else:
                anchor = False
            if anchor:
                out.append(
                    Candidate(
                        field_name=spec.name,
                        value=zero,
                        raw_text=match.group(0),
                        score=0.72,
                        extractor=self.name,
                        matcher="no_change_sentence",
                        start=match.start(),
                        end=match.end(),
                        page=doc.page_of(match.start()),
                        notes=["explicit_no_change"],
                    )
                )
        return out

    def _currency_candidate(self, doc: Document) -> list[Candidate]:
        code, confidence = detect_currency(doc.text)
        return [
            Candidate(
                field_name="currency",
                value=code,
                raw_text=code,
                score=confidence,
                extractor=self.name,
                matcher="currency_frequency",
                start=0,
                end=0,
                page=1,
                notes=["inferred_from_symbol_frequency"],
            )
        ]


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse candidates that are the same field, value and span."""
    best: dict[tuple[str, str, Optional[int]], Candidate] = {}
    for cand in candidates:
        key = (cand.field_name, cand.value_key(), cand.start)
        current = best.get(key)
        if current is None or cand.score > current.score:
            best[key] = cand
    return list(best.values())


def evidence_snippet(doc: Document, cand: Candidate) -> str:
    if cand.start is None or cand.end is None:
        return cand.raw_text
    return snippet_around(doc.text, cand.start, cand.end)
