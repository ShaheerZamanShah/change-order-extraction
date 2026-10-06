"""Optional LLM extractor with a quote verification guard.

The pipeline runs fine without this. It exists for the long tail: narrative
change orders, email threads and letters where the value a human would read off
the page is never next to a recognisable label.

The important part is not the prompt, it is the guard. The model must return a
verbatim quote from the document for every value it reports. We then look that
quote up in the document text. If we cannot find it, the value is dropped. A
model that invents a contract sum almost always invents the quote with it, so
this single check removes the failure mode that makes LLM extraction unsafe for
money fields.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Protocol

from ..fields import FIELD_SPECS, FieldSpec
from ..ingest import Document
from ..normalize.parse import (
    clean_string_value,
    parse_currency,
    parse_date_value,
    parse_days,
    parse_money,
    parse_percent,
)
from .base import Candidate, apply_note_penalties

log = logging.getLogger(__name__)

MAX_SCORE = 0.88  # an unverifiable model is never allowed to look certain
MIN_QUOTE_RATIO = 0.82


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - protocol
        ...


class NullLLMClient:
    """Default client. Returns nothing so the extractor is a no-op."""

    name = "null"

    def complete(self, system: str, user: str) -> str:
        return ""


class OpenAIChatClient:
    """Minimal chat completions client over urllib, no SDK dependency.

    Works against the OpenAI API or anything that speaks the same shape,
    including a local server, by setting ``OPENAI_BASE_URL``.
    """

    name = "openai_chat"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 90,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {exc.code} {exc.reason}") from exc
        return body["choices"][0]["message"]["content"]


SYSTEM_PROMPT = (
    "You read construction change orders and return structured data. "
    "You only report a value if it is supported by text that appears in the document. "
    "For every value you must copy a short verbatim quote from the document that contains it. "
    "Never paraphrase a quote, never reformat numbers inside a quote, and never guess. "
    "If a field is not stated in the document, omit it entirely. "
    "Respond with a single JSON object and nothing else."
)


def _field_catalogue() -> str:
    lines = []
    for spec in FIELD_SPECS:
        hint = spec.description or ", ".join(alias.replace("\\", "") for alias, _ in spec.aliases[:3])
        lines.append(f"- {spec.name} ({spec.kind}): {hint}")
    return "\n".join(lines)


def build_prompt(text: str, max_chars: int = 18000) -> str:
    body = text if len(text) <= max_chars else text[: max_chars // 2] + "\n...\n" + text[-max_chars // 2 :]
    return (
        "Extract the following fields from the change order below.\n\n"
        f"{_field_catalogue()}\n\n"
        "Return JSON of this shape:\n"
        '{"fields": {"<field_name>": {"value": <string|number>, "quote": "<verbatim text from the document>", '
        '"confidence": <0..1>}}}\n\n'
        "Rules:\n"
        "- money as a plain number, negative for deducts and credits\n"
        "- dates as YYYY-MM-DD\n"
        "- schedule_impact_days as an integer, 0 if the document says there is no time impact\n"
        "- omit any field that is not stated\n\n"
        "DOCUMENT:\n"
        "-----\n"
        f"{body}\n"
        "-----"
    )


def _coerce(spec: FieldSpec, value: Any) -> tuple[Any, list[str]]:
    raw = str(value).strip()
    if spec.kind == "money":
        return parse_money(raw)
    if spec.kind == "date":
        return parse_date_value(raw)
    if spec.kind == "days":
        return parse_days(raw)
    if spec.kind == "percent":
        return parse_percent(raw)
    if spec.kind == "currency":
        return parse_currency(raw)
    if spec.kind == "enum":
        allowed = {enum_value for _, enum_value, _ in spec.enum_cues}
        token = raw.lower().replace(" ", "_")
        return (token, []) if token in allowed else (None, ["enum_value_not_allowed"])
    return clean_string_value(raw, max_chars=spec.max_chars)


def _locate_quote(text: str, quote: str) -> tuple[Optional[int], Optional[int], list[str]]:
    """Find a quote in the document, allowing for whitespace drift."""
    quote = " ".join(quote.split())
    if len(quote) < 4:
        return None, None, ["quote_too_short"]

    index = text.find(quote)
    if index >= 0:
        return index, index + len(quote), []

    lowered = text.lower()
    index = lowered.find(quote.lower())
    if index >= 0:
        return index, index + len(quote), []

    # Whitespace insensitive search: match on a collapsed copy and map back.
    collapsed_chars: list[str] = []
    positions: list[int] = []
    previous_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if previous_space:
                continue
            collapsed_chars.append(" ")
            positions.append(i)
            previous_space = True
        else:
            collapsed_chars.append(ch)
            positions.append(i)
            previous_space = False
    collapsed = "".join(collapsed_chars)
    index = collapsed.lower().find(quote.lower())
    if index >= 0:
        start = positions[index]
        end = positions[min(index + len(quote), len(positions) - 1)]
        return start, end, []

    # Last resort: closest line by similarity ratio.
    best_ratio = 0.0
    best_span: tuple[Optional[int], Optional[int]] = (None, None)
    cursor = 0
    for line in text.split("\n"):
        if line.strip():
            ratio = difflib.SequenceMatcher(None, quote.lower(), line.strip().lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (cursor, cursor + len(line))
        cursor += len(line) + 1
    if best_ratio >= MIN_QUOTE_RATIO:
        return best_span[0], best_span[1], ["quote_fuzzy_match"]
    return None, None, ["quote_not_found_in_document"]


def _value_digits_in_document(
    text: str, value: Any, start: Optional[int], end: Optional[int], window: int = 30
) -> bool:
    """Is this number actually written in the document near the cited span?

    Both the full digit string and the integer part are accepted, because the
    document and the model often disagree about whether to write the cents.
    """
    if start is None:
        return False
    lo = max(0, start - window)
    hi = min(len(text), (end or start) + window)
    nearby = re.sub(r"\D", "", text[lo:hi])
    if not nearby:
        return False

    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return False

    full = re.sub(r"\D", "", f"{abs(number)}")
    integral = re.sub(r"\D", "", f"{abs(number).to_integral_value()}")
    return bool((full and full in nearby) or (integral and integral in nearby))


def _parse_json_block(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class LLMExtractor:
    name = "llm"

    def __init__(self, client: Optional[LLMClient] = None, max_chars: int = 18000) -> None:
        self.client = client or NullLLMClient()
        self.max_chars = max_chars
        self.dropped: list[tuple[str, str]] = []  # (field, reason) for the run report

    @property
    def enabled(self) -> bool:
        return not isinstance(self.client, NullLLMClient)

    def extract(self, doc: Document) -> list[Candidate]:
        self.dropped = []
        if not self.enabled:
            return []
        try:
            raw = self.client.complete(SYSTEM_PROMPT, build_prompt(doc.text, self.max_chars))
        except Exception as exc:
            log.warning("LLM extractor failed, continuing without it: %s", exc)
            self.dropped.append(("*", f"llm_call_failed:{type(exc).__name__}"))
            return []

        payload = _parse_json_block(raw)
        fields = payload.get("fields") or payload
        if not isinstance(fields, dict):
            return []

        specs = {spec.name: spec for spec in FIELD_SPECS}
        candidates: list[Candidate] = []

        for name, entry in fields.items():
            spec = specs.get(name)
            if spec is None or entry is None:
                continue
            if not isinstance(entry, dict):
                entry = {"value": entry, "quote": "", "confidence": 0.5}
            if entry.get("value") in (None, "", []):
                continue

            value, notes = _coerce(spec, entry.get("value"))
            if value is None:
                self.dropped.append((name, "value_failed_type_check"))
                continue

            start, end, quote_notes = _locate_quote(doc.text, str(entry.get("quote") or ""))
            if start is None:
                self.dropped.append((name, quote_notes[0] if quote_notes else "quote_not_found"))
                continue

            # For a number, the digits have to be present in the document text
            # at the place the quote resolved to.
            #
            # Checking against the model's own quote is not enough, and that
            # mistake is worth spelling out: a model that invents an amount
            # invents a quote containing it, so the two agree perfectly. Worse,
            # the invented quote is usually a near copy of a real line, so it
            # fuzzy matches that line and arrives with real looking evidence
            # attached. Verifying against the document instead of the quote is
            # what makes the check mean anything.
            if spec.kind in ("money", "days", "percent"):
                if not _value_digits_in_document(doc.text, value, start, end):
                    self.dropped.append((name, "value_not_present_in_document"))
                    continue

            model_confidence = entry.get("confidence")
            try:
                model_confidence = float(model_confidence)
            except (TypeError, ValueError):
                model_confidence = 0.5
            model_confidence = max(0.0, min(1.0, model_confidence))

            notes = notes + quote_notes
            score = apply_note_penalties(0.5 + 0.38 * model_confidence, notes)
            candidates.append(
                Candidate(
                    field_name=name,
                    value=value,
                    raw_text=str(entry.get("value")),
                    score=min(MAX_SCORE, score),
                    extractor=self.name,
                    matcher=f"{self.client.name}:quote_verified",
                    start=start,
                    end=end,
                    page=doc.page_of(start),
                    notes=notes,
                )
            )
        return candidates
