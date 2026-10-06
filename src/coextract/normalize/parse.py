"""Value parsers.

Each parser returns ``(value, notes)``. Notes are short machine readable tags
such as ``ambiguous_decimal_separator`` that the confidence scorer uses to
discount a candidate. Parsers never raise on bad input, they return ``None``.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from dateutil import parser as dateparser

_CURRENCY_WORDS = re.compile(r"(?i)\b(?:usd|eur|gbp|cad|aud)\b|us\$|[$€£]")
_NON_NUMERIC = re.compile(r"[^\d,.\-+]")

CURRENCY_BY_SYMBOL = {"$": "USD", "€": "EUR", "£": "GBP", "us$": "USD"}

MIN_YEAR = 1980
MAX_YEAR = 2100


def _strip_currency(raw: str) -> tuple[str, str | None]:
    symbol = None
    match = _CURRENCY_WORDS.search(raw)
    if match:
        token = match.group(0).lower()
        symbol = CURRENCY_BY_SYMBOL.get(token, token.upper())
    return _CURRENCY_WORDS.sub("", raw), symbol


def parse_money(raw: str) -> tuple[Decimal | None, list[str]]:
    """Parse a money string into a Decimal.

    Handles ``$1,234.56``, ``(2,500.00)`` for negatives, trailing minus,
    European ``1.234,56`` grouping, OCR noise and bare integers.
    """
    notes: list[str] = []
    if raw is None:
        return None, ["empty"]
    text = raw.strip()
    if not text:
        return None, ["empty"]

    negative = False
    if text.startswith("(") and text.rstrip().endswith(")"):
        negative = True
        notes.append("parenthesised_negative")
    text = text.strip("()").strip()

    text, _ = _strip_currency(text)
    text = _NON_NUMERIC.sub("", text)

    if text.endswith("-"):
        negative = True
        text = text[:-1]
    while text.startswith(("-", "+")):
        if text[0] == "-":
            negative = not negative
        text = text[1:]
    text = text.replace("+", "")

    if not text or not any(c.isdigit() for c in text):
        return None, notes + ["no_digits"]

    has_comma = "," in text
    has_dot = "." in text
    if has_comma and has_dot:
        decimal_sep = "," if text.rindex(",") > text.rindex(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        if decimal_sep == ",":
            notes.append("european_grouping")
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma:
        parts = text.split(",")
        tail = parts[-1]
        if len(parts) > 2 or (len(tail) == 3 and len(parts[0]) <= 3):
            text = text.replace(",", "")
        else:
            notes.append("decimal_comma")
            text = text.replace(",", ".")
    elif has_dot:
        parts = text.split(".")
        tail = parts[-1]
        if len(parts) > 2:
            text = text.replace(".", "")
            notes.append("european_grouping")
        elif len(tail) == 3 and len(parts[0]) <= 3:
            # "1.500" is either 1500 (EU grouping) or 1.5. Grouping is far more
            # likely on a change order, but flag it so confidence drops.
            text = text.replace(".", "")
            notes.append("ambiguous_decimal_separator")

    try:
        value = Decimal(text)
    except InvalidOperation:
        return None, notes + ["unparseable"]

    if negative:
        value = -value
    if abs(value) > Decimal("1e12"):
        return None, notes + ["implausible_magnitude"]
    return value.quantize(Decimal("0.01")), notes


def parse_percent(raw: str) -> tuple[Decimal | None, list[str]]:
    value, notes = parse_money(raw.replace("%", ""))
    if value is None:
        return None, notes
    if not (Decimal("0") <= value <= Decimal("100")):
        return None, notes + ["percent_out_of_range"]
    return value, notes


def parse_days(raw: str) -> tuple[int | None, list[str]]:
    notes: list[str] = []
    if not raw:
        return None, ["empty"]
    match = re.search(r"[-+]?\d{1,4}", raw.replace(",", ""))
    if not match:
        return None, ["no_digits"]
    value = int(match.group(0))
    if abs(value) > 1825:
        return None, notes + ["implausible_duration"]
    return value, notes


def parse_date_value(raw: str, dayfirst: bool = False) -> tuple[date | None, list[str]]:
    """Parse a date, retrying with day-first order when the US order is impossible."""
    notes: list[str] = []
    if not raw:
        return None, ["empty"]
    text = re.sub(r"(?i)\b(\d{1,2})(st|nd|rd|th)\b", r"\1", raw.strip().rstrip(".,;"))
    text = re.sub(r"\s+", " ", text)

    numeric = re.match(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\s*$", text)
    if numeric:
        first, second = int(numeric.group(1)), int(numeric.group(2))
        if first > 12 and second <= 12:
            dayfirst = True
            notes.append("dayfirst_inferred")
        elif first <= 12 and second <= 12 and not dayfirst:
            notes.append("ambiguous_day_month_order")

    for attempt_dayfirst in (dayfirst, not dayfirst):
        try:
            parsed = dateparser.parse(text, dayfirst=attempt_dayfirst, fuzzy=False)
        except (ValueError, OverflowError, TypeError):
            parsed = None
        if parsed is not None:
            break
    else:
        return None, notes + ["unparseable"]

    if parsed is None:
        return None, notes + ["unparseable"]
    if not (MIN_YEAR <= parsed.year <= MAX_YEAR):
        return None, notes + ["implausible_year"]
    if isinstance(parsed, datetime):
        return parsed.date(), notes
    return parsed, notes


def parse_currency(raw: str) -> tuple[str | None, list[str]]:
    if not raw:
        return None, ["empty"]
    token = raw.strip().lower()
    mapped = CURRENCY_BY_SYMBOL.get(token)
    if mapped:
        return mapped, []
    if re.fullmatch(r"[a-z]{3}", token):
        return token.upper(), []
    return None, ["unrecognised_currency"]


def detect_currency(text: str) -> tuple[str, float]:
    """Document level currency guess with a crude confidence."""
    counts = {
        "USD": len(re.findall(r"(?i)\bUSD\b|US\$|\$", text)),
        "EUR": len(re.findall(r"(?i)\bEUR\b|€", text)),
        "GBP": len(re.findall(r"(?i)\bGBP\b|£", text)),
        "CAD": len(re.findall(r"(?i)\bCAD\b|C\$", text)),
    }
    total = sum(counts.values())
    if total == 0:
        return "USD", 0.35
    best = max(counts, key=lambda k: counts[k])
    return best, min(0.95, 0.55 + counts[best] / max(total, 1) * 0.4)


_LABEL_ONLY = re.compile(
    r"(?i)^(?:n/?a|none|tbd|t\.b\.d\.?|see attached|as noted|same|unknown|[-_.,:;]*)$"
)
_LABEL_WORDS = re.compile(
    r"(?i)^(?:project|owner|contractor|architect|engineer|date|amount|total|description|scope|"
    r"change order|contract|status|reason|signature|title|subject|no|number|page)\b[\s:.#]*$"
)
_TRAILING_LABEL = re.compile(
    r"(?i)\s{2,}(?:project|owner|contractor|architect|date|amount|total|contract|change order|"
    r"page|status|reason|phone|fax|email)\b.*$"
)
# A capture that begins with a label fragment means the matcher latched onto
# the wrong caption: "Project" matched but the line was "Project No.: X".
_LEADING_LABEL_FRAGMENT = re.compile(
    r"(?i)^(?:nos?|number|num|#|name|title|id|code|type)\b[\s.:#]*"
)


def clean_string_value(raw: str, max_chars: int = 200) -> tuple[str | None, list[str]]:
    """Tidy a free text capture and reject the obvious non-values."""
    notes: list[str] = []
    if raw is None:
        return None, ["empty"]
    text = raw.replace("\t", " ").strip()
    text = text.lstrip(":-=\u2013 ").strip()

    # Column layouts leave the neighbouring cell in the capture. Only do this
    # for single line captures: a paragraph is allowed to contain indentation.
    if "\n" not in text and re.search(r"\s{3,}", text):
        head = re.split(r"\s{3,}", text)[0].strip()
        if len(head) >= 2:
            if head != text:
                notes.append("trimmed_adjacent_column")
            text = head
    text = _TRAILING_LABEL.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip(" .,;|")

    if not text or _LABEL_ONLY.match(text):
        return None, notes + ["placeholder_value"]
    if _LABEL_WORDS.match(text) or _LEADING_LABEL_FRAGMENT.match(text):
        return None, notes + ["captured_label_not_value"]
    if len(text) < 2:
        return None, notes + ["too_short"]

    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0]
        text = (cut or text[:max_chars]).rstrip() + "..."
        notes.append("truncated")
    return text, notes
