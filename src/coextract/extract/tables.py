"""Line item extraction.

Two paths, because change orders arrive both ways:

1. real PDF tables lifted by pdfplumber, where we map columns from the header;
2. text that only looks like a table, where columns are runs of spaces, pipes
   or tabs and we have to infer the layout from the row itself.

The total row is treated separately. It is not a line item, but it is one of
the better signals for the change order amount, so it becomes a candidate.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Optional

from ..fields import MONEY
from ..ingest import Document
from ..normalize.parse import clean_string_value, parse_money
from ..schema import LineItem
from .base import Candidate

MONEY_TOKEN = re.compile(MONEY)

# Column cues per canonical name. "index" is recognised only so that an item
# number column does not get mistaken for a quantity.
HEADER_CUES = {
    "index": ("item", "item no", "no", "no.", "#", "line", "line no", "ref"),
    "description": ("description", "item description", "scope", "work", "detail", "particulars"),
    "quantity": ("qty", "quantity", "qnty", "units"),
    "unit": ("unit", "uom", "u/m", "unit of measure", "um"),
    "unit_price": ("unit price", "unit cost", "rate", "price", "cost each", "each", "u/price", "unit rate"),
    "amount": ("amount", "total", "extended", "ext amount", "ext. amount", "line total", "extension", "cost"),
}

# Anchored at the start but not at the end, because total rows carry trailing
# words: "Total credit to Owner", "Total this Change Order".
TOTAL_ROW_CUES = re.compile(
    r"(?i)^\s*(?:grand\s+total|sub-?total|total|net\s+(?:change|cost|total)|"
    r"sum\s+of\s+(?:the\s+)?(?:above|items))\b"
)

# Rows that are legitimate cost components rather than a final total.
COMPONENT_CUES = re.compile(
    r"(?i)\b(?:overhead|profit|markup|mark-up|bond|insurance|tax|fee|escalation|general conditions)\b"
)

_TABLE_HEADER_LINE = re.compile(
    r"(?i)(?=.*\b(?:description|item|scope)\b)(?=.*\b(?:amount|total|cost|price)\b)"
)
_SPLIT_COLUMNS = re.compile(r"\s{2,}|\s*\|\s*|\t+")
_LEADING_INDEX = re.compile(r"^\s*(\d{1,3})[.)]\s+")

# Rows from the contract summary block look exactly like line items once you
# split on whitespace: a caption and a money amount. They are not line items,
# and letting them in corrupts the line item total check that the summary
# itself is supposed to be validated against.
_SUMMARY_ROW_RE = re.compile(
    r"(?i)\b(?:"
    r"original\s+contract|new\s+contract|revised\s+contract|adjusted\s+contract|current\s+contract|"
    r"contract\s+(?:sum|amount|price|value|time)|net\s+change|previous(?:ly)?\s+(?:approved|authori[sz]ed)?"
    r"\s*change\s+orders?|prior\s+change\s+orders?|amount\s+of\s+this\s+change|this\s+change\s+order"
    r"(?:\s+amount)?|change\s+order\s+(?:amount|total)|retainage|retention|schedule\s+impact|"
    r"approved\s+changes\s+to\s+date|sum\s+of\s+(?:all\s+)?previous|amount\s+of|"
    r"sum\s+will\s+be|awarded\s+(?:sum|amount|value)|guaranteed\s+maximum\s+price"
    r")\b"
)

# A money value worth trusting as a column: grouped thousands or real cents.
_WELL_FORMED_MONEY = re.compile(r"\d,\d{3}|\.\d{2}\b|\d+,\d{2}\b")
# A cell that is purely a number, with optional currency and bracket negation.
_PURE_NUMBER_CELL = re.compile(
    r"^[(\[]?\s*[-+]?\s*(?:USD|US\$|\$|€|£)?\s*[-+]?[\d][\d,. \u00a0]*\s*(?:%|USD|EUR|GBP)?\s*[)\]]?$"
)
# A cell that is entirely a date. Breakdown rows do not carry dates, so a row
# that does is almost certainly a change order log entry rather than a cost item.
_DATE_CELL = re.compile(
    r"(?i)^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"\d{1,2}[-\s][A-Za-z]{3,9}[-\s,]*\d{2,4}|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4})\s*$"
)
# A header that belongs to a change order log, not a cost breakdown.
_LOG_HEADER_RE = re.compile(r"(?i)\b(?:status|date)\b")

UNIT_TOKENS = {
    "EA", "LS", "LF", "SF", "SY", "CY", "HR", "HRS", "DAY", "DAYS", "LOT", "TON", "GAL",
    "MO", "WK", "CF", "CWT", "ROLL", "SET", "LUMP", "HOURLY", "DAILY", "ALLOW",
}


def _cue_score(key: str, cues: tuple[str, ...]) -> int:
    """Rank how well a header cell matches a column's cues.

    Exact beats prefix beats substring. Without this ordering an "Item" column
    matches the description cue list by substring and steals the slot from the
    actual "Description" column, which then silently drops every description.
    """
    best = 0
    for cue in cues:
        if key == cue:
            best = max(best, 3)
        elif key.startswith(cue) or cue.startswith(key):
            best = max(best, 2)
        elif cue in key:
            best = max(best, 1)
    return best


def _match_header(cells: list[str]) -> Optional[dict[str, int]]:
    scored: list[tuple[int, str, int]] = []  # (score, canonical, column index)
    for index, cell in enumerate(cells):
        key = re.sub(r"[^a-z0-9/. ]", " ", cell.lower()).strip()
        key = re.sub(r"\s+", " ", key).rstrip(".")
        if not key:
            continue
        for canonical, cues in HEADER_CUES.items():
            score = _cue_score(key, tuple(c.rstrip(".") for c in cues))
            if score:
                scored.append((score, canonical, index))

    mapping: dict[str, int] = {}
    taken: set[int] = set()
    for score, canonical, index in sorted(scored, key=lambda s: -s[0]):
        if canonical in mapping or index in taken:
            continue
        mapping[canonical] = index
        taken.add(index)

    mapping.pop("index", None)
    if "amount" in mapping and ("description" in mapping or "unit_price" in mapping):
        return mapping
    return None


def _numeric_cells(cells: list[str]) -> list[tuple[int, Decimal]]:
    """Cells that are a number and nothing else.

    The "and nothing else" is the point. A description like "Overhead and
    profit at 10 percent" contains a number, and treating it as a numeric
    column puts 10.00 where a quantity belongs.
    """
    out: list[tuple[int, Decimal]] = []
    for index, cell in enumerate(cells):
        text = cell.strip()
        if not text or not any(c.isdigit() for c in text):
            continue
        if not _PURE_NUMBER_CELL.match(text):
            continue
        value, notes = parse_money(text)
        if value is not None and "unparseable" not in notes:
            out.append((index, value))
    return out


def _row_from_mapping(cells: list[str], mapping: dict[str, int]) -> dict[str, object]:
    def cell(key: str) -> str:
        idx = mapping.get(key)
        if idx is None or idx >= len(cells):
            return ""
        return cells[idx].strip()

    description, _ = clean_string_value(cell("description"), max_chars=300)
    quantity, _ = parse_money(cell("quantity")) if cell("quantity") else (None, [])
    unit_price, _ = parse_money(cell("unit_price")) if cell("unit_price") else (None, [])
    amount, _ = parse_money(cell("amount")) if cell("amount") else (None, [])
    unit = cell("unit") or None
    return {
        "description": description or "",
        "quantity": quantity,
        "unit": unit,
        "unit_price": unit_price,
        "amount": amount,
    }


def _row_by_heuristic(cells: list[str]) -> Optional[dict[str, object]]:
    """Infer the layout when there is no usable header.

    Convention in almost every change order breakdown: the rightmost money
    column is the extended amount, the one before it is the unit price, and the
    leftmost long text cell is the description.
    """
    numbers = _numeric_cells(cells)
    if not numbers:
        return None
    text_cells = [(i, c) for i, c in enumerate(cells) if len(c.strip()) >= 4 and not c.strip().replace(",", "").replace(".", "").replace("$", "").isdigit()]
    if not text_cells:
        return None

    description = max(text_cells, key=lambda t: len(t[1]))[1]
    description, _ = clean_string_value(description, max_chars=300)
    amount = numbers[-1][1]
    unit_price = numbers[-2][1] if len(numbers) >= 2 else None
    quantity = numbers[-3][1] if len(numbers) >= 3 else None

    # If quantity times unit price lands on the amount we have the layout right.
    if quantity is not None and unit_price is not None:
        if abs(quantity * unit_price - amount) > Decimal("1.00"):
            if len(numbers) >= 2 and abs(numbers[0][1] * numbers[-2][1] - amount) <= Decimal("1.00"):
                quantity = numbers[0][1]

    unit = None
    for cell in cells:
        token = cell.strip().upper()
        if token in UNIT_TOKENS:
            unit = token
            break
    return {
        "description": description or "",
        "quantity": quantity,
        "unit": unit,
        "unit_price": unit_price,
        "amount": amount,
    }


def _row(cells: list[str], mapping: Optional[dict[str, int]]) -> Optional[dict[str, object]]:
    """Map a row using the header, falling back to layout inference.

    Markup, bond and tax rows typically fill only the description and amount
    columns, so a strict header mapping finds nothing in the amount column and
    drops them. Dropping them is what makes the line item total disagree with
    the change order amount, so the fallback is not optional.
    """
    row = _row_from_mapping(cells, mapping) if mapping else None
    if row is None or row.get("amount") is None or not row.get("description"):
        row = _row_by_heuristic(cells)
    return row


def _score_row(row: dict[str, object], base: float) -> float:
    score = base
    if not row.get("description"):
        score -= 0.25
    if row.get("amount") is None:
        score -= 0.35
    quantity, unit_price, amount = row.get("quantity"), row.get("unit_price"), row.get("amount")
    if isinstance(quantity, Decimal) and isinstance(unit_price, Decimal) and isinstance(amount, Decimal):
        if abs(quantity * unit_price - amount) <= Decimal("1.00"):
            score += 0.12
        else:
            score -= 0.15
    return max(0.05, min(0.98, score))


class TableExtractor:
    """Produces line items plus a change amount candidate from total rows."""

    name = "tables"

    def __init__(self) -> None:
        self.line_items: list[LineItem] = []

    def extract(self, doc: Document) -> list[Candidate]:
        self.line_items = []
        candidates: list[Candidate] = []

        for table in doc.tables:
            items, totals = self._from_grid(table.rows, table.page)
            self.line_items.extend(items)
            candidates.extend(self._total_candidates(totals, doc, table.page, base=0.84))

        if not self.line_items:
            items, totals = self._from_text(doc)
            self.line_items.extend(items)
            candidates.extend(self._total_candidates(totals, doc, None, base=0.7))

        for index, item in enumerate(self.line_items, start=1):
            item.index = index
        return candidates

    # -- grid path ----------------------------------------------------------
    def _from_grid(self, rows: list[list[str]], page: int) -> tuple[list[LineItem], list[tuple[str, Decimal]]]:
        items: list[LineItem] = []
        totals: list[tuple[str, Decimal]] = []
        mapping: Optional[dict[str, int]] = None

        for raw_row in rows:
            cells = [(c or "").strip() for c in raw_row]
            if not any(cells):
                continue
            if mapping is None:
                mapping = _match_header(cells)
                if mapping is not None:
                    continue

            joined = " ".join(c for c in cells if c)
            label = next((c for c in cells if c.strip()), "")
            numbers = _numeric_cells(cells)
            is_total = bool(TOTAL_ROW_CUES.match(label) or TOTAL_ROW_CUES.match(joined.strip()))
            if is_total and numbers:
                if not COMPONENT_CUES.search(joined):
                    totals.append((label or joined.strip(), numbers[-1][1]))
                continue
            if any(_DATE_CELL.match(c) for c in cells if c):
                continue
            if _SUMMARY_ROW_RE.search(joined):
                continue

            row = _row(cells, mapping)
            if not row or row.get("amount") is None:
                continue
            if not row.get("description") and row.get("unit_price") is None:
                continue
            items.append(
                LineItem(
                    index=0,
                    description=str(row.get("description") or ""),
                    quantity=row.get("quantity"),  # type: ignore[arg-type]
                    unit=row.get("unit"),  # type: ignore[arg-type]
                    unit_price=row.get("unit_price"),  # type: ignore[arg-type]
                    amount=row.get("amount"),  # type: ignore[arg-type]
                    confidence=round(_score_row(row, 0.86), 3),
                    page=page,
                    raw_row=" | ".join(cells),
                )
            )
        return items, totals

    # -- text path ----------------------------------------------------------
    def _from_text(self, doc: Document) -> tuple[list[LineItem], list[tuple[str, Decimal]]]:
        """Find breakdown rows in text that only looks like a table.

        Plenty of change orders have a cost breakdown with no header row at
        all, so a row cannot be recognised on its own. What identifies one is
        company: breakdown rows arrive in consecutive runs. A single isolated
        line with a caption and an amount is almost always part of the contract
        summary, not a breakdown, so a run of one is not accepted unless a
        header or a line number vouches for it.
        """
        lines = doc.text.split("\n")
        offsets: list[int] = []
        offset = 0
        for line in lines:
            offsets.append(offset)
            offset += len(line) + 1

        totals: list[tuple[str, Decimal]] = []
        rows: dict[int, dict] = {}
        mapping: Optional[dict[str, int]] = None
        header_line: Optional[int] = None
        in_log = False
        blank_run = 0

        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                blank_run += 1
                # One blank line ends the table. Breakdown tables do not
                # contain blank rows, and letting a header stay in scope past
                # the gap lets the contract summary inherit it.
                mapping, header_line, in_log = None, None, False
                continue
            blank_run = 0

            cells = [c for c in _SPLIT_COLUMNS.split(stripped) if c.strip()]

            if _TABLE_HEADER_LINE.match(stripped) and len(cells) >= 2:
                if any(_LOG_HEADER_RE.search(c) for c in cells):
                    # A change order log, not a breakdown. Skip the block.
                    mapping, header_line = None, None
                    in_log = True
                    continue
                in_log = False
                mapping = _match_header(cells)
                header_line = index
                continue

            numbers = _numeric_cells(cells)
            label = cells[0].strip() if cells else ""
            if numbers and (
                TOTAL_ROW_CUES.match(label)
                or TOTAL_ROW_CUES.match(stripped.rstrip("0123456789$,.() "))
            ):
                if not COMPONENT_CUES.search(stripped):
                    totals.append((label, numbers[-1][1]))
                continue

            if in_log or len(cells) < 2 or not numbers:
                continue
            if _SUMMARY_ROW_RE.search(stripped):
                continue
            if any(_DATE_CELL.match(c) for c in cells):
                continue
            if not _WELL_FORMED_MONEY.search(cells[numbers[-1][0]]):
                continue

            indexed = bool(_LEADING_INDEX.match(line))
            row_cells = ([_LEADING_INDEX.sub("", cells[0])] + cells[1:]) if indexed else cells
            row = _row(row_cells, mapping)
            if not row or row.get("amount") is None or not row.get("description"):
                continue
            rows[index] = {
                "row": row,
                "vouched": bool(mapping) or indexed,
                "raw": stripped,
            }

        accepted = self._accept_runs(rows, header_line)
        items = [
            LineItem(
                index=0,
                description=str(entry["row"].get("description") or ""),
                quantity=entry["row"].get("quantity"),  # type: ignore[arg-type]
                unit=entry["row"].get("unit"),  # type: ignore[arg-type]
                unit_price=entry["row"].get("unit_price"),  # type: ignore[arg-type]
                amount=entry["row"].get("amount"),  # type: ignore[arg-type]
                confidence=round(_score_row(entry["row"], 0.78 if entry["vouched"] else 0.68), 3),
                page=doc.page_of(offsets[line_index]),
                raw_row=entry["raw"],
            )
            for line_index, entry in accepted
        ]
        return items, totals

    @staticmethod
    def _accept_runs(
        rows: dict[int, dict], header_line: Optional[int]
    ) -> list[tuple[int, dict]]:
        """Keep rows that sit in a run of at least two, or that are vouched for."""
        accepted: list[tuple[int, dict]] = []
        for line_index in sorted(rows):
            entry = rows[line_index]
            neighbour = (line_index - 1) in rows or (line_index + 1) in rows
            if entry["vouched"] or neighbour or header_line == line_index - 1:
                accepted.append((line_index, entry))
        return accepted

    # -- totals -------------------------------------------------------------
    def _total_candidates(
        self,
        totals: list[tuple[str, Decimal]],
        doc: Document,
        page: Optional[int],
        base: float,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for label, value in totals:
            strength = base
            if re.search(r"(?i)grand total|total this change|net change", label):
                strength = min(0.92, base + 0.06)
            elif re.search(r"(?i)^sub-?total", label):
                strength = base - 0.2
            out.append(
                Candidate(
                    field_name="change_amount",
                    value=value,
                    raw_text=f"{label} {value}",
                    score=max(0.1, strength),
                    extractor=self.name,
                    matcher="table_total_row",
                    page=page,
                    notes=["derived_from_table_total"],
                )
            )
        return out
