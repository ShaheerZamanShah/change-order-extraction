"""Candidate model shared by every extractor.

Extractors never decide anything. They propose candidates with a local score
and the evidence behind them, and arbitration picks a winner. This keeps new
extractors cheap to add and makes disagreement visible instead of hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional, Protocol, runtime_checkable

from ..ingest import Document


@dataclass
class Candidate:
    field_name: str
    value: Any
    raw_text: str = ""
    score: float = 0.0
    extractor: str = ""
    matcher: str = ""
    start: Optional[int] = None
    end: Optional[int] = None
    page: Optional[int] = None
    notes: list[str] = field(default_factory=list)

    def value_key(self) -> str:
        """Normalised key used to cluster candidates that mean the same thing."""
        value = self.value
        if value is None:
            return "<none>"
        if isinstance(value, Decimal):
            return f"{value:.2f}"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return str(value)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        text = str(value).casefold()
        return " ".join(text.split()).rstrip(".,;:")


@runtime_checkable
class Extractor(Protocol):
    name: str

    def extract(self, doc: Document) -> list[Candidate]:  # pragma: no cover - protocol
        ...


# Parse note tags that should reduce a candidate's score, and by how much.
NOTE_PENALTIES: dict[str, float] = {
    "ambiguous_decimal_separator": 0.25,
    "ambiguous_day_month_order": 0.10,
    "decimal_comma": 0.06,
    "european_grouping": 0.04,
    "dayfirst_inferred": 0.03,
    "trimmed_adjacent_column": 0.08,
    "truncated": 0.05,
    "sign_from_context": 0.07,
    "value_below_label": 0.05,
    "quote_fuzzy_match": 0.12,
    "derived_from_table_total": 0.05,
}


def apply_note_penalties(score: float, notes: list[str]) -> float:
    for note in notes:
        score *= 1.0 - NOTE_PENALTIES.get(note, 0.0)
    return max(0.0, min(1.0, score))
