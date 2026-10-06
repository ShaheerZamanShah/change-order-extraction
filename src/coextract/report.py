"""Human readable summary of a result, for the terminal and for review queues."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from .fields import FIELD_BY_NAME
from .schema import ExtractionResult, ReviewDecision, Severity

_BAR_WIDTH = 10


def _bar(confidence: float) -> str:
    filled = int(round(confidence * _BAR_WIDTH))
    return "#" * filled + "." * (_BAR_WIDTH - filled)


def _format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return f"{value:,.2f}"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value)
    return text if len(text) <= 70 else text[:67] + "..."


def render_text_report(result: ExtractionResult, show_evidence: bool = False) -> str:
    lines: list[str] = []
    meta = result.document
    lines.append(f"Document     {meta.source_name}")
    lines.append(
        f"Ingest       {meta.ingest_method.value}, {meta.page_count} page(s), "
        f"{meta.char_count} chars, text quality {meta.text_quality:.2f}"
    )
    if meta.quality_notes:
        lines.append(f"Ingest notes {', '.join(meta.quality_notes[:4])}")
    lines.append(f"Extractors   {', '.join(result.extractors_used)}  ({result.runtime_ms} ms)")
    lines.append("")

    decision = result.review.decision
    marker = {
        ReviewDecision.AUTO_ACCEPT: "AUTO ACCEPT",
        ReviewDecision.REVIEW: "NEEDS REVIEW",
        ReviewDecision.REJECT: "REJECT",
    }[decision]
    lines.append(f"{marker}   document confidence {result.review.document_confidence:.2f}")
    for reason in result.review.reasons:
        lines.append(f"  - {reason}")
    lines.append("")

    lines.append(f"{'field':<28}{'conf':>6}  {'':<{_BAR_WIDTH}}  value")
    lines.append("-" * 92)
    record = result.change_order
    for name in record.field_names():
        extracted = record.get(name)
        spec = FIELD_BY_NAME.get(name)
        flag = "*" if spec and spec.required and not extracted.is_present else " "
        lines.append(
            f"{flag}{name:<27}{extracted.confidence:>6.2f}  {_bar(extracted.confidence)}  "
            f"{_format_value(extracted.value)}"
        )
        if show_evidence and extracted.evidence:
            evidence = extracted.evidence[0]
            lines.append(f"{'':<28}        p{evidence.page} via {evidence.matcher}: {evidence.snippet}")

    if record.line_items:
        lines.append("")
        lines.append(f"Line items ({len(record.line_items)})")
        lines.append(f"{'#':>3}  {'description':<44}{'qty':>8}{'unit':>6}{'price':>12}{'amount':>14}{'conf':>6}")
        lines.append("-" * 94)
        total = Decimal("0.00")
        for item in record.line_items:
            total += item.amount or Decimal("0.00")
            lines.append(
                f"{item.index:>3}  {(item.description or '')[:44]:<44}"
                f"{_format_value(item.quantity):>8}{(item.unit or '-'):>6}"
                f"{_format_value(item.unit_price):>12}{_format_value(item.amount):>14}"
                f"{item.confidence:>6.2f}"
            )
        lines.append(f"{'':>3}  {'sum of line items':<44}{'':>26}{total:>14,.2f}")

    lines.append("")
    validation = result.validation
    lines.append(
        f"Validation   {len(validation.checks_run)} check group(s) run, "
        f"{validation.error_count} error(s), {validation.warning_count} warning(s)"
    )
    if validation.corroborated_fields:
        lines.append(f"Corroborated {', '.join(validation.corroborated_fields)}")
    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    for issue in sorted(validation.issues, key=lambda i: order[i.severity]):
        tag = issue.severity.value.upper()
        repaired = " (auto repaired)" if issue.auto_repaired else ""
        lines.append(f"  [{tag:<7}] {issue.code}: {issue.message}{repaired}")
        if issue.observed is not None or issue.expected is not None:
            lines.append(f"            observed {issue.observed}, expected {issue.expected}")

    return "\n".join(lines)


def render_csv_row(result: ExtractionResult) -> dict[str, object]:
    """Flat row for batch summaries and spreadsheets."""
    row: dict[str, object] = {
        "source": result.document.source_name,
        "decision": result.review.decision.value,
        "document_confidence": result.review.document_confidence,
        "text_quality": result.document.text_quality,
        "errors": result.validation.error_count,
        "warnings": result.validation.warning_count,
        "line_item_count": len(result.change_order.line_items),
    }
    for name, value in result.flat_values().items():
        row[name] = value
        row[f"{name}__confidence"] = result.change_order.get(name).confidence
    return row
