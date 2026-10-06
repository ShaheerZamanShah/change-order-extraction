"""Arithmetic repair.

Change orders are redundant by design, so a missing number can often be
recovered exactly from the ones that are present. Recovering it is better than
leaving a hole, but a derived value is not the same as a read value, so it is
capped at a lower confidence, tagged in the notes, and reported as an issue
with ``auto_repaired`` set. Nothing in the output pretends we read it off the
page.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from ..config import PipelineConfig
from ..schema import (
    ChangeOrderRecord,
    Evidence,
    ExtractedField,
    Severity,
    ValidationIssue,
)

DERIVED_EXTRACTOR = "derived"
DERIVED_PENALTY = 0.9


def _set_derived(
    record: ChangeOrderRecord,
    name: str,
    value,
    sources: list[str],
    config: PipelineConfig,
    explanation: str,
) -> ValidationIssue:
    source_confidences = [record.get(s).confidence for s in sources if record.get(s).is_present]
    confidence = min(config.derived_value_ceiling, DERIVED_PENALTY * min(source_confidences or [0.5]))
    setattr(
        record,
        name,
        ExtractedField(
            value=value,
            confidence=round(confidence, 3),
            extractor=DERIVED_EXTRACTOR,
            raw_text=None,
            evidence=[Evidence(snippet=explanation, matcher="arithmetic_identity")],
            notes=["derived_not_read_from_document", f"derived_from:{'+'.join(sources)}"],
        ),
    )
    return ValidationIssue(
        code="FIELD_DERIVED",
        severity=Severity.INFO,
        message=explanation,
        fields=[name] + sources,
        auto_repaired=True,
    )


def repair_record(record: ChangeOrderRecord, config: PipelineConfig) -> list[ValidationIssue]:
    """Fill gaps that follow from the values we already have."""
    issues: list[ValidationIssue] = []
    if not config.enable_repair:
        return issues

    def money(name: str):
        value = record.get(name).value
        return value if isinstance(value, Decimal) else None

    # Two passes so a value derived in the first pass can feed the second.
    for _ in range(2):
        original = money("original_contract_sum")
        previous = money("net_previous_changes")
        before = money("contract_sum_before")
        change = money("change_amount")
        after = money("contract_sum_after")

        if before is None and original is not None and previous is not None:
            issues.append(
                _set_derived(
                    record,
                    "contract_sum_before",
                    original + previous,
                    ["original_contract_sum", "net_previous_changes"],
                    config,
                    "contract sum prior to this change derived as original sum plus net previous changes",
                )
            )
        elif previous is None and original is not None and before is not None:
            issues.append(
                _set_derived(
                    record,
                    "net_previous_changes",
                    before - original,
                    ["contract_sum_before", "original_contract_sum"],
                    config,
                    "net previous changes derived as prior contract sum minus the original sum",
                )
            )
        elif original is None and before is not None and previous is not None:
            issues.append(
                _set_derived(
                    record,
                    "original_contract_sum",
                    before - previous,
                    ["contract_sum_before", "net_previous_changes"],
                    config,
                    "original contract sum derived as prior contract sum minus net previous changes",
                )
            )

        before = money("contract_sum_before")
        if after is None and before is not None and change is not None:
            issues.append(
                _set_derived(
                    record,
                    "contract_sum_after",
                    before + change,
                    ["contract_sum_before", "change_amount"],
                    config,
                    "new contract sum derived as prior contract sum plus this change",
                )
            )
        elif change is None and before is not None and after is not None:
            issues.append(
                _set_derived(
                    record,
                    "change_amount",
                    after - before,
                    ["contract_sum_after", "contract_sum_before"],
                    config,
                    "change amount derived as new contract sum minus prior contract sum",
                )
            )
        elif before is None and after is not None and change is not None:
            issues.append(
                _set_derived(
                    record,
                    "contract_sum_before",
                    after - change,
                    ["contract_sum_after", "change_amount"],
                    config,
                    "prior contract sum derived as new contract sum minus this change",
                )
            )

    # Change amount from line items when the document never states a total.
    if record.get("change_amount").value is None and record.line_items:
        amounts = [i.amount for i in record.line_items if i.amount is not None]
        if len(amounts) >= 2:
            total = sum(amounts, Decimal("0.00"))
            confidence = min(
                config.derived_value_ceiling,
                0.85 * (sum(i.confidence for i in record.line_items) / len(record.line_items)),
            )
            record.change_amount = ExtractedField(
                value=total,
                confidence=round(confidence, 3),
                extractor=DERIVED_EXTRACTOR,
                evidence=[
                    Evidence(snippet=f"sum of {len(amounts)} line items", matcher="line_item_sum")
                ],
                notes=["derived_not_read_from_document", "derived_from:line_items"],
            )
            issues.append(
                ValidationIssue(
                    code="FIELD_DERIVED",
                    severity=Severity.INFO,
                    message=f"change amount derived by summing {len(amounts)} line items",
                    fields=["change_amount", "line_items"],
                    auto_repaired=True,
                )
            )

    # Schedule: the two dates and the day count are three views of one fact.
    days = record.get("schedule_impact_days").value
    original_completion = record.get("original_completion_date").value
    revised_completion = record.get("revised_completion_date").value

    if days is None and original_completion and revised_completion:
        issues.append(
            _set_derived(
                record,
                "schedule_impact_days",
                (revised_completion - original_completion).days,
                ["original_completion_date", "revised_completion_date"],
                config,
                "time impact derived from the gap between the original and revised completion dates",
            )
        )
    elif revised_completion is None and original_completion and isinstance(days, int):
        issues.append(
            _set_derived(
                record,
                "revised_completion_date",
                original_completion + timedelta(days=days),
                ["original_completion_date", "schedule_impact_days"],
                config,
                "revised completion date derived from the original date plus the stated time impact",
            )
        )

    return issues
