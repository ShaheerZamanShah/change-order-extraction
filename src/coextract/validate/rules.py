"""Validation.

Change orders carry their own redundancy. The same numbers appear two or three
times in different roles, and they have to add up:

    contract sum prior to this change = original sum + net previous changes
    new contract sum                  = contract sum prior + this change
    sum of the line items             = this change

That redundancy is the most valuable thing in the document. It lets us catch a
misread digit without any external reference data, which is why the validator
is not an afterthought here: it is how confidence gets earned.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Optional

from ..config import PipelineConfig
from ..fields import REQUIRED_FIELDS
from ..schema import (
    ChangeOrderRecord,
    ReasonCode,
    Severity,
    ValidationIssue,
    ValidationReport,
)

DEDUCTIVE_REASONS = {ReasonCode.SCOPE_DEDUCTION.value, ReasonCode.ALLOWANCE_ADJUSTMENT.value}


def _val(record: ChangeOrderRecord, name: str) -> Any:
    return record.get(name).value


def _money(record: ChangeOrderRecord, name: str) -> Optional[Decimal]:
    value = _val(record, name)
    return value if isinstance(value, Decimal) else None


def _as_date(record: ChangeOrderRecord, name: str) -> Optional[date]:
    value = _val(record, name)
    return value if isinstance(value, date) else None


def _fmt(value: Any) -> str:
    if isinstance(value, Decimal):
        return f"{value:,.2f}"
    return str(value)


def validate_record(record: ChangeOrderRecord, config: PipelineConfig) -> ValidationReport:
    issues: list[ValidationIssue] = []
    checks: list[str] = []
    corroborated: list[str] = []

    _check_required(record, issues, checks, config)
    _check_money_identities(record, issues, checks, config, corroborated)
    _check_line_items(record, issues, checks, config, corroborated)
    _check_dates(record, issues, checks)
    _check_schedule(record, issues, checks, corroborated)
    _check_plausibility(record, issues, checks, config)
    _check_status(record, issues, checks)

    errors = sum(1 for i in issues if i.severity == Severity.ERROR)
    warnings = sum(1 for i in issues if i.severity == Severity.WARNING)
    return ValidationReport(
        passed=errors == 0,
        error_count=errors,
        warning_count=warnings,
        issues=issues,
        checks_run=checks,
        corroborated_fields=sorted(set(corroborated)),
    )


def _corroborate(
    record: ChangeOrderRecord, fields: list[str], corroborated: list[str]
) -> None:
    """Credit fields for passing a cross check.

    A field that we derived from the very identity being checked cannot
    corroborate anything, since it was computed to satisfy it. Letting a
    derived value hand out confidence to its own inputs would be circular, so
    any group containing one is skipped entirely.
    """
    known = type(record).model_fields
    if any(record.get(name).extractor == "derived" for name in fields if name in known):
        return
    corroborated.extend(fields)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def _check_required(record, issues, checks, config) -> None:
    checks.append("required_fields_present")
    for name in REQUIRED_FIELDS:
        if _val(record, name) is None:
            hard = name in config.hard_required
            issues.append(
                ValidationIssue(
                    code="MISSING_REQUIRED_FIELD",
                    severity=Severity.ERROR if hard else Severity.WARNING,
                    message=f"{name} was not found in the document",
                    fields=[name],
                )
            )


def _compare(
    label: str,
    code: str,
    left: Decimal,
    right: Decimal,
    fields: list[str],
    config: PipelineConfig,
    issues: list[ValidationIssue],
) -> bool:
    tolerance = config.money_tolerance_for(max(abs(left), abs(right)))
    difference = abs(left - right)
    if difference <= tolerance:
        return True
    severity = Severity.ERROR if difference > tolerance * 10 else Severity.WARNING
    issues.append(
        ValidationIssue(
            code=code,
            severity=severity,
            message=f"{label} does not balance, off by {_fmt(difference)}",
            fields=fields,
            observed=_fmt(left),
            expected=_fmt(right),
        )
    )
    return False


def _check_money_identities(record, issues, checks, config, corroborated) -> None:
    checks.append("money_identities")
    original = _money(record, "original_contract_sum")
    previous = _money(record, "net_previous_changes")
    before = _money(record, "contract_sum_before")
    change = _money(record, "change_amount")
    after = _money(record, "contract_sum_after")

    if original is not None and previous is not None and before is not None:
        group = ["contract_sum_before", "original_contract_sum", "net_previous_changes"]
        if _compare(
            "contract sum prior to this change order",
            "SUM_BEFORE_MISMATCH",
            before,
            original + previous,
            group,
            config,
            issues,
        ):
            _corroborate(record, group, corroborated)

    if before is not None and change is not None and after is not None:
        group = ["contract_sum_after", "contract_sum_before", "change_amount"]
        if _compare(
            "new contract sum",
            "SUM_AFTER_MISMATCH",
            after,
            before + change,
            group,
            config,
            issues,
        ):
            _corroborate(record, group, corroborated)

    if (
        original is not None
        and previous is not None
        and change is not None
        and after is not None
        and before is None
    ):
        group = [
            "contract_sum_after",
            "original_contract_sum",
            "net_previous_changes",
            "change_amount",
        ]
        if _compare(
            "new contract sum against the full chain",
            "SUM_CHAIN_MISMATCH",
            after,
            original + previous + change,
            group,
            config,
            issues,
        ):
            _corroborate(record, group, corroborated)

    if after is not None and after <= 0:
        issues.append(
            ValidationIssue(
                code="NON_POSITIVE_CONTRACT_SUM",
                severity=Severity.ERROR,
                message="the new contract sum is zero or negative, which is almost always a misread value",
                fields=["contract_sum_after"],
                observed=_fmt(after),
            )
        )


def _check_line_items(record, issues, checks, config, corroborated) -> None:
    if not record.line_items:
        return
    checks.append("line_item_totals")
    amounts = [item.amount for item in record.line_items if item.amount is not None]
    if not amounts:
        return
    total = sum(amounts, Decimal("0.00"))
    change = _money(record, "change_amount")

    if change is not None:
        tolerance = config.money_tolerance_for(max(abs(total), abs(change)))
        difference = abs(total - change)
        if difference > tolerance:
            severity = (
                Severity.ERROR
                if difference > max(tolerance * 10, abs(change) * Decimal("0.10"))
                else Severity.WARNING
            )
            issues.append(
                ValidationIssue(
                    code="LINE_ITEM_TOTAL_MISMATCH",
                    severity=severity,
                    message=(
                        f"the {len(amounts)} line items add up to {_fmt(total)} "
                        f"but the change order amount reads {_fmt(change)}"
                    ),
                    fields=["change_amount", "line_items"],
                    observed=_fmt(total),
                    expected=_fmt(change),
                )
            )
        else:
            _corroborate(record, ["change_amount"], corroborated)

    checks.append("line_item_arithmetic")
    for item in record.line_items:
        if item.quantity is None or item.unit_price is None or item.amount is None:
            continue
        expected = (item.quantity * item.unit_price).quantize(Decimal("0.01"))
        # Compare magnitudes. On a credit line the unit price is quoted
        # positive and only the extended amount carries the minus sign, which
        # is a formatting convention rather than an arithmetic error.
        if abs(abs(expected) - abs(item.amount)) > config.money_tolerance_for(item.amount):
            issues.append(
                ValidationIssue(
                    code="LINE_ITEM_ARITHMETIC",
                    severity=Severity.WARNING,
                    message=(
                        f"line {item.index} quantity times unit price gives {_fmt(expected)} "
                        f"but the amount column reads {_fmt(item.amount)}"
                    ),
                    fields=["line_items"],
                    observed=_fmt(item.amount),
                    expected=_fmt(expected),
                )
            )


def _check_dates(record, issues, checks) -> None:
    checks.append("date_ordering")
    issued = _as_date(record, "date_issued")
    executed = _as_date(record, "date_executed")
    original_completion = _as_date(record, "original_completion_date")
    revised_completion = _as_date(record, "revised_completion_date")

    if issued and executed and executed < issued:
        issues.append(
            ValidationIssue(
                code="EXECUTED_BEFORE_ISSUED",
                severity=Severity.WARNING,
                message="the execution date is earlier than the issue date",
                fields=["date_issued", "date_executed"],
                observed=executed.isoformat(),
                expected=f"on or after {issued.isoformat()}",
            )
        )

    if original_completion and revised_completion and revised_completion < original_completion:
        days = _val(record, "schedule_impact_days")
        if not isinstance(days, int) or days >= 0:
            issues.append(
                ValidationIssue(
                    code="COMPLETION_DATE_REGRESSION",
                    severity=Severity.WARNING,
                    message="the revised completion date is earlier than the original with no acceleration stated",
                    fields=["original_completion_date", "revised_completion_date"],
                    observed=revised_completion.isoformat(),
                    expected=f"on or after {original_completion.isoformat()}",
                )
            )

    if issued and revised_completion and revised_completion < issued:
        issues.append(
            ValidationIssue(
                code="COMPLETION_BEFORE_ISSUE",
                severity=Severity.WARNING,
                message="the revised completion date is before the change order was issued",
                fields=["date_issued", "revised_completion_date"],
            )
        )


def _check_schedule(record, issues, checks, corroborated) -> None:
    days = _val(record, "schedule_impact_days")
    original_completion = _as_date(record, "original_completion_date")
    revised_completion = _as_date(record, "revised_completion_date")
    if not isinstance(days, int) or not original_completion or not revised_completion:
        return
    checks.append("schedule_consistency")
    delta = (revised_completion - original_completion).days
    if abs(delta - days) <= 2:
        _corroborate(
            record,
            ["schedule_impact_days", "original_completion_date", "revised_completion_date"],
            corroborated,
        )
    else:
        issues.append(
            ValidationIssue(
                code="SCHEDULE_DAYS_MISMATCH",
                severity=Severity.WARNING,
                message=(
                    f"the stated time impact is {days} day(s) but the completion dates are "
                    f"{delta} day(s) apart"
                ),
                fields=["schedule_impact_days", "original_completion_date", "revised_completion_date"],
                observed=str(days),
                expected=str(delta),
            )
        )


def _check_plausibility(record, issues, checks, config) -> None:
    checks.append("plausibility")
    change = _money(record, "change_amount")
    original = _money(record, "original_contract_sum")
    days = _val(record, "schedule_impact_days")

    if change is not None and original is not None and original > 0:
        ratio = abs(change) / original
        if ratio > Decimal("2"):
            issues.append(
                ValidationIssue(
                    code="CHANGE_EXCEEDS_CONTRACT",
                    severity=Severity.ERROR,
                    message=(
                        f"this change is {ratio:.1f} times the original contract sum, "
                        "which usually means a label was matched to the wrong number"
                    ),
                    fields=["change_amount", "original_contract_sum"],
                    observed=_fmt(change),
                )
            )
        elif ratio > Decimal("0.5"):
            issues.append(
                ValidationIssue(
                    code="LARGE_CHANGE_RATIO",
                    severity=Severity.WARNING,
                    message=f"this change is {ratio * 100:.0f} percent of the original contract sum",
                    fields=["change_amount", "original_contract_sum"],
                    observed=_fmt(change),
                )
            )

    if change is not None and change == 0 and (days in (0, None)):
        issues.append(
            ValidationIssue(
                code="NO_COST_NO_TIME",
                severity=Severity.WARNING,
                message="this change order has no cost and no time impact, confirm it is not a misread",
                fields=["change_amount", "schedule_impact_days"],
            )
        )

    reason = _val(record, "reason_code")
    if change is not None and change < 0 and reason is not None:
        reason_value = reason.value if hasattr(reason, "value") else str(reason)
        if reason_value not in DEDUCTIVE_REASONS:
            issues.append(
                ValidationIssue(
                    code="DEDUCT_WITH_ADDITIVE_REASON",
                    severity=Severity.INFO,
                    message=f"the amount is a credit but the reason reads as {reason_value}",
                    fields=["change_amount", "reason_code"],
                )
            )

    number = _val(record, "change_order_number")
    if isinstance(number, str) and not any(c.isdigit() for c in number):
        issues.append(
            ValidationIssue(
                code="SUSPICIOUS_CO_NUMBER",
                severity=Severity.WARNING,
                message="the change order number contains no digits",
                fields=["change_order_number"],
                observed=number,
            )
        )


def _check_status(record, issues, checks) -> None:
    checks.append("status_consistency")
    status = _val(record, "status")
    status_value = status.value if hasattr(status, "value") else status
    executed = _as_date(record, "date_executed")
    if status_value in ("executed", "approved") and executed is None:
        issues.append(
            ValidationIssue(
                code="EXECUTED_WITHOUT_DATE",
                severity=Severity.WARNING,
                message=f"status reads {status_value} but no execution or approval date was found",
                fields=["status", "date_executed"],
            )
        )
