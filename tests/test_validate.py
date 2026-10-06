"""Validation, repair and confidence feedback tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from coextract.config import PipelineConfig
from coextract.confidence import apply_validation_feedback, document_confidence
from coextract.schema import (
    ChangeOrderRecord,
    ExtractedField,
    LineItem,
    Severity,
)
from coextract.validate import repair_record, validate_record


def money_field(amount: str, confidence: float = 0.9) -> ExtractedField:
    return ExtractedField(value=Decimal(amount), confidence=confidence, extractor="rules")


def balanced_record() -> ChangeOrderRecord:
    record = ChangeOrderRecord()
    record.change_order_number = ExtractedField(value="007", confidence=0.95, extractor="rules")
    record.project_name = ExtractedField(value="Riverside School", confidence=0.9, extractor="rules")
    record.scope_description = ExtractedField(value="New fire main", confidence=0.9, extractor="rules")
    record.date_issued = ExtractedField(value=date(2024, 3, 14), confidence=0.95, extractor="rules")
    record.original_contract_sum = money_field("8742000.00")
    record.net_previous_changes = money_field("214380.00")
    record.contract_sum_before = money_field("8956380.00")
    record.change_amount = money_field("35667.00")
    record.contract_sum_after = money_field("8992047.00")
    return record


CONFIG = PipelineConfig()


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_balanced_record_passes_and_is_corroborated():
    report = validate_record(balanced_record(), CONFIG)
    assert report.passed
    assert report.error_count == 0
    assert "change_amount" in report.corroborated_fields
    assert "contract_sum_after" in report.corroborated_fields


def test_broken_sum_chain_is_caught():
    record = balanced_record()
    record.contract_sum_after = money_field("8995047.00")  # out by 3,000
    report = validate_record(record, CONFIG)
    assert "SUM_AFTER_MISMATCH" in codes(report)
    assert report.error_count >= 1


def test_tolerance_does_not_scale_away_a_real_error_on_a_big_contract():
    """Regression: a relative tolerance let a 3,000 error through on a 1.9m contract."""
    record = ChangeOrderRecord()
    record.contract_sum_before = money_field("1887100.00")
    record.change_amount = money_field("25568.00")
    record.contract_sum_after = money_field("1915668.00")  # should be 1,912,668
    report = validate_record(record, CONFIG)
    assert "SUM_AFTER_MISMATCH" in codes(report)


def test_rounding_difference_is_tolerated():
    record = ChangeOrderRecord()
    record.contract_sum_before = money_field("1887100.00")
    record.change_amount = money_field("25568.00")
    record.contract_sum_after = money_field("1912668.49")
    report = validate_record(record, CONFIG)
    assert "SUM_AFTER_MISMATCH" not in codes(report)


def test_line_item_total_must_match_the_change_amount():
    record = balanced_record()
    record.line_items = [
        LineItem(index=1, description="pipe", amount=Decimal("10000.00"), confidence=0.9),
        LineItem(index=2, description="valves", amount=Decimal("5000.00"), confidence=0.9),
    ]
    report = validate_record(record, CONFIG)
    assert "LINE_ITEM_TOTAL_MISMATCH" in codes(report)


def test_credit_line_item_arithmetic_is_not_flagged_for_its_sign():
    record = balanced_record()
    record.change_amount = money_field("-54940.00")
    record.contract_sum_after = money_field("8901440.00")
    record.line_items = [
        LineItem(
            index=1,
            description="deduct terrazzo",
            quantity=Decimal("4100"),
            unit_price=Decimal("13.40"),
            amount=Decimal("-54940.00"),
            confidence=0.9,
        )
    ]
    report = validate_record(record, CONFIG)
    assert "LINE_ITEM_ARITHMETIC" not in codes(report)


def test_schedule_days_must_agree_with_the_completion_dates():
    record = balanced_record()
    record.schedule_impact_days = ExtractedField(value=5, confidence=0.9, extractor="rules")
    record.original_completion_date = ExtractedField(value=date(2024, 9, 13), confidence=0.9, extractor="rules")
    record.revised_completion_date = ExtractedField(value=date(2024, 9, 25), confidence=0.9, extractor="rules")
    report = validate_record(record, CONFIG)
    assert "SCHEDULE_DAYS_MISMATCH" in codes(report)


def test_execution_date_before_issue_date_is_flagged():
    record = balanced_record()
    record.date_executed = ExtractedField(value=date(2024, 3, 1), confidence=0.9, extractor="rules")
    report = validate_record(record, CONFIG)
    assert "EXECUTED_BEFORE_ISSUED" in codes(report)


def test_change_larger_than_the_contract_is_an_error():
    record = balanced_record()
    record.change_amount = money_field("20000000.00")
    report = validate_record(record, CONFIG)
    assert "CHANGE_EXCEEDS_CONTRACT" in codes(report)


def test_missing_hard_required_field_is_an_error():
    record = balanced_record()
    record.change_amount = ExtractedField()
    report = validate_record(record, CONFIG)
    errors = [i for i in report.issues if i.severity == Severity.ERROR]
    assert any(i.code == "MISSING_REQUIRED_FIELD" for i in errors)


# --- repair ---------------------------------------------------------------


def test_repair_derives_the_missing_contract_sum():
    record = balanced_record()
    record.contract_sum_after = ExtractedField()
    issues = repair_record(record, CONFIG)
    assert record.contract_sum_after.value == Decimal("8992047.00")
    assert record.contract_sum_after.extractor == "derived"
    assert any(i.code == "FIELD_DERIVED" and i.auto_repaired for i in issues)


def test_repair_derives_the_missing_original_sum():
    record = balanced_record()
    record.original_contract_sum = ExtractedField()
    repair_record(record, CONFIG)
    assert record.original_contract_sum.value == Decimal("8742000.00")


def test_derived_values_are_capped_below_a_read_value():
    record = balanced_record()
    record.change_amount = ExtractedField()
    repair_record(record, CONFIG)
    assert record.change_amount.value == Decimal("35667.00")
    assert record.change_amount.confidence <= CONFIG.derived_value_ceiling
    assert "derived_not_read_from_document" in record.change_amount.notes


def test_derived_field_does_not_corroborate_its_own_inputs():
    """A value computed from an identity will always satisfy that identity.

    Letting it hand confidence back to the numbers it came from would be
    circular, and would make a half read document look fully verified.
    """
    record = balanced_record()
    record.contract_sum_after = ExtractedField()
    repair_record(record, CONFIG)
    report = validate_record(record, CONFIG)
    assert "contract_sum_after" not in report.corroborated_fields
    assert "change_amount" not in report.corroborated_fields


def test_repair_derives_schedule_days_from_the_dates():
    record = balanced_record()
    record.original_completion_date = ExtractedField(value=date(2024, 8, 16), confidence=0.9, extractor="rules")
    record.revised_completion_date = ExtractedField(value=date(2024, 8, 27), confidence=0.9, extractor="rules")
    repair_record(record, CONFIG)
    assert record.schedule_impact_days.value == 11


def test_repair_can_be_turned_off():
    config = PipelineConfig(enable_repair=False)
    record = balanced_record()
    record.contract_sum_after = ExtractedField()
    assert repair_record(record, config) == []
    assert record.contract_sum_after.value is None


# --- confidence feedback --------------------------------------------------


def test_corroboration_raises_confidence_and_failure_lowers_it():
    good = balanced_record()
    good_report = validate_record(good, CONFIG)
    before = good.change_amount.confidence
    apply_validation_feedback(good, good_report, CONFIG)
    assert good.change_amount.confidence > before

    bad = balanced_record()
    bad.contract_sum_after = money_field("8995047.00")
    bad_report = validate_record(bad, CONFIG)
    baseline = bad.change_amount.confidence
    apply_validation_feedback(bad, bad_report, CONFIG)
    assert bad.change_amount.confidence < baseline


def test_every_field_named_in_a_failed_check_takes_the_penalty():
    """We cannot tell which of the three numbers is wrong, so all three drop."""
    record = balanced_record()
    record.contract_sum_after = money_field("8995047.00")
    report = validate_record(record, CONFIG)
    apply_validation_feedback(record, report, CONFIG)
    for name in ("contract_sum_after", "contract_sum_before", "change_amount"):
        assert record.get(name).confidence < 0.9, name


def test_document_confidence_drops_when_a_required_field_is_missing():
    full = balanced_record()
    partial = balanced_record()
    partial.project_name = ExtractedField()
    report = validate_record(partial, CONFIG)
    assert document_confidence(partial, report, CONFIG) < document_confidence(
        full, validate_record(full, CONFIG), CONFIG
    )


def test_money_tolerance_is_capped_in_absolute_terms():
    config = PipelineConfig()
    assert config.money_tolerance_for(Decimal("100.00")) == Decimal("1.00")
    assert config.money_tolerance_for(Decimal("50000000.00")) == config.max_money_tolerance
