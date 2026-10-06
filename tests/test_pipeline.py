"""End to end tests over the sample documents."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from coextract import ChangeOrderPipeline, PipelineConfig, extract_file, extract_many
from coextract.fields import FIELD_SPECS
from coextract.schema import ChangeOrderRecord, ExtractionResult, ReviewDecision

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


@pytest.fixture(scope="module")
def results() -> dict[str, ExtractionResult]:
    pipeline = ChangeOrderPipeline()
    return {p.name: pipeline.run_path(p) for p in sorted(SAMPLES.glob("*.txt"))}


def test_registry_and_output_schema_agree():
    """One source of truth for the field list, checked rather than assumed."""
    registry = {spec.name for spec in FIELD_SPECS}
    model = set(ChangeOrderRecord().field_names())
    assert registry == model


def test_clean_form_is_read_completely(results):
    result = results["co_001_aia_clean.txt"]
    record = result.change_order
    assert record.change_order_number.value == "007"
    assert record.project_name.value == "Riverside Elementary School Addition"
    assert record.original_contract_sum.value == Decimal("8742000.00")
    assert record.change_amount.value == Decimal("35667.00")
    assert record.contract_sum_after.value == Decimal("8992047.00")
    assert record.schedule_impact_days.value == 11
    assert result.validation.error_count == 0
    assert sum(i.amount for i in record.line_items if i.amount) == Decimal("35667.00")


def test_ocr_sample_survives_character_confusion(results):
    record = results["co_003_ocr_scan.txt"].change_order
    assert record.change_order_number.value == "03"
    assert record.project_name.value == "Lakeshore Transit Maintenance Facility"
    assert record.original_contract_sum.value == Decimal("2480000.00")
    assert record.change_amount.value == Decimal("12475.50")
    assert record.contract_sum_after.value == Decimal("2492475.50")


def test_european_formatting_is_read_correctly(results):
    record = results["co_008_eu_format.txt"].change_order
    assert record.currency.value == "EUR"
    assert record.original_contract_sum.value == Decimal("4850000.00")
    assert record.change_amount.value == Decimal("231875.50")
    assert record.date_issued.value.isoformat() == "2024-03-14"


def test_deductive_change_order_keeps_its_sign(results):
    record = results["co_005_deduct_credit.txt"].change_order
    assert record.change_amount.value == Decimal("-62480.00")
    assert record.contract_sum_after.value < record.contract_sum_before.value


def test_no_cost_change_order_reports_an_explicit_zero(results):
    record = results["co_006_no_cost_time_only.txt"].change_order
    assert record.change_amount.value == Decimal("0.00")
    assert record.schedule_impact_days.value == 14


def test_internally_inconsistent_document_is_not_auto_accepted(results):
    result = results["co_010_arithmetic_conflict.txt"]
    assert result.validation.error_count >= 1
    assert result.review.decision != ReviewDecision.AUTO_ACCEPT
    assert "change_amount" in result.review.fields_to_check or result.review.fields_to_check


def test_fragment_is_rejected_rather_than_guessed_at(results):
    result = results["co_011_fragment.txt"]
    assert result.review.decision == ReviewDecision.REJECT
    assert result.change_order.change_amount.value is None
    assert result.review.document_confidence < 0.4


def test_change_order_log_does_not_become_line_items(results):
    record = results["co_012_prior_co_noise.txt"].change_order
    assert record.change_amount.value == Decimal("214660.00")
    assert record.net_previous_changes.value == Decimal("130935.00")
    assert record.line_items == []


def test_multipage_document_strips_repeated_furniture(results):
    result = results["co_007_multipage.txt"]
    assert result.document.page_count == 3
    assert any("header/footer" in note for note in result.document.quality_notes)
    assert result.change_order.change_amount.value == Decimal("76350.00")


def test_evidence_offsets_point_at_the_real_text(results):
    """Evidence has to be usable, which means the span must resolve."""
    pipeline = ChangeOrderPipeline()
    document_text = (SAMPLES / "co_001_aia_clean.txt").read_text(encoding="utf-8")
    result = pipeline.run_text(document_text, "co_001_aia_clean.txt")
    field = result.change_order.change_amount
    assert field.evidence
    evidence = field.evidence[0]
    assert evidence.page == 1
    assert evidence.snippet
    assert "35,667" in evidence.snippet or "35667" in evidence.snippet


def test_alternatives_are_kept_for_audit(results):
    """Losing candidates are retained so a reviewer can see the competition."""
    assert any(
        result.change_order.get(name).alternatives
        for result in results.values()
        for name in result.change_order.field_names()
    )


def test_confidence_tracks_document_quality(results):
    clean = results["co_001_aia_clean.txt"].review.document_confidence
    memo = results["co_009_field_memo.txt"].review.document_confidence
    fragment = results["co_011_fragment.txt"].review.document_confidence
    assert clean > memo > fragment


def test_auto_accepted_documents_have_no_errors_and_no_blockers(results):
    for name, result in results.items():
        if result.review.decision == ReviewDecision.AUTO_ACCEPT:
            assert result.validation.error_count == 0, name
            for field_name in result.change_order.field_names():
                field = result.change_order.get(field_name)
                spec = next(s for s in FIELD_SPECS if s.name == field_name)
                if spec.required:
                    assert field.value is not None, f"{name}:{field_name}"


def test_result_serialises_and_round_trips(results):
    result = results["co_001_aia_clean.txt"]
    payload = json.loads(result.to_json())
    assert payload["schema_version"]
    assert payload["change_order"]["change_amount"]["value"] == "35667.00"
    assert payload["change_order"]["change_amount"]["confidence"] > 0
    assert payload["review"]["decision"] in ("auto_accept", "review", "reject")
    restored = ExtractionResult.model_validate(payload)
    assert restored.change_order.change_amount.value == Decimal("35667.00")


def test_flat_values_are_plain_json_types(results):
    for name, value in results["co_001_aia_clean.txt"].flat_values().items():
        assert value is None or isinstance(value, (str, int, float)), name


def test_batch_survives_an_unreadable_file(tmp_path):
    good = SAMPLES / "co_001_aia_clean.txt"
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf at all")
    results = extract_many([good, broken])
    assert len(results) == 2
    assert results[0].change_order.change_amount.value == Decimal("35667.00")
    assert results[1].validation.error_count >= 1


def test_unsupported_file_type_is_reported_clearly(tmp_path):
    odd = tmp_path / "drawing.dwg"
    odd.write_bytes(b"\x00\x01")
    with pytest.raises(Exception) as excinfo:
        extract_file(odd)
    assert "dwg" in str(excinfo.value).lower() or "unsupported" in str(excinfo.value).lower()


def test_thresholds_are_configurable():
    strict = PipelineConfig(auto_accept_threshold=0.99)
    lenient = PipelineConfig(auto_accept_threshold=0.1, field_review_threshold=0.0)
    path = SAMPLES / "co_001_aia_clean.txt"
    assert extract_file(path, config=strict).review.decision != ReviewDecision.AUTO_ACCEPT
    assert extract_file(path, config=lenient).review.decision == ReviewDecision.AUTO_ACCEPT


def test_every_sample_runs_without_raising(results):
    assert len(results) >= 12
    for name, result in results.items():
        assert result.runtime_ms >= 0, name
        assert result.extractors_used
