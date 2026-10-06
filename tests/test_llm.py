"""Tests for the optional LLM extractor, focused on the quote guard.

The guard is the only reason an LLM is allowed near a money field here, so it
gets tested harder than the happy path does.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from coextract.extract.llm import LLMExtractor, NullLLMClient, _locate_quote, build_prompt
from coextract.ingest import load_text
from coextract.pipeline import ChangeOrderPipeline

DOCUMENT = """CHANGE ORDER

Project Name: Harbour Point Terminal
Change Order Number: 014
Amount of this Change Order: $12,480.00
The Contract Time will be increased by 6 calendar days.
"""


class FakeClient:
    """Returns a canned payload so the guard can be tested without a network."""

    name = "fake"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return json.dumps(self.payload)


class ExplodingClient:
    name = "exploding"

    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("upstream model is down")


def extractor_for(payload: dict) -> tuple[LLMExtractor, list]:
    extractor = LLMExtractor(FakeClient(payload))
    candidates = extractor.extract(load_text(DOCUMENT))
    return extractor, candidates


def test_null_client_means_the_extractor_is_a_no_op():
    extractor = LLMExtractor(NullLLMClient())
    assert not extractor.enabled
    assert extractor.extract(load_text(DOCUMENT)) == []


def test_value_with_a_real_quote_is_accepted():
    _, candidates = extractor_for(
        {
            "fields": {
                "change_amount": {
                    "value": "12480.00",
                    "quote": "Amount of this Change Order: $12,480.00",
                    "confidence": 0.9,
                }
            }
        }
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.value == Decimal("12480.00")
    assert candidate.start is not None and candidate.end is not None
    assert candidate.score <= 0.88


def test_hallucinated_quote_is_discarded():
    """The invented quote fuzzy matches a real line, so the amount must be
    checked against the document rather than against the quote."""
    extractor, candidates = extractor_for(
        {
            "fields": {
                "change_amount": {
                    "value": "999999.00",
                    "quote": "Amount of this Change Order: $999,999.00",
                    "confidence": 0.99,
                }
            }
        }
    )
    assert candidates == []
    assert extractor.dropped


def test_quote_from_a_different_document_is_discarded():
    extractor, candidates = extractor_for(
        {
            "fields": {
                "change_amount": {
                    "value": "41250.00",
                    "quote": "Total of this authorization is forty one thousand two hundred fifty",
                    "confidence": 0.99,
                }
            }
        }
    )
    assert candidates == []
    assert any("quote_not_found" in reason for _, reason in extractor.dropped)


def test_cents_written_differently_still_verifies():
    """The document says 12,480.00 and the model says 12480; same number."""
    _, candidates = extractor_for(
        {
            "fields": {
                "change_amount": {
                    "value": "12480",
                    "quote": "Amount of this Change Order: $12,480.00",
                    "confidence": 0.8,
                }
            }
        }
    )
    assert len(candidates) == 1
    assert candidates[0].value == Decimal("12480.00")


def test_value_that_does_not_appear_in_its_own_quote_is_discarded():
    """The quote exists but does not contain the number the model reported."""
    extractor, candidates = extractor_for(
        {
            "fields": {
                "change_amount": {
                    "value": "25000.00",
                    "quote": "Project Name: Harbour Point Terminal",
                    "confidence": 0.95,
                }
            }
        }
    )
    assert candidates == []
    assert ("change_amount", "value_not_present_in_document") in extractor.dropped


def test_value_failing_its_type_check_is_discarded():
    extractor, candidates = extractor_for(
        {
            "fields": {
                "date_issued": {
                    "value": "sometime in the spring",
                    "quote": "Project Name: Harbour Point Terminal",
                    "confidence": 0.8,
                }
            }
        }
    )
    assert candidates == []
    assert ("date_issued", "value_failed_type_check") in extractor.dropped


def test_enum_value_outside_the_allowed_set_is_discarded():
    extractor, candidates = extractor_for(
        {
            "fields": {
                "reason_code": {
                    "value": "because the owner felt like it",
                    "quote": "Project Name: Harbour Point Terminal",
                    "confidence": 0.8,
                }
            }
        }
    )
    assert candidates == []


def test_unknown_field_names_are_ignored():
    _, candidates = extractor_for(
        {"fields": {"vibe": {"value": "good", "quote": "CHANGE ORDER", "confidence": 1.0}}}
    )
    assert candidates == []


def test_quote_matching_tolerates_whitespace_drift():
    document = load_text(DOCUMENT)
    start, end, notes = _locate_quote(document.text, "Amount of this  Change   Order: $12,480.00")
    assert start is not None
    assert "quote_not_found_in_document" not in notes


def test_a_failing_model_never_breaks_the_run():
    extractor = LLMExtractor(ExplodingClient())
    assert extractor.extract(load_text(DOCUMENT)) == []
    assert extractor.dropped and extractor.dropped[0][0] == "*"


def test_pipeline_runs_with_the_llm_enabled_and_merges_candidates():
    payload = {
        "fields": {
            "change_amount": {
                "value": "12480.00",
                "quote": "Amount of this Change Order: $12,480.00",
                "confidence": 0.9,
            }
        }
    }
    pipeline = ChangeOrderPipeline(llm_client=FakeClient(payload))
    result = pipeline.run_text(DOCUMENT, "harbour.txt")
    assert result.change_order.change_amount.value == Decimal("12480.00")
    assert any("llm" in used for used in result.extractors_used)
    # Rules and the model agreed, which should show up in the notes.
    assert "agreed_by_multiple_extractors" in result.change_order.change_amount.notes


def test_pipeline_records_discarded_model_values_as_issues():
    payload = {
        "fields": {
            "contract_sum_after": {
                "value": "50000000.00",
                "quote": "The new Contract Sum will be $50,000,000.00",
                "confidence": 0.99,
            }
        }
    }
    pipeline = ChangeOrderPipeline(llm_client=FakeClient(payload))
    result = pipeline.run_text(DOCUMENT, "harbour.txt")
    assert result.change_order.contract_sum_after.value is None
    assert any(issue.code == "LLM_CANDIDATE_REJECTED" for issue in result.validation.issues)


def test_prompt_lists_the_fields_and_includes_the_document():
    prompt = build_prompt(DOCUMENT)
    assert "change_amount" in prompt
    assert "schedule_impact_days" in prompt
    assert "Harbour Point Terminal" in prompt
    assert "verbatim" in prompt.lower() or "quote" in prompt.lower()


def test_prompt_truncates_a_very_long_document():
    prompt = build_prompt("x" * 100000, max_chars=2000)
    assert len(prompt) < 10000
