"""Extractor tests: rule patterns, enum cues and table reading."""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from coextract.extract.rules import RuleExtractor
from coextract.extract.tables import TableExtractor
from coextract.fields import FIELD_SPECS, compiled_extra_patterns, compiled_label_patterns
from coextract.ingest import load_text


def candidates_for(text: str, field: str):
    doc = load_text(text)
    return [c for c in RuleExtractor().extract(doc) if c.field_name == field]


def best_value(text: str, field: str):
    found = candidates_for(text, field)
    if not found:
        return None
    return max(found, key=lambda c: c.score).value


def test_label_with_filler_words_between_caption_and_value():
    text = "THE ORIGINAL CONTRACT SUM WAS                 8,742,000.00\n"
    assert best_value(text, "original_contract_sum") == Decimal("8742000.00")


def test_label_with_long_filler_phrase():
    text = "THE NEW CONTRACT SUM INCLUDING THIS CHANGE ORDER WILL BE   8,992,047.00\n"
    assert best_value(text, "contract_sum_after") == Decimal("8992047.00")


def test_value_on_the_line_below_the_caption():
    text = "DESCRIPTION OF CHANGE:\nFurnish and install additional fire main piping.\n"
    value = best_value(text, "scope_description")
    assert value is not None and "fire main piping" in value


def test_pipe_delimited_label():
    assert best_value("Job No. | HVM-2201\n", "project_number") == "HVM-2201"


def test_generic_project_label_does_not_steal_the_project_number():
    """Regression: "project" matched the "Project No." line and captured "No.: X"."""
    text = "PROJECT: Riverside Elementary School\nPROJECT NO.: RES-2023-114\n"
    assert best_value(text, "project_name") == "Riverside Elementary School"
    assert best_value(text, "project_number") == "RES-2023-114"


def test_short_alias_does_not_match_the_start_of_a_longer_word():
    """The "re" alias must not fire on "REASON FOR CHANGE"."""
    text = "REASON FOR CHANGE: Code official requirement.\n"
    titles = [c.value for c in candidates_for(text, "change_order_title")]
    assert not any(v and "ason for change" in str(v).lower() for v in titles)


def test_decrease_wording_produces_a_negative_amount():
    text = "The Contract Sum will be decreased by this Change Order in the amount of $62,480.00.\n"
    assert best_value(text, "change_amount") == Decimal("-62480.00")


def test_bracketed_amount_stays_negative():
    assert best_value("Amount of this Change Order ... ($62,480.00)\n", "change_amount") == Decimal("-62480.00")


def test_no_change_wording_becomes_an_explicit_zero():
    """A stated zero and a missing value are different facts."""
    text = "There is no change to the Contract Time as a result of this Change Order.\n"
    assert best_value(text, "schedule_impact_days") == 0


def test_time_extension_phrase():
    assert best_value("THE CONTRACT TIME WILL BE INCREASED BY 11 CALENDAR DAYS.\n", "schedule_impact_days") == 11


def test_enum_cue_with_label_beats_cue_in_prose():
    text = "Status: Pending owner signature\nPreviously approved work was approved last year.\n"
    assert best_value(text, "status") == "pending"


def test_enum_cue_exclusion_ignores_a_signature_caption():
    """Regression: "Date Executed:" set the status to executed."""
    text = "Date Executed: 2024-06-06\nStatus: Executed\n"
    statuses = {c.value for c in candidates_for(text, "status")}
    assert statuses == {"executed"}


def test_enum_cue_exclusion_is_scoped_to_its_own_line():
    """A nearby date caption must not veto a real status line."""
    text = "Date Issued: 2024-05-30\nDate Executed: 2024-06-06\nStatus: Executed\n"
    assert best_value(text, "status") == "executed"


def test_prior_change_orders_approved_does_not_set_the_status():
    text = "We have eleven prior change orders approved totaling $386,440.00.\n"
    assert best_value(text, "status") is None


def test_change_order_number_is_normalised_to_the_number():
    assert best_value("Subject: RE: Northgate Hub CO 12 pricing\n", "change_order_number") == "12"


def test_narrative_amounts_are_found_with_lower_confidence():
    text = (
        "Pricing came back at $48,920.00 all in, which includes the levelers.\n"
        "For the record, the contract started at $14,250,000.00 and we are sitting at\n"
        "$14,636,440.00 before this one. With CO 12 the contract goes to $14,685,360.00.\n"
    )
    assert best_value(text, "change_amount") == Decimal("48920.00")
    assert best_value(text, "original_contract_sum") == Decimal("14250000.00")
    assert best_value(text, "contract_sum_after") == Decimal("14685360.00")

    labelled = "Amount of this Change Order: $48,920.00\n"
    narrative_score = max(c.score for c in candidates_for(text, "change_amount"))
    labelled_score = max(c.score for c in candidates_for(labelled, "change_amount"))
    assert narrative_score < labelled_score


TABLE_WITH_HEADER = """ITEMIZED BREAKDOWN

  Item  Description                       Qty   Unit   Unit Price      Amount
  1     4 inch ductile iron fire main     260    LF        68.50    17,810.00
  2     Post indicator valve assembly       2    EA     3,240.00     6,480.00
  3     Trench excavation and backfill    260    LF        22.00     5,720.00
        Subtotal                                                   30,010.00
        Overhead and profit at 10 percent                           3,001.00
        Total this Change Order                                    33,011.00
"""


def test_table_with_header_maps_every_column():
    extractor = TableExtractor()
    doc = load_text(TABLE_WITH_HEADER)
    extractor.extract(doc)
    items = extractor.line_items

    assert len(items) == 4  # three detail rows plus the markup row
    first = items[0]
    assert first.description == "4 inch ductile iron fire main"
    assert first.quantity == Decimal("260.00")
    assert first.unit == "LF"
    assert first.unit_price == Decimal("68.50")
    assert first.amount == Decimal("17810.00")


def test_markup_row_is_kept_so_the_total_reconciles():
    extractor = TableExtractor()
    extractor.extract(load_text(TABLE_WITH_HEADER))
    total = sum(i.amount for i in extractor.line_items if i.amount)
    assert total == Decimal("33011.00")


def test_total_rows_become_change_amount_candidates_not_line_items():
    extractor = TableExtractor()
    candidates = extractor.extract(load_text(TABLE_WITH_HEADER))
    descriptions = {i.description.lower() for i in extractor.line_items}
    assert "subtotal" not in descriptions
    assert not any(d.startswith("total this") for d in descriptions)
    assert any(c.field_name == "change_amount" and c.value == Decimal("33011.00") for c in candidates)


def test_description_column_is_not_lost_to_the_item_column():
    """Regression: "Item" matched the description cue list by substring."""
    extractor = TableExtractor()
    extractor.extract(load_text(TABLE_WITH_HEADER))
    assert all(i.description for i in extractor.line_items)


def test_unheaded_breakdown_is_found_by_consecutive_rows():
    text = """BREAKDOWN
  Metal deck replacement, 20 ga      1,240 SF     14.75     18,290.00
  Welding and fastening                  1 LS   3,620.00     3,620.00
  Primer and base sheet over repair  1,240 SF       2.95      3,658.00
"""
    extractor = TableExtractor()
    extractor.extract(load_text(text))
    assert len(extractor.line_items) == 3
    assert sum(i.amount for i in extractor.line_items if i.amount) == Decimal("25568.00")


def test_contract_summary_block_is_not_read_as_line_items():
    text = """CONTRACT SUMMARY
Original Contract Sum                             $1,845,000.00
Net change by previously authorized change orders    $42,100.00
Contract Sum prior to this Change Order           $1,887,100.00
New Contract Sum                                  $1,915,668.00
"""
    extractor = TableExtractor()
    extractor.extract(load_text(text))
    assert extractor.line_items == []


def test_change_order_log_is_not_read_as_line_items():
    text = """CHANGE ORDER LOG FOR REFERENCE ONLY
  CO No.  Date        Description                      Amount        Status
  011     2024-06-14  Stair pressurization revision     68,400.00    Approved
  012     2024-07-02  Added bike room enclosure         31,255.00    Approved
"""
    extractor = TableExtractor()
    extractor.extract(load_text(text))
    assert extractor.line_items == []


def test_bracketed_line_items_stay_negative():
    text = """CREDIT BREAKDOWN
  Deduct terrazzo flooring, material and labor    4,100 SF    13.40    (54,940.00)
  Deduct terrazzo base and transitions              320 LF    11.00     (3,520.00)
"""
    extractor = TableExtractor()
    extractor.extract(load_text(text))
    assert sum(i.amount for i in extractor.line_items if i.amount) == Decimal("-58460.00")


def test_every_registry_pattern_compiles():
    for spec in FIELD_SPECS:
        assert compiled_label_patterns(spec) is not None
        assert compiled_extra_patterns(spec) is not None


def test_patterns_do_not_blow_up_on_a_long_document():
    """Regression guard for catastrophic backtracking.

    Two patterns that opened with the money group once took forty seconds
    across twelve short samples. A loose upper bound here is enough to catch
    that class of mistake before it reaches a batch of real documents.
    """
    text = (TABLE_WITH_HEADER + "The Contract Sum will be increased by $12,480.00.\n") * 40
    doc = load_text(text)
    started = time.perf_counter()
    RuleExtractor().extract(doc)
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"rule extraction took {elapsed:.1f}s on a {len(text)} character document"
