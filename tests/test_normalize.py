"""Normalisation tests, including regression tests for two real bugs.

The two named regression tests below both come from mistakes this pipeline
actually made: a dollar sign repair that ate the leading digit of an amount,
and a digit inside a caption that silently cost a whole field.
"""

from __future__ import annotations

import pytest

from coextract.ingest.quality import assess_text_quality
from coextract.normalize.text import (
    collapse_spaced_capitals,
    fix_alpha_ocr,
    fix_numeric_ocr,
    normalize_document_text,
    normalize_unicode,
    page_for_offset,
    strip_repeated_furniture,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("l,85O.OO", "1,850.00"),
        ("S2,48O,OOO.OO", "$2,480,000.00"),
        ("SO.OO", "$0.00"),
        ("O3", "03"),
        ("O7/22/2O24", "07/22/2024"),
        ("l2,475.5O", "12,475.50"),
    ],
)
def test_fix_numeric_ocr_repairs_misread_digits(raw, expected):
    assert fix_numeric_ocr(raw)[0] == expected


@pytest.mark.parametrize("raw", ["O'Brien", "Lot I", "Suite O", "LS 1,850.00", "4,100 SF 13.40", "2.5"])
def test_fix_numeric_ocr_leaves_real_text_alone(raw):
    assert fix_numeric_ocr(raw)[0] == raw


def test_fix_numeric_ocr_does_not_eat_a_leading_five():
    """Regression: a leading digit 5 was being rewritten as a dollar sign.

    "5,720.00" became "$,720.00" and then parsed as 720.00, which quietly
    changed a line item by five thousand dollars and then showed up as a line
    item total mismatch somewhere else entirely.
    """
    assert fix_numeric_ocr("5,720.00")[0] == "5,720.00"
    assert fix_numeric_ocr("260 LF 22.00 5,720.00")[0] == "260 LF 22.00 5,720.00"


def test_fix_alpha_ocr_repairs_digits_inside_words():
    """Regression: "Pr0ject Name:" matched no label, losing the project fields."""
    assert fix_alpha_ocr("Pr0ject Name:")[0] == "Project Name:"
    assert fix_alpha_ocr("PR0JECT NO.")[0] == "PROJECT NO."


@pytest.mark.parametrize("raw", ["G701", "Phase 2", "LTM-19-03", "RES-2023-114", "Building C"])
def test_fix_alpha_ocr_leaves_codes_alone(raw):
    assert fix_alpha_ocr(raw)[0] == raw


def test_normalize_unicode_flattens_typography():
    raw = "Contract\u00a0Sum \u2013 \u201cfinal\u201d \ufb01xture"
    cleaned = normalize_unicode(raw)
    assert "\u00a0" not in cleaned
    assert "-" in cleaned
    assert '"final"' in cleaned
    assert "fixture" in cleaned


def test_dehyphenation_joins_words_split_across_lines():
    text, _, _ = normalize_document_text(["replace the bus wash equip-\nment pad today"])
    assert "equipment pad" in text


def test_collapse_spaced_capitals():
    assert collapse_spaced_capitals("C H A N G E  O R D E R").split() == ["CHANGE", "ORDER"]


def test_strip_repeated_furniture_removes_headers_and_page_numbers():
    header = "ACME PARTNERS CONFIDENTIAL CHANGE ORDER 041"
    pages = [f"{header}\nbody line {i}\nPage {i} of 3" for i in range(1, 4)]
    cleaned, removed = strip_repeated_furniture(pages)
    assert header in removed
    for page in cleaned:
        assert header not in page
        assert "Page" not in page
        assert "body line" in page


def test_page_offsets_map_back_to_page_numbers():
    text, offsets, _ = normalize_document_text(["first page text", "second page text", "third"])
    assert len(offsets) == 3
    assert page_for_offset(offsets, 0) == 1
    assert page_for_offset(offsets, offsets[1]) == 2
    assert page_for_offset(offsets, len(text) - 1) == 3


def test_text_quality_scores_a_real_change_order_highly():
    good = (
        "CHANGE ORDER\nProject: Riverside School\nOwner: Marion County\n"
        "Contractor: Hollis Brothers\nThe original contract sum was 8,742,000.00 and the "
        "total scope of work includes a new date for substantial completion. Signature below."
    ) * 3
    score, _ = assess_text_quality(good, page_count=1)
    assert score > 0.8


def test_text_quality_punishes_garbage():
    score, notes = assess_text_quality("xkcd zzz pqrst vwxyz bcdfg hjklm", page_count=1)
    assert score < 0.6
    assert notes

    score, notes = assess_text_quality("", page_count=1)
    assert score == 0.0
    assert "no_text_extracted" in notes


def test_text_quality_flags_a_sparse_text_layer():
    score, notes = assess_text_quality("Change Order", page_count=4)
    assert score < 0.6
    assert any("sparse_text" in n for n in notes)
