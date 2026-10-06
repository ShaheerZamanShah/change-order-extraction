"""Value parser tests.

Most extraction bugs are really parsing bugs, so these are the cheapest tests
in the project and the ones worth having most.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from coextract.normalize.parse import (
    clean_string_value,
    detect_currency,
    parse_currency,
    parse_date_value,
    parse_days,
    parse_money,
    parse_percent,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$1,234.56", Decimal("1234.56")),
        ("1,234.56", Decimal("1234.56")),
        ("USD 8,742,000.00", Decimal("8742000.00")),
        ("8742000", Decimal("8742000.00")),
        ("$ 35,667.00", Decimal("35667.00")),
        ("0.00", Decimal("0.00")),
        ("$0", Decimal("0.00")),
        # Brackets and trailing minus are both deductions.
        ("(62,480.00)", Decimal("-62480.00")),
        ("-62,480.00", Decimal("-62480.00")),
        ("62,480.00-", Decimal("-62480.00")),
        ("($2,500.00)", Decimal("-2500.00")),
        # European grouping.
        ("4.850.000,00", Decimal("4850000.00")),
        ("231.875,50", Decimal("231875.50")),
        ("96.400,00", Decimal("96400.00")),
        ("218,25", Decimal("218.25")),
    ],
)
def test_parse_money(raw, expected):
    value, _ = parse_money(raw)
    assert value == expected


def test_parse_money_rejects_junk():
    for raw in ("", "   ", "n/a", "see attached", "$"):
        value, notes = parse_money(raw)
        assert value is None, raw
        assert notes


def test_parse_money_flags_ambiguous_separator():
    """1.500 is either 1500 or 1.5 and we must say so rather than pick quietly."""
    value, notes = parse_money("1.500")
    assert value == Decimal("1500.00")
    assert "ambiguous_decimal_separator" in notes


def test_parse_money_rejects_implausible_magnitude():
    value, notes = parse_money("99,999,999,999,999.00")
    assert value is None
    assert "implausible_magnitude" in notes


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-03-14", date(2024, 3, 14)),
        ("March 14, 2024", date(2024, 3, 14)),
        ("Mar 14 2024", date(2024, 3, 14)),
        ("14 March 2024", date(2024, 3, 14)),
        ("08 Aug 2024", date(2024, 8, 8)),
        ("06/18/2024", date(2024, 6, 18)),
        ("2024/06/18", date(2024, 6, 18)),
    ],
)
def test_parse_date(raw, expected):
    value, _ = parse_date_value(raw)
    assert value == expected


def test_parse_date_infers_dayfirst_when_us_order_is_impossible():
    value, notes = parse_date_value("14.03.2024")
    assert value == date(2024, 3, 14)
    assert "dayfirst_inferred" in notes


def test_parse_date_flags_genuinely_ambiguous_order():
    value, notes = parse_date_value("03/04/2024")
    assert value == date(2024, 3, 4)
    assert "ambiguous_day_month_order" in notes


def test_parse_date_rejects_implausible_year():
    value, notes = parse_date_value("01/01/1835")
    assert value is None
    assert "implausible_year" in notes


@pytest.mark.parametrize(
    "raw,expected",
    [("11 calendar days", 11), ("0", 0), ("14", 14), ("-3 days", -3), ("7 working days", 7)],
)
def test_parse_days(raw, expected):
    value, _ = parse_days(raw)
    assert value == expected


def test_parse_days_rejects_absurd_duration():
    value, notes = parse_days("9999 days")
    assert value is None
    assert "implausible_duration" in notes


def test_parse_percent_range():
    assert parse_percent("10")[0] == Decimal("10.00")
    assert parse_percent("5.5%")[0] == Decimal("5.50")
    assert parse_percent("250")[0] is None


def test_parse_currency():
    assert parse_currency("$")[0] == "USD"
    assert parse_currency("EUR")[0] == "EUR"
    assert parse_currency("zz")[0] is None


def test_detect_currency_prefers_the_dominant_symbol():
    code, confidence = detect_currency("Amount $1,000.00 and $2,000.00 USD")
    assert code == "USD"
    assert confidence > 0.5
    code, _ = detect_currency("Betrag 4.850.000,00 EUR")
    assert code == "EUR"


def test_clean_string_value_trims_adjacent_column():
    value, notes = clean_string_value("Riverside Elementary    Project No.: RES-114")
    assert value == "Riverside Elementary"
    assert "trimmed_adjacent_column" in notes


def test_clean_string_value_rejects_placeholders_and_labels():
    for raw in ("N/A", "TBD", "---", "Project", "Owner:", "no.", "number:"):
        value, _ = clean_string_value(raw)
        assert value is None, raw


def test_clean_string_value_rejects_label_fragment_captures():
    """A capture starting with "No.:" means we matched the wrong caption."""
    value, notes = clean_string_value("No.: RES-2023-114")
    assert value is None
    assert "captured_label_not_value" in notes


def test_clean_string_value_keeps_paragraph_indentation():
    paragraph = "Furnish and install the main\n   including two valves and testing"
    value, notes = clean_string_value(paragraph, max_chars=2000)
    assert value is not None
    assert "including two valves" in value
    assert "trimmed_adjacent_column" not in notes


def test_clean_string_value_truncates_long_text():
    value, notes = clean_string_value("word " * 200, max_chars=50)
    assert value is not None and len(value) <= 54
    assert "truncated" in notes
