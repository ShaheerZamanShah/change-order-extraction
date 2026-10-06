"""Field registry: what we extract, how we recognise it, how much we trust it.

The registry is the single place that defines the extraction target. Rule
extraction, the LLM prompt, validation and the eval harness all read from here,
so adding a field means editing one file.

Each field declares *label aliases* with a strength between 0 and 1. The
strength is a prior on how much a match on that label should be trusted before
any other signal is considered. A very specific label such as
"net change by previously authorized change orders" earns a high prior. A bare
"amount" earns a low one because it shows up all over a change order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import ChangeOrderStatus, ReasonCode

# ---------------------------------------------------------------------------
# Value level regex building blocks
# ---------------------------------------------------------------------------

# US grouping (1,234.56), plain (1234.56) and EU grouping (1.234,56).
_NUM_US = r"\d{1,3}(?:[,\u00a0\u202f ]\d{3})+(?:\.\d{1,2})?"
_NUM_EU = r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?"
_NUM_PLAIN = r"\d+(?:[.,]\d{1,2})?"
NUMBER = rf"(?:{_NUM_EU}|{_NUM_US}|{_NUM_PLAIN})"

# "S" is included as a currency symbol because OCR reads "$" as "S" constantly.
CURRENCY_SYMBOL = r"(?:USD|US\$|\$|S(?=[^\S\n]?\d)|€|£)"

# The leading lookahead is a cheap gate. Without it the engine tries every
# permutation of the optional bracket, sign, symbol and spacing at every
# position in the document before discovering there is no digit nearby, which
# is slow enough to matter on a long multi page change order.
MONEY = (
    rf"(?=[(\s$€£A-Z+\-]{{0,8}}\d)"
    rf"\(?[^\S\n]*[-+]?[^\S\n]*{CURRENCY_SYMBOL}?[^\S\n]*[-+]?[^\S\n]*{NUMBER}"
    rf"[^\S\n]*\)?(?:[^\S\n]*(?:USD|EUR|GBP|CAD))?"
)

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE = (
    r"(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # 2024-03-14
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"  # 03/14/2024 or 14.03.2024
    r"|\d{1,2}[-\s]" + _MONTHS + r"[-\s,]*\d{2,4}"  # 14-Mar-2024
    r"|" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?[,\s]+\d{2,4}"  # March 14, 2024
    r")"
)

DAYS = r"\d{1,4}"
IDENT = r"[A-Za-z0-9][A-Za-z0-9\-_./#]{0,30}"
LINE_TAIL = r"[^\n\r]{1,160}"
# A paragraph: this line plus following lines up to the next blank line.
BLOCK = r"[^\n]{1,200}(?:\n(?![^\S\n]*\n)[^\n]{1,200}){0,15}"

# Separator between a label and its value. Change orders use colons, dashes,
# dot leaders, long runs of spaces (column layout) and nothing at all.
# Horizontal whitespace only, so a same-line pattern can never reach down and
# grab the number from the line below.
# The punctuation set includes the pipe, because pipe delimited exports are
# everywhere, and the comma, because "Sent: Tuesday, 11 June 2024" is a date.
_H = r"[^\S\n]"
SEP = rf"(?:{_H}*[:,|\u2013\u2014\-=]{_H}*|{_H}*\.{{2,}}{_H}*|{_H}{{2,}}|{_H}*)"

# Connective words that sit between a caption and its number. Real forms are
# full of them: "The original Contract Sum was", "the new Contract Sum
# including this Change Order will be". The list is a closed vocabulary with
# no digits in it, so letting the matcher skip a few cannot pull in a number
# from somewhere else on the line.
_FILLER_WORDS = (
    r"was|were|is|are|shall|will|would|to|be|been|of|at|on|for|by|in|the|a|an|this|that|"
    r"and|as|per|net|sum|total|amount|amounts|change|order|orders|including|included|"
    r"increased|decreased|adjusted|revised|equals|equal|hereby|now|then|date|dated|day|"
    r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
)
FILLER = rf"(?:{_H}+(?:{_FILLER_WORDS})(?![A-Za-z])){{0,16}}"

# Guard placed immediately after a label. The first half stops "re" matching
# the start of "reason"; the second stops the generic "project" label from
# swallowing the value that belongs to "project no.".
LABEL_GUARD = rf"(?![A-Za-z])(?!{_H}*(?:nos?|number|num|#|name|title|id|code|type)(?![A-Za-z]))"

NO_CHANGE = re.compile(
    r"\b(?:no|zero|nil|n/?a|none)\b[\s\w]{0,20}?\b(?:change|cost|impact|adjustment|extension|delay)\b"
    r"|\b(?:change|cost|impact|adjustment|extension)\b\s*[:\-]?\s*\b(?:none|n/?a|nil|zero)\b"
    r"|\bno\s+(?:cost|time|schedule|price)\b",
    re.IGNORECASE,
)

DECREASE_CUES = re.compile(
    r"\b(?:decreas\w*|deduct\w*|credit\w*|reduc\w*|less|back\s?charge|descoped?)\b",
    re.IGNORECASE,
)
INCREASE_CUES = re.compile(r"\b(?:increas\w*|add(?:ed|ition(?:al)?)?|extra)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Field specs
# ---------------------------------------------------------------------------

Kind = str  # one of: money, date, days, string, text, enum, currency, percent


@dataclass(frozen=True)
class ExtraPattern:
    """A hand written pattern for phrasing that labels do not cover."""

    name: str
    regex: str
    strength: float
    sign: Optional[str] = None  # "negative" forces a deduct reading


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: Kind
    aliases: tuple[tuple[str, float], ...] = ()
    extra_patterns: tuple[ExtraPattern, ...] = ()
    enum_cues: tuple[tuple[str, str, float], ...] = ()  # (regex, enum value, strength)
    # Contexts in which an enum cue should be ignored. "Date Executed" is a
    # signature block caption, not a statement that the status is executed.
    cue_exclusions: tuple[str, ...] = ()
    required: bool = False
    importance: float = 1.0
    zero_on_no_change: bool = False
    max_chars: int = 200
    description: str = ""
    negative_lookbehind: tuple[str, ...] = field(default_factory=tuple)


def _alias_group(aliases: tuple[tuple[str, float], ...]) -> list[tuple[str, float]]:
    return sorted(aliases, key=lambda a: (-a[1], -len(a[0])))


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="change_order_number",
        kind="string",
        required=True,
        importance=2.0,
        max_chars=32,
        description="Identifier of this change order, for example CO-014 or 7.",
        aliases=(
            ("change order (?:no|number|#|num)\\.?", 0.96),
            ("change order", 0.72),
            ("c\\.?o\\.? (?:no|number|#)\\.?", 0.9),
            ("co(?:\\s|_|-)?(?:no|number|#)\\.?", 0.88),
            ("pco (?:no|number|#)\\.?", 0.8),
            ("amendment (?:no|number|#)\\.?", 0.78),
            ("field order (?:no|number|#)\\.?", 0.74),
            # Contractors each have their own name for the same document.
            ("cca (?:no|number|#)\\.?", 0.84),
            ("ccd (?:no|number|#)\\.?", 0.82),
            ("cor (?:no|number|#)\\.?", 0.78),
            ("(?:contract )?change (?:authorization|authorisation|directive) (?:no|number|#)\\.?", 0.86),
            ("change request (?:no|number|#)\\.?", 0.78),
        ),
        extra_patterns=(
            # Capture the number itself, not the prefix. "CO 12" and "CO-12"
            # and "12" are the same change order, and a downstream system
            # matching on the identifier should not see three of them.
            ExtraPattern(
                "co_token",
                r"\b(?:CO|PCO|FO|CCD|CCA)[\s\-_#]?(?P<value>\d{1,4}[A-Za-z]?)\b",
                0.7,
            ),
            ExtraPattern(
                "change_order_hash",
                r"change\s*order\s*#\s*(?P<value>\d{1,4}[A-Za-z]?)",
                0.9,
            ),
        ),
    ),
    FieldSpec(
        name="change_order_title",
        kind="text",
        importance=0.6,
        max_chars=120,
        description="Short title or subject line of the change.",
        aliases=(
            ("(?:change order )?title", 0.88),
            ("subject", 0.84),
            ("re", 0.55),
        ),
    ),
    FieldSpec(
        name="project_name",
        kind="text",
        required=True,
        importance=1.5,
        max_chars=120,
        description="Name of the project the change order belongs to.",
        aliases=(
            ("project (?:name|title)", 0.95),
            ("project", 0.8),
            ("job (?:name|title)", 0.85),
            ("the project", 0.7),
            ("job", 0.7),
        ),
    ),
    FieldSpec(
        name="project_number",
        kind="string",
        importance=1.0,
        max_chars=40,
        description="Owner or contractor project/job number.",
        aliases=(
            ("project (?:no|number|#)\\.?", 0.95),
            ("job (?:no|number|#)\\.?", 0.92),
            ("project id", 0.88),
            ("wbs", 0.6),
        ),
    ),
    FieldSpec(
        name="contract_number",
        kind="string",
        importance=0.9,
        max_chars=40,
        aliases=(
            ("contract (?:no|number|#)\\.?", 0.95),
            ("agreement (?:no|number|#)\\.?", 0.88),
            ("purchase order (?:no|number|#)\\.?", 0.7),
            ("po (?:no|number|#)\\.?", 0.65),
        ),
        extra_patterns=(
            # "contract 2023-NG-17" with no caption word. The value has to look
            # like a reference code, otherwise a bare "contract" label would
            # happily capture the "Sum" out of "Contract Sum".
            ExtraPattern(
                "bare_contract_code",
                r"\bcontract\s+(?P<value>[A-Z0-9][A-Z0-9]*[-/][A-Z0-9\-/]*\d[A-Z0-9\-/]*)\b",
                0.62,
            ),
        ),
    ),
    FieldSpec(
        name="owner_name",
        kind="text",
        importance=1.2,
        max_chars=120,
        aliases=(
            ("owner", 0.92),
            ("to owner", 0.9),
            ("client", 0.85),
            ("employer", 0.8),
            ("purchaser", 0.7),
        ),
    ),
    FieldSpec(
        name="contractor_name",
        kind="text",
        importance=1.2,
        max_chars=120,
        aliases=(
            ("(?:general )?contractor", 0.92),
            ("to contractor", 0.9),
            ("gc", 0.65),
            ("vendor", 0.7),
            ("subcontractor", 0.72),
            ("supplier", 0.6),
        ),
    ),
    FieldSpec(
        name="architect_name",
        kind="text",
        importance=0.8,
        max_chars=120,
        aliases=(
            ("architect(?:/engineer)?", 0.92),
            ("a/?e", 0.7),
            ("engineer(?: of record)?", 0.82),
            ("design(?:er| professional)", 0.72),
        ),
    ),
    FieldSpec(
        name="date_issued",
        kind="date",
        required=True,
        importance=1.5,
        aliases=(
            ("date of issuance", 0.96),
            ("(?:date )?issued(?: on| date)?", 0.92),
            ("change order date", 0.9),
            ("dated", 0.78),
            ("date sent", 0.72),
            ("sent", 0.58),
            ("date", 0.6),
        ),
        extra_patterns=(
            ExtraPattern(
                "issued_phrase",
                r"issued\s+(?:on|this)\s+(?P<value>" + DATE + r")",
                0.9,
            ),
        ),
    ),
    FieldSpec(
        name="date_executed",
        kind="date",
        importance=1.0,
        aliases=(
            ("date (?:executed|of execution)", 0.95),
            ("executed(?: on| date)?", 0.9),
            ("(?:date )?approved(?: on| date)?", 0.85),
            ("signed(?: on| date)?", 0.82),
            ("effective date", 0.75),
        ),
        extra_patterns=(
            # Signature blocks rarely carry a caption, just a name and a date.
            ExtraPattern(
                "signature_block_date",
                r"(?:approved|signed|executed|accepted)\s+by[^\n]{0,70}?"
                r"(?:\bon\b\s*)?(?P<value>" + DATE + r")",
                0.78,
            ),
        ),
    ),
    FieldSpec(
        name="status",
        kind="enum",
        importance=0.9,
        enum_cues=(
            (r"\bfully\s+executed\b|\bexecuted\s+copy\b|\bexecuted\b", ChangeOrderStatus.EXECUTED.value, 0.85),
            (r"\bapproved\b(?!\s*by\s*:?\s*$)|\bapproval\s+granted\b", ChangeOrderStatus.APPROVED.value, 0.8),
            (r"\bpending\b|\bawaiting\s+(?:approval|signature)\b|\bfor\s+review\b|\bsubmitted\s+for\b", ChangeOrderStatus.PENDING.value, 0.78),
            (r"\bdraft\b|\bpreliminary\b|\bproposed\s+change\b|\bnot\s+for\s+construction\b", ChangeOrderStatus.DRAFT.value, 0.72),
            (r"\breject(?:ed|ion)\b|\bdenied\b|\bdisapproved\b", ChangeOrderStatus.REJECTED.value, 0.85),
            (r"\bvoid(?:ed)?\b|\bcancell?ed\b|\bwithdrawn\b", ChangeOrderStatus.VOID.value, 0.85),
        ),
        cue_exclusions=(
            r"date\s+(?:of\s+)?(?:execut|approv|sign)",
            r"(?:execut|approv|sign)\w*\s*(?:date\b|on\s+\d)",
            # "eleven prior change orders approved" describes other change
            # orders, not the status of this one.
            r"change\s+orders?\s+(?:approved|executed|authori[sz]ed)",
            r"_{4,}",
        ),
        aliases=(("status", 0.9), ("co status", 0.9), ("state", 0.6)),
    ),
    FieldSpec(
        name="reason_code",
        kind="enum",
        importance=0.8,
        enum_cues=(
            (r"\bunforeseen\b|\bdiffering site condition\b|\bconcealed condition\b|\bhidden\b.{0,20}\bcondition\b|\brock\b|\bgroundwater\b|\bundocumented\b", ReasonCode.UNFORESEEN_CONDITION.value, 0.8),
            # The same thing said without the term of art, which is how site
            # staff actually write it up.
            (
                r"\bnot\s+(?:known|shown|identified|indicated|disclosed)\b[^.]{0,50}\bbid\b"
                r"|\bdiscovered\s+(?:during|after)\s+(?:demolition|excavation|tear\s?[- ]?off|construction|removal|trenching)",
                ReasonCode.UNFORESEEN_CONDITION.value,
                0.72,
            ),
            (r"\bdesign error\b|\bomission\b|\berror(?:s)? (?:and|or|/) omission\b|\bcoordination error\b|\bRFI\b.{0,30}\berror\b", ReasonCode.DESIGN_ERROR_OMISSION.value, 0.8),
            (r"\bowner (?:request|requested|directive|direction|elected)\b|\bat the owner'?s? request\b|\bclient requested\b", ReasonCode.OWNER_REQUESTED.value, 0.82),
            (r"\bcode (?:official|requirement|compliance)\b|\bbuilding department\b|\bauthority having jurisdiction\b|\bAHJ\b|\bfire marshal\b|\binspector required\b", ReasonCode.CODE_OR_AUTHORITY.value, 0.8),
            (r"\bweather\b|\brain delay\b|\bforce majeure\b|\bhurricane\b|\bsnow\b", ReasonCode.WEATHER_DELAY.value, 0.78),
            (r"\ballowance (?:adjustment|reconcil\w+|true[- ]?up)\b|\breconcile.{0,20}allowance\b", ReasonCode.ALLOWANCE_ADJUSTMENT.value, 0.8),
            (r"\bdescope\w*\b|\bscope (?:deduction|removal)\b|\bdeductive change\b|\bcredit to owner\b|\bvalue engineer\w*\b", ReasonCode.SCOPE_DEDUCTION.value, 0.75),
            (r"\bdesign (?:change|revision|development)\b|\brevised drawings?\b|\bbulletin\b|\bASI\b|\bsketch\b", ReasonCode.DESIGN_CHANGE.value, 0.7),
        ),
        aliases=(
            ("reason(?: for change| code)?", 0.9),
            ("cause", 0.8),
            ("justification", 0.8),
            ("basis of change", 0.8),
        ),
    ),
    FieldSpec(
        name="scope_description",
        kind="block",
        required=True,
        importance=1.3,
        max_chars=2000,
        description="Narrative of the work added, removed or modified.",
        aliases=(
            ("description of (?:the )?(?:change|work)", 0.95),
            ("scope of (?:the )?(?:change|work)", 0.95),
            ("(?:the )?change (?:consists of|is as follows)", 0.9),
            ("work included", 0.85),
            ("description", 0.7),
            ("scope", 0.7),
            ("details", 0.55),
            ("remarks", 0.55),
            ("narrative", 0.6),
        ),
    ),
    FieldSpec(
        name="currency",
        kind="currency",
        importance=0.5,
        aliases=(("currency", 0.9),),
    ),
    FieldSpec(
        name="original_contract_sum",
        kind="money",
        importance=1.4,
        aliases=(
            ("(?:the )?original contract (?:sum|amount|price|value)", 0.97),
            ("base contract (?:sum|amount|price|value)", 0.92),
            ("original (?:gmp|guaranteed maximum price)", 0.9),
            ("contract sum as awarded", 0.9),
            ("awarded (?:sum|amount|value)", 0.82),
            ("original (?:sum|amount)", 0.75),
        ),
        extra_patterns=(
            # Narrative phrasing from letters and email threads. It lives in
            # the phrase list rather than the alias list because it appears
            # mid sentence, where a caption never does. Weaker on purpose:
            # these earn their confidence from the arithmetic checks rather
            # than from the strength of the cue.
            ExtraPattern(
                "narrative_started_at",
                r"contract\s+(?:started|commenced|began|was\s+awarded)\s+at\s*(?P<value>" + MONEY + r")",
                0.74,
            ),
        ),
    ),
    FieldSpec(
        name="net_previous_changes",
        kind="money",
        importance=1.2,
        zero_on_no_change=True,
        aliases=(
            ("net change by previously authori[sz]ed change orders", 0.98),
            ("net (?:change|changes) by previous change orders", 0.97),
            ("(?:sum|total) of (?:all )?previous(?:ly approved)? change orders", 0.95),
            ("previous(?:ly)? (?:approved |authori[sz]ed )?change orders?", 0.88),
            ("prior change orders?", 0.88),
            ("net previous changes", 0.9),
            ("approved changes to date", 0.85),
        ),
        extra_patterns=(
            ExtraPattern(
                "prior_cos_totaling",
                r"(?:prior|previous|earlier)\s+change\s+orders?[^.\n]{0,50}?"
                r"(?:totaling|totalling|totals|worth|of)\s*(?P<value>" + MONEY + r")",
                0.82,
            ),
        ),
    ),
    FieldSpec(
        name="contract_sum_before",
        kind="money",
        importance=1.1,
        aliases=(
            ("(?:the )?contract (?:sum|amount|price) prior to this change order", 0.98),
            ("contract (?:sum|amount) before this change", 0.96),
            ("(?:the )?contract sum prior to this change", 0.96),
            ("adjusted contract (?:sum|amount) to date", 0.88),
            ("current contract (?:sum|amount|value)", 0.84),
            ("contract sum to date", 0.84),
        ),
        extra_patterns=(
            ExtraPattern(
                "narrative_before_this",
                r"\b(?:at|to|of|is|was|were)\s+(?P<value>" + MONEY + r")"
                r"[^.\n]{0,12}?before\s+this\s+(?:one|change)",
                0.76,
            ),
        ),
    ),
    FieldSpec(
        name="change_amount",
        kind="money",
        required=True,
        importance=2.5,
        zero_on_no_change=True,
        aliases=(
            ("(?:net )?(?:cost|amount|value|total) of this change order", 0.97),
            ("this change order (?:amount|total|sum)", 0.96),
            ("(?:total |net )?change order (?:amount|value|total)", 0.92),
            ("amount of this change", 0.95),
            ("net change (?:amount|to contract sum)", 0.9),
            ("total this change order", 0.95),
            ("change order total", 0.92),
            ("co (?:amount|total|value)", 0.85),
            ("total adjustment", 0.82),
            ("net cost", 0.8),
            ("(?:grand )?total", 0.5),
            ("amount", 0.42),
        ),
        extra_patterns=(
            ExtraPattern(
                "will_be_increased",
                r"contract\s+(?:sum|price|amount)\s+will\s+be\s+(?:increased|adjusted upward)\s+by\s*(?P<value>" + MONEY + r")",
                0.96,
            ),
            ExtraPattern(
                "will_be_decreased",
                r"contract\s+(?:sum|price|amount)\s+will\s+be\s+(?:decreased|reduced|adjusted downward)\s+by\s*(?P<value>" + MONEY + r")",
                0.96,
                sign="negative",
            ),
            ExtraPattern(
                "add_deduct_line",
                r"\b(?:add|addition)\b\s*[:\-]?\s*(?P<value>" + MONEY + r")",
                0.6,
            ),
            ExtraPattern(
                "deduct_line",
                r"\b(?:deduct|credit|less)\b\s*[:\-]?\s*(?P<value>" + MONEY + r")",
                0.65,
                sign="negative",
            ),
            ExtraPattern(
                "in_the_amount_of",
                r"in the (?:lump sum )?amount of\s*(?P<value>" + MONEY + r")",
                0.8,
            ),
            ExtraPattern(
                "narrative_priced_at",
                r"(?:pric(?:ing|e|ed)|quot(?:e|ed|ation)|proposal|cost)\s+"
                r"(?:came\s+(?:back\s+)?(?:in\s+)?at|came\s+to|is|comes\s+to|at|of)\s*"
                r"(?P<value>" + MONEY + r")",
                0.7,
            ),
            # Anchored on a leading preposition rather than on the amount.
            # A pattern that opens with the money group has to be attempted at
            # every character in the document; this one does not.
            ExtraPattern(
                "narrative_all_in",
                r"\b(?:at|of|for|is|was)\s+(?P<value>" + MONEY + r")\s*,?\s*all\s+in\b",
                0.68,
            ),
        ),
    ),
    FieldSpec(
        name="contract_sum_after",
        kind="money",
        importance=1.4,
        aliases=(
            ("(?:the )?new contract (?:sum|amount|price|value)", 0.97),
            ("revised contract (?:sum|amount|price|value)", 0.96),
            ("contract (?:sum|amount) including this change order", 0.97),
            ("(?:the )?contract sum (?:will be|as) (?:revised|adjusted) to", 0.9),
            ("adjusted contract (?:sum|amount|price)", 0.88),
            ("total contract (?:sum|amount|value) after this change", 0.95),
            ("new (?:gmp|guaranteed maximum price)", 0.9),
            ("final contract (?:sum|amount)", 0.82),
            ("revised (?:sum|total)", 0.74),
        ),
        extra_patterns=(
            ExtraPattern(
                "narrative_goes_to",
                r"contract\s+(?:goes|rises|increases|moves|changes)\s+(?:up\s+)?to\s*"
                r"(?P<value>" + MONEY + r")",
                0.76,
            ),
        ),
    ),
    FieldSpec(
        name="retainage_percent",
        kind="percent",
        importance=0.4,
        aliases=(("retainage", 0.9), ("retention", 0.85)),
    ),
    FieldSpec(
        name="schedule_impact_days",
        kind="days",
        importance=1.3,
        zero_on_no_change=True,
        aliases=(
            ("(?:the )?contract time will be (?:increased|extended) by", 0.97),
            ("contract time (?:change|adjustment|impact)", 0.92),
            ("(?:schedule|time) (?:impact|extension|adjustment)", 0.94),
            ("(?:additional|added) (?:calendar |working |work )?days", 0.92),
            ("days? (?:added|extension|granted)", 0.88),
            ("extension of time", 0.9),
            ("time (?:added|granted)", 0.84),
        ),
        extra_patterns=(
            ExtraPattern(
                "extended_by_days",
                r"(?:contract time|schedule|completion)\D{0,40}?(?:increased|extended|delayed)\s+by\s+(?P<value>\d{1,4})\s*(?:calendar|working|business|work)?\s*days?",
                0.96,
            ),
            ExtraPattern(
                "reduced_by_days",
                r"(?:contract time|schedule)\D{0,40}?(?:decreased|reduced|accelerated)\s+by\s+(?P<value>\d{1,4})\s*(?:calendar|working|business|work)?\s*days?",
                0.94,
                sign="negative",
            ),
            ExtraPattern(
                "bare_days",
                r"(?P<value>\d{1,4})\s*(?:additional\s+)?(?:calendar|working|business|work)\s*days?",
                0.6,
            ),
        ),
    ),
    FieldSpec(
        name="original_completion_date",
        kind="date",
        importance=0.8,
        aliases=(
            ("original (?:date of )?substantial completion", 0.96),
            ("original completion date", 0.96),
            ("(?:current|existing) substantial completion", 0.88),
            ("contract completion date", 0.88),
            ("original (?:end|finish) date", 0.85),
        ),
        extra_patterns=(
            ExtraPattern(
                "completion_moves_from",
                r"completion[^.\n]{0,30}?(?:moves|changes|shifts|slips|goes|is revised)\s+from\s+"
                r"(?P<value>" + DATE + r")",
                0.8,
            ),
        ),
    ),
    FieldSpec(
        name="revised_completion_date",
        kind="date",
        importance=0.9,
        aliases=(
            ("(?:new|revised) (?:date of )?substantial completion", 0.96),
            ("revised completion date", 0.96),
            ("new completion date", 0.96),
            ("(?:substantial completion|completion) (?:will be|is) (?:revised|adjusted|extended) to", 0.92),
            ("revised (?:end|finish) date", 0.85),
        ),
        extra_patterns=(
            ExtraPattern(
                "completion_moves_to",
                r"completion[^.\n]{0,30}?from\s+" + DATE + r"\s+to\s+(?P<value>" + DATE + r")",
                0.8,
            ),
        ),
    ),
)

FIELD_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}

MONEY_FIELDS: tuple[str, ...] = tuple(s.name for s in FIELD_SPECS if s.kind == "money")
DATE_FIELDS: tuple[str, ...] = tuple(s.name for s in FIELD_SPECS if s.kind == "date")
REQUIRED_FIELDS: tuple[str, ...] = tuple(s.name for s in FIELD_SPECS if s.required)


def value_regex(kind: Kind) -> str:
    """Regex fragment (named group ``value``) for a field kind."""
    if kind == "money":
        return rf"(?P<value>{MONEY})"
    if kind == "date":
        return rf"(?P<value>{DATE})"
    if kind == "days":
        return rf"(?P<value>[-+]?{DAYS})[^\S\n]*(?:calendar|working|business|work)?[^\S\n]*(?:days?|d\b)?"
    if kind == "percent":
        return r"(?P<value>\d{1,3}(?:\.\d{1,2})?)[^\S\n]*%?"
    if kind == "currency":
        return r"(?P<value>USD|EUR|GBP|CAD|AUD|\$|€|£)"
    if kind == "string":
        return rf"(?P<value>{IDENT})"
    if kind == "block":
        return rf"(?P<value>{BLOCK})"
    return rf"(?P<value>{LINE_TAIL})"


def compiled_label_patterns(spec: FieldSpec) -> list[tuple[str, re.Pattern[str], float]]:
    """Build label anchored patterns for one field.

    Two shapes are produced per alias: the value on the same line as the label,
    and the value on the line immediately below it. Form layouts put the value
    under the caption often enough that the second shape pays for itself.

    Enum fields get no label patterns at all. Their value is a classification,
    not a span of text, so they are handled by the cue list instead and the
    labels only serve to boost a nearby cue.
    """
    if spec.kind == "enum":
        return []

    out: list[tuple[str, re.Pattern[str], float]] = []
    val = value_regex(spec.kind)
    for alias, strength in _alias_group(spec.aliases):
        same_line = (
            # A label sits at the start of a line, after a pipe, after a run of
            # spaces that marks a column boundary, or at the start of a
            # sentence or clause. The last two cases are what make narrative
            # documents readable: "Project is Northgate, job number NLH-0442".
            rf"(?:(?:^|\n){_H}{{0,12}}(?:\d{{1,2}}[.)]{_H}*)?|(?<={_H}{_H})|(?<=\|)"
            rf"|(?<=[.;,]{_H}))"
            rf"(?P<label>{alias}){LABEL_GUARD}{_H}*[:\u2013\u2014\-]?"
            rf"{FILLER}{SEP}{val}"
        )
        out.append((f"label:{alias}", re.compile(same_line, re.IGNORECASE), strength))
        next_line = (
            rf"(?:^|\n){_H}{{0,12}}(?P<label>{alias}){LABEL_GUARD}{_H}*:?{_H}*\n{_H}*{val}"
        )
        out.append((f"label_below:{alias}", re.compile(next_line, re.IGNORECASE), strength * 0.82))
    return out


def compiled_extra_patterns(spec: FieldSpec) -> list[tuple[str, re.Pattern[str], float, Optional[str]]]:
    return [
        (f"phrase:{p.name}", re.compile(p.regex, re.IGNORECASE), p.strength, p.sign)
        for p in spec.extra_patterns
    ]
