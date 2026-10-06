"""Text quality scoring.

The score answers one question: how much should we trust anything we read out
of this text? It feeds directly into field confidence, and a very low score is
itself a reason to route a document to a human instead of guessing.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_TOKEN_RE = re.compile(r"\S+")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")

# Words that appear in essentially every real change order. Seeing them is a
# strong signal that the text layer is intact and in the right language.
_ANCHOR_WORDS = (
    "contract",
    "change",
    "order",
    "project",
    "owner",
    "contractor",
    "amount",
    "work",
    "date",
    "sum",
    "total",
    "scope",
    "signature",
)


def _garbled_token_share(text: str) -> float:
    words = ["".join(c for c in t if c.isalpha()) for t in _TOKEN_RE.findall(text)]
    words = [w for w in words if len(w) >= 4]
    if not words:
        return 1.0
    bad = 0
    for word in words:
        if not _VOWEL_RE.search(word):
            bad += 1
            continue
        # "CoNtRacT" style case salad is a classic OCR failure.
        case_flips = sum(1 for i in range(1, len(word)) if word[i].isupper() != word[i - 1].isupper())
        if case_flips >= 3:
            bad += 1
    return bad / len(words)


def assess_text_quality(text: str, page_count: int) -> tuple[float, list[str]]:
    notes: list[str] = []
    if not text or not text.strip():
        return 0.0, ["no_text_extracted"]

    chars = len(text)
    per_page = chars / max(page_count, 1)
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    printable_share = printable / chars

    words = _WORD_RE.findall(text)
    alpha_chars = sum(len(w) for w in words)
    alpha_share = alpha_chars / chars

    lowered = text.lower()
    anchors_found = sum(1 for w in _ANCHOR_WORDS if w in lowered)
    anchor_share = anchors_found / len(_ANCHOR_WORDS)

    garbled = _garbled_token_share(text)
    replacement_share = text.count("\ufffd") / chars

    score = 1.0
    if per_page < 200:
        score -= 0.35
        notes.append(f"sparse_text_{per_page:.0f}_chars_per_page")
    elif per_page < 600:
        score -= 0.12
        notes.append("thin_text_layer")

    if printable_share < 0.98:
        score -= 0.2
        notes.append("non_printable_characters")
    if alpha_share < 0.35:
        score -= 0.15
        notes.append("low_alphabetic_density")
    if anchor_share < 0.4:
        score -= 0.25
        notes.append(f"few_expected_keywords_{anchors_found}_of_{len(_ANCHOR_WORDS)}")
    elif anchor_share < 0.7:
        score -= 0.08
        notes.append("some_expected_keywords_missing")
    if garbled > 0.08:
        score -= min(0.3, garbled * 2)
        notes.append(f"garbled_token_share_{garbled:.2f}")
    if replacement_share > 0.001:
        score -= 0.1
        notes.append("unicode_replacement_characters")

    return max(0.0, min(1.0, round(score, 3))), notes
