"""Text clean up.

Everything downstream works on the *normalised* text, and all evidence offsets
refer to that normalised text. That is a deliberate trade: we lose the ability
to map a span back to an exact byte in the original PDF, but we gain stable
offsets that a reviewer can still use because the normalised text is what we
store alongside the result.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that PDF text layers love to emit and nothing downstream wants.
_CHAR_FIXES = {
    "\u00a0": " ",
    "\u202f": " ",
    "\u2007": " ",
    "\u2009": " ",
    "\u200b": "",
    "\ufeff": "",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2022": "-",
    "\u00b7": "-",
    "\u2039": "<",
    "\u203a": ">",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2044": "/",
    "\ufffd": " ",
}

# Dashes are normalised to a plain hyphen except the en dash used in ranges,
# which we also flatten because change orders use all of them interchangeably.
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"

_LIGATURE_RE = re.compile("|".join(map(re.escape, _CHAR_FIXES)))
_HYPHEN_BREAK_RE = re.compile(r"(\w)[-\u2010\u2011]\s*\n\s*(\w)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[^\S\n]+\n")
_DOT_LEADER_RE = re.compile(r"\.{3,}")
_PAGE_FURNITURE_RE = re.compile(
    r"^\s*(?:page\s+\d+\s*(?:of\s*\d+)?|\d+\s*/\s*\d+|-\s*\d+\s*-)\s*$",
    re.IGNORECASE,
)

# OCR confusions. Only the classic letter/digit shape collisions are included.
# Adding plausible looking extras such as B/8 or Z/2 does more harm than good
# because they fire on real words far more often than on misread digits.
_OCR_DIGIT_MAP = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})

# A token that might be a misread number, with an optional currency prefix.
_OCR_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([S$]?)([0-9OoIl|][0-9OoIl|,.]{0,18})(?![A-Za-z0-9])")
# A repair is only accepted if the result is unambiguously a number. This is
# what keeps the pass from turning "Suite O" into "Suite 0".
_NUMERIC_OK = re.compile(r"^(?:\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)$")
_SPACED_CAPS_RE = re.compile(r"\b(?:[A-Z]\s){3,}[A-Z]\b")

# The reverse confusion: a digit sitting inside a word, as in "Pr0ject Name".
# Only a digit with a letter on both sides is touched, which leaves part
# numbers, drawing references and anything like "G701" or "Phase 2" alone.
_DIGIT_IN_WORD_RE = re.compile(r"(?<=[A-Za-z])([015])(?=[A-Za-z])")
_DIGIT_TO_LETTER = {"0": "o", "1": "l", "5": "s"}


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _LIGATURE_RE.sub(lambda m: _CHAR_FIXES[m.group(0)], text)
    for dash in _DASHES:
        text = text.replace(dash, "-")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Drop control characters other than newline and tab.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    return text


def fix_numeric_ocr(text: str) -> tuple[str, int]:
    """Repair letter/digit confusion inside numeric looking tokens.

    The rule is conservative on purpose: a token is only rewritten if the
    rewrite turns it into something that is unambiguously a number. So
    ``l,5OO.OO`` becomes ``1,500.00`` and ``S2,48O.OO`` becomes ``$2,480.00``,
    while ``O'Brien``, ``Lot I`` and ``Suite O`` are left alone.

    Returns the repaired text and the number of tokens changed.
    """
    fixes = 0

    def _fix_token(match: re.Match[str]) -> str:
        nonlocal fixes
        prefix, body = match.group(1), match.group(2)
        if len(body) < 2:
            return match.group(0)
        if not any(c.isdigit() for c in body) and not any(c in body for c in ",."):
            return match.group(0)

        # A trailing period is usually sentence punctuation, not part of the number.
        trailing = ""
        core = body
        if core.endswith("."):
            core, trailing = core[:-1], "."

        repaired = core.translate(_OCR_DIGIT_MAP)
        if not _NUMERIC_OK.match(repaired):
            return match.group(0)

        looks_like_money = "," in repaired or "." in repaired
        new_prefix = "$" if prefix == "S" and looks_like_money and len(repaired) >= 4 else prefix
        rebuilt = f"{new_prefix}{repaired}{trailing}"
        if rebuilt != match.group(0):
            fixes += 1
        return rebuilt

    return _OCR_TOKEN_RE.sub(_fix_token, text), fixes


def fix_alpha_ocr(text: str) -> tuple[str, int]:
    """Repair digits that should be letters inside a word.

    "Pr0ject Name" and "Pr0ject No." are extremely common in OCR output, and a
    single wrong character there costs you the project name and number for the
    whole document, because every label pattern for those fields stops
    matching.
    """
    fixes = 0

    def _fix(match: re.Match[str]) -> str:
        nonlocal fixes
        digit = match.group(1)
        start, end = match.span(1)
        before, after = text[start - 1], text[end] if end < len(text) else "a"
        replacement = _DIGIT_TO_LETTER[digit]
        if before.isupper() and after.isupper():
            replacement = replacement.upper()
        fixes += 1
        return replacement

    return _DIGIT_IN_WORD_RE.sub(_fix, text), fixes


def collapse_spaced_capitals(text: str) -> str:
    """``C H A N G E  O R D E R`` -> ``CHANGE ORDER`` (common in scanned headers)."""
    return _SPACED_CAPS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def strip_repeated_furniture(pages: list[str], min_share: float = 0.6) -> tuple[list[str], list[str]]:
    """Remove headers/footers that repeat across most pages.

    Returns the cleaned pages and the list of removed lines, which is useful as
    a quality note: a document whose every page is 40 percent furniture usually
    needs a different ingest path.
    """
    if len(pages) < 3:
        cleaned = []
        for page in pages:
            kept = [ln for ln in page.split("\n") if not _PAGE_FURNITURE_RE.match(ln)]
            cleaned.append("\n".join(kept))
        return cleaned, []

    counts: dict[str, int] = {}
    for page in pages:
        lines = page.split("\n")
        edge = set(ln.strip() for ln in lines[:3] + lines[-3:] if len(ln.strip()) > 3)
        for line in edge:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(2, int(len(pages) * min_share))
    repeated = {line for line, n in counts.items() if n >= threshold}

    cleaned_pages = []
    for page in pages:
        kept = [
            ln
            for ln in page.split("\n")
            if ln.strip() not in repeated and not _PAGE_FURNITURE_RE.match(ln)
        ]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages, sorted(repeated)


def normalize_document_text(pages: list[str]) -> tuple[str, list[int], list[str]]:
    """Normalise page texts into one string.

    Returns the joined text, the start offset of each page inside it, and notes
    describing what had to be repaired.
    """
    notes: list[str] = []
    pages = [normalize_unicode(p or "") for p in pages]
    pages, removed = strip_repeated_furniture(pages)
    if removed:
        notes.append(f"removed {len(removed)} repeated header/footer line(s)")

    total_fixes = 0
    total_word_fixes = 0
    out_pages: list[str] = []
    for page in pages:
        page = _HYPHEN_BREAK_RE.sub(r"\1\2", page)
        page = collapse_spaced_capitals(page)
        page, fixes = fix_numeric_ocr(page)
        total_fixes += fixes
        page, word_fixes = fix_alpha_ocr(page)
        total_word_fixes += word_fixes
        page = _DOT_LEADER_RE.sub("  ", page)
        page = _TRAILING_WS_RE.sub("\n", page)
        page = _MULTI_BLANK_RE.sub("\n\n", page)
        out_pages.append(page.strip("\n"))

    if total_fixes:
        notes.append(f"repaired {total_fixes} numeric token(s) with OCR style character confusion")
    if total_word_fixes:
        notes.append(f"repaired {total_word_fixes} word(s) containing a digit where a letter belongs")

    joined_parts: list[str] = []
    offsets: list[int] = []
    cursor = 0
    separator = "\n\n"
    for i, page in enumerate(out_pages):
        offsets.append(cursor)
        joined_parts.append(page)
        cursor += len(page)
        if i != len(out_pages) - 1:
            joined_parts.append(separator)
            cursor += len(separator)

    return "".join(joined_parts), offsets, notes


def page_for_offset(offsets: list[int], offset: int) -> int:
    """1 based page number containing a character offset."""
    page = 1
    for i, start in enumerate(offsets):
        if offset >= start:
            page = i + 1
        else:
            break
    return page


def snippet_around(text: str, start: int, end: int, window: int = 45) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return " ".join(text[lo:hi].split())
