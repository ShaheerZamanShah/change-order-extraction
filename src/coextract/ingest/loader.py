"""Document loading.

Supported inputs: plain text, markdown, email bodies, PDFs with a text layer,
PDFs without one (OCR fallback) and standalone images.

The loader is deliberately tolerant. If pdfplumber is missing it falls back to
pypdf; if both are missing it raises a clear error instead of silently
returning nothing. If a PDF turns out to be a scan, it reports that in the
document metadata so the confidence scorer can discount everything.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..normalize.text import normalize_document_text
from ..schema import DocumentMeta, IngestMethod
from .quality import assess_text_quality

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".text", ".md", ".eml", ".log", ".csv"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Below this many characters per page we assume the page is a scan.
SCAN_THRESHOLD_CHARS_PER_PAGE = 120


class UnsupportedDocument(RuntimeError):
    pass


@dataclass
class LoadedTable:
    page: int
    rows: list[list[str]]

    def as_text(self) -> str:
        return "\n".join(" | ".join(cell or "" for cell in row) for row in self.rows)


@dataclass
class Document:
    """Normalised text plus everything needed to point back at a location."""

    text: str
    pages: list[str]
    page_offsets: list[int] = field(default_factory=list)
    tables: list[LoadedTable] = field(default_factory=list)
    meta: DocumentMeta = field(default_factory=DocumentMeta)

    def page_of(self, offset: int) -> int:
        page = 1
        for i, start in enumerate(self.page_offsets):
            if offset >= start:
                page = i + 1
            else:
                break
        return page


def _finalise(
    raw_pages: list[str],
    tables: list[LoadedTable],
    method: IngestMethod,
    source: Path | None,
    raw_bytes: bytes | None,
    extra_notes: list[str],
) -> Document:
    text, offsets, notes = normalize_document_text(raw_pages)
    notes = extra_notes + notes
    quality, quality_notes = assess_text_quality(text, max(len(raw_pages), 1))

    meta = DocumentMeta(
        source_path=str(source) if source else "<memory>",
        source_name=source.name if source else "<memory>",
        sha256=hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else hashlib.sha256(text.encode()).hexdigest(),
        byte_size=len(raw_bytes) if raw_bytes else len(text.encode()),
        page_count=max(len(raw_pages), 1),
        ingest_method=method,
        char_count=len(text),
        text_quality=quality,
        quality_notes=notes + quality_notes,
    )
    return Document(text=text, pages=raw_pages, page_offsets=offsets, tables=tables, meta=meta)


def load_text(text: str, name: str = "<memory>") -> Document:
    """Load an in memory string. Form feeds are treated as page breaks."""
    raw_pages = text.split("\f") if "\f" in text else [text]
    source = Path(name) if name != "<memory>" else None
    return _finalise(raw_pages, [], IngestMethod.TEXT_FILE, source, text.encode("utf-8", "replace"), [])


def _page_text_with_columns(
    page, line_tolerance: float = 2.5, gap_ratio: float = 1.8, min_gap: float = 4.0
) -> str:
    """Rebuild page text with column gaps preserved.

    A PDF text layer has no concept of a column. ``extract_text`` puts a single
    space between two words whether they are adjacent in a sentence or sitting
    in different columns of a cost breakdown, which destroys exactly the signal
    the table reader needs.

    So the text is rebuilt from word positions instead: consecutive words get
    one space if they are close together and a column separator if the gap
    between them is wide relative to the character width on that line. The
    result reads like the original form, and the rest of the pipeline can treat
    a PDF and a plain text file the same way.
    """
    words = page.extract_words(x_tolerance=1.2, y_tolerance=3, keep_blank_chars=False)
    if not words:
        return page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""

    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if rows and abs(word["top"] - rows[-1][0]["top"]) <= line_tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])

    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda w: w["x0"])
        widths = [
            (w["x1"] - w["x0"]) / max(len(w["text"]), 1) for w in row if w["text"].strip()
        ]
        char_width = sum(widths) / len(widths) if widths else 5.0
        threshold = max(min_gap, char_width * gap_ratio)

        parts = [row[0]["text"]]
        for previous, current in zip(row, row[1:]):
            gap = current["x0"] - previous["x1"]
            parts.append("   " if gap > threshold else " ")
            parts.append(current["text"])
        lines.append("".join(parts))
    return "\n".join(lines)


def _extract_pdf_pdfplumber(path: Path) -> tuple[list[str], list[LoadedTable]]:
    import pdfplumber

    pages: list[str] = []
    tables: list[LoadedTable] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            pages.append(_page_text_with_columns(page))
            try:
                found = page.extract_tables()
            except Exception as exc:  # pdfplumber can trip on odd layouts
                log.debug("table extraction failed on page %s: %s", page_number, exc)
                found = []
            for raw_rows in found:
                rows = [[(cell or "").strip() for cell in row] for row in raw_rows if row]
                if len(rows) >= 2:
                    tables.append(LoadedTable(page=page_number, rows=rows))
    return pages, tables


def _extract_pdf_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _ocr_pdf(path: Path, dpi: int = 300) -> list[str]:
    """OCR every page. Requires pytesseract, pdf2image and their binaries."""
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(str(path), dpi=dpi)
    return [pytesseract.image_to_string(img, config="--psm 6") for img in images]


def _ocr_image(path: Path) -> list[str]:
    import pytesseract
    from PIL import Image

    return [pytesseract.image_to_string(Image.open(str(path)), config="--psm 6")]


def _load_pdf(path: Path, allow_ocr: bool) -> Document:
    raw_bytes = path.read_bytes()
    notes: list[str] = []
    tables: list[LoadedTable] = []
    pages: list[str] = []

    try:
        pages, tables = _extract_pdf_pdfplumber(path)
    except ImportError:
        notes.append("pdfplumber_unavailable_no_table_extraction")
    except Exception as exc:
        notes.append(f"pdfplumber_failed:{type(exc).__name__}")

    if not any(p.strip() for p in pages):
        try:
            pages = _extract_pdf_pypdf(path)
            if any(p.strip() for p in pages):
                notes.append("used_pypdf_fallback")
        except ImportError:
            if not pages:
                raise UnsupportedDocument(
                    "No PDF backend available. Install pdfplumber or pypdf."
                ) from None
        except Exception as exc:
            notes.append(f"pypdf_failed:{type(exc).__name__}")

    chars_per_page = sum(len(p) for p in pages) / max(len(pages), 1)
    if chars_per_page < SCAN_THRESHOLD_CHARS_PER_PAGE:
        notes.append(f"text_layer_sparse:{chars_per_page:.0f}_chars_per_page")
        if allow_ocr:
            try:
                ocr_pages = _ocr_pdf(path)
                if sum(len(p) for p in ocr_pages) > sum(len(p) for p in pages):
                    notes.append("ocr_fallback_used")
                    return _finalise(ocr_pages, tables, IngestMethod.PDF_OCR, path, raw_bytes, notes)
            except ImportError:
                notes.append("ocr_requested_but_pytesseract_missing")
            except Exception as exc:
                notes.append(f"ocr_failed:{type(exc).__name__}")
        else:
            notes.append("ocr_disabled")

    return _finalise(pages, tables, IngestMethod.PDF_TEXT_LAYER, path, raw_bytes, notes)


def load_document(path: str | Path, allow_ocr: bool = True) -> Document:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")
        raw_pages = text.split("\f") if "\f" in text else [text]
        return _finalise(raw_pages, [], IngestMethod.TEXT_FILE, path, raw_bytes, [])

    if suffix in PDF_SUFFIXES:
        return _load_pdf(path, allow_ocr=allow_ocr)

    if suffix in IMAGE_SUFFIXES:
        raw_bytes = path.read_bytes()
        try:
            pages = _ocr_image(path)
        except ImportError as exc:
            raise UnsupportedDocument(
                "Image input needs the ocr extra: pip install 'coextract[ocr]'"
            ) from exc
        return _finalise(pages, [], IngestMethod.IMAGE_OCR, path, raw_bytes, ["image_ocr"])

    raise UnsupportedDocument(f"Unsupported file type: {suffix or path.name}")
