"""Generate the sample documents that cannot be stored as plain text.

Two things happen here:

1. a genuine multi page sample is written with form feed page breaks and the
   repeated header and footer that real PDFs carry, so the furniture stripping
   in the normaliser has something to chew on;
2. the text samples are rendered to PDFs, which exercises the real ingest path
   including pdfplumber table detection.

    python scripts/make_samples.py
    python scripts/make_samples.py --pdf-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
PDF_DIR = SAMPLES / "pdf"

HEADER = "ALDERGROVE INNOVATION CAMPUS  |  CHANGE ORDER 041  |  CONTRACT AIC-2022-9"
FOOTER = "Confidential - Aldergrove Partners LLP and Rennick Construction"

PAGE_1 = """CHANGE ORDER

Project Name: Aldergrove Innovation Campus Lab Fit Out
Project No.: AIC-2022-9
Contract No.: RC-9914-B
Owner: Aldergrove Partners LLP
Contractor: Rennick Construction Company
Architect: Hollenbeck Ilves Architects

Change Order Number: 041
Date of Issuance: 2024-04-09
Status: Approved

DESCRIPTION OF CHANGE
Add four variable air volume fume hoods in lab suites 3.08 and 3.09, modify
the adjacent casework to receive them, extend the exhaust ductwork to the
existing manifold, and integrate the new hoods into the building controls
system. The additional hoods were added by the owner after the research
programme for the third floor was finalised.

Reason for change: Owner requested addition.

(continued on the following page)"""

PAGE_2 = """ITEMIZED COST BREAKDOWN

  Item  Description                                Qty  Unit   Unit Price      Amount
  1     Fume hood, 6 foot, variable air volume       4    EA     9,450.00    37,800.00
  2     Lab casework modification and refit          1    LS    14,200.00    14,200.00
  3     Exhaust ductwork and transitions           180    LF        62.50    11,250.00
  4     Controls integration and air balancing       1    LS    13,100.00    13,100.00
        Total this Change Order                                             76,350.00

CONTRACT SUM SUMMARY

The original Contract Sum was                                        12,340,000.00
Net change by previously authorized Change Orders                       587,210.00
The Contract Sum prior to this Change Order was                      12,927,210.00
The Contract Sum will be increased by this Change Order by               76,350.00
The new Contract Sum including this Change Order will be             13,003,560.00"""

PAGE_3 = """CONTRACT TIME

There is no change to the Contract Time as a result of this Change Order.
Original date of Substantial Completion: 2025-01-31
The date of Substantial Completion is unchanged.

ATTACHMENTS
  A. Rennick Construction proposal RCP-0412 dated 2024-03-28
  B. Hollenbeck Ilves sketch SK-118 dated 2024-03-21
  C. Mechanical subcontractor quotation from Pell Air Systems

EXECUTION

Owner        Aldergrove Partners LLP        By: H. Aldergrove   Date: 2024-04-15
Contractor   Rennick Construction Company   By: S. Rennick      Date: 2024-04-12
Architect    Hollenbeck Ilves Architects    By: M. Ilves        Date: 2024-04-11

Date Executed: 2024-04-15"""


def write_multipage() -> Path:
    pages = []
    for number, body in enumerate((PAGE_1, PAGE_2, PAGE_3), start=1):
        pages.append(f"{HEADER}\n\n{body}\n\n{FOOTER}\nPage {number} of 3")
    target = SAMPLES / "co_007_multipage.txt"
    target.write_text("\f".join(pages) + "\n", encoding="utf-8")
    return target


def render_pdfs(names: list[str] | None = None) -> list[Path]:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
    except ImportError:
        print("reportlab is not installed, skipping PDF rendering")
        return []

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sources = sorted(SAMPLES.glob("*.txt"))
    if names:
        sources = [p for p in sources if p.stem in names]

    for source in sources:
        text = source.read_text(encoding="utf-8")
        target = PDF_DIR / f"{source.stem}.pdf"
        pdf = canvas.Canvas(str(target), pagesize=LETTER)
        pdf.setTitle(source.stem)
        width, height = LETTER
        margin, leading, font_size = 48, 11.2, 8.6

        for page_text in text.split("\f"):
            pdf.setFont("Courier", font_size)
            y = height - margin
            for line in page_text.split("\n"):
                if y < margin:
                    pdf.showPage()
                    pdf.setFont("Courier", font_size)
                    y = height - margin
                # Courier is monospaced, so column alignment survives the trip.
                pdf.drawString(margin, y, line[:118])
                y -= leading
            pdf.showPage()
        pdf.save()
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--only", nargs="*", help="limit PDF rendering to these sample stems")
    args = parser.parse_args()

    if not args.pdf_only:
        print(f"wrote {write_multipage().relative_to(ROOT)}")
    if not args.no_pdf:
        written = render_pdfs(args.only)
        print(f"rendered {len(written)} PDF(s) into {PDF_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
