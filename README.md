# coextract

Pull structured, validated, confidence scored fields out of messy construction change orders. Input is a PDF or a text file. Output is JSON where every field carries its own confidence, the evidence it came from, and the candidates that lost.

Read [WRITEUP.md](WRITEUP.md) for the approach and the failure modes.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Core extraction needs only `pydantic` and `python-dateutil`. PDFs need `pdfplumber`. Scanned PDFs additionally need `pytesseract`, `pdf2image` and the Tesseract and Poppler binaries on PATH.

## Use it

```bash
# one document, human readable report
coextract extract data/samples/co_001_aia_clean.txt

# show the evidence snippet behind each field
coextract extract data/samples/co_003_ocr_scan.txt --evidence

# full JSON
coextract extract scan.pdf --json out/scan.json

# a whole folder, plus a flat CSV summary
coextract batch data/samples --out out/ --summary out/summary.csv

# what the schema is, and what the knobs are
coextract fields
coextract config
```

`coextract extract` exits 0 when validation found no errors and 2 when it did, so it drops straight into a shell pipeline. `coextract batch` always exits 0 and reports per document.

From Python:

```python
from coextract import extract_file

result = extract_file("change_order.pdf")

print(result.review.decision)                      # auto_accept | review | reject
print(result.change_order.change_amount.value)      # Decimal("35667.00")
print(result.change_order.change_amount.confidence) # 0.97
print(result.change_order.change_amount.evidence[0].snippet)

for issue in result.validation.issues:
    print(issue.severity, issue.code, issue.message)
```

As a service:

```bash
pip install "coextract[api]"
coextract serve
curl -F file=@change_order.pdf http://127.0.0.1:8000/extract
```

## What comes out

```json
{
  "schema_version": "1.0.0",
  "document": {
    "source_name": "co_001_aia_clean.txt",
    "ingest_method": "text_file",
    "page_count": 1,
    "text_quality": 1.0,
    "quality_notes": ["repaired 1 numeric token(s) with OCR style character confusion"]
  },
  "change_order": {
    "change_amount": {
      "value": "35667.00",
      "confidence": 0.97,
      "extractor": "rules",
      "evidence": [
        {
          "page": 1,
          "start": 1704,
          "end": 1713,
          "snippet": "AMOUNT OF 35,667.00 THE NEW CONTRACT SUM",
          "matcher": "label:amount"
        }
      ],
      "alternatives": [
        {"value": "32040.00", "score": 0.64, "matcher": "table_total_row", "snippet": "Subtotal 32,040.00"}
      ],
      "notes": ["supported_by_3_patterns", "corroborated_by_cross_check"]
    }
  },
  "validation": {
    "passed": true,
    "error_count": 0,
    "warning_count": 0,
    "corroborated_fields": ["change_amount", "contract_sum_after", "contract_sum_before"],
    "issues": []
  },
  "review": {
    "decision": "auto_accept",
    "document_confidence": 0.94,
    "reasons": ["all checks passed and every field cleared the confidence threshold"],
    "fields_to_check": []
  }
}
```

### Fields

Identity and parties: `change_order_number`, `change_order_title`, `project_name`, `project_number`, `contract_number`, `owner_name`, `contractor_name`, `architect_name`.

Dates and classification: `date_issued`, `date_executed`, `status`, `reason_code`, `scope_description`.

Money: `currency`, `original_contract_sum`, `net_previous_changes`, `contract_sum_before`, `change_amount`, `contract_sum_after`, `retainage_percent`.

Schedule: `schedule_impact_days`, `original_completion_date`, `revised_completion_date`.

Plus `line_items`, each with description, quantity, unit, unit price, amount and its own confidence.

Markup, bond, insurance and tax rows are kept as line items rather than thrown away. They are part of what the change order costs, and keeping them is what lets the line item total reconcile against the stated change amount. Subtotal and total rows are not line items; they become candidates for `change_amount` instead.

## How it works

```
ingest -> normalise -> extract candidates -> arbitrate -> repair -> validate -> score and route
```

- **Ingest** reads text files, PDFs with a text layer, and scans through OCR. PDF text is rebuilt from word positions so that column gaps survive, because `extract_text` puts a single space between two words whether they are adjacent in a sentence or in different columns of a cost breakdown. Every document gets a text quality score that caps the confidence of everything read from it.
- **Normalise** fixes what OCR and PDF exporters break: ligatures, hyphenated line breaks, repeated page headers, `l,85O.OO` for `1,850.00`, `S2,480.00` for `$2,480.00`, and `Pr0ject Name` for `Project Name`.
- **Extract** runs three extractors that propose candidates rather than answers. Rule based patterns built from the field registry, a table reader for line items, and an optional LLM that is only allowed to report a value if it can quote the document and the quote checks out against the document text.
- **Arbitrate** clusters candidates by value, rewards independent agreement, and penalises a close contest between two different values.
- **Repair** fills a gap that follows arithmetically from the values already found, and marks it as derived rather than read.
- **Validate** checks the identities a change order carries inside itself, the date ordering, the schedule against the dates, and plausibility.
- **Score and route** folds validation back into per field confidence in both directions, then decides auto accept, review or reject.

## Results

Twelve sample documents with hand written ground truth in `data/gold/gold.json`, covering a clean AIA form, an email thread, OCR output from a scan, a pipe delimited export, a deductive change order, a no cost time extension, a three page document, European number formatting, a transcribed field memo, a deliberately inconsistent document, a page fragment with nothing in it, and a document containing a change order log built to look like a cost breakdown.

```bash
python eval/run_eval.py                 # evaluate the text samples
python eval/run_eval.py --pdf           # evaluate the rendered PDFs
python -m pytest                        # 150 tests
```

| | text input | PDF input |
|---|---|---|
| stated values graded | 192 | 192 |
| correct | 191 | 191 |
| wrong | 0 | 0 |
| missed | 1 | 1 |
| spurious | 0 | 0 |
| precision / recall / f1 | 1.000 / 0.995 / 0.997 | 1.000 / 0.995 / 0.997 |
| line item totals matching gold | 12 of 12 | 12 of 12 |
| wrong fields inside auto accepted documents | 0 | 0 |

The single miss is a contractor name that appears only in an email signature block. The last row is the number that matters operationally: nothing wrong went through unchecked.

Confidence is conservative rather than well calibrated. Every reported value in this corpus was correct, so observed accuracy is 1.0 in all five confidence bins while mean confidence runs from 0.35 to 0.95, which gives an expected calibration error of 0.116 entirely in the safe direction. Twelve documents is not enough to calibrate on and the harness prints the reliability table so it can be redone against a real corpus. See the writeup.

## Configuration

```bash
coextract config > my_config.json
coextract batch inbox/ --config my_config.json
```

The settings that matter most:

| setting | default | what it does |
|---|---|---|
| `money_tolerance` | 1.00 | absolute floor for comparing two amounts |
| `relative_money_tolerance` | 0.0005 | percentage rounding allowance |
| `max_money_tolerance` | 25.00 | cap, so tolerance does not scale with contract size |
| `auto_accept_threshold` | 0.85 | document confidence needed to skip review |
| `field_review_threshold` | 0.70 | below this, a field is listed for checking |
| `reject_threshold` | 0.40 | below this, the document is rejected |
| `require_review_for_ocr` | true | OCR output is never auto accepted |
| `enable_repair` | true | derive values that follow from arithmetic |
| `enable_llm` | false | turn on the optional LLM extractor |

## Optional LLM extractor

Off by default, and the pipeline is fully functional without it. It exists for narrative documents where no value sits next to a recognisable label.

```bash
export OPENAI_API_KEY=...
coextract extract letter.pdf --llm
```

The model has to return a verbatim quote for every value. The quote is then located in the document, and for any number the digits must actually appear in the document text near the quote. Anything that fails is dropped and recorded as an `LLM_CANDIDATE_REJECTED` issue. A model value can never exceed 0.88 confidence on its own.

`OPENAI_BASE_URL` points it at a local or self hosted model. Implementing `complete(system, user)` on your own class replaces the client entirely.

## Layout

```
src/coextract/
  fields.py            field registry: what we extract and how we recognise it
  schema.py            output models
  config.py            thresholds
  ingest/              loading, PDF column reconstruction, OCR, text quality
  normalize/           text clean up and value parsers
  extract/             rule, table and LLM extractors, plus arbitration
  validate/            cross field checks and arithmetic repair
  confidence/          scoring and review routing
  pipeline.py          orchestration
  cli.py  api.py  report.py
data/samples/          twelve messy sample documents
data/gold/gold.json    hand written ground truth
eval/run_eval.py       accuracy, calibration and routing report
scripts/make_samples.py
tests/                 150 tests
```

Adding a field means adding one entry to `FIELD_SPECS` in `fields.py` and one line to `ChangeOrderRecord` in `schema.py`. A test asserts the two stay in sync.
