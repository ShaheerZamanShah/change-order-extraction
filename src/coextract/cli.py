"""Command line interface.

    coextract extract data/samples/co_001_aia_clean.txt
    coextract extract scan.pdf --json out/scan.json --evidence
    coextract batch data/samples --out out/ --summary out/summary.csv
    coextract fields
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, PipelineConfig
from .fields import FIELD_SPECS
from .ingest.loader import IMAGE_SUFFIXES, PDF_SUFFIXES, TEXT_SUFFIXES
from .pipeline import ChangeOrderPipeline
from .report import render_csv_row, render_text_report

SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | IMAGE_SUFFIXES


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()
    if getattr(args, "llm", False):
        config.enable_llm = True
    if getattr(args, "llm_model", None):
        config.llm_model = args.llm_model
    if getattr(args, "no_ocr", False):
        config.allow_ocr = False
    if getattr(args, "dayfirst", False):
        config.dayfirst_dates = True
    if getattr(args, "no_repair", False):
        config.enable_repair = False
    if getattr(args, "accept_threshold", None) is not None:
        config.auto_accept_threshold = args.accept_threshold
    return config


def cmd_extract(args: argparse.Namespace) -> int:
    pipeline = ChangeOrderPipeline(_build_config(args))
    result = pipeline.run_path(args.path)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.to_json(), encoding="utf-8")
        print(f"wrote {out_path}")
    if args.stdout_json:
        print(result.to_json())
    if not args.stdout_json:
        print(render_text_report(result, show_evidence=args.evidence))

    return 0 if result.validation.error_count == 0 else 2


def cmd_batch(args: argparse.Namespace) -> int:
    root = Path(args.path)
    paths = (
        sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if root.is_dir()
        else [root]
    )
    if not paths:
        print(f"no supported documents found under {root}", file=sys.stderr)
        return 1

    pipeline = ChangeOrderPipeline(_build_config(args))
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"auto_accept": 0, "review": 0, "reject": 0}
    for path in paths:
        try:
            result = pipeline.run_path(path)
        except Exception as exc:
            print(f"{path.name}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        counts[result.review.decision.value] += 1
        rows.append(render_csv_row(result))
        if out_dir:
            (out_dir / f"{path.stem}.json").write_text(result.to_json(), encoding="utf-8")
        print(
            f"{path.name:<42} {result.review.decision.value:<12} "
            f"conf {result.review.document_confidence:.2f}  "
            f"{result.validation.error_count} err  {result.validation.warning_count} warn"
        )

    print(
        f"\n{len(rows)} document(s): "
        f"{counts['auto_accept']} auto accept, {counts['review']} review, {counts['reject']} reject"
    )

    if args.summary and rows:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {summary_path}")
    return 0


def cmd_fields(args: argparse.Namespace) -> int:
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": spec.name,
                        "kind": spec.kind,
                        "required": spec.required,
                        "importance": spec.importance,
                        "aliases": [alias.replace("\\", "") for alias, _ in spec.aliases],
                        "description": spec.description,
                    }
                    for spec in FIELD_SPECS
                ],
                indent=2,
            )
        )
        return 0
    print(f"{'field':<28}{'type':<10}{'req':<5}{'weight':<8}top labels")
    print("-" * 100)
    for spec in FIELD_SPECS:
        labels = ", ".join(alias.replace("\\", "") for alias, _ in spec.aliases[:3])
        print(
            f"{spec.name:<28}{spec.kind:<10}{'yes' if spec.required else '':<5}"
            f"{spec.importance:<8.1f}{labels[:52]}"
        )
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    print(json.dumps(DEFAULT_CONFIG.to_dict(), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("the api extra is required: pip install 'coextract[api]'", file=sys.stderr)
        return 1
    uvicorn.run("coextract.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coextract",
        description="Extract validated, confidence scored fields from change orders",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log extraction detail")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="path to a JSON config file")
        p.add_argument("--llm", action="store_true", help="enable the optional LLM extractor")
        p.add_argument("--llm-model", help="model name for the LLM extractor")
        p.add_argument("--no-ocr", action="store_true", help="never fall back to OCR")
        p.add_argument("--no-repair", action="store_true", help="do not derive missing values")
        p.add_argument("--dayfirst", action="store_true", help="read ambiguous dates as day first")
        p.add_argument("--accept-threshold", type=float, help="override the auto accept threshold")

    p_extract = sub.add_parser("extract", help="extract one document")
    p_extract.add_argument("path")
    p_extract.add_argument("--json", help="write the full JSON result to this path")
    p_extract.add_argument("--stdout-json", action="store_true", help="print JSON instead of a report")
    p_extract.add_argument("--evidence", action="store_true", help="show the evidence snippet per field")
    add_common(p_extract)
    p_extract.set_defaults(func=cmd_extract)

    p_batch = sub.add_parser("batch", help="extract every document in a folder")
    p_batch.add_argument("path")
    p_batch.add_argument("--out", help="directory for per document JSON")
    p_batch.add_argument("--summary", help="write a flat CSV summary here")
    add_common(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_fields = sub.add_parser("fields", help="list the extraction schema")
    p_fields.add_argument("--json", action="store_true")
    p_fields.set_defaults(func=cmd_fields)

    p_config = sub.add_parser("config", help="print the default configuration")
    p_config.set_defaults(func=cmd_config)

    p_serve = sub.add_parser("serve", help="run the HTTP service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
