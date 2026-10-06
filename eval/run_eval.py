"""Evaluation harness.

Three questions get answered here, and they are not the same question.

1. Accuracy. When the pipeline reports a value, is it right? When the document
   states a value, does the pipeline find it?
2. Calibration. When the pipeline says 0.9, is it right about nine times out of
   ten? An uncalibrated confidence score is worse than no score, because people
   build thresholds on it.
3. Routing. Of the documents sent through without review, how many carried a
   wrong value? That is the number that decides whether this can be trusted in
   a real workflow, and it is the one worth optimising.

    python eval/run_eval.py
    python eval/run_eval.py --pdf --json eval/reports/latest.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coextract import PipelineConfig, extract_file  # noqa: E402
from coextract.fields import FIELD_BY_NAME  # noqa: E402

GOLD_PATH = ROOT / "data" / "gold" / "gold.json"
SAMPLES = ROOT / "data" / "samples"
TEXT_MATCH_RATIO = 0.85
CONFIDENCE_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


@dataclass
class FieldOutcome:
    document: str
    field: str
    gold: Any
    predicted: Any
    confidence: float
    verdict: str  # correct | wrong | missed | spurious | correct_absent


@dataclass
class Totals:
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    spurious: int = 0
    correct_absent: int = 0

    def add(self, verdict: str) -> None:
        setattr(self, verdict, getattr(self, verdict) + 1)

    @property
    def reported(self) -> int:
        return self.correct + self.wrong + self.spurious

    @property
    def stated(self) -> int:
        return self.correct + self.wrong + self.missed

    @property
    def precision(self) -> Optional[float]:
        return self.correct / self.reported if self.reported else None

    @property
    def recall(self) -> Optional[float]:
        return self.correct / self.stated if self.stated else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if not p or not r:
            return None
        return 2 * p * r / (p + r)


def _normalise_text(value: Any) -> str:
    return " ".join(str(value).lower().replace(",", " ").split()).strip(" .;:")


def _to_decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def values_match(field_name: str, gold: Any, predicted: Any) -> bool:
    spec = FIELD_BY_NAME.get(field_name)
    kind = spec.kind if spec else "text"

    if kind in ("money", "percent"):
        gold_decimal, predicted_decimal = _to_decimal(gold), _to_decimal(predicted)
        if gold_decimal is None or predicted_decimal is None:
            return False
        return abs(gold_decimal - predicted_decimal) <= Decimal("0.01")

    if kind == "days":
        return _to_decimal(gold) == _to_decimal(predicted)

    if kind in ("date", "enum", "currency"):
        return _normalise_text(gold) == _normalise_text(predicted)

    gold_text, predicted_text = _normalise_text(gold), _normalise_text(predicted)
    if not gold_text or not predicted_text:
        return False
    if kind == "string":
        return gold_text == predicted_text
    # Free text: the gold value is a key phrase, so containment counts, and
    # near misses on punctuation or a legal suffix count too.
    if gold_text in predicted_text or predicted_text in gold_text:
        return True
    return difflib.SequenceMatcher(None, gold_text, predicted_text).ratio() >= TEXT_MATCH_RATIO


def evaluate(use_pdf: bool, config: PipelineConfig) -> dict[str, Any]:
    gold_data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    outcomes: list[FieldOutcome] = []
    per_field: dict[str, Totals] = defaultdict(Totals)
    overall = Totals()
    decisions: list[tuple[str, str, str]] = []
    line_item_results: list[tuple[str, Optional[str], Optional[str], bool]] = []
    failures: list[str] = []

    for name, entry in sorted(gold_data.items()):
        if name.startswith("_"):
            continue
        path = (SAMPLES / "pdf" / name).with_suffix(".pdf") if use_pdf else SAMPLES / name
        if not path.exists():
            failures.append(f"{name}: no input file at {path}")
            continue

        result = extract_file(path, config=config)
        predicted_values = result.flat_values()

        for field_name, gold_value in entry["fields"].items():
            extracted = result.change_order.get(field_name)
            predicted = predicted_values.get(field_name)
            confidence = extracted.confidence

            if gold_value is None and predicted is None:
                verdict = "correct_absent"
            elif gold_value is None:
                verdict = "spurious"
            elif predicted is None:
                verdict = "missed"
            elif values_match(field_name, gold_value, predicted):
                verdict = "correct"
            else:
                verdict = "wrong"

            per_field[field_name].add(verdict)
            overall.add(verdict)
            outcomes.append(
                FieldOutcome(name, field_name, gold_value, predicted, confidence, verdict)
            )

        decisions.append((name, entry.get("expected_decision", ""), result.review.decision.value))

        gold_total = entry.get("line_item_total")
        amounts = [i.amount for i in result.change_order.line_items if i.amount is not None]
        predicted_total = str(sum(amounts, Decimal("0.00"))) if amounts else None
        if gold_total is None:
            matched = predicted_total is None
        else:
            matched = predicted_total is not None and abs(
                Decimal(gold_total) - Decimal(predicted_total)
            ) <= Decimal("0.01")
        line_item_results.append((name, gold_total, predicted_total, matched))

    return {
        "outcomes": outcomes,
        "per_field": per_field,
        "overall": overall,
        "decisions": decisions,
        "line_items": line_item_results,
        "failures": failures,
        "source": "pdf" if use_pdf else "text",
    }


def calibration_table(outcomes: list[FieldOutcome]) -> list[dict[str, Any]]:
    """Reliability table over fields where a value was actually reported."""
    reported = [o for o in outcomes if o.verdict in ("correct", "wrong", "spurious")]
    rows = []
    for low, high in CONFIDENCE_BINS:
        bucket = [o for o in reported if low <= o.confidence < high]
        if not bucket:
            rows.append({"bin": f"{low:.2f}-{min(high, 1.0):.2f}", "n": 0})
            continue
        correct = sum(1 for o in bucket if o.verdict == "correct")
        mean_confidence = sum(o.confidence for o in bucket) / len(bucket)
        rows.append(
            {
                "bin": f"{low:.2f}-{min(high, 1.0):.2f}",
                "n": len(bucket),
                "mean_confidence": round(mean_confidence, 3),
                "observed_accuracy": round(correct / len(bucket), 3),
                "gap": round(mean_confidence - correct / len(bucket), 3),
            }
        )
    return rows


def expected_calibration_error(outcomes: list[FieldOutcome]) -> Optional[float]:
    reported = [o for o in outcomes if o.verdict in ("correct", "wrong", "spurious")]
    if not reported:
        return None
    total = 0.0
    for low, high in CONFIDENCE_BINS:
        bucket = [o for o in reported if low <= o.confidence < high]
        if not bucket:
            continue
        accuracy = sum(1 for o in bucket if o.verdict == "correct") / len(bucket)
        mean_confidence = sum(o.confidence for o in bucket) / len(bucket)
        total += (len(bucket) / len(reported)) * abs(mean_confidence - accuracy)
    return round(total, 4)


def routing_quality(report: dict[str, Any]) -> dict[str, Any]:
    """The operational question: what gets through unchecked, and is it right?"""
    by_document = defaultdict(list)
    for outcome in report["outcomes"]:
        by_document[outcome.document].append(outcome)

    decision_by_document = {name: actual for name, _, actual in report["decisions"]}
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "bad_fields": 0, "clean_documents": 0})

    for name, outcomes in by_document.items():
        decision = decision_by_document.get(name, "unknown")
        bad = sum(1 for o in outcomes if o.verdict in ("wrong", "spurious", "missed"))
        stats[decision]["documents"] += 1
        stats[decision]["bad_fields"] += bad
        if bad == 0:
            stats[decision]["clean_documents"] += 1
    return dict(stats)


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    overall: Totals = report["overall"]
    outcomes: list[FieldOutcome] = report["outcomes"]

    lines.append(f"Change order extraction evaluation  (input: {report['source']})")
    lines.append("=" * 88)
    lines.append("")
    lines.append(
        f"{overall.stated} stated field values across {len(report['decisions'])} documents, "
        f"{overall.correct_absent} correctly left empty"
    )
    lines.append(
        f"correct {overall.correct}   wrong {overall.wrong}   missed {overall.missed}   "
        f"spurious {overall.spurious}"
    )
    if overall.precision is not None:
        lines.append(
            f"precision {overall.precision:.3f}   recall {overall.recall:.3f}   "
            f"f1 {overall.f1:.3f}"
        )
    lines.append("")

    lines.append("Per field")
    lines.append(f"{'field':<28}{'ok':>4}{'wrong':>7}{'miss':>6}{'spur':>6}{'prec':>8}{'rec':>7}")
    lines.append("-" * 88)
    for field_name, totals in sorted(report["per_field"].items()):
        precision = f"{totals.precision:.2f}" if totals.precision is not None else "  - "
        recall = f"{totals.recall:.2f}" if totals.recall is not None else "  - "
        lines.append(
            f"{field_name:<28}{totals.correct:>4}{totals.wrong:>7}{totals.missed:>6}"
            f"{totals.spurious:>6}{precision:>8}{recall:>7}"
        )
    lines.append("")

    lines.append("Confidence calibration (reported values only)")
    lines.append(f"{'bin':<14}{'n':>5}{'mean conf':>12}{'accuracy':>11}{'gap':>8}")
    lines.append("-" * 88)
    for row in calibration_table(outcomes):
        if not row["n"]:
            lines.append(f"{row['bin']:<14}{0:>5}")
            continue
        lines.append(
            f"{row['bin']:<14}{row['n']:>5}{row['mean_confidence']:>12.3f}"
            f"{row['observed_accuracy']:>11.3f}{row['gap']:>8.3f}"
        )
    ece = expected_calibration_error(outcomes)
    lines.append(f"expected calibration error {ece}")
    lines.append("")

    lines.append("Routing")
    lines.append(f"{'decision':<14}{'docs':>6}{'clean docs':>12}{'bad fields':>12}")
    lines.append("-" * 88)
    for decision, stats in sorted(routing_quality(report).items()):
        lines.append(
            f"{decision:<14}{stats['documents']:>6}{stats['clean_documents']:>12}"
            f"{stats['bad_fields']:>12}"
        )
    mismatched = [(n, e, a) for n, e, a in report["decisions"] if e and e != a]
    if mismatched:
        lines.append("")
        lines.append("Routing differs from the annotation on:")
        for name, expected, actual in mismatched:
            lines.append(f"  {name}: expected {expected}, got {actual}")
    lines.append("")

    lines.append("Line item totals")
    lines.append(f"{'document':<34}{'gold':>16}{'extracted':>16}{'match':>8}")
    lines.append("-" * 88)
    for name, gold, predicted, matched in report["line_items"]:
        lines.append(
            f"{name:<34}{(gold or '-'):>16}{(predicted or '-'):>16}{('yes' if matched else 'NO'):>8}"
        )
    lines.append("")

    problems = [o for o in outcomes if o.verdict in ("wrong", "spurious", "missed")]
    if problems:
        lines.append(f"Every miss in detail ({len(problems)})")
        lines.append("-" * 88)
        for outcome in sorted(problems, key=lambda o: (o.verdict, o.document, o.field)):
            lines.append(
                f"  [{outcome.verdict:<8}] {outcome.document:<32} {outcome.field:<26} "
                f"gold={outcome.gold!r} got={outcome.predicted!r} conf={outcome.confidence:.2f}"
            )

    if report["failures"]:
        lines.append("")
        lines.append("Inputs that could not be evaluated:")
        lines.extend(f"  {f}" for f in report["failures"])

    return "\n".join(lines)


def to_json(report: dict[str, Any]) -> dict[str, Any]:
    overall: Totals = report["overall"]
    return {
        "source": report["source"],
        "summary": {
            "documents": len(report["decisions"]),
            "correct": overall.correct,
            "wrong": overall.wrong,
            "missed": overall.missed,
            "spurious": overall.spurious,
            "correct_absent": overall.correct_absent,
            "precision": overall.precision,
            "recall": overall.recall,
            "f1": overall.f1,
            "expected_calibration_error": expected_calibration_error(report["outcomes"]),
        },
        "per_field": {
            name: {
                "correct": t.correct,
                "wrong": t.wrong,
                "missed": t.missed,
                "spurious": t.spurious,
                "precision": t.precision,
                "recall": t.recall,
            }
            for name, t in sorted(report["per_field"].items())
        },
        "calibration": calibration_table(report["outcomes"]),
        "routing": routing_quality(report),
        "decisions": [
            {"document": n, "expected": e, "actual": a} for n, e, a in report["decisions"]
        ],
        "line_items": [
            {"document": n, "gold": g, "extracted": p, "match": m}
            for n, g, p, m in report["line_items"]
        ],
        "misses": [
            {
                "document": o.document,
                "field": o.field,
                "verdict": o.verdict,
                "gold": o.gold,
                "predicted": o.predicted,
                "confidence": o.confidence,
            }
            for o in report["outcomes"]
            if o.verdict in ("wrong", "spurious", "missed")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="evaluate the rendered PDFs instead of the text")
    parser.add_argument("--json", help="also write the machine readable report here")
    parser.add_argument("--llm", action="store_true", help="enable the optional LLM extractor")
    parser.add_argument(
        "--fail-under", type=float, default=None, help="exit non zero if f1 falls below this"
    )
    args = parser.parse_args()

    config = PipelineConfig()
    if args.llm:
        config.enable_llm = True

    report = evaluate(args.pdf, config)
    print(render(report))

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(to_json(report), indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {out_path}")

    f1 = report["overall"].f1 or 0.0
    if args.fail_under is not None and f1 < args.fail_under:
        print(f"\nf1 {f1:.3f} is below the floor of {args.fail_under}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
