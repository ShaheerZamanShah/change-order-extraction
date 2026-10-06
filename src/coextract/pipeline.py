"""The pipeline.

    ingest -> normalise -> extract candidates -> arbitrate -> repair
           -> validate -> score and route

Each stage is independently testable and none of them throw away information
the next stage might want. Candidates that lose arbitration are kept as
alternatives, repairs are tagged rather than hidden, and failed checks are
reported with the fields they implicate.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, Optional

from .config import DEFAULT_CONFIG, PipelineConfig
from .confidence import apply_validation_feedback, route_for_review
from .extract import Candidate, LLMExtractor, NullLLMClient, RuleExtractor, TableExtractor
from .extract.arbitrate import arbitrate_field
from .extract.llm import LLMClient, OpenAIChatClient
from .fields import FIELD_SPECS
from .ingest import Document, load_document, load_text
from .schema import ChangeOrderRecord, ExtractionResult, Severity, ValidationIssue
from .validate import repair_record, validate_record

log = logging.getLogger(__name__)


class ChangeOrderPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.rules = RuleExtractor(dayfirst=self.config.dayfirst_dates)
        self.tables = TableExtractor()

        if llm_client is None and self.config.enable_llm:
            llm_client = OpenAIChatClient(
                model=self.config.llm_model, base_url=self.config.llm_base_url
            )
        self.llm = LLMExtractor(llm_client or NullLLMClient(), max_chars=self.config.max_llm_chars)

    # -- entry points -------------------------------------------------------
    def run_path(self, path: str | Path) -> ExtractionResult:
        doc = load_document(path, allow_ocr=self.config.allow_ocr)
        return self.run_document(doc)

    def run_text(self, text: str, name: str = "<memory>") -> ExtractionResult:
        return self.run_document(load_text(text, name))

    # -- core ---------------------------------------------------------------
    def run_document(self, doc: Document) -> ExtractionResult:
        started = time.perf_counter()

        candidates: list[Candidate] = []
        extractors_used: list[str] = []

        candidates.extend(self.rules.extract(doc))
        extractors_used.append(self.rules.name)

        candidates.extend(self.tables.extract(doc))
        extractors_used.append(self.tables.name)

        if self.llm.enabled:
            candidates.extend(self.llm.extract(doc))
            extractors_used.append(f"{self.llm.name}:{self.llm.client.name}")

        record = ChangeOrderRecord()
        for spec in FIELD_SPECS:
            extracted = arbitrate_field(spec.name, candidates, doc)
            if extracted.is_present and extracted.confidence < self.config.min_confidence_to_keep:
                extracted.notes.append("dropped_below_minimum_confidence")
                extracted.value = None
                extracted.confidence = 0.0
            setattr(record, spec.name, extracted)

        record.line_items = list(self.tables.line_items)

        extra_issues: list[ValidationIssue] = repair_record(record, self.config)

        report = validate_record(record, self.config)
        report.issues = extra_issues + report.issues
        report.checks_run = sorted(set(report.checks_run))

        if self.llm.dropped:
            for field_name, reason in self.llm.dropped:
                report.issues.append(
                    ValidationIssue(
                        code="LLM_CANDIDATE_REJECTED",
                        severity=Severity.INFO,
                        message=f"a model proposed value for {field_name} was discarded: {reason}",
                        fields=[field_name] if field_name != "*" else [],
                    )
                )

        apply_validation_feedback(record, report, self.config)
        review = route_for_review(record, doc.meta, report, self.config)

        return ExtractionResult(
            document=doc.meta,
            change_order=record,
            validation=report,
            review=review,
            extractors_used=extractors_used,
            runtime_ms=int((time.perf_counter() - started) * 1000),
        )


def extract_file(
    path: str | Path,
    config: Optional[PipelineConfig] = None,
    llm_client: Optional[LLMClient] = None,
) -> ExtractionResult:
    """One shot helper for a single document."""
    return ChangeOrderPipeline(config, llm_client).run_path(path)


def extract_many(
    paths: Iterable[str | Path],
    config: Optional[PipelineConfig] = None,
    llm_client: Optional[LLMClient] = None,
) -> list[ExtractionResult]:
    """Batch helper. One bad document never stops the batch."""
    pipeline = ChangeOrderPipeline(config, llm_client)
    results: list[ExtractionResult] = []
    for path in paths:
        try:
            results.append(pipeline.run_path(path))
        except Exception as exc:
            log.error("failed on %s: %s", path, exc)
            failed = ExtractionResult()
            failed.document.source_path = str(path)
            failed.document.source_name = Path(path).name
            failed.validation.passed = False
            failed.validation.error_count = 1
            failed.validation.issues = [
                ValidationIssue(
                    code="INGEST_FAILED",
                    severity=Severity.ERROR,
                    message=f"{type(exc).__name__}: {exc}",
                )
            ]
            failed.review.reasons = ["the document could not be read at all"]
            results.append(failed)
    return results
