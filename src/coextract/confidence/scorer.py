"""Confidence scoring and review routing.

Confidence here is built from four things, in this order:

1. how good the cue was that found the value (pattern strength),
2. whether anything else independently agreed with it (arbitration),
3. how trustworthy the underlying text is (ingest quality),
4. whether the value survived the cross checks (validation).

Step four cuts both ways, which is the part people usually leave out. A number
that balances against two other numbers deserves more confidence than a number
that was merely matched by a good regex, and a number that breaks an identity
deserves much less even when the regex was perfect.

The scores are deliberately interpretable rather than learned. Every penalty
and bonus below is a stated rule you can argue with, and the eval harness
reports how well the resulting numbers track real accuracy so the constants
can be retuned against your own document mix.
"""

from __future__ import annotations

from ..config import PipelineConfig
from ..fields import FIELD_BY_NAME
from ..schema import (
    ChangeOrderRecord,
    DocumentMeta,
    IngestMethod,
    ReviewDecision,
    ReviewRouting,
    Severity,
    ValidationReport,
)

CORROBORATION_GAIN = 0.30
MAX_DOC_ERROR_PENALTY = 0.30
PER_ERROR_DOC_PENALTY = 0.10


def apply_validation_feedback(
    record: ChangeOrderRecord, report: ValidationReport, config: PipelineConfig
) -> None:
    """Fold validation outcomes back into per field confidence.

    When a check fails we usually cannot tell which of the participating
    fields is wrong, so every field named in the issue takes the penalty. That
    is intentional. The honest statement is "one of these three is wrong", and
    sending all three to review is the correct response to it.
    """
    corroborated = set(report.corroborated_fields)

    for name in record.field_names():
        extracted = record.get(name)
        if not extracted.is_present:
            continue

        confidence = extracted.confidence
        touched_by_error = False

        for issue in report.issues:
            if name not in issue.fields:
                continue
            if issue.severity == Severity.ERROR:
                confidence *= 1.0 - config.error_penalty
                touched_by_error = True
                extracted.notes.append(f"failed_check:{issue.code}")
            elif issue.severity == Severity.WARNING:
                confidence *= 1.0 - config.warning_penalty
                extracted.notes.append(f"flagged_by_check:{issue.code}")

        if name in corroborated and not touched_by_error:
            confidence += (1.0 - confidence) * CORROBORATION_GAIN
            extracted.notes.append("corroborated_by_cross_check")

        if extracted.extractor == "derived":
            confidence = min(confidence, config.derived_value_ceiling)

        extracted.confidence = round(max(0.0, min(1.0, confidence)), 3)
        extracted.notes = list(dict.fromkeys(extracted.notes))


def document_confidence(
    record: ChangeOrderRecord, report: ValidationReport, config: PipelineConfig
) -> float:
    """Importance weighted score for the record as a whole.

    Missing required fields count as zero at full weight. Missing optional
    fields are left out of the average entirely, because a change order that
    genuinely has no architect is not a worse extraction for lacking one.
    """
    weighted_total = 0.0
    weight_sum = 0.0

    for name in record.field_names():
        spec = FIELD_BY_NAME.get(name)
        if spec is None:
            continue
        extracted = record.get(name)
        if extracted.is_present:
            weighted_total += extracted.confidence * spec.importance
            weight_sum += spec.importance
        elif spec.required:
            weight_sum += spec.importance

    if weight_sum == 0:
        return 0.0

    score = weighted_total / weight_sum
    score *= 1.0 - min(MAX_DOC_ERROR_PENALTY, PER_ERROR_DOC_PENALTY * report.error_count)
    return round(max(0.0, min(1.0, score)), 3)


def route_for_review(
    record: ChangeOrderRecord,
    meta: DocumentMeta,
    report: ValidationReport,
    config: PipelineConfig,
) -> ReviewRouting:
    score = document_confidence(record, report, config)
    reasons: list[str] = []
    to_check: list[str] = []
    # Everything low confidence goes on the list a reviewer sees, but only a
    # required field, a failed check or bad source text actually blocks auto
    # accept. Otherwise a softly inferred currency code is enough to send a
    # perfectly good document to a human, and the queue fills with noise.
    blockers: list[str] = []

    missing_hard = [n for n in config.hard_required if not record.get(n).is_present]
    for name in missing_hard:
        reasons.append(f"{name} is missing and is required")
        to_check.append(name)

    for issue in report.issues:
        if issue.severity == Severity.ERROR:
            reasons.append(issue.message)
            fields = [f for f in issue.fields if f in type(record).model_fields]
            to_check.extend(fields)
            blockers.extend(fields or [issue.code])

    for name in record.field_names():
        spec = FIELD_BY_NAME.get(name)
        extracted = record.get(name)
        required = bool(spec and spec.required)

        if not extracted.is_present:
            # A required field we never found is the single most likely way for
            # a wrong record to look clean: there are no values to disagree
            # with each other, so no check fails.
            if required and name not in missing_hard:
                reasons.append(f"{name} is required but was not found in the document")
                to_check.append(name)
                blockers.append(name)
            continue

        if extracted.confidence < config.field_review_threshold:
            to_check.append(name)
            if required:
                reasons.append(f"{name} was read with low confidence ({extracted.confidence:.2f})")
                blockers.append(name)

    if meta.text_quality < 0.6:
        reasons.append(
            f"the extracted text is poor quality ({meta.text_quality:.2f}), "
            "every value should be spot checked"
        )
        blockers.append("document_text_quality")

    if config.require_review_for_ocr and meta.ingest_method in (
        IngestMethod.PDF_OCR,
        IngestMethod.IMAGE_OCR,
    ):
        reasons.append("the text came from OCR, so the numbers need a second pair of eyes")
        blockers.append("document_ingest_method")

    if missing_hard or score < config.reject_threshold:
        decision = ReviewDecision.REJECT
    elif (
        score >= config.auto_accept_threshold
        and report.error_count == 0
        and not blockers
    ):
        decision = ReviewDecision.AUTO_ACCEPT
    else:
        decision = ReviewDecision.REVIEW

    if decision == ReviewDecision.AUTO_ACCEPT and not reasons:
        reasons.append("all checks passed and every field cleared the confidence threshold")

    return ReviewRouting(
        decision=decision,
        document_confidence=score,
        reasons=list(dict.fromkeys(reasons))[:12],
        fields_to_check=sorted(dict.fromkeys(to_check)),
    )
