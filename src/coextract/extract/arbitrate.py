"""Arbitration: turn many candidates into one answer per field.

Candidates are clustered by normalised value. A cluster that several
independent patterns or extractors agree on gets a bonus, because independent
agreement is the cheapest reliable signal we have. The gap between the winning
cluster and the runner up then decides how much of that score survives: two
clusters that are neck and neck mean we genuinely do not know, and the
confidence should say so.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Optional

from ..fields import FIELD_BY_NAME
from ..ingest import Document
from ..normalize.text import snippet_around
from ..schema import Alternative, Evidence, ExtractedField
from .base import Candidate

AGREEMENT_PER_MATCHER = 0.05
MAX_AGREEMENT_BONUS = 0.12
CROSS_EXTRACTOR_BONUS = 0.06
TEXT_MERGE_RATIO = 0.90

# Confidence is scaled by document text quality. A perfect text layer keeps the
# full score; a bad scan caps everything well below auto accept.
QUALITY_FLOOR = 0.55


@dataclass
class Cluster:
    key: str
    value: Any
    members: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate:
        return max(self.members, key=lambda c: c.score)

    @property
    def matchers(self) -> set[str]:
        return {m.matcher for m in self.members}

    @property
    def extractors(self) -> set[str]:
        return {m.extractor for m in self.members}

    def combined_score(self) -> float:
        base = self.best.score
        bonus = min(MAX_AGREEMENT_BONUS, AGREEMENT_PER_MATCHER * (len(self.matchers) - 1))
        if len(self.extractors) > 1:
            bonus += CROSS_EXTRACTOR_BONUS
        return min(0.99, base + bonus)


def _cluster(candidates: list[Candidate], kind: str) -> list[Cluster]:
    clusters: dict[str, Cluster] = {}
    for cand in candidates:
        key = cand.value_key()
        if kind in ("text", "string", "block") and key not in clusters:
            # Merge near duplicate names and titles ("Acme Builders, Inc" and
            # "Acme Builders Inc" are the same company).
            match = next(
                (
                    existing
                    for existing in clusters
                    if difflib.SequenceMatcher(None, existing, key).ratio() >= TEXT_MERGE_RATIO
                ),
                None,
            )
            if match is not None:
                key = match
        cluster = clusters.get(key)
        if cluster is None:
            cluster = Cluster(key=key, value=cand.value)
            clusters[key] = cluster
        cluster.members.append(cand)
        if cand.score > max((m.score for m in cluster.members if m is not cand), default=-1.0):
            cluster.value = cand.value
    return list(clusters.values())


def _margin_factor(top: float, runner_up: Optional[float]) -> tuple[float, Optional[str]]:
    if runner_up is None:
        return 1.0, None
    margin = top - runner_up
    if margin < 0.03:
        return 0.78, "competing_value_with_near_equal_support"
    if margin < 0.10:
        return 0.90, "competing_value_with_similar_support"
    if margin < 0.20:
        return 0.96, None
    return 1.0, None


def arbitrate_field(
    field_name: str,
    candidates: list[Candidate],
    doc: Document,
) -> ExtractedField:
    spec = FIELD_BY_NAME.get(field_name)
    kind = spec.kind if spec else "text"
    relevant = [c for c in candidates if c.field_name == field_name and c.value is not None]
    if not relevant:
        return ExtractedField(value=None, confidence=0.0, notes=["not_found"])

    clusters = sorted(_cluster(relevant, kind), key=lambda c: c.combined_score(), reverse=True)
    winner = clusters[0]
    runner_up = clusters[1].combined_score() if len(clusters) > 1 else None

    quality_factor = QUALITY_FLOOR + (1.0 - QUALITY_FLOOR) * doc.meta.text_quality
    margin_factor, margin_note = _margin_factor(winner.combined_score(), runner_up)
    confidence = winner.combined_score() * quality_factor * margin_factor

    best = winner.best
    notes = list(dict.fromkeys(best.notes))
    if margin_note:
        notes.append(margin_note)
    if len(winner.matchers) > 1:
        notes.append(f"supported_by_{len(winner.matchers)}_patterns")
    if len(winner.extractors) > 1:
        notes.append("agreed_by_multiple_extractors")

    evidence: list[Evidence] = []
    seen_spans: set[tuple[Optional[int], Optional[int]]] = set()
    for member in sorted(winner.members, key=lambda c: c.score, reverse=True):
        span = (member.start, member.end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        snippet = (
            snippet_around(doc.text, member.start, member.end)
            if member.start is not None and member.end is not None
            else member.raw_text
        )
        evidence.append(
            Evidence(
                page=member.page,
                start=member.start,
                end=member.end,
                snippet=snippet,
                matcher=member.matcher,
            )
        )
        if len(evidence) >= 3:
            break

    alternatives = [
        Alternative(
            value=_plain(other.value),
            score=round(other.combined_score(), 3),
            extractor=other.best.extractor,
            matcher=other.best.matcher,
            snippet=(
                snippet_around(doc.text, other.best.start, other.best.end)
                if other.best.start is not None and other.best.end is not None
                else other.best.raw_text
            ),
        )
        for other in clusters[1:4]
    ]

    return ExtractedField(
        value=winner.value,
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        extractor=best.extractor,
        raw_text=best.raw_text,
        evidence=evidence,
        alternatives=alternatives,
        notes=notes,
    )


def _plain(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
