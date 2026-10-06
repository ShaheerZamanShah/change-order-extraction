"""Pipeline configuration.

Thresholds live here rather than being scattered through the code because they
are the knobs you actually turn when you tune the pipeline against your own
document mix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    # --- money comparison -------------------------------------------------
    # A change order that is out by a cent is a rounding artefact. One that is
    # out by a few hundred dollars is a real problem, and it stays a real
    # problem on a fifty million dollar contract. So the relative term exists
    # only to absorb percentage rounding on markup lines, and it is capped in
    # absolute terms rather than being allowed to scale with the contract.
    money_tolerance: Decimal = Decimal("1.00")
    relative_money_tolerance: Decimal = Decimal("0.0005")
    max_money_tolerance: Decimal = Decimal("25.00")

    # --- review routing ---------------------------------------------------
    auto_accept_threshold: float = 0.85
    reject_threshold: float = 0.40
    field_review_threshold: float = 0.70

    # --- ingest -----------------------------------------------------------
    allow_ocr: bool = True
    dayfirst_dates: bool = False
    # A document that reached us through OCR never gets auto accepted, however
    # well it scored. OCR failures are silent and systematic rather than
    # random: a misread digit produces a perfectly plausible number, and if the
    # same digit is misread consistently the arithmetic checks balance anyway.
    require_review_for_ocr: bool = True

    # --- extractors -------------------------------------------------------
    enable_llm: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_base_url: Optional[str] = None
    max_llm_chars: int = 18000

    # --- post processing --------------------------------------------------
    enable_repair: bool = True
    min_confidence_to_keep: float = 0.15

    # --- confidence penalties applied per validation outcome --------------
    error_penalty: float = 0.35
    warning_penalty: float = 0.12
    derived_value_ceiling: float = 0.80

    # Fields that must be present for the record to be usable at all.
    hard_required: tuple[str, ...] = field(
        default_factory=lambda: ("change_order_number", "change_amount")
    )

    def money_tolerance_for(self, magnitude: Decimal) -> Decimal:
        scaled = (self.relative_money_tolerance * abs(magnitude)).quantize(Decimal("0.01"))
        return max(self.money_tolerance, min(self.max_money_tolerance, scaled))

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("money_tolerance", "relative_money_tolerance", "max_money_tolerance"):
            if key in data:
                data[key] = Decimal(str(data[key]))
        if "hard_required" in data:
            data["hard_required"] = tuple(data["hard_required"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        out = asdict(self)
        for key in ("money_tolerance", "relative_money_tolerance", "max_money_tolerance"):
            out[key] = str(getattr(self, key))
        out["hard_required"] = list(self.hard_required)
        return out


DEFAULT_CONFIG = PipelineConfig()
