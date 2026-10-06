"""Output schema for the change-order extraction pipeline.

Every extracted field is wrapped in :class:`ExtractedField` so that a value,
its confidence, the evidence it came from, and the runner-up candidates all
travel together. Downstream consumers can therefore route a document to human
review on a per-field basis instead of trusting or discarding the whole record.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

SCHEMA_VERSION = "1.0.0"


class ChangeOrderStatus(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    VOID = "void"
    UNKNOWN = "unknown"


class ReasonCode(str, Enum):
    OWNER_REQUESTED = "owner_requested"
    DESIGN_CHANGE = "design_change"
    DESIGN_ERROR_OMISSION = "design_error_omission"
    UNFORESEEN_CONDITION = "unforeseen_condition"
    CODE_OR_AUTHORITY = "code_or_authority"
    WEATHER_DELAY = "weather_delay"
    ALLOWANCE_ADJUSTMENT = "allowance_adjustment"
    SCOPE_DEDUCTION = "scope_deduction"
    OTHER = "other"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IngestMethod(str, Enum):
    TEXT_FILE = "text_file"
    PDF_TEXT_LAYER = "pdf_text_layer"
    PDF_OCR = "pdf_ocr"
    IMAGE_OCR = "image_ocr"


class Evidence(BaseModel):
    """Where a value was found, so a reviewer can jump straight to it."""

    model_config = ConfigDict(extra="forbid")

    page: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    snippet: str = ""
    matcher: str = ""


class Alternative(BaseModel):
    """A candidate that lost arbitration. Kept for audit and review UIs."""

    model_config = ConfigDict(extra="forbid")

    value: Any = None
    score: float = 0.0
    extractor: str = ""
    matcher: str = ""
    snippet: str = ""


class ExtractedField(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: Optional[T] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extractor: Optional[str] = None
    raw_text: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def is_present(self) -> bool:
        return self.value is not None


def empty_field() -> Any:
    """Default factory so declaring 20 fields stays readable."""
    return Field(default_factory=ExtractedField)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    description: str = ""
    quantity: Optional[Decimal] = None
    unit: Optional[str] = None
    unit_price: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    page: Optional[int] = None
    raw_row: str = ""


class Signature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = ""
    name: str = ""
    signed: Optional[bool] = None
    signed_date: Optional[date] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Severity
    message: str
    fields: list[str] = Field(default_factory=list)
    observed: Optional[str] = None
    expected: Optional[str] = None
    auto_repaired: bool = False


class ChangeOrderRecord(BaseModel):
    """The structured change order itself.

    ``validate_assignment`` is on deliberately. The pipeline builds this record
    field by field, and without it a wrongly typed value (a free text reason
    where an enum belongs, say) would travel all the way to the output JSON
    unchallenged. Here it fails at the assignment that caused it.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Identity
    change_order_number: ExtractedField[str] = empty_field()
    change_order_title: ExtractedField[str] = empty_field()
    project_name: ExtractedField[str] = empty_field()
    project_number: ExtractedField[str] = empty_field()
    contract_number: ExtractedField[str] = empty_field()

    # Parties
    owner_name: ExtractedField[str] = empty_field()
    contractor_name: ExtractedField[str] = empty_field()
    architect_name: ExtractedField[str] = empty_field()

    # Dates and status
    date_issued: ExtractedField[date] = empty_field()
    date_executed: ExtractedField[date] = empty_field()
    status: ExtractedField[ChangeOrderStatus] = empty_field()
    reason_code: ExtractedField[ReasonCode] = empty_field()
    scope_description: ExtractedField[str] = empty_field()

    # Money
    currency: ExtractedField[str] = empty_field()
    original_contract_sum: ExtractedField[Decimal] = empty_field()
    net_previous_changes: ExtractedField[Decimal] = empty_field()
    contract_sum_before: ExtractedField[Decimal] = empty_field()
    change_amount: ExtractedField[Decimal] = empty_field()
    contract_sum_after: ExtractedField[Decimal] = empty_field()
    retainage_percent: ExtractedField[Decimal] = empty_field()

    # Schedule
    schedule_impact_days: ExtractedField[int] = empty_field()
    original_completion_date: ExtractedField[date] = empty_field()
    revised_completion_date: ExtractedField[date] = empty_field()

    # Repeating structures
    line_items: list[LineItem] = Field(default_factory=list)
    signatures: list[Signature] = Field(default_factory=list)

    def field_names(self) -> list[str]:
        return [n for n in type(self).model_fields if n not in ("line_items", "signatures")]

    def get(self, name: str) -> ExtractedField:
        return getattr(self, name)


class DocumentMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = ""
    source_name: str = ""
    sha256: str = ""
    byte_size: int = 0
    page_count: int = 0
    ingest_method: IngestMethod = IngestMethod.TEXT_FILE
    char_count: int = 0
    text_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool = True
    error_count: int = 0
    warning_count: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    # Fields that took part in a cross check that balanced. These earn a
    # confidence bonus because an independent number agreed with them.
    corroborated_fields: list[str] = Field(default_factory=list)


class ReviewDecision(str, Enum):
    AUTO_ACCEPT = "auto_accept"
    REVIEW = "review"
    REJECT = "reject"


class ReviewRouting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecision = ReviewDecision.REVIEW
    document_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    fields_to_check: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    document: DocumentMeta = Field(default_factory=DocumentMeta)
    change_order: ChangeOrderRecord = Field(default_factory=ChangeOrderRecord)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    review: ReviewRouting = Field(default_factory=ReviewRouting)
    extractors_used: list[str] = Field(default_factory=list)
    runtime_ms: int = 0

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=False)

    def flat_values(self) -> dict[str, Any]:
        """Field name to plain value, for eval and spreadsheet export."""
        out: dict[str, Any] = {}
        for name in self.change_order.field_names():
            value = self.change_order.get(name).value
            if isinstance(value, Enum):
                value = value.value
            if isinstance(value, Decimal):
                value = str(value)
            if isinstance(value, date):
                value = value.isoformat()
            out[name] = value
        return out
