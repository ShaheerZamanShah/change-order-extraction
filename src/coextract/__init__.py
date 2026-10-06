"""coextract: structured extraction from messy construction change orders."""

from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import ChangeOrderPipeline, extract_file, extract_many
from .schema import (
    SCHEMA_VERSION,
    ChangeOrderRecord,
    ExtractedField,
    ExtractionResult,
    LineItem,
    ReviewDecision,
    ValidationIssue,
)

__version__ = "0.1.0"

__all__ = [
    "ChangeOrderPipeline",
    "ChangeOrderRecord",
    "DEFAULT_CONFIG",
    "ExtractedField",
    "ExtractionResult",
    "LineItem",
    "PipelineConfig",
    "ReviewDecision",
    "SCHEMA_VERSION",
    "ValidationIssue",
    "extract_file",
    "extract_many",
    "__version__",
]
