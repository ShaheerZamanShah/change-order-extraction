"""Optional HTTP service.

    pip install "coextract[api]"
    coextract serve

POST /extract with a file, or POST /extract/text with a JSON body, and you get
the same ExtractionResult the CLI prints. The response status is 200 even for a
rejected document, because "we read this badly and here is why" is a successful
answer. Only an unreadable file is an error.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .config import PipelineConfig
from .ingest.loader import UnsupportedDocument
from .pipeline import ChangeOrderPipeline
from .schema import ExtractionResult

app = FastAPI(
    title="coextract",
    version="0.1.0",
    description="Structured extraction from construction change orders",
)

_pipeline = ChangeOrderPipeline(PipelineConfig())


class TextRequest(BaseModel):
    text: str
    name: str = "inline.txt"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract/text", response_model=ExtractionResult)
def extract_text(request: TextRequest) -> ExtractionResult:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    return _pipeline.run_text(request.text, request.name)


@app.post("/extract", response_model=ExtractionResult)
async def extract_upload(file: UploadFile = File(...)) -> ExtractionResult:
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        return _pipeline.run_path(temp_path)
    except UnsupportedDocument as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
