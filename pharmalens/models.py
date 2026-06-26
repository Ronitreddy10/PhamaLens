from typing import Any
import uuid

from pydantic import BaseModel, Field


class DocMetadata(BaseModel):
    """Document-level metadata extracted during classification."""

    doc_id: str
    filename: str
    filepath: str
    doc_type: str
    regulatory_body: str
    product_name: str = "Unknown"
    api: str = "Unknown"
    formulation: str = "Unknown"
    route: str = "Unknown"
    version_date: str | None = None
    total_pages: int = 0


class ChunkMetadata(BaseModel):
    """Chunk and its denormalized document metadata."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    filename: str
    doc_type: str
    regulatory_body: str
    product_name: str
    api: str
    formulation: str
    route: str
    version_date: str | None = None
    section_title: str = ""
    section_number: str = ""
    page_number: int = 0
    chunk_index: int = 0
    token_count: int = 0
    text: str
    chunk_kind: str = "text"
    dominant_modality: str = "NEUTRAL"


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    delta_alerts: list[str] = Field(default_factory=list)
