from typing import Literal

from pydantic import BaseModel, Field


class UploadItemResult(BaseModel):
    filename: str
    status: Literal["processed", "duplicate", "failed"]
    message: str
    document_id: str | None = None
    page_count: int = 0
    chunk_count: int = 0


class UploadResponse(BaseModel):
    results: list[UploadItemResult]
    total_documents: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    filename: str
    page_number: int | None = None
    text: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    used_chunks: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    size_bytes: int
    uploaded_at: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    total_documents: int
    total_chunks: int
    embedding_model: str
