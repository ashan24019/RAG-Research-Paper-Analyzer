from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentMetadata(BaseModel):
    title: str
    authors: str = "Unknown"
    year: Optional[int] = None
    total_pages: int


class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    total_chunks: int
    metadata: DocumentMetadata
    message: str
    upload_date: str


class Source(BaseModel):
    text: str
    page: int
    chunk_id: str
    distance: float = Field(..., description="Similarity distance (lower = more similar)")


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    chunk_used: int
    document_id: Optional[str] = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    upload_date: str
    total_chunks: int
    metadata: DocumentMetadata


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    success: bool
    document_id: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None