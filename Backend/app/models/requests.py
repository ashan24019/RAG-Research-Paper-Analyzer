from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ChatRequest(BaseModel):
    """Request model for asking questions to the chatbot."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Question to ask about the document(s)"
    )

    document_id: Optional[str] = Field(
        None,
        description="Optional: Specific document ID to search. If not provided, searches all documents."
    )

    @field_validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the key findings of the report?",
                "document_id": "a1b2c3d4e5f6g7h8"
            }
        }