import logging

from fastapi import APIRouter, HTTPException

from app.services import llm_service
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    try:
        logger.info(f"Processing chat request: {request.question[:50]}...")

        # Generate answer using LLM service
        result = await llm_service.generate_answer(
            question=request.question,
            document_id=request.document_id
        )

        logger.info(f"Answer generated using {result['chunks_used']} chunks")

        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")