from fastapi import APIRouter, HTTPException, UploadFile, File, status
from app.models.responses import UploadResponse, DocumentMetadata, ErrorResponse
from app.services.pdf_service import pdf_processor
from app.services.vector_service import vector_service
import logging

logger = logging.getLogger(__name__)

# Initialize router and services
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload")
):
    try:
        # Basic filename validation (case-insensitive)
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Basic content-type hint check if client provided it
        if getattr(file, "content_type", None) and "pdf" not in file.content_type.lower():
            raise HTTPException(status_code=415, detail="Unsupported media type; expected PDF")

        logger.info(f"Uploading document: {file.filename}")

        pdf_result = await pdf_processor.process_pdf(file, user_id="default")

        document_id = pdf_result["document_id"]
        logger.info(f"Document processed: {document_id}")

        # Step 2: Store in vector database (create embeddings)
        vector_result = await vector_service.store_document(
            document_id=document_id,
            chunks=pdf_result["chunks"],
            metadata=pdf_result["metadata"],
        )

        logger.info(f"Embeddings stored: {vector_result['chunks_stored']} chunks")

        # Step 3: Prepare response
        response = UploadResponse(
            success=True,
            document_id=document_id,
            filename=pdf_result['filename'],
            total_chunks=pdf_result['total_chunks'],
            metadata=DocumentMetadata(**pdf_result['metadata']),
            message="Document uploaded and processed successfully",
            upload_date=pdf_result['upload_date']
        )
        logger.info(f"Upload complete: {document_id}")
        return response

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error uploading document")
        # Return generic message to avoid leaking internals
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document",
        )


@router.get("/")
async def list_documents():
    try:
        documents = []
        for doc_id in vector_service.list_document():
            try:
                stats = vector_service.get_collection_stats(doc_id)
                documents.append(stats)
            except Exception:
                logger.warning("Skipping document stats for %s", doc_id)
        return {"documents": documents, "total": len(documents)}
    except Exception:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=500, detail="Failed to list documents")