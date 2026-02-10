from fastapi import APIRouter, HTTPException, UploadFile, File, status
from app.models.responses import UploadResponse, DocumentMetadata, ErrorResponse
from app.services.pdf_service import PDFProcessor
from app.services.vector_service import VectorService
import logging

logger = logging.getLogger(__name__)

# Initialize router and services
router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload")
):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        

        logger.info(f"Uploading document: {file.filename}")

        pdf_result = await PDFProcessor.process_pdf(file, user_id="default")

        document_id = pdf_result["document_id"]
        logger.info(f"Document processed: {document_id}")

        # Step 2: Store in vector database (create embeddings)

        vector_result = await VectorService.store_document(
            document_id=document_id,
            text_chunks=pdf_result["chunks"],
            metadata=pdf_result["metadata"]
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
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to process document: {str(e)}"
        )