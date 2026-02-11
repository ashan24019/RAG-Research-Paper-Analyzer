from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import asyncio
import tempfile
import os
import hashlib
from datetime import datetime
import re
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size =chunk_size or  settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " "]  # Try to split on natural boundaries
        )

        logger.info(f"PDFProcessor initialized with chunk_size={self.chunk_size} and chunk_overlap={self.chunk_overlap}")


    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA256 hash of the file for unique identification."""
        return hashlib.sha256(file_content).hexdigest()
    


    
    def extract_metadata(self, pages: List, filename: str) -> Dict[str, Any]:

        """
        Extract metadata from the document.
        Uses simple heuristics to find title, authors, etc.

        """
        metadata = {
            "title": filename.replace(".pdf", ""),
            "authors": "Unknown",
            "year": None,
            "total_pages": len(pages)
        }

        if not pages:
            return metadata
        
        # Get first page content for metadata extraction
        first_page = pages[0].page_content
        lines = first_page.split("\n")

        # Try to extract title (usually first non-empty line)
        for line in lines[: 10]:
            line = line.strip()
            if len(line) > 10 and not line.isupper():
                metadata["title"] = line
                break
        
        # Try to extract year (look for 4-digit number between 1900-2099)
        year_pattern = r'\b(19|20)\d{2}\b'
        year_match = re.search(year_pattern, first_page)
        if year_match:
            metadata["year"] = int(year_match.group(0))
        
        logger.info(f"Extracted metadata: title='{metadata['title'][:50]}...', pages={metadata['total_pages']}")
        return metadata
    
    async def process_pdf(self, file, user_id: str = "default") -> Dict[str, Any]:
        """
        Main method to process a PDF file.
        
        Steps:
        1. Save uploaded file temporarily
        2. Calculate file hash
        3. Extract text using PyPDFLoader
        4. Split text into chunks
        5. Extract metadata
        6. Clean up temporary file
        
        Args:
            file: Uploaded file object (from FastAPI)
            user_id: ID of user uploading (for future multi-user support)
            
        Returns:
            Dictionary containing:
                - document_id: Unique identifier (file hash)
                - filename: Original filename
                - chunks: List of text chunks with metadata
                - metadata: Extracted document metadata
                - user_id: User who uploaded
                - upload_date: Timestamp
        """

        temp_path = None

        try:
            # Read file content
            file_content = await file.read()

            # Calculate hash for unique ID and duplicate detection
            file_hash = self.calculate_file_hash(file_content)
            document_id = file_hash[:16]  # Use first 16 chars as ID

            logger.info(f"Processing PDF: {file.filename} (ID: {document_id})")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                temp_path = tmp.name

             # Load PDF and extract text
            loader = PyPDFLoader(temp_path)
            # loader.load() is blocking; run in a thread to avoid blocking the event loop
            pages = await asyncio.to_thread(loader.load)

            if not pages:
                raise ValueError("No text extracted from PDF.")
            
            logger.info(f"Loaded {len(pages)} pages from PDF")

            # splitting can be CPU-bound; run in a thread
            chunks = await asyncio.to_thread(self.text_splitter.split_documents, pages)
            logger.info(f"Split into {len(chunks)} chunks")

            #Extract metadata
            metadata = self.extract_metadata(pages, file.filename)

            processed_chunks = []
            for idx, chunk in enumerate(chunks):
                processed_chunks.append({
                    "text": chunk.page_content,
                    "metadata": {
                        "chunk_index": idx,
                        "page_number": chunk.metadata.get("page", 0),
                        "document_id": document_id,
                        "source": file.filename
                    }
                })

            # Return processed data
            result = {
                "document_id": document_id,
                "filename": file.filename,
                "chunks": processed_chunks,
                "metadata": metadata,
                "user_id": user_id,
                "upload_date": datetime.utcnow().isoformat(),
                "total_chunks": len(processed_chunks)
            }

            logger.info(f"PDF processing complete: {document_id}")

            return result
        
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise Exception(f"Failed to process PDF: {str(e)}")
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
 
pdf_processor = PDFProcessor()