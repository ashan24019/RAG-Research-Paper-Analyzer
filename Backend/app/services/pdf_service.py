from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
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
            separators=["\n\n", "\n", " ", ""] # Try to split on natural boundaries
        )

        logger.info(f"PDFProcessor initialized with chunk_size={self.chunk_size} and chunk_overlap={self.chunk_overlap}")


    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA256 hash of the file for unique identification."""
        return hashlib.sha256(file_content).hexdigest()