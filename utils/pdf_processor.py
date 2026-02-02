import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

__all__ = ["PDFProcessor"]


class PDFProcessor:
    """Lightweight PDF processor that loads pages and returns text chunks.

    This module-level class provides two convenience methods:
    - `process_pdf(uploaded_file)` expects a Streamlit-like UploadedFile with
      a `getvalue()` method.
    - `process_pdf_bytes(pdf_bytes)` accepts raw bytes (useful for API uploads).
    Both methods delegate to `_process_file_path` which uses `PyPDFLoader`.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, uploaded_file) -> list:
        """Process a Streamlit `UploadedFile`-like object and return chunks."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            return self._process_file_path(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def process_pdf_bytes(self, pdf_bytes: bytes) -> list:
        """Process raw PDF bytes and return chunks."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            return self._process_file_path(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _process_file_path(self, file_path: str) -> list:
        """Load the PDF at `file_path` and split pages into text chunks."""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        chunks = self.text_splitter.split_documents(pages)
        return chunks
