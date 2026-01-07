"""Robust PDF processing with fallbacks.

This module prefers langchain's `PyPDFLoader` when available, falls back
to `langchain_community` loaders, and finally uses `pypdf` (PdfReader)
if none of the langchain loaders are present. It returns a list of
document chunks via `RecursiveCharacterTextSplitter`.
"""
import tempfile
import os

# Try to import a LangChain-style PDF loader from a few possible packages.
LoaderClass = None
try:
    from langchain.document_loaders import PyPDFLoader
    LoaderClass = PyPDFLoader
except Exception:
    try:
        # Some environments have langchain_community with a similar loader
        from langchain_community.document_loaders import PyPDFLoader
        LoaderClass = PyPDFLoader
    except Exception:
        try:
            from langchain_community.document_loaders import PDFLoader
            LoaderClass = PDFLoader
        except Exception:
            LoaderClass = None

from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, uploaded_file):
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            # If we have a LoaderClass (from langchain or langchain_community), use it
            if LoaderClass is not None:
                loader = LoaderClass(temp_file_path)
                # Most LangChain loaders expose `load()` which returns a list of Documents
                pages = loader.load()
            else:
                # Fallback: use pypdf (PdfReader) to extract text per page and
                # create simple Document-like objects compatible with
                # `RecursiveCharacterTextSplitter.split_documents` which expects
                # objects with `.page_content` and `.metadata` attributes.
                try:
                    from pypdf import PdfReader
                except Exception:
                    raise RuntimeError(
                        "No compatible langchain PDF loader found and 'pypdf' is not installed. "
                        "Install with `pip install pypdf` or install `langchain`/`langchain_community`."
                    )

                class SimpleDocument:
                    def __init__(self, text, metadata=None):
                        self.page_content = text or ""
                        self.metadata = metadata or {}

                reader = PdfReader(temp_file_path)
                pages = []
                for i, p in enumerate(reader.pages):
                    text = p.extract_text() or ""
                    pages.append(SimpleDocument(text, {"page": i + 1}))

            # Split the documents into chunks
            chunks = self.text_splitter.split_documents(pages)

            return chunks
        except Exception as e:
            raise RuntimeError(f"Failed to load or process PDF: {e}")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass