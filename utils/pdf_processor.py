from langchain_community.document_loaders import PDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os


class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, uploaded_file) :
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            # Load the PDF document
            loader = PDFLoader(temp_file_path)
            pages = loader.load()

            # Split the documents into chunks

            chunks = self.text_splitter.split_documents(pages)

            return chunks
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)