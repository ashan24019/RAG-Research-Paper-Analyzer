from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        # Use free sentence transformers model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None

    def create_vectorstore(self, documents):
        """Create vector store from documents"""
        self.vectorstore = Chroma.from_documents(
            documents = documents,
            embedding = self.embeddings,
            persist_directory=self.persist_directory
        )
        return self.vectorstore

    def similarity_search(self, query, k=4):
        """Search for similar documents"""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)
