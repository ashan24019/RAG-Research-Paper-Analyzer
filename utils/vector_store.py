import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


class VectorStoreManager:
    """Manage a Chroma vector store using OpenAI embeddings.

    The manager will read `OPENAI_API_KEY` from the environment if no key
    is provided explicitly.
    """

    def __init__(self, persist_directory: str = "./chroma_db", openai_api_key: str | None = None, embedding_model: str = "text-embedding-3-small"):
        self.persist_directory = persist_directory
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        # OpenAIEmbeddings will use the provided key (or the environment variable)
        self.embeddings = OpenAIEmbeddings(openai_api_key=key, model=embedding_model)
        self.vectorstore = None

    def create_vectorstore(self, documents):
        """Create and persist a Chroma vectorstore from documents."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        return self.vectorstore

    def load_vectorstore(self):
        """Load an existing Chroma vectorstore from the persist directory."""
        if self.vectorstore is None:
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return self.vectorstore

    def similarity_search(self, query, k=4):
        """Search for similar documents."""
        if not self.vectorstore:
            try:
                self.load_vectorstore()
            except Exception:
                return []
        return self.vectorstore.similarity_search(query, k=k)
