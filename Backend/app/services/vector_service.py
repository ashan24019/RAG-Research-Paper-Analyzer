from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings
import asyncio

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self):
        """
        Initialize vector service with OpenAI embeddings and ChromaDB client.
        """
        self.embeddings = OpenAIEmbeddings(
            model = settings.embedding_model,
            api_key=settings.openai_api_key 
        )

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )

        logger.info(f"VectorService initialized with model: {settings.embedding_model}")
        logger.info(f"ChromaDB path: {settings.chroma_persist_dir}")

    
    async def store_document(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:

        try:
            logger.info(f"Storing document {document_id} with {len(chunks)} chunks")

            # Create collection for this document
            collection_name = f"doc_{document_id}"
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"document_metadata": str(metadata)}
            )

            # Prepare data for ChromaDB
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk.get("metadata", {}) for chunk in chunks]
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]

            # Generate embeddings (calls OpenAI API) - run in thread to avoid blocking
            logger.info(f"Generating embeddings for document {document_id}")
            embeddings_list = await asyncio.to_thread(self.embeddings.embed_documents, texts)
            logger.info(f"Generated {len(embeddings_list)} embeddings")

            # Add to collection - run in thread if blocking
            await asyncio.to_thread(
                collection.add,
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully stored document {document_id}")

            return {
                "success": True,
                "document_id": document_id,
                "chunks_stored": len(chunks),
                "collection_name": collection_name
            }

        except Exception:
            logger.exception(f"Error storing document {document_id}")
            raise
    
    async def search(
            self,
            query: str,
            document_id: Optional[str] = None,
            n_results: int = None
    ) -> List[Dict[str, Any]]:
        
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: User's question or search query
            document_id: Optional - search in specific document only
            n_results: Number of results to return (defaults to settings.max_chunks_retrieval)
            
        Returns:
            List of matching chunks with text, metadata, and similarity scores

        """

        try:
            n_results = n_results or settings.max_chunks_retrieval
            logger.info(f"Searching for: '{query[:50]}...' (max results: {n_results})")

            # Generate embedding for the query
            query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)

            if document_id:
                collection_name = f"doc_{document_id}"
                try:
                    collection = self.chroma_client.get_collection(collection_name)
                    collections = [collection]
                except Exception as e:
                    logger.error(f"Collection for document {document_id} not found: {str(e)}")
                    return []
            else:
                collections = self.chroma_client.list_collections()

            all_results = []

            for collection in collections:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                )

                # Process results
                if results and results.get('documents') and results['documents'][0]:
                    for i in range(len(results['documents'][0])):
                        all_results.append({
                            "document": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "distance": results['distances'][0][i],
                            "chunk_id": results['ids'][0][i],
                            "collection_name": getattr(collection, 'name', None)
                        })
            all_results.sort(key=lambda x: x['distance'])
            all_results = all_results[:n_results]

            return all_results
        
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise Exception(f"Search failed: {str(e)}")
        

    async def delete_document(self, document_id: str) -> bool:

        try:
            collection_name = f"doc_{document_id}"
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
        
    
    def list_document(self) -> List[str]:
        collections = self.chroma_client.list_collections()
        document_ids = [col.name.replace("doc_", "") for col in collections]
        return document_ids

    def get_collection_stats(self, document_id: str) -> Dict[str, Any]:
        collection_name = f"doc_{document_id}"
        collection = self.chroma_client.get_collection(collection_name)

        return {
            "document_id": document_id,
            "total_chunks": collection.count(),
            "collection_name": collection_name
        }

vector_service = VectorService()
