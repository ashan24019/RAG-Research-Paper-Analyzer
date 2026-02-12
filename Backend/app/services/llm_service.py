from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings
from app.services.vector_service import vector_service
import httpx

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """
        Initialize LLM service with OpenAI chat model.
        """
        self.chat_model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.2
        )

        self.prompt_template = self.build_prompt()

        logger.info(f"LLMService initialized with model: {settings.chat_model}")
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        formatted_chunks = []
        for chunk in chunks:
            # Support multiple possible result shapes from the vector search
            # older code expected `text`; search returns `document`.
            text = chunk.get("text") or chunk.get("document") or ""
            metadata = chunk.get("metadata", {})
            # prefer page_number if present
            page = metadata.get("page_number") or metadata.get("page") or "N/A"
            source_info = f"(Source: {metadata.get('source', 'unknown')}, Page: {page})"
            formatted_chunks.append(f"{text}\n{source_info}\n")
        
        return "\n\n---\n\n".join(formatted_chunks)
    
    def build_prompt(self) -> PromptTemplate:

        template = """You are an AI assistant specialized in analyzing research papers. Your role is to provide accurate, well-cited answers based on the provided context.
        

        context from the research paper:
        {context}

        Question: {question}

        Instructions:
            - Answer based ONLY on the provided context above
            - If the information needed to answer is not in the context, clearly state: "The provided excerpts do not contain information about [topic]."
            - Be specific and reference relevant parts of the paper
            - When possible, mention the excerpt number or page number in your answer
            - Use clear, academic language
            - Do not make up, infer, or assume information not explicitly stated in the context
            - If the context is ambiguous, acknowledge the ambiguity
        
        Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
            )
    
    async def generate_answer(
            self,
            question: str,
            document_id: Optional[str] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM based on question and optional document context.
        """
        try:
            chunks = await vector_service.search(
                query=question,
                document_id=document_id,
                n_results=settings.max_chunks_retrieval
            )

            if not chunks:
                    logger.warning("No relevant chunks found")
                    return {
                        "answer": "I couldn't find relevant information in the uploaded paper(s) to answer your question. Please try rephrasing your question or upload a different paper.",
                        "sources": [],
                        "chunks_used": 0,
                        "document_id": document_id
                    }
            context = self.format_context(chunks)

                # Step 4: Build complete prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )

            # Step 5: Get answer from LLM via OpenAI Chat Completions API
            logger.info("Calling OpenAI Chat Completions for answer generation...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "model": settings.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": float(settings.temperature)
                }
                headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
                resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                # Extract assistant text
                answer = data["choices"][0]["message"]["content"]

            sources = []
            for chunk in chunks:
                text = chunk.get("text") or chunk.get("document") or ""
                metadata = chunk.get("metadata") or {}
                # Ensure page is an int for the response model
                page_val = metadata.get("page_number") or metadata.get("page")
                try:
                    page = int(page_val) if page_val is not None else 0
                except Exception:
                    page = 0

                # Ensure distance is float
                try:
                    distance = float(chunk.get("distance", 0.0))
                except Exception:
                    distance = 0.0

                sources.append({
                    "text": text,
                    "page": page,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "distance": distance
                })
                
            return {
                "answer": answer,
                "sources": sources,
                # Provide both keys to satisfy route logging and Pydantic model
                "chunks_used": len(chunks),
                "chunk_used": len(chunks),
                "document_id": document_id
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise Exception(f"Failed to generate answer: {str(e)}")
        
    async def generate_summary(
        self,
        document_id: str) -> Dict[str, Any]:

        summary_question = """Please provide a comprehensive summary of this research paper, including:
        1. Main research question or objective
        2. Methodology used
        3. Key findings
        4. Conclusions and implications"""

        return await self.generate_answer(
            question=summary_question,
            document_id=document_id
            
        )

llm_service = LLMService()


# Module-level convenience wrapper so callers that import the module
# (e.g. `from app.services import llm_service`) can call `llm_service.generate_answer(...)`
# as expected by the routes.
async def generate_answer(question: str, document_id: Optional[str] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    return await llm_service.generate_answer(question=question, document_id=document_id, conversation_history=conversation_history)