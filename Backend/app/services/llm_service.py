from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings
from app.services.vector_service import VectorService

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
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
            source_info = f"(Source: {metadata.get('source', 'unknown')}, Page: {metadata.get('page', 'N/A')})"
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
            chunks = await VectorService.search(
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

            # Step 5: Get answer from LLM
            logger.info("Calling LLM for answer generation...")
            response = self.llm.invoke(prompt)
            answer = response.content

            sources = [
                {
                    "text": chunk["text"],
                    "page": chunk["metadata"].get("page_number", "Unknown"),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "distance": chunk.get("distance", 0)
                }
                for chunk in chunks
            ]
                
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(chunks),
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