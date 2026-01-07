from langchain_groq import ChatGroq
from langchain_community.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os


class LLMHandler:
    def __init__(self, api_key):
        # Groq provides free, fast inference
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )

        # Custom prompt for research papers
        self.prompt_template = """You are an AI assistant specialized in analyzing research papers.
        Use the following context from the paper to answer the question.
        If you don't know the answer, say so. Don't make up information.

        Context: {context}

        Question: {question}

        Answer: Let me analyze this based on the paper..."""

    def create_qa_chain(self, vectorstore):
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return qa_chain
