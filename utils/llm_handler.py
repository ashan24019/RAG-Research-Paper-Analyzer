import os
from langchain_openai import ChatOpenAI


class LLMHandler:

    def __init__(self, openai_api_key: str | None = None, model_name: str = "gpt-5-nano-2025-08-07", temperature: float = 0.0):
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=key)

        self.prompt_template = (
            "Use the following extracted passages from a research paper to answer the question.\n\n"
            "Passages:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite sources."
        )

    def _call_llm(self, prompt_text: str) -> str:
        # Prefer predict() which returns a string for chat-style wrappers.
        if hasattr(self.llm, "predict"):
            try:
                return self.llm.predict(prompt_text)
            except Exception:
                pass

        # Fallback to generate() and try to extract text
        if hasattr(self.llm, "generate"):
            try:
                res = self.llm.generate([prompt_text])
                if hasattr(res, "generations"):
                    gens = res.generations
                    if isinstance(gens, list) and len(gens) > 0 and len(gens[0]) > 0:
                        gen0 = gens[0][0]
                        text = getattr(gen0, "text", None) or getattr(gen0, "generation_text", None)
                        if text:
                            return text
                return str(res)
            except Exception:
                pass

        # Last resort: try calling the object
        try:
            return str(self.llm(prompt_text))
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def create_qa_chain(self, vectorstore):
        """Return a simple callable `qa(inputs)` where `inputs` is a dict with `query`.

        The callable returns a dict: {"result": str, "source_documents": list}
        """

        def simple_qa(inputs):
            query = inputs.get("query") if isinstance(inputs, dict) else inputs
            if not query:
                return {"result": "", "source_documents": []}

            # Retrieve documents
            try:
                docs = vectorstore.similarity_search(query, k=4)
            except Exception:
                try:
                    retr = vectorstore.as_retriever(search_kwargs={"k": 4})
                    docs = retr.get_relevant_documents(query)
                except Exception:
                    docs = []

            # Build context from retrieved docs
            context = "\n\n---\n\n".join(
                (getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)) for d in docs
            )

            prompt_text = self.prompt_template.format(context=context, question=query)

            llm_resp = self._call_llm(prompt_text)

            answer_text = llm_resp if isinstance(llm_resp, str) else str(llm_resp)

            return {"result": answer_text, "source_documents": docs}

        return simple_qa
