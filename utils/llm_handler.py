from langchain_groq import ChatGroq


class LLMHandler:
    def __init__(self, api_key):
        # Groq provides free, fast inference
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )

        # Custom prompt for research papers
        self.prompt_template = (
            "Use the following extracted passages from a research paper to answer the question.\n\n"
            "Passages:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite sources."
        )

    def _extract_text_from_llm_response(self, resp):
        # support common return shapes: plain str, object with .content, LLMResult with .generations, dict, list
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        # langchain-style LLMResult with .generations -> list[list[Generation]] where Generation has .text
        if hasattr(resp, "generations"):
            try:
                gens = resp.generations
                if isinstance(gens, list) and len(gens) > 0 and len(gens[0]) > 0:
                    gen0 = gens[0][0]
                    if hasattr(gen0, "text"):
                        return gen0.text or ""
                    if hasattr(gen0, "generation_text"):
                        return gen0.generation_text or ""
            except Exception:
                pass
        # simple chat objects with .content
        if hasattr(resp, "content"):
            return getattr(resp, "content") or ""
        # some SDKs return dicts
        if isinstance(resp, dict):
            for k in ("content", "text", "message", "response"):
                if k in resp and isinstance(resp[k], str):
                    return resp[k]
        # list-like responses
        if isinstance(resp, (list, tuple)) and len(resp) > 0:
            return self._extract_text_from_llm_response(resp[0])
        # fallback to string conversion
        try:
            return str(resp)
        except Exception:
            return ""

    def _call_llm_with_prompt(self, prompt_text):
        # Try common call methods in order
        # 1) predict(prompt)
        if hasattr(self.llm, "predict"):
            try:
                return self.llm.predict(prompt_text)
            except Exception:
                pass
        # 2) generate([prompt]) -> LLMResult
        if hasattr(self.llm, "generate"):
            try:
                return self.llm.generate([prompt_text])
            except Exception:
                pass
        # 3) run / invoke / call style methods
        for name in ("run", "invoke", "call"):
            if hasattr(self.llm, name):
                try:
                    return getattr(self.llm, name)(prompt_text)
                except Exception:
                    pass
        # 4) attempt direct attribute 'complete' or 'complete_prompt'
        for name in ("complete", "complete_prompt"):
            if hasattr(self.llm, name):
                try:
                    return getattr(self.llm, name)(prompt_text)
                except Exception:
                    pass
        # no supported method found
        raise RuntimeError("LLM does not expose a supported call method (predict/generate/run/invoke).")

    def create_qa_chain(self, vectorstore):

        # Fallback simple QA callable using the vectorstore directly
        def simple_qa(inputs):
            # accept dict {"query": "..."} or plain string
            query = inputs.get("query") if isinstance(inputs, dict) else inputs
            if not query:
                return {"result": "", "source_documents": []}
            # retrieve top docs
            try:
                docs = vectorstore.similarity_search(query, k=4)
            except Exception:
                # try as_retriever if similarity_search not present
                try:
                    retr = vectorstore.as_retriever(search_kwargs={"k": 4})
                    docs = retr.get_relevant_documents(query)
                except Exception:
                    docs = []

            # build context from retrieved docs
            context_parts = []
            for d in docs:
                if isinstance(d, dict):
                    text = d.get("page_content", "") or d.get("content", "")
                else:
                    text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                context_parts.append(text)
            context = "\n\n---\n\n".join(context_parts) if context_parts else ""

            prompt_text = self.prompt_template.format(context=context, question=query)

            # Call the LLM using robust wrapper and normalize response
            try:
                llm_resp = self._call_llm_with_prompt(prompt_text)
            except Exception as e:
                return {"result": f"LLM call failed: {e}", "source_documents": docs}

            answer_text = self._extract_text_from_llm_response(llm_resp)

            return {"result": answer_text, "source_documents": docs}

        return simple_qa
