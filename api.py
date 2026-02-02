from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv

from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from utils.llm_handler import LLMHandler

load_dotenv()

API_KEY = os.getenv("API_KEY")
if API_KEY:
    API_KEY = API_KEY.strip()

app = FastAPI(title="Research Paper RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple in-memory session store: session_id -> metadata
session_store = {}


class AskRequest(BaseModel):
    session_id: str
    query: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured in environment")

    contents = await file.read()
    processor = PDFProcessor()
    try:
        chunks = processor.process_pdf_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {e}")

    session_id = str(uuid.uuid4())

    persist_dir = Path("chroma_db") / session_id
    persist_dir.mkdir(parents=True, exist_ok=True)

    vector_manager = VectorStoreManager(persist_directory=str(persist_dir))
    try:
        vectorstore = vector_manager.create_vectorstore(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store creation failed: {e}")

    llm_handler = LLMHandler(API_KEY)
    qa_chain = llm_handler.create_qa_chain(vectorstore)

    session_store[session_id] = {
        "vectorstore": vectorstore,
        "qa_chain": qa_chain,
        "chunks": len(chunks),
    }

    return {"session_id": session_id, "chunks": len(chunks), "status": "processed"}


@app.post("/ask")
async def ask(req: AskRequest):
    meta = session_store.get(req.session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="session_id not found")

    qa = meta.get("qa_chain")
    try:
        resp = qa({"query": req.query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM/QA call failed: {e}")

    # normalize response
    result = resp.get("result") if isinstance(resp, dict) else str(resp)
    source_docs = resp.get("source_documents") if isinstance(resp, dict) else []

    serialized_sources = []
    for d in source_docs:
        try:
            page = None
            if hasattr(d, "metadata") and isinstance(d.metadata, dict):
                page = d.metadata.get("page")
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            serialized_sources.append({"page": page, "text": text[:1000]})
        except Exception:
            continue

    return {"result": result, "source_documents": serialized_sources}


@app.get("/status/{session_id}")
async def status(session_id: str):
    meta = session_store.get(session_id)
    if not meta:
        return {"session_id": session_id, "processed": False, "chunks": 0}
    return {"session_id": session_id, "processed": True, "chunks": meta.get("chunks", 0)}


@app.get("/sources/{session_id}")
async def sources(session_id: str, limit: int = 10):
    meta = session_store.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="session_id not found")

    # attempt to retrieve a small set of documents from the vectorstore
    vs = meta.get("vectorstore")
    if not vs:
        raise HTTPException(status_code=500, detail="vectorstore missing for session")

    try:
        docs = vs.similarity_search("", k=limit)
    except Exception:
        # some vectorstores expose as_retriever
        try:
            retr = vs.as_retriever(search_kwargs={"k": limit})
            docs = retr.get_relevant_documents("")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to fetch sources: {e}")

    out = []
    for d in docs:
        page = None
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            page = d.metadata.get("page")
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        out.append({"page": page, "text": text[:2000]})

    return {"sources": out}
