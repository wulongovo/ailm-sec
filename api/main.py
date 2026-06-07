"""AILLM-SEC - FastAPI Backend."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List

from lib.logger import log
from core.embeddings import EmbeddingService
from core.rag import KnowledgeBase
from core.llm import LLMEngine
from core.ingestion import DocumentIngestor

app = FastAPI(
    title="AILLM-SEC",
    description="AI-Powered Security Analysis Assistant with RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Global instances ============
embedding_service = EmbeddingService()
knowledge_base = KnowledgeBase(embedding_service)
llm_engine = LLMEngine()
ingestor = DocumentIngestor(knowledge_base)


# ============ Request/Response Models ============
class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    use_rag: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None

class IngestRequest(BaseModel):
    file_path: str
    file_type: str = "auto"  # auto, cve, qa, markdown

class StatsResponse(BaseModel):
    knowledge_base: dict
    llm_status: str


# ============ API Endpoints ============
@app.get("/")
async def root():
    return {"name": "AILLM-SEC", "version": "1.0.0", "status": "running"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    kb_stats = knowledge_base.get_stats()
    llm_status = "connected" if llm_engine.client else "not configured"
    return StatsResponse(knowledge_base=kb_stats, llm_status=llm_status)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with the security AI assistant."""
    # RAG retrieval
    context_docs = None
    if req.use_rag:
        context_docs = knowledge_base.query(req.message, k=5)

    if req.stream:
        def generate():
            for chunk in llm_engine.chat(req.message, context_docs, stream=True):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")

    answer = llm_engine.chat(req.message, context_docs, stream=False)
    sources = context_docs if context_docs else None
    return ChatResponse(answer=answer, sources=sources)


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Ingest documents into the knowledge base."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    file_type = req.file_type
    if file_type == "auto":
        if req.file_path.endswith(".jsonl"):
            if "cve" in req.file_path.lower():
                file_type = "cve"
            else:
                file_type = "qa"
        elif req.file_path.endswith((".md", ".markdown")):
            file_type = "markdown"
        else:
            file_type = "markdown"

    if file_type == "cve":
        count = ingestor.ingest_cve_jsonl(req.file_path)
    elif file_type == "qa":
        count = ingestor.ingest_qa_jsonl(req.file_path)
    else:
        count = ingestor.ingest_markdown(req.file_path)

    return {"status": "ok", "file_type": file_type, "documents_ingested": count}


@app.get("/api/search")
async def search(q: str, k: int = 5):
    """Search the knowledge base."""
    results = knowledge_base.query(q, k=k)
    return {"query": q, "results": results, "count": len(results)}


@app.get("/api/ui", response_class=HTMLResponse)
async def ui():
    """Serve the web UI."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AILLM-SEC</h1><p>Frontend not found. Place index.html in frontend/</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
