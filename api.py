"""
api.py — EvoRAG Phase 3: FastAPI Application

Endpoints:
    GET  /health   → {"status": "ok", "index_vectors": N, "model": "phi3:latest"}
    POST /query    → {"query": "...", "top_k": 5}
                  ← {"answer": "...", "sources": [...], "query": "..."}

Run:
    python api.py
    -- or --
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    API_HOST,
    API_PORT,
    OLLAMA_MODEL,
    RETRIEVAL_TOP_K,
)
from embedder import load_index
from retriever import answer_async, retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Shared state loaded once at startup ───────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index once when the server starts."""
    log.info("EvoRAG API starting — loading FAISS index…")
    try:
        index, metadata = load_index()
        _state["index"] = index
        _state["metadata"] = metadata
        log.info(f"Index ready: {index.ntotal} vectors, dim={index.d}")
    except FileNotFoundError as e:
        log.error(str(e))
        log.error("Run 'python embedder.py --build' to create the index first.")
        raise RuntimeError(str(e))
    yield
    log.info("EvoRAG API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EvoRAG API",
    description="Evolutionary Multi-Personality RAG — retrieval + LLM endpoint",
    version="0.3.0",
    lifespan=lifespan,
)

# Allow all origins for local development (Phase 8 React frontend will need this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: int = Field(RETRIEVAL_TOP_K, ge=1, le=20, description="Chunks to retrieve")


class SourceItem(BaseModel):
    title:  str
    url:    str
    source: str
    score:  float


class QueryResponse(BaseModel):
    query:   str
    answer:  str
    sources: List[SourceItem]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    """Returns server status and the number of vectors in the loaded index."""
    index = _state.get("index")
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded yet.")
    return {
        "status": "ok",
        "index_vectors": index.ntotal,
        "embedding_dim": index.d,
        "model": OLLAMA_MODEL,
    }


@app.post("/query", response_model=QueryResponse, summary="RAG query")
async def query_endpoint(req: QueryRequest):
    """
    Full retrieval-augmented generation pipeline:
      1. Embed the query
      2. Retrieve top-k chunks from FAISS
      3. Assemble RAG prompt
      4. Send to Ollama (phi3:latest)
      5. Return answer + source attribution
    """
    index    = _state.get("index")
    metadata = _state.get("metadata")

    if index is None or metadata is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    log.info(f"POST /query — '{req.query[:60]}' (top_k={req.top_k})")

    try:
        result = await answer_async(req.query, top_k=req.top_k)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during query processing")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
