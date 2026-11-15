"""Part 3e - FastAPI service used by the Node.js frontend.

    uvicorn api:app --port 8080          (run from backend/)

Endpoints
  GET  /health      liveness + index status
  GET  /api/info    models, documents and chunk counts
  POST /api/query   {"question": "...", "top_k": 5} -> answer + citations
"""
from __future__ import annotations

import logging
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import pipeline
from rag.config import settings
from rag.generator import get_generator
from rag.reranker import get_reranker
from rag.retriever import get_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load every model at startup (Cloud Run startup CPU boost helps here) so
    # the first user request isn't the one that pays for model loading.
    import time

    r = get_retriever()
    rr = get_reranker()
    get_generator()
    # Warm-up: the first forward pass of a torch model is much slower
    # (kernel selection, memory allocation). Pay that cost here, not on the
    # first user question.
    t = time.perf_counter()
    r.retrieve("How do I register an FIR?")
    rr.score("How do I register an FIR?", ["An FIR is registered at the police station."])
    log.info("models warmed up in %.1fs", time.perf_counter() - t)
    meta = r.store.meta()
    if meta and meta.get("embed_model") != r.embedder.name:
        log.warning("index was built with %s but querying with %s - rebuild the index!",
                    meta.get("embed_model"), r.embedder.name)
    log.info("ready: %d chunks, store=%s", len(r.chunks), r.store.mode)
    yield


app = FastAPI(title="Police Complaints RAG API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class Query(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=10)


@app.get("/health")
def health():
    r = get_retriever()
    return {"status": "ok", "chunks": len(r.chunks), "vectors": r.store.count()}


@app.get("/api/info")
def info():
    r = get_retriever()
    docs = Counter((c["source"], c["title"], c["doc_type"]) for c in r.chunks)
    gen = get_generator()
    return {
        "models": {"embedding": r.embedder.name, "reranker": get_reranker().name,
                   "llm": gen.model if gen.mode == "gemini" else "none (extractive)"},
        "vector_store": f"qdrant ({r.store.mode})",
        "documents": [{"source": s, "title": t, "type": d, "chunks": n} for (s, t, d), n in sorted(docs.items())],
        "chunk_count": len(r.chunks),
        "settings": {"final_k": settings.final_k, "rerank_candidates": settings.rerank_candidates,
                     "min_relevance": settings.min_relevance},
    }


@app.post("/api/query")
def query(q: Query):
    try:
        return pipeline.answer(q.question.strip(), q.top_k)
    except RuntimeError as exc:  # e.g. Gemini quota / auth problems
        raise HTTPException(status_code=502, detail=str(exc)) from exc
