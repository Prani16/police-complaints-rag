"""Part 3e - FastAPI service used by the Node.js frontend.

    uvicorn api:app --port 8080          (run from backend/)

Endpoints
  GET    /health                 liveness + index status
  GET    /api/info               models, documents and chunk counts
  POST   /api/query              {"question": "...", "top_k": 5, "scope": "all"|"uploads"}
  POST   /api/upload             raw file bytes, filename in the X-Filename header
  GET    /api/documents          every indexed document (built-in + uploads)
  DELETE /api/documents/{doc_id} remove an uploaded document
  GET    /api/admin/eval         evaluation results (needs X-Admin-Key = ADMIN_PASSWORD)
"""
from __future__ import annotations

import hmac
import json
import logging
from collections import Counter
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import unquote

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import pipeline
from rag.config import settings
from rag.generator import get_generator
from rag.reranker import get_reranker
from rag.retriever import get_retriever
from rag.uploads import UploadError, delete_upload, ingest_upload, is_upload, upload_doc_ids

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


class Query(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=10)
    scope: Literal["all", "uploads"] = "all"


@app.get("/health")
def health():
    r = get_retriever()
    return {"status": "ok", "chunks": len(r.chunks), "vectors": r.store.count()}


@app.get("/api/info")
def info():
    r = get_retriever()
    gen = get_generator()
    return {
        "models": {"embedding": r.embedder.name, "reranker": get_reranker().name,
                   "llm": gen.model if gen.mode == "gemini" else "none (extractive)"},
        "vector_store": f"qdrant ({r.store.mode})",
        "documents": list_documents(),
        "chunk_count": len(r.chunks),
        "settings": {"final_k": settings.final_k, "rerank_candidates": settings.rerank_candidates,
                     "min_relevance": settings.min_relevance},
    }


def list_documents() -> list[dict]:
    r = get_retriever()
    docs = Counter((c["doc_id"], c["source"], c["title"], c["doc_type"]) for c in r.chunks)
    return [{"doc_id": i, "source": s, "title": t, "type": d, "chunks": n, "uploaded": is_upload(i)}
            for (i, s, t, d), n in sorted(docs.items(), key=lambda x: (is_upload(x[0][0]), x[0][0]))]


@app.post("/api/query")
def query(q: Query):
    doc_ids = None
    if q.scope == "uploads":
        doc_ids = upload_doc_ids()
        if not doc_ids:
            raise HTTPException(status_code=400, detail="No uploaded documents yet - upload a file first.")
    try:
        return pipeline.answer(q.question.strip(), q.top_k, doc_ids)
    except RuntimeError as exc:  # e.g. Gemini quota / auth problems
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/upload")
async def upload(request: Request):
    """Body = the raw file bytes; X-Filename = URL-encoded original file name.
    (Raw bytes instead of multipart keeps the backend free of extra packages.)"""
    filename = unquote(request.headers.get("x-filename", "")).strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Missing X-Filename header.")
    data = await request.body()
    try:
        # embedding is CPU-heavy: run it in a worker thread, not on the event loop
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(ingest_upload, filename, data)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/documents")
def documents():
    return list_documents()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        if not delete_upload(doc_id):
            raise HTTPException(status_code=404, detail="Document not found.")
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": doc_id}


# --------------------------------------------------------------------------- admin
def _check_admin(key: str | None) -> None:
    if not settings.admin_password:
        raise HTTPException(status_code=403, detail="Admin is disabled: set ADMIN_PASSWORD.")
    if not key or not hmac.compare_digest(key.encode(), settings.admin_password.encode()):
        raise HTTPException(status_code=401, detail="Wrong admin key.")


@app.get("/api/admin/eval")
def admin_eval(x_admin_key: str | None = Header(default=None)):
    """Latest evaluation run plus a short summary of earlier runs (for trends)."""
    _check_admin(x_admin_key)
    d = settings.eval_results_dir
    latest = d / "latest.json"
    history = []
    for f in sorted((d / "history").glob("run-*.json"))[-20:]:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            history.append({"file": f.name, "created_at": r.get("created_at"), "summary": r.get("summary", {}),
                            "config": {k: r.get("config", {}).get(k) for k in ("reranker", "llm", "k")}})
        except (OSError, ValueError):
            continue
    return {"latest": json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else None,
            "history": history}
