"""Part 3d - the full query pipeline:

  question -> hybrid retrieve (dense + BM25, RRF, top 20)
           -> rerank (Qwen3-Reranker, keep top 5)
           -> relevance gate (all scores low -> "not found", no LLM call)
           -> Gemini grounded answer with [n] citations
"""
from __future__ import annotations

import logging
import re
import time

from .config import settings
from .generator import NOT_FOUND, get_generator
from .reranker import get_reranker
from .retriever import get_retriever

log = logging.getLogger(__name__)


def answer(question: str, top_k: int | None = None) -> dict:
    top_k = top_k or settings.final_k
    timings: dict[str, float] = {}

    t = time.perf_counter()
    cands = get_retriever().retrieve(question)
    timings["retrieve_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    reranker = get_reranker()
    scores = reranker.score(question, [c.chunk["embed_text"] for c in cands]) if cands else None
    if scores is not None:
        for c, s in zip(cands, scores):
            c.rerank = float(s)
        cands.sort(key=lambda c: c.rerank, reverse=True)
    timings["rerank_ms"] = (time.perf_counter() - t) * 1000

    top = cands[:top_k]
    gated = scores is not None and (not top or top[0].rerank < settings.min_relevance)
    if scores is not None:  # drop weak passages so they can't be cited
        top = [c for c in top if c.rerank >= settings.min_relevance / 2] or top[:1]

    t = time.perf_counter()
    gen = get_generator()
    text = NOT_FOUND if gated else gen.generate(question, [c.chunk for c in top])
    timings["generate_ms"] = (time.perf_counter() - t) * 1000

    log.info("query timings: retrieve %.0fms | rerank %.0fms (%d passages) | generate %.0fms",
             timings["retrieve_ms"], timings["rerank_ms"], len(cands), timings["generate_ms"])
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    sources = []
    for i, c in enumerate(top, 1):
        ch = c.chunk
        sources.append({
            "n": i, "cited": i in cited, "source": ch["source"], "doc_type": ch["doc_type"],
            "title": ch["title"], "section": ch["section"], "pages": ch["pages"],
            "text": ch["text"], "rerank_score": c.rerank, "rrf_score": round(c.rrf, 5),
            "dense_rank": c.dense_rank, "sparse_rank": c.sparse_rank,
        })
    return {
        "question": question,
        "answer": text,
        "found": text.strip() != NOT_FOUND,
        "sources": [] if gated else sources,
        "mode": gen.mode,
        "models": {"embedding": get_retriever().embedder.name, "reranker": reranker.name,
                   "llm": gen.model if gen.mode == "gemini" else "none (extractive)"},
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
    }
