"""Deterministic retrieval metrics computed against the labelled test set.

Relevance is judged per chunk, so the numbers don't depend on how many chunks a
document happens to be split into:

    a chunk is relevant to a question  <=>  it comes from one of the question's
    `relevant_docs` AND contains at least one of its `evidence` phrases.

For the top-k retrieved chunks of one question:
    Hit@k       1 if any relevant chunk is in the top k, else 0
    MRR@k       1 / rank of the first relevant chunk (0 if none in the top k)
    Precision@k relevant chunks in the top k / k
    Recall@k    relevant chunks in the top k / all relevant chunks in the corpus
"""
from __future__ import annotations

from statistics import mean


def relevant_chunk_ids(question: dict, chunks: list[dict]) -> set[str]:
    docs = set(question.get("relevant_docs", []))
    ev = [e.lower() for e in question.get("evidence", [])]
    return {c["chunk_id"] for c in chunks
            if c["doc_id"] in docs and any(e in c["text"].lower() for e in ev)}


def retrieval_scores(ranked_ids: list[str], relevant: set[str], k: int) -> dict:
    top = ranked_ids[:k]
    hits = [i for i, cid in enumerate(top, 1) if cid in relevant]
    n_rel = len(relevant)
    return {
        "hit": 1.0 if hits else 0.0,
        "mrr": 1.0 / hits[0] if hits else 0.0,
        "precision": len(hits) / k if k else 0.0,
        "recall": len(hits) / n_rel if n_rel else 0.0,
    }


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(vals), 4) if vals else None
