"""Part 3a - hybrid retrieval: dense (Qdrant) + sparse (BM25), fused with
Reciprocal Rank Fusion.

Why hybrid for this domain: legal questions mix meaning ("what can I do if
police won't file my complaint") with exact identifiers ("173(3)", "D.K.
Basu", "CCTNS", "Section 41D") that dense embeddings blur. BM25 catches the
exact tokens, dense catches paraphrases, RRF combines ranks without having
to calibrate two incompatible score scales.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import get_embedder
from .vectorstore import get_store

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the of to in and or for on is are be by with as at that this it from was were
what which who whom how when where why do does did can could should would will i my me we our you your
if about into than then there their they them its""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


@dataclass
class Candidate:
    chunk: dict
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf: float = 0.0
    rerank: float | None = None
    debug: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in chunks])
        self.embedder = get_embedder()
        self.store = get_store()

    def dense(self, query: str, k: int) -> list[str]:
        hits = self.store.search(self.embedder.embed_query(query), k)
        return [p["chunk_id"] for p, _ in hits]

    def sparse(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i]["chunk_id"] for i in order[:k] if scores[i] > 0]

    def retrieve(self, query: str, k: int | None = None, rrf_k: int = 60) -> list[Candidate]:
        k = k or settings.rerank_candidates
        cands: dict[str, Candidate] = {}
        for rank, cid in enumerate(self.dense(query, settings.dense_k), 1):
            c = cands.setdefault(cid, Candidate(self.by_id.get(cid) or self._payload(cid)))
            c.dense_rank = rank
            c.rrf += 1.0 / (rrf_k + rank)
        for rank, cid in enumerate(self.sparse(query, settings.sparse_k), 1):
            c = cands.setdefault(cid, Candidate(self.by_id[cid]))
            c.sparse_rank = rank
            c.rrf += 1.0 / (rrf_k + rank)
        return sorted(cands.values(), key=lambda c: c.rrf, reverse=True)[:k]

    def _payload(self, cid: str) -> dict:  # chunk only in the vector store
        raise KeyError(f"chunk {cid} missing from chunks.jsonl - rebuild the index")


def load_chunks() -> list[dict]:
    if not settings.chunks_path.exists():
        raise FileNotFoundError(f"{settings.chunks_path} missing - run scripts.ingest")
    return [json.loads(line) for line in settings.chunks_path.open(encoding="utf-8")]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(load_chunks())
