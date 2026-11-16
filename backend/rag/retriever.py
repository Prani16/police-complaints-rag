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
import threading
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
        self.embedder = get_embedder()
        self.store = get_store()
        self._lock = threading.Lock()
        self._set_chunks(chunks)

    # ---------------------------------------------------------------- index state
    def _set_chunks(self, chunks: list[dict]) -> None:
        """Rebuild the in-memory BM25 index and swap it in atomically, so a
        query running in another thread never sees a half-updated index."""
        bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in chunks]) if chunks else None
        self.chunks, self.by_id, self.bm25 = chunks, {c["chunk_id"]: c for c in chunks}, bm25

    def add_chunks(self, new: list[dict]) -> None:
        with self._lock:
            ids = {c["doc_id"] for c in new}
            self._set_chunks([c for c in self.chunks if c["doc_id"] not in ids] + new)
            save_chunks(self.chunks)

    def remove_doc(self, doc_id: str) -> int:
        with self._lock:
            kept = [c for c in self.chunks if c["doc_id"] != doc_id]
            removed = len(self.chunks) - len(kept)
            self._set_chunks(kept)
            save_chunks(self.chunks)
        return removed

    # ---------------------------------------------------------------- search
    def dense(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[str]:
        hits = self.store.search(self.embedder.embed_query(query), k, doc_ids)
        return [p["chunk_id"] for p, _ in hits]

    def sparse(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[str]:
        chunks, bm25 = self.chunks, self.bm25
        if bm25 is None:
            return []
        scores = bm25.get_scores(tokenize(query))
        allowed = set(doc_ids) if doc_ids is not None else None
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = [chunks[i]["chunk_id"] for i in order
               if scores[i] > 0 and (allowed is None or chunks[i]["doc_id"] in allowed)]
        return out[:k]

    def retrieve(self, query: str, k: int | None = None, rrf_k: int = 60,
                 doc_ids: list[str] | None = None) -> list[Candidate]:
        """doc_ids=None searches everything; a list restricts the search to
        those documents (e.g. only the user's uploads)."""
        k = k or settings.rerank_candidates
        by_id = self.by_id
        cands: dict[str, Candidate] = {}
        for rank, cid in enumerate(self.dense(query, settings.dense_k, doc_ids), 1):
            if cid not in by_id:  # deleted between search and lookup
                continue
            c = cands.setdefault(cid, Candidate(by_id[cid]))
            c.dense_rank = rank
            c.rrf += 1.0 / (rrf_k + rank)
        for rank, cid in enumerate(self.sparse(query, settings.sparse_k, doc_ids), 1):
            c = cands.setdefault(cid, Candidate(by_id[cid]))
            c.sparse_rank = rank
            c.rrf += 1.0 / (rrf_k + rank)
        return sorted(cands.values(), key=lambda c: c.rrf, reverse=True)[:k]



def save_chunks(chunks: list[dict]) -> None:
    """Persist the chunk list so uploads survive a restart."""
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.chunks_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(settings.chunks_path)


def load_chunks() -> list[dict]:
    if not settings.chunks_path.exists():
        raise FileNotFoundError(f"{settings.chunks_path} missing - run scripts.ingest")
    return [json.loads(line) for line in settings.chunks_path.open(encoding="utf-8")]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(load_chunks())
