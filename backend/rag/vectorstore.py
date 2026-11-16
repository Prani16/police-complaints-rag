"""Part 2b - Qdrant vector store.

Two modes, same code:
  * embedded (QDRANT_URL empty): Qdrant runs in-process and persists to
    data/qdrant/. The index is built at Docker build time and baked into the
    image, so Cloud Run needs no database server at all -> $0.
  * remote (QDRANT_URL set): Qdrant Cloud free tier (1 GB) or a self-hosted
    Qdrant container, for when the corpus is updated at runtime.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector, MatchAny,
                                  MatchValue, PointStruct, VectorParams)

from .config import settings

log = logging.getLogger(__name__)
_NS = uuid.UUID("6f1c2b1e-6a51-4c55-9a0e-6a3f0b3c1d77")


class VectorStore:
    def __init__(self):
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
            self.mode = "remote"
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
            self.mode = "embedded"
        self.collection = settings.collection
        # embedded Qdrant is not thread-safe; FastAPI runs requests in threads
        self.lock = threading.RLock()

    # ---------------------------------------------------------------- write
    def rebuild(self, chunks: list[dict], vectors: np.ndarray, model_name: str) -> None:
        dim = int(vectors.shape[1])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        points = [PointStruct(id=str(uuid.uuid5(_NS, c["chunk_id"])), vector=v.tolist(),
                              payload={k: c[k] for k in c if k != "embed_text"})
                  for c, v in zip(chunks, vectors)]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i:i + 128])
        self._write_meta({"embed_model": model_name, "dim": dim, "count": len(points)})
        log.info("indexed %d chunks (%d-dim) into %s [%s]", len(points), dim, self.collection, self.mode)

    def upsert(self, chunks: list[dict], vectors: np.ndarray) -> None:
        """Add chunks to the existing collection (used for uploaded files)."""
        points = [PointStruct(id=str(uuid.uuid5(_NS, c["chunk_id"])), vector=v.tolist(),
                              payload={k: c[k] for k in c if k != "embed_text"})
                  for c, v in zip(chunks, vectors)]
        with self.lock:
            for i in range(0, len(points), 128):
                self.client.upsert(self.collection, points=points[i:i + 128])

    def delete_doc(self, doc_id: str) -> None:
        with self.lock:
            self.client.delete(self.collection, points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]))
            )

    # ---------------------------------------------------------------- read
    def search(self, query_vector: np.ndarray, k: int,
               doc_ids: list[str] | None = None) -> list[tuple[dict, float]]:
        flt = None
        if doc_ids is not None:  # restrict the search to these documents
            flt = Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids)))])
        with self.lock:
            res = self.client.query_points(self.collection, query=query_vector.tolist(),
                                           query_filter=flt, limit=k, with_payload=True)
        return [(p.payload, float(p.score)) for p in res.points]

    def count(self) -> int:
        with self.lock:
            if not self.client.collection_exists(self.collection):
                return 0
            return self.client.count(self.collection).count

    # ---------------------------------------------------------------- meta
    def _meta_path(self):
        return settings.qdrant_path / "index_meta.json"

    def _write_meta(self, meta: dict) -> None:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        self._meta_path().write_text(json.dumps(meta, indent=2))

    def meta(self) -> dict:
        p = self._meta_path()
        return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()
