"""Part 2 tests (offline, hashing embedder): index builds and search returns
the right document."""
import json

import numpy as np

from rag.config import settings
from rag.embeddings import get_embedder
from rag.vectorstore import VectorStore


def _chunks():
    return [json.loads(line) for line in settings.chunks_path.open()]


def test_embeddings_are_normalised():
    e = get_embedder()
    v = e.embed_documents(["custodial death reporting", "zero fir transfer"])
    assert v.shape == (2, e.dim)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_index_build_and_search(tmp_path):
    original = settings.qdrant_path
    object.__setattr__(settings, "qdrant_path", tmp_path)  # Settings is frozen
    try:
        chunks = _chunks()
        e = get_embedder()
        store = VectorStore()
        store.rebuild(chunks, e.embed_documents([c["embed_text"] for c in chunks]), e.name)
        assert store.count() == len(chunks)
        assert store.meta()["embed_model"] == e.name
        hits = store.search(e.embed_query("Zero FIR transfer within 24 hours through CCTNS"), 3)
        assert any(h[0]["doc_id"].startswith("08_zero_fir") for h in hits)
        assert "embed_text" not in hits[0][0]
        store.client.close()
    finally:
        object.__setattr__(settings, "qdrant_path", original)
