"""Part 2a - embedding models.

Default: Qwen3-Embedding-0.6B (Apache-2.0, free, runs on CPU, 1024-dim,
32K context, instruction-aware). Swap to Qwen3-Embedding-4B/8B with the
EMBED_MODEL env var when a GPU is available - same code, same API.

`HashingEmbedder` exists only so tests and CI can run offline without
downloading a model. It is lexical, not semantic - never use it for real.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache

import numpy as np

from .config import settings

log = logging.getLogger(__name__)

# Qwen3-Embedding is instruction-aware: queries get a task instruction,
# documents are embedded as-is (per the model card this adds ~1-5% retrieval).
QUERY_INSTRUCTION = (
    "Instruct: Given a question about filing police complaints, FIRs, arrests, custody "
    "or police accountability in India, retrieve the passages that answer it\nQuery: "
)


class Embedder:
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class QwenEmbedder(Embedder):
    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        log.info("loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name, device=_device())
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def embed_documents(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 64, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode([text], prompt=QUERY_INSTRUCTION, normalize_embeddings=True,
                                 convert_to_numpy=True)[0]


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in and or for on is are be by with as at that this it from was".split())


class HashingEmbedder(Embedder):
    """Offline test double: hashed unigrams + bigrams, L2-normalised."""

    def __init__(self, dim: int = 1024):
        self.name, self.dim = "hashing-bow-test-only", dim

    def _vec(self, text: str) -> np.ndarray:
        toks = [t for t in _WORD.findall(text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embed_backend == "hash":
        return HashingEmbedder()
    return QwenEmbedder(settings.embed_model, settings.embed_batch_size)
