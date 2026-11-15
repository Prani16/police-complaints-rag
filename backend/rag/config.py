"""Central configuration. Every value can be overridden with an environment
variable so the same code runs locally, in tests and on Cloud Run."""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional .env support for local development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # ---- paths -------------------------------------------------------------
    raw_dir: Path = Path(_env("RAW_DIR", str(ROOT / "data" / "raw")))
    processed_dir: Path = Path(_env("PROCESSED_DIR", str(ROOT / "data" / "processed")))

    # ---- chunking (Part 1) --------------------------------------------------
    chunk_max_tokens: int = int(_env("CHUNK_MAX_TOKENS", "400"))
    chunk_overlap_tokens: int = int(_env("CHUNK_OVERLAP_TOKENS", "60"))
    chunk_min_tokens: int = int(_env("CHUNK_MIN_TOKENS", "80"))

    # ---- embeddings + vector store (Part 2) --------------------------------
    # "qwen" = Qwen3-Embedding via sentence-transformers (production)
    # "hash" = dependency-free hashing embedder (offline tests only)
    embed_backend: str = _env("EMBED_BACKEND", "qwen")
    embed_model: str = _env("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embed_batch_size: int = int(_env("EMBED_BATCH_SIZE", "16"))
    qdrant_url: str = _env("QDRANT_URL", "")          # empty -> embedded local mode
    qdrant_api_key: str = _env("QDRANT_API_KEY", "")
    qdrant_path: Path = Path(_env("QDRANT_PATH", str(ROOT / "data" / "qdrant")))
    collection: str = _env("QDRANT_COLLECTION", "police_complaints")

    # ---- retrieval + reranking (Part 3) ------------------------------------
    dense_k: int = int(_env("DENSE_K", "30"))
    sparse_k: int = int(_env("SPARSE_K", "30"))
    # 10 candidates keeps CPU reranking to a few seconds; raise on a GPU
    rerank_candidates: int = int(_env("RERANK_CANDIDATES", "10"))
    # passages are truncated to this many tokens for reranking (chunks are ~400)
    rerank_max_tokens: int = int(_env("RERANK_MAX_TOKENS", "512"))
    final_k: int = int(_env("FINAL_K", "5"))
    # at most this many chunks from one document in the final context, so the
    # LLM sees several sources instead of five near-identical chunks
    max_per_doc: int = int(_env("MAX_PER_DOC", "2"))
    # "qwen" = Qwen3-Reranker (default) | "bge" = bge-reranker-v2-m3 | "none"
    rerank_backend: str = _env("RERANK_BACKEND", "qwen")
    rerank_model: str = _env("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    # below this reranker relevance probability we answer "not found"
    min_relevance: float = float(_env("MIN_RELEVANCE", "0.15"))

    # ---- generation (Part 3) -----------------------------------------------
    # Demo default = cheapest current model; set GEMINI_MODEL=gemini-4-argon
    # for best quality.
    gemini_model: str = _env("GEMINI_MODEL", "gemini-3.5-flash-lite")
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    use_vertex: bool = _env("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    gcp_project: str = _env("GOOGLE_CLOUD_PROJECT", "")
    gcp_location: str = _env("GOOGLE_CLOUD_LOCATION", "global")
    temperature: float = float(_env("GEMINI_TEMPERATURE", "0.2"))

    extra: dict = field(default_factory=dict)

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"


settings = Settings()
