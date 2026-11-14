"""Part 2 entry point: embed data/processed/chunks.jsonl and (re)build the
Qdrant collection.

    python -m scripts.build_index          (run from backend/)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import settings  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.vectorstore import get_store  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not settings.chunks_path.exists():
        sys.exit("chunks.jsonl not found - run `python -m scripts.ingest` first")
    chunks = [json.loads(line) for line in settings.chunks_path.open(encoding="utf-8")]

    t0 = time.time()
    embedder = get_embedder()
    vectors = embedder.embed_documents([c["embed_text"] for c in chunks])
    t1 = time.time()
    store = get_store()
    store.rebuild(chunks, vectors, embedder.name)
    print(f"embedder : {embedder.name} ({embedder.dim}-dim)")
    print(f"embedded : {len(chunks)} chunks in {t1 - t0:.1f}s")
    print(f"store    : Qdrant [{store.mode}] collection '{store.collection}' -> {store.count()} points")
    store.client.close()  # release the embedded-mode file lock cleanly
    get_store.cache_clear()  # a later get_store() in the same process reopens it


if __name__ == "__main__":
    main()
