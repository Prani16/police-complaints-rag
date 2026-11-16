"""Part 1 entry point: load data/raw/* (pdf, html, md, txt) -> clean ->
structure-aware chunks -> data/processed/chunks.jsonl

    python -m scripts.ingest            (run from backend/)
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunker import chunk_document, dedupe  # noqa: E402
from rag.config import settings  # noqa: E402
from rag.loaders import load_directory  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    docs, failures = load_directory(settings.raw_dir)
    if settings.upload_dir.exists():  # keep files uploaded through the UI
        up_docs, up_fail = load_directory(settings.upload_dir)
        docs, failures = docs + up_docs, failures + up_fail
    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d, settings.chunk_max_tokens,
                                     settings.chunk_overlap_tokens, settings.chunk_min_tokens))
    before = len(chunks)
    chunks = dedupe(chunks)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with settings.chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    sizes = [c.n_tokens for c in chunks]
    print(f"documents loaded : {len(docs)}  ({dict(Counter(d.doc_type for d in docs))})")
    print(f"failed files     : {failures or 'none'}")
    print(f"chunks           : {len(chunks)} (removed {before - len(chunks)} duplicates)")
    if sizes:
        print(f"tokens per chunk : min {min(sizes)}  avg {sum(sizes) // len(sizes)}  max {max(sizes)}")
    for d in docs:
        n = sum(1 for c in chunks if c.doc_id == d.doc_id)
        print(f"  {d.doc_type:4} {n:3} chunks  {d.title[:70]}")
    print(f"wrote {settings.chunks_path}")


if __name__ == "__main__":
    main()
