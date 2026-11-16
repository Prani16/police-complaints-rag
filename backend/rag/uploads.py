"""User uploads: save a file, run it through the same Part 1 + Part 2 pipeline
(load -> clean -> chunk -> embed -> Qdrant + BM25) and make it searchable
immediately, without rebuilding the whole index.

Uploaded documents get a doc_id starting with "upload_" so they can be told
apart from the built-in corpus, searched on their own, and deleted.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from .chunker import chunk_document
from .config import settings
from .loaders import SUPPORTED, load_file
from .retriever import get_retriever

log = logging.getLogger(__name__)
PREFIX = "upload_"
MAX_CHUNKS = 400  # ~a 150-page document; keeps CPU embedding under a few minutes


class UploadError(ValueError):
    """A problem with the uploaded file that the user can fix (shown in the UI)."""


def _safe_name(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name          # drop any path parts
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80] or "document"
    return f"{stem}{ext}"


def is_upload(doc_id: str) -> bool:
    return doc_id.startswith(PREFIX)


def ingest_upload(filename: str, data: bytes) -> dict:
    name = _safe_name(filename)
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise UploadError(f"Unsupported file type '{ext or 'none'}'. Upload PDF, HTML, Markdown or TXT.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise UploadError(f"File is larger than {settings.max_upload_mb:g} MB.")

    t0 = time.perf_counter()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"{PREFIX}{name}"   # file stem becomes the doc_id
    path.write_bytes(data)
    try:
        doc = load_file(path)
    except Exception as exc:  # noqa: BLE001 - corrupt PDF, bad encoding, ...
        path.unlink(missing_ok=True)
        raise UploadError(f"Could not read the file: {exc}") from exc
    if not doc.blocks:
        path.unlink(missing_ok=True)
        raise UploadError("No text found in the file. Scanned PDFs need OCR first.")

    chunks = [c.to_dict() for c in chunk_document(doc, settings.chunk_max_tokens,
                                                    settings.chunk_overlap_tokens, settings.chunk_min_tokens)]
    if len(chunks) > MAX_CHUNKS:
        path.unlink(missing_ok=True)
        raise UploadError(f"Document is too long ({len(chunks)} chunks, max {MAX_CHUNKS}).")

    r = get_retriever()
    vectors = r.embedder.embed_documents([c["embed_text"] for c in chunks])
    r.store.delete_doc(doc.doc_id)          # re-uploading the same file replaces it
    r.store.upsert(chunks, vectors)
    r.add_chunks(chunks)
    secs = time.perf_counter() - t0
    log.info("indexed upload %s: %d chunks in %.1fs", name, len(chunks), secs)
    return {"doc_id": doc.doc_id, "source": doc.source, "title": doc.title, "type": doc.doc_type,
            "chunks": len(chunks), "seconds": round(secs, 1)}


def delete_upload(doc_id: str) -> bool:
    if not is_upload(doc_id):
        raise UploadError("Only uploaded documents can be removed.")
    r = get_retriever()
    removed = r.remove_doc(doc_id)
    r.store.delete_doc(doc_id)
    for f in settings.upload_dir.glob(f"{doc_id}.*"):
        f.unlink(missing_ok=True)
    return removed > 0


def upload_doc_ids() -> list[str]:
    return sorted({c["doc_id"] for c in get_retriever().chunks if is_upload(c["doc_id"])})
