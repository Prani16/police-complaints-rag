"""Part 1b - structure-aware recursive chunking.

Strategy (see docs/DESIGN_DECISIONS.md for the reasoning):
  1. Group blocks by section (heading path). A chunk never crosses a section
     boundary, so "Directive Six" never bleeds into "Directive Seven".
  2. Pack whole blocks (paragraphs / list items / table rows) into a chunk up
     to `max_tokens`. Blocks are only cut when a single block is larger than
     the budget; then we fall back recursively: sentences -> words.
  3. Overlap: when a section needs several chunks, the next chunk starts with
     the last ~`overlap_tokens` of the previous one (whole sentences), so a
     fact that straddles the cut is retrievable from either side.
  4. Tiny trailing chunks (< `min_tokens`) are merged into their neighbour so
     we never index 20-token fragments.
  5. Contextual header: the text that is *embedded* is prefixed with
     "Document title > Section path" so a chunk that just says "must be
     completed within 14 days" still carries what it is about.
"""
from __future__ import annotations

import hashlib
import re
from itertools import groupby

from .schema import Block, Chunk, Document

_TOKEN = re.compile(r"\w+|[^\w\s]")
_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])")


def count_tokens(text: str) -> int:
    """Cheap tokenizer-free estimate (~= subword count for English legal text)."""
    return len(_TOKEN.findall(text))


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Recursive fallback for one block that is bigger than the budget."""
    if count_tokens(text) <= max_tokens:
        return [text]
    sentences = _SENT.split(text)
    if len(sentences) > 1:
        out, cur = [], ""
        for s in sentences:
            cand = f"{cur} {s}".strip()
            if cur and count_tokens(cand) > max_tokens:
                out.extend(_split_oversized(cur, max_tokens))
                cur = s
            else:
                cur = cand
        if cur:
            out.extend(_split_oversized(cur, max_tokens))
        return out
    words = text.split()
    out, cur = [], []
    for w in words:
        cur.append(w)
        if count_tokens(" ".join(cur)) >= max_tokens:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def _tail(text: str, overlap_tokens: int) -> str:
    """Last whole sentences of `text` totalling <= overlap_tokens."""
    if overlap_tokens <= 0:
        return ""
    sents = _SENT.split(text)
    tail: list[str] = []
    for s in reversed(sents):
        if count_tokens(" ".join([s] + tail)) > overlap_tokens:
            break
        tail.insert(0, s)
    return " ".join(tail)


def _section_label(headings: list[str], title: str) -> str:
    # The document title is usually heading[0]; don't repeat it in the section.
    hs = [h for h in headings if h != title]
    return " > ".join(hs)


def _join_sections(sections: list[str]) -> str:
    """["A > B", "A > C"] -> "A > B; C" (shared parent written once)."""
    if not sections:
        return ""
    if len(sections) == 1:
        return sections[0]
    parts = [s.split(" > ") for s in sections]
    prefix = []
    for level in zip(*parts):
        if len(set(level)) == 1:
            prefix.append(level[0])
        else:
            break
    leaves = [" > ".join(p[len(prefix):]) or p[-1] for p in parts]
    return " > ".join(prefix + ["; ".join(leaves)])


def chunk_document(doc: Document, max_tokens: int = 400, overlap_tokens: int = 60,
                   min_tokens: int = 80) -> list[Chunk]:
    pieces: list[tuple[str, str, list[int]]] = []  # (section, text, pages)

    for headings, group in groupby(doc.blocks, key=lambda b: tuple(b.headings)):
        section = _section_label(list(headings), doc.title)
        units: list[tuple[str, int | None]] = []
        for b in group:  # type: Block
            for part in _split_oversized(b.text, max_tokens):
                units.append((part, b.page))

        cur: list[str] = []
        cur_pages: list[int] = []
        for text, page in units:
            candidate = "\n".join(cur + [text])
            if cur and count_tokens(candidate) > max_tokens:
                body = "\n".join(cur)
                pieces.append((section, body, cur_pages))
                overlap = _tail(cur[-1], overlap_tokens)
                cur = [overlap] if overlap and overlap != text else []
                cur_pages = list(cur_pages[-1:]) if overlap else []
            cur.append(text)
            if page and page not in cur_pages:
                cur_pages.append(page)
        if cur:
            pieces.append((section, "\n".join(cur), cur_pages))

    # merge tiny fragments into a neighbour (same document). When sections are
    # merged, the sub-heading is kept inline as "[Heading]" so the LLM still
    # sees the structure.
    merged: list[tuple[list[str], str, list[int]]] = []
    for section, text, pages in pieces:
        if merged and count_tokens(text) < min_tokens and \
                count_tokens(merged[-1][1]) + count_tokens(text) <= max_tokens:
            psecs, ptext, ppages = merged[-1]
            new = section not in psecs
            label = f"[{section.split(' > ')[-1]}] " if new and section else ""
            merged[-1] = (psecs + ([section] if new else []), f"{ptext}\n{label}{text}",
                          sorted(set(ppages + pages)))
        else:
            merged.append(([section], text, pages))
    # a tiny piece that could not merge backwards is merged forward instead
    i = 0
    while i < len(merged) - 1:
        s0, t0, p0 = merged[i]
        s1, t1, p1 = merged[i + 1]
        if count_tokens(t0) < min_tokens and count_tokens(t0) + count_tokens(t1) <= max_tokens:
            label = f"[{s1[0].split(' > ')[-1]}] " if s1[0] and s1[0] not in s0 else ""
            merged[i] = (s0 + [s for s in s1 if s not in s0], f"{t0}\n{label}{t1}", sorted(set(p0 + p1)))
            merged.pop(i + 1)
        else:
            i += 1

    chunks = []
    for i, (sections, text, pages) in enumerate(merged):
        section = _join_sections([s for s in sections if s])
        header = f"{doc.title}" + (f" > {section}" if section else "")
        cid = hashlib.sha1(f"{doc.doc_id}:{i}:{text}".encode()).hexdigest()[:12]
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{i:03d}-{cid}", doc_id=doc.doc_id, source=doc.source,
            doc_type=doc.doc_type, title=doc.title, section=section, pages=pages,
            text=text, embed_text=f"{header}\n\n{text}", n_tokens=count_tokens(text)))
    return chunks


def dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """Drop exact-duplicate chunk bodies (the same content often exists as both
    an HTML page and a PDF download)."""
    seen, out = set(), []
    for c in chunks:
        key = hashlib.sha1(re.sub(r"\W+", " ", c.text.lower()).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
