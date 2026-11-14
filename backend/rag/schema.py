"""Common data structures shared by every loader and the chunker.

Every loader, whatever the file format, returns a `Document` made of ordered
`Block`s. A block is one paragraph, list item or table row, and remembers the
heading path it sits under and (for PDFs) its page. Keeping structure at this
level is what lets the chunker cut on section boundaries instead of at an
arbitrary character count."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Block:
    text: str
    headings: list[str] = field(default_factory=list)  # e.g. ["Guidelines", "Time limit"]
    page: int | None = None
    kind: str = "para"  # para | list | table


@dataclass
class Document:
    doc_id: str          # file stem, stable across runs
    source: str          # file name
    doc_type: str        # pdf | html | md | txt
    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    doc_type: str
    title: str
    section: str         # "Heading > Subheading"
    pages: list[int]
    text: str            # what the LLM sees and what is shown as citation
    embed_text: str      # text + contextual header, what gets embedded
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)
