"""Part 1 tests: every format loads, noise is removed, chunks respect limits."""
from rag.chunker import chunk_document, count_tokens, dedupe
from rag.config import settings
from rag.loaders import load_directory, load_file

RAW = settings.raw_dir


def test_all_ten_files_load_in_four_formats():
    docs, failures = load_directory(RAW)
    assert failures == []
    assert len(docs) == 10
    assert {d.doc_type for d in docs} == {"pdf", "html", "md", "txt"}


def test_pdf_running_header_and_page_numbers_removed():
    doc = load_file(RAW / "01_lalita_kumari_fir_guidelines.pdf")
    text = " ".join(b.text for b in doc.blocks)
    assert "Practice Corpus" not in text          # running header
    assert not any(b.text.startswith("Page ") for b in doc.blocks)
    assert doc.title.startswith("Lalita Kumari")
    assert any(b.page == 2 for b in doc.blocks)   # page numbers kept as metadata


def test_html_noise_removed_and_tables_linearised():
    doc = load_file(RAW / "05_police_complaints_authority_india.html")
    text = " ".join(b.text for b in doc.blocks)
    for noise in ("Random", "Privacy policy", "tracker.js"):
        assert noise not in text
    assert any(b.kind == "table" and b.text.startswith("State / UT: Delhi") for b in doc.blocks)


def test_markdown_headings_become_sections():
    doc = load_file(RAW / "07_bnss_section_173_fir.md")
    assert any("173(3) – Preliminary enquiry" in b.headings for b in doc.blocks)
    assert not any("**" in b.text for b in doc.blocks)


def test_chunks_respect_section_boundaries_and_size():
    docs, _ = load_directory(RAW)
    chunks = dedupe([c for d in docs for c in chunk_document(d, 400, 60, 80)])
    assert all(count_tokens(c.text) <= 400 + 80 for c in chunks)
    assert all(c.embed_text.startswith(c.title) for c in chunks)
    # the 14-day BNSS preliminary enquiry rule must live in one chunk with its heading
    hit = [c for c in chunks if "14 days" in c.text and c.doc_id.startswith("07")]
    assert hit and "Preliminary enquiry" in hit[0].section


def test_oversized_block_is_split_recursively():
    from rag.schema import Block, Document

    long = " ".join(f"Sentence number {i} is about police complaints." for i in range(200))
    doc = Document("x", "x.txt", "txt", "X", [Block(long, ["S"])])
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=20, min_tokens=10)
    assert len(chunks) > 5
    assert all(c.n_tokens <= 100 + 20 for c in chunks)
