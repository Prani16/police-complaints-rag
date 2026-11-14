"""Part 1a - format-specific loaders (PDF, HTML, Markdown, TXT).

Each loader normalises its format into the same `Document`/`Block` structure
and removes format-specific noise:
  * PDF  - running headers/footers and page numbers repeated on every page;
           headings detected from font size.
  * HTML - <script>, <style>, <nav>, <header>, <footer>, <aside>, ads, forms;
           tables linearised as "column: value" rows so they stay retrievable.
  * MD   - heading hierarchy kept, emphasis markers and table pipes removed.
  * TXT  - ALL-CAPS / underlined lines treated as headings.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median

from .schema import Block, Document

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}


# --------------------------------------------------------------------------- utils
def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("­", "")                 # soft hyphen
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)      # de-hyphenate line breaks
    s = re.sub(r"[ \t ]+", " ", s)
    return s.strip()


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def _push_heading(stack: list[tuple[int, str]], level: int, text: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))


def _path(stack: list[tuple[int, str]]) -> list[str]:
    return [t for _, t in stack]


# --------------------------------------------------------------------------- PDF
_PAGE_NO = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)


def load_pdf(path: Path) -> Document:
    import pdfplumber

    pages: list[list[tuple[str, float]]] = []  # per page: (line_text, font_size)
    with pdfplumber.open(path) as pdf:
        meta_title = (pdf.metadata or {}).get("Title") or ""
        for page in pdf.pages:
            lines = []
            for ln in page.extract_text_lines(keep_blank_chars=False):
                sizes = [c.get("size", 0) for c in ln.get("chars", []) if c.get("text", "").strip()]
                lines.append((ln["text"].strip(), median(sizes) if sizes else 0.0))
            pages.append(lines)

    # Running header/footer = same (digit-normalised) line at the top/bottom of
    # most pages. Real PDFs from government sites almost always have these and
    # they pollute every chunk if left in.
    def norm(t: str) -> str:
        return re.sub(r"\d+", "#", t.lower())

    edge = Counter()
    for lines in pages:
        for t, _ in lines[:2] + lines[-2:]:
            edge[norm(t)] += 1
    threshold = max(2, int(0.6 * len(pages))) if len(pages) > 1 else 10**9
    boiler = {k for k, v in edge.items() if v >= threshold}

    all_sizes = [s for lines in pages for _, s in lines if s]
    body_size = Counter(round(s, 1) for s in all_sizes).most_common(1)[0][0] if all_sizes else 10

    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    title = ""
    para: list[str] = []
    para_page = 1

    def flush():
        nonlocal para
        if para:
            text = clean_text(" ".join(para))
            if text:
                kind = "list" if re.match(r"^(-|•|\d+\.|\([a-z]\))\s", text) else "para"
                blocks.append(Block(text, _path(stack), para_page, kind))
        para = []

    last_level = 0  # level of the previous line if it was a heading, else 0
    for pno, lines in enumerate(pages, start=1):
        for t, size in lines:
            if not t or norm(t) in boiler or _PAGE_NO.match(t):
                continue
            level = 1 if size >= body_size * 1.35 else 2 if size >= body_size * 1.12 else 0
            if level:
                flush()
                if last_level == level and stack and stack[-1][0] == level:
                    # heading wrapped onto a second line -> join it
                    stack[-1] = (level, f"{stack[-1][1]} {t}")
                    if level == 1 and title:
                        title = f"{title} {t}"
                else:
                    if level == 1:
                        title = title or t
                    _push_heading(stack, level, t)
                last_level = level
                continue
            last_level = 0
            # a new list item starts a new block; long paragraphs are cut at a
            # sentence end so one block never gets huge
            if para and re.match(r"^(-|•|\d+\.|\([a-z]\))\s", t):
                flush()
            if not para:
                para_page = pno
            para.append(t)
            if re.search(r"[.:;]$", t) and len(" ".join(para)) > 300:
                flush()
        flush()  # never let a paragraph span a page break silently

    title = title or meta_title or path.stem
    return Document(path.stem, path.name, "pdf", clean_text(title), _merge_list_blocks(blocks))


def _merge_list_blocks(blocks: list[Block]) -> list[Block]:
    """PDF line extraction splits wrapped list items; glue continuation lines
    (no bullet, same headings, same page) back onto the previous item."""
    out: list[Block] = []
    for b in blocks:
        if (out and b.kind == "para" and out[-1].kind == "list"
                and out[-1].headings == b.headings and out[-1].page == b.page
                and b.text[:1].islower()):
            out[-1].text = f"{out[-1].text} {b.text}"
        else:
            out.append(b)
    return out


# --------------------------------------------------------------------------- HTML
_NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside",
               "form", "iframe", "svg", "button"]
_NOISE_CLASS = re.compile(r"(^|\b)(ad|ads|advert|banner|cookie|newsletter|share|social|related|sidebar|breadcrumb)(\b|$)", re.I)


def _table_rows(table) -> list[str]:
    rows = table.find_all("tr")
    if not rows:
        return []
    header = [clean_text(c.get_text(" ")) for c in rows[0].find_all(["th", "td"])]
    has_header = bool(rows[0].find_all("th"))
    out = []
    for r in rows[1:] if has_header else rows:
        cells = [clean_text(c.get_text(" ")) for c in r.find_all(["th", "td"])]
        if has_header and len(cells) == len(header):
            out.append("; ".join(f"{h}: {v}" for h, v in zip(header, cells) if v))
        else:
            out.append(" | ".join(c for c in cells if c))
    return [r for r in out if r]


def load_html(path: Path) -> Document:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text(path), "html.parser")
    page_title = clean_text(soup.title.get_text()) if soup.title else ""
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": _NOISE_CLASS}):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    h1 = ""
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "table", "pre", "blockquote"]):
        if el.find_parent("table") is not None and el.name != "table":
            continue  # handled by the table branch
        if el.name in ("p", "dd", "dt") and el.find_parent("li") is not None:
            continue
        if el.name[0] == "h" and el.name[1:].isdigit():
            text = clean_text(el.get_text(" "))
            if text:
                level = int(el.name[1])
                h1 = h1 or (text if level == 1 else "")
                _push_heading(stack, level, text)
            continue
        if el.name == "table":
            for row in _table_rows(el):
                blocks.append(Block(row, _path(stack), None, "table"))
            continue
        text = clean_text(el.get_text(" "))
        if not text:
            continue
        if el.name == "dt":       # question/term: join with its answer
            blocks.append(Block(text, _path(stack), None, "para"))
        elif el.name == "dd" and blocks and blocks[-1].kind == "para":
            blocks[-1].text = f"{blocks[-1].text} {text}"
        else:
            kind = "list" if el.name == "li" else "para"
            if kind == "list" and el.find_parent("ol") is not None:
                idx = len(el.find_previous_siblings("li")) + 1
                text = f"{idx}. {text}"
            blocks.append(Block(text, _path(stack), None, kind))

    return Document(path.stem, path.name, "html", h1 or page_title or path.stem, blocks)


# --------------------------------------------------------------------------- Markdown
_MD_INLINE = [(re.compile(r"\*\*(.+?)\*\*"), r"\1"), (re.compile(r"__(.+?)__"), r"\1"),
              (re.compile(r"(?<!\w)\*(.+?)\*(?!\w)"), r"\1"), (re.compile(r"`(.+?)`"), r"\1"),
              (re.compile(r"\[(.+?)\]\((.+?)\)"), r"\1")]


def _md_inline(s: str) -> str:
    for pat, rep in _MD_INLINE:
        s = pat.sub(rep, s)
    return clean_text(s)


def load_markdown(path: Path) -> Document:
    lines = _read_text(path).splitlines()
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    title = ""
    para: list[str] = []
    table: list[list[str]] = []

    def flush_para():
        nonlocal para
        if para:
            text = _md_inline(" ".join(para))
            if text:
                kind = "list" if re.match(r"^(\d+\.|[-*+])\s", para[0]) else "para"
                text = re.sub(r"^[-*+]\s+", "", text)
                blocks.append(Block(text, _path(stack), None, kind))
        para = []

    def flush_table():
        nonlocal table
        if table:
            header, rows = table[0], table[1:]
            for r in rows:
                if len(r) == len(header):
                    blocks.append(Block("; ".join(f"{_md_inline(h)}: {_md_inline(v)}" for h, v in zip(header, r) if v),
                                        _path(stack), None, "table"))
                else:
                    blocks.append(Block(" | ".join(_md_inline(c) for c in r), _path(stack), None, "table"))
        table = []

    for raw in lines:
        line = raw.rstrip()
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para(); flush_table()
            level, text = len(m.group(1)), _md_inline(m.group(2))
            title = title or (text if level == 1 else "")
            _push_heading(stack, level, text)
            continue
        if line.strip().startswith("|"):
            flush_para()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue  # separator row
            table.append(cells)
            continue
        flush_table()
        stripped = line.strip().lstrip(">").strip()
        if not stripped:
            flush_para()
        elif re.match(r"^(\d+\.|[-*+])\s", stripped):
            flush_para()
            para = [stripped]
        else:
            para.append(stripped)
    flush_para(); flush_table()
    return Document(path.stem, path.name, "md", title or path.stem, blocks)


# --------------------------------------------------------------------------- TXT
def _is_txt_heading(line: str, nxt: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90:
        return False
    if re.fullmatch(r"[=\-~]{3,}", nxt.strip()):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters) and not s.endswith(".")


def load_txt(path: Path) -> Document:
    lines = _read_text(path).splitlines() + [""]
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    title = ""
    para: list[str] = []

    def flush():
        nonlocal para
        if para:
            text = clean_text(" ".join(para))
            kind = "list" if re.match(r"^(-|\d+\.|\([a-z]\)|Step \d+:)\s?", text) else "para"
            blocks.append(Block(text, _path(stack), None, kind))
        para = []

    i = 0
    prev_heading = False
    while i < len(lines) - 1:
        line, nxt = lines[i], lines[i + 1]
        if re.fullmatch(r"\s*[=\-~]{3,}\s*", line):
            i += 1
            continue
        if _is_txt_heading(line, nxt):
            flush()
            text = clean_text(line)
            underlined = bool(re.fullmatch(r"\s*[=\-~]{3,}\s*", nxt))
            if not title or (prev_heading and stack and stack[-1][0] == 1):
                # consecutive heading lines at the top form one title block
                title = f"{title} - {text}" if title else text
                stack[:] = [(1, title)]
            else:
                _push_heading(stack, 2, text)
            prev_heading = True
            i += 2 if underlined else 1
            continue
        prev_heading = False
        s = line.strip()
        if not s:
            flush()
        elif re.match(r"^(-|\d+\.|\([a-z]\)|Step \d+:)\s?", s):
            flush()
            para = [s]
        else:
            para.append(s)
        i += 1
    flush()
    return Document(path.stem, path.name, "txt", (title or path.stem)[:200], blocks)


# --------------------------------------------------------------------------- router
LOADERS = {".pdf": load_pdf, ".html": load_html, ".htm": load_html,
           ".md": load_markdown, ".markdown": load_markdown, ".txt": load_txt}


def load_file(path: Path) -> Document:
    return LOADERS[path.suffix.lower()](path)


def load_directory(folder: Path) -> tuple[list[Document], list[tuple[str, str]]]:
    """Load every supported file. A broken file is logged and skipped so one
    bad upload never kills the whole ingestion run."""
    docs, failures = [], []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        try:
            doc = load_file(path)
            if not doc.blocks:
                raise ValueError("no extractable text (scanned PDF? needs OCR)")
            docs.append(doc)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", path.name, exc)
            failures.append((path.name, str(exc)))
    return docs, failures
