"""PDF text extraction using PyMuPDF.

Extracts text page-by-page so that page numbers can be carried into chunks and
surfaced as citations later.

Two robustness features for these (mostly two-column, justified) SGS documents:

* **De-hyphenation** — justified text splits words across line breaks with soft
  hyphens ("affili-\\nated"). We rejoin them, while keeping genuine hyphenated
  compounds (e.g. "non-performance") intact via a small prefix allow-list.
* **Column-aware ordering ("auto")** — PyMuPDF's default text order is correct for
  the seed PDFs, but an arbitrary uploaded PDF could interleave columns. When a page
  is confidently detected as multi-column, we re-order text column-by-column
  (left→right, top→bottom) as a safety net. Single-column pages use the default.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Strategies for reading a page's text.
Strategy = str  # "auto" | "text" | "columns"

# Prefixes that legitimately keep a hyphen when a word wraps at a line break.
# For these we drop the newline but keep the hyphen ("non-\nperformance" ->
# "non-performance"); everything else is treated as a soft hyphen and merged
# ("affili-\nated" -> "affiliated").
_KEEP_HYPHEN_PREFIXES = {
    "non", "self", "co", "anti", "pre", "post", "multi", "sub", "cross",
    "inter", "intra", "semi", "ex", "well", "off", "over", "under", "out",
    "up", "all", "near", "mid", "long", "short", "high", "low", "full",
}

# word-hyphen-linebreak-word  (allows spaces around the hyphen/newline)
_HYPHEN_LINEBREAK = re.compile(r"([A-Za-z]{2,})-[ \t]*\n[ \t]*([a-z]{2,})")


@dataclass
class Page:
    """A single extracted PDF page."""

    page_number: int  # 1-based
    text: str


def _dehyphenate(text: str) -> str:
    """Rejoin words split by a soft hyphen at a line break."""

    def repl(match: re.Match) -> str:
        prefix, suffix = match.group(1), match.group(2)
        if prefix.lower() in _KEEP_HYPHEN_PREFIXES:
            return f"{prefix}-{suffix}"  # genuine compound: keep hyphen, drop newline
        return f"{prefix}{suffix}"  # soft hyphen: merge into one word

    return _HYPHEN_LINEBREAK.sub(repl, text)


def _text_blocks(page: "pymupdf.Page") -> list[tuple]:
    """Return non-empty text blocks: (x0, y0, x1, y1, text, block_no, block_type)."""
    return [
        b for b in page.get_text("blocks")
        if b[4].strip() and b[6] == 0
    ]


def _extract_columns(page: "pymupdf.Page") -> str | None:
    """Re-order a multi-column page column-by-column.

    Returns ordered text if the page is confidently multi-column, else None so the
    caller can fall back to the default extraction.
    """
    page_width = page.rect.width
    blocks = _text_blocks(page)
    if len(blocks) < 4:
        return None

    # Cluster blocks into columns by their left edge (x0). Full-width title/footer
    # blocks share the left margin and naturally land in the leftmost column.
    gap_threshold = page_width * 0.12
    xs = sorted(b[0] for b in blocks)
    cluster_edges = [xs[0]]
    for x in xs[1:]:
        if x - cluster_edges[-1] > gap_threshold:
            cluster_edges.append(x)
    if len(cluster_edges) < 2:
        return None  # single column

    def column_index(block: tuple) -> int:
        return min(range(len(cluster_edges)), key=lambda i: abs(block[0] - cluster_edges[i]))

    # Require at least two columns with real content (>=2 blocks each), otherwise
    # this is probably a single column with a stray indented block.
    counts = Counter(column_index(b) for b in blocks)
    if sum(1 for c in counts.values() if c >= 2) < 2:
        return None

    ordered = sorted(blocks, key=lambda b: (column_index(b), round(b[1], 1)))
    return "\n".join(b[4].strip() for b in ordered)


def extract_page_text(page: "pymupdf.Page", strategy: Strategy = "auto") -> str:
    """Extract one page's text using the chosen strategy, then de-hyphenate.

    - "text":    PyMuPDF default order (proven correct for the seed PDFs).
    - "columns": force column-aware ordering (falls back to default if not multi-col).
    - "auto":    use column-aware ordering only when a page is detected multi-column.
    """
    if strategy in ("auto", "columns"):
        columns = _extract_columns(page)
        if columns is not None:
            return _dehyphenate(columns).strip()
        if strategy == "columns":
            # explicit request but not multi-column: fall through to default
            pass
    return _dehyphenate(page.get_text("text")).strip()


def extract_pages(pdf_path: str | Path, strategy: Strategy = "auto") -> list[Page]:
    """Extract non-empty text pages from a PDF.

    Returns pages in reading order; pages with no extractable text are skipped
    (e.g. purely image-based pages that PyMuPDF can't read without OCR).
    """
    path = Path(pdf_path)
    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            text = extract_page_text(page, strategy)
            if text:
                pages.append(Page(page_number=index + 1, text=text))
    return pages


def discover_pdfs(dirs: list[str], root: str | Path = ".") -> list[Path]:
    """Find unique PDF files across the given directories (non-recursive per dir).

    ``root`` is the project root the directories are relative to. Results are
    de-duplicated (a file reachable via two dir entries appears once) and sorted.
    """
    root_path = Path(root).resolve()
    found: set[Path] = set()
    for d in dirs:
        dir_path = (root_path / d).resolve()
        if not dir_path.is_dir():
            continue
        for pdf in dir_path.glob("*.pdf"):
            if pdf.is_file():
                found.add(pdf.resolve())
    return sorted(found)
