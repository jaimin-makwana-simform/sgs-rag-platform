"""Token-based chunking.

Uses LangChain's RecursiveCharacterTextSplitter with a tiktoken length function so
CHUNK_SIZE / CHUNK_OVERLAP are measured in tokens (matching embedding-model limits),
while splits still prefer natural boundaries (paragraphs, sentences, words).
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_loader import Page

# text-embedding-3-* and gpt-4o* all use the o200k_base / cl100k_base families.
# cl100k_base is a safe, widely-compatible tokenizer for length estimation.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class Chunk:
    """A chunk of text with provenance for citation."""

    text: str
    page_number: int
    chunk_index: int  # position within the document


def chunk_pages(
    pages: list[Page],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split a document's pages into token-sized, overlapping chunks.

    Each page is split independently so a chunk maps cleanly to a single page for
    citation purposes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    running_index = 0
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    page_number=page.page_number,
                    chunk_index=running_index,
                )
            )
            running_index += 1
    return chunks
