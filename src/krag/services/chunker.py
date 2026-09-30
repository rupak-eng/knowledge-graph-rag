"""Chunking for SEC 10-K HTML filings: section-aware, deterministic chunk IDs."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from krag.domain.schemas import Chunk, Document, chunk_id_for

logger = logging.getLogger(__name__)

# 10-K item headings, e.g. "ITEM 1. Business", "ITEM 7. Management's Discussion..."
ITEM_RE = re.compile(r"^\s*ITEM\s+(\d+[A-Z]?)\s*[.\-–—]?\s*(.*)$", re.IGNORECASE)

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 150


def parse_10k_html(path: Path, doc_id: str, title: str) -> tuple[Document, str]:
    """Extract clean text from a 10-K HTML/XML filing, dropping XBRL tags."""
    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")

    # Drop XBRL facts and non-content elements
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()
    for ix in soup.find_all(lambda t: t.name and t.name.startswith("ix:")):
        # keep visible text of inline xbrl, drop the tag wrapper
        ix.unwrap()

    text = soup.get_text(separator="\n")
    # collapse whitespace but keep paragraph-ish structure
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    clean = "\n".join(lines)
    content_hash = hashlib.sha256(clean.encode()).hexdigest()
    return Document(
        doc_id=doc_id,
        title=title,
        source="SEC EDGAR 10-K FY2024",
        content_hash=content_hash,
    ), clean


def split_sections(clean_text: str) -> list[tuple[str, str]]:
    """Split filing text into (section_title, section_text) at ITEM headings."""
    sections: list[tuple[str, list[str]]] = []
    current_title = "FRONT MATTER"
    current_lines: list[str] = []
    for line in clean_text.split("\n"):
        m = ITEM_RE.match(line.strip())
        if m and len(line.strip()) < 120:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = f"ITEM {m.group(1)}"
            rest = m.group(2).strip()
            if rest:
                current_title += f" {rest}"
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return [(t, "\n".join(ls)) for t, ls in sections]


def chunk_section(
    doc_id: str, section: str, text: str, start_index: int
) -> tuple[list[Chunk], int]:
    """Sliding-window chunking over a section. Returns (chunks, next_index)."""
    chunks: list[Chunk] = []
    idx = start_index
    step = CHUNK_CHARS - CHUNK_OVERLAP
    pos = 0
    text = text.strip()
    if not text:
        return [], idx
    while pos < len(text):
        piece = text[pos : pos + CHUNK_CHARS].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(doc_id, idx),
                    doc_id=doc_id,
                    index=idx,
                    section=section,
                    text=piece,
                )
            )
            idx += 1
        pos += step
    return chunks, idx


def chunk_document(doc: Document, clean_text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for section, section_text in split_sections(clean_text):
        sec_chunks, idx = chunk_section(doc.doc_id, section, section_text, idx)
        chunks.extend(sec_chunks)
    logger.info("chunked %s into %d chunks", doc.doc_id, len(chunks))
    return chunks
