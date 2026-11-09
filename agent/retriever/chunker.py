"""Structure-aware, page-aware chunking for Indian legal documents."""

from __future__ import annotations

import bisect
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document

from .config import DocumentSpec
from .pdf_extractor import ExtractedPage
from .text_cleaner import clean_text


@dataclass(frozen=True)
class Boundary:
    position: int
    header: str
    kind: str  # chapter, part, or section
    section_number: str | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class TextSpan:
    text: str
    start: int
    end: int


_CHAPTER_RE = re.compile(
    r"(?im)^\s*((?:CHAPTER|PART)\s+[IVXLC0-9A-Z]+\b[^\n]*)"
)
_EXPLICIT_SECTION_RE = re.compile(
    r"(?im)^\s*((?:SECTION|SEC\.?|RULE|REGULATION|ARTICLE)\s+"
    r"(\d+[A-Z]?(?:[-/]\d+)?)\s*(?:[.\-:\u2013\u2014]\s*)?([^\n]{2,180}))"
)
_NUMBERED_SECTION_RE = re.compile(
    r"(?m)^\s*(\d{1,3}[A-Z]?(?:[-/]\d+)?)\s*[.)]\s+([^\n]{3,180})"
)
_SUBSECTION_RE = re.compile(r"(?m)(?=^\s*(?:\(\d+[A-Za-z]?\)|[a-z]\))\s+)")
_PARAGRAPH_RE = re.compile(r"(?m)(?=^\s*\n)")


def _detect_boundaries(text: str) -> list[Boundary]:
    found: list[Boundary] = []
    for match in _CHAPTER_RE.finditer(text):
        found.append(Boundary(match.start(1), match.group(1).strip(), "chapter"))
    for match in _EXPLICIT_SECTION_RE.finditer(text):
        prefix = match.group(1).strip()
        number = match.group(2)
        title = match.group(3).strip()
        found.append(Boundary(match.start(1), prefix, "section", number, title))
    for match in _NUMBERED_SECTION_RE.finditer(text):
        number, title = match.group(1), match.group(2).strip()
        # Subsections are normally parenthesized; this guard avoids obvious
        # list lines while retaining conventional Act headings such as "1. Title".
        if title.startswith((".", ")")):
            continue
        found.append(
            Boundary(match.start(1), f"{number}. {title}", "section", number, title)
        )

    found.sort(key=lambda item: (item.position, 0 if item.kind != "section" else 1))
    deduped: list[Boundary] = []
    for item in found:
        if deduped and item.position - deduped[-1].position < 3:
            if item.kind == "section" and deduped[-1].kind != "section":
                deduped[-1] = item
            continue
        deduped.append(item)
    return deduped


def detect_section_boundaries(text: str) -> list[tuple[int, str]]:
    """Public compatibility view of detected structure boundaries."""

    return [(boundary.position, boundary.header) for boundary in _detect_boundaries(text)]


def _joined_pages(pages: Sequence[ExtractedPage]) -> tuple[str, list[int], list[int]]:
    pieces: list[str] = []
    starts: list[int] = []
    page_numbers: list[int] = []
    offset = 0
    for page in pages:
        cleaned = clean_text(page.text)
        starts.append(offset)
        page_numbers.append(page.page_number)
        pieces.append(cleaned)
        offset += len(cleaned) + 2
    return "\n\n".join(pieces), starts, page_numbers


def _page_for_position(
    starts: Sequence[int], position: int, page_numbers: Sequence[int] | None = None
) -> int:
    index = max(0, bisect.bisect_right(starts, max(0, position)) - 1)
    if page_numbers:
        return page_numbers[min(index, len(page_numbers) - 1)]
    return index + 1


def _word_count(text: str) -> int:
    return len(text.split())


def _token_count(text: str) -> int:
    """Conservative token estimate for legal prose without a model tokenizer."""

    return max(1, math.ceil(_word_count(text) * 1.3))


def _span_parts(text: str, start: int, end: int, pattern: re.Pattern[str]) -> list[TextSpan]:
    body = text[start:end]
    matches = list(pattern.finditer(body))
    if not matches:
        return [TextSpan(body.strip(), start, end)] if body.strip() else []
    starts = [match.start() for match in matches] + [len(body)]
    parts: list[TextSpan] = []
    for left, right in zip(starts, starts[1:]):
        value = body[left:right].strip()
        if value:
            leading = len(body[left:right]) - len(body[left:right].lstrip())
            parts.append(TextSpan(value, start + left + leading, start + right))
    return parts


def _word_windows(span: TextSpan, max_words: int, overlap_words: int = 40) -> list[TextSpan]:
    matches = list(re.finditer(r"\S+", span.text))
    if not matches:
        return []
    step = max(1, max_words - overlap_words)
    result: list[TextSpan] = []
    for start_index in range(0, len(matches), step):
        selected = matches[start_index : start_index + max_words]
        if not selected:
            break
        result.append(
            TextSpan(
                span.text[selected[0].start() : selected[-1].end()],
                span.start + selected[0].start(),
                span.start + selected[-1].end(),
            )
        )
        if start_index + max_words >= len(matches):
            break
    return result


def _pack_spans(spans: Iterable[TextSpan], max_tokens: int, prefix: str) -> list[TextSpan]:
    packed: list[TextSpan] = []
    current: list[TextSpan] = []
    current_words = _token_count(prefix)
    for span in spans:
        span_words = _token_count(span.text)
        if span_words + _token_count(prefix) > max_tokens:
            if current:
                packed.append(TextSpan("\n\n".join(item.text for item in current), current[0].start, current[-1].end))
                current, current_words = [], _token_count(prefix)
            packed.extend(_word_windows(span, max(80, int(max_tokens / 1.3) - _word_count(prefix))))
            continue
        if current and current_words + span_words > max_tokens:
            packed.append(TextSpan("\n\n".join(item.text for item in current), current[0].start, current[-1].end))
            current, current_words = [], _token_count(prefix)
        current.append(span)
        current_words += span_words
    if current:
        packed.append(TextSpan("\n\n".join(item.text for item in current), current[0].start, current[-1].end))
    if len(packed) > 1 and _token_count(packed[-1].text) < 80:
        previous, last = packed[-2], packed[-1]
        merged = f"{previous.text}\n\n{last.text}"
        if _token_count(prefix + merged) <= max_tokens:
            packed[-2] = TextSpan(merged, previous.start, last.end)
            packed.pop()
    return packed


def _metadata(
    *,
    collection: str,
    spec: DocumentSpec,
    source_path: str,
    source_hash: str,
    page_start: int,
    page_end: int,
    boundary: Boundary | None,
    content: str,
    chunk_type: str,
    chunk_index: int,
    domain: str,
    chapter: str | None = None,
) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "collection": collection,
        "domain": domain,
        "document_name": spec.document_name,
        "act_name": spec.document_name,
        "document_type": spec.document_type,
        "source_file": Path(source_path).name,
        "source_filename": Path(source_path).name,
        "source_path": source_path,
        "source_sha256": source_hash,
        "page_start": page_start,
        "page_end": page_end,
        "page_number": page_start,
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
    }
    if chapter:
        metadata["chapter"] = chapter
    if spec.act_year is not None:
        metadata["act_year"] = spec.act_year
    if boundary is not None:
        if boundary.section_number is not None:
            metadata["section_number"] = boundary.section_number
            metadata["section"] = boundary.section_number
        if boundary.section_title:
            metadata["section_title"] = boundary.section_title
        subsection = re.search(r"(?m)^\s*\((\d+[A-Za-z]?)\)\s+", content)
        if subsection:
            metadata["subsection"] = subsection.group(1)
    return metadata


def chunk_pages(
    pages: Sequence[ExtractedPage],
    *,
    collection: str,
    spec: DocumentSpec,
    source_path: str,
    source_hash: str,
    domain: str | None = None,
    max_tokens: int = 650,
) -> list[Document]:
    """Chunk extracted pages while retaining structure and page ranges."""

    text, page_starts, page_numbers = _joined_pages(pages)
    boundaries = _detect_boundaries(text)
    sections = [item for item in boundaries if item.kind == "section"]
    chapters = [item for item in boundaries if item.kind in {"chapter", "part"}]
    chunks: list[Document] = []

    resolved_domain = domain or collection

    def context_for(position: int) -> tuple[str, str | None]:
        active = [item.header for item in chapters if item.position < position]
        chapter = active[-1] if active else None
        return f"Document: {spec.document_name}\n" + (f"Chapter: {chapter}\n" if chapter else ""), chapter

    if sections:
        for section_index, boundary in enumerate(sections):
            end = len(text)
            for next_boundary in boundaries:
                if next_boundary.position > boundary.position:
                    end = next_boundary.position
                    break
            body = text[boundary.position:end].strip()
            # Short provisions such as commencement or citation sections are
            # legally meaningful; discard only heading-only noise.
            if _word_count(body) < 8:
                continue
            context, chapter = context_for(boundary.position)
            prefix = context + f"Section: {boundary.header}\n"
            units = _span_parts(text, boundary.position, end, _SUBSECTION_RE)
            if len(units) == 1 and _token_count(units[0].text) > max_tokens:
                units = _span_parts(text, boundary.position, end, _PARAGRAPH_RE)
            section_chunks = _pack_spans(units, max_tokens, prefix)
            for part_index, span in enumerate(section_chunks, start=1):
                content = f"{prefix}\n{span.text}".strip()
                page_start = _page_for_position(page_starts, span.start, page_numbers)
                page_end = _page_for_position(
                    page_starts, max(span.start, span.end - 1), page_numbers
                )
                chunks.append(
                    Document(
                        page_content=content,
                        metadata=_metadata(
                            collection=collection,
                            spec=spec,
                            source_path=source_path,
                            source_hash=source_hash,
                            page_start=page_start,
                            page_end=page_end,
                            boundary=boundary,
                            content=span.text,
                            chunk_type="section" if len(section_chunks) == 1 else "subsection",
                            chunk_index=part_index,
                            domain=resolved_domain,
                            chapter=chapter,
                        ),
                    )
                )
    else:
        # Paragraph fallback still uses real character spans for page metadata.
        fallback_prefix = f"Document: {spec.document_name}\n"
        paragraph_units = _span_parts(text, 0, len(text), _PARAGRAPH_RE)
        if not paragraph_units and text.strip():
            paragraph_units = [TextSpan(text.strip(), 0, len(text))]
        for index, span in enumerate(_pack_spans(paragraph_units, min(max_tokens, 600), fallback_prefix), start=1):
            content = f"{fallback_prefix}\n{span.text}".strip()
            chunks.append(
                Document(
                    page_content=content,
                    metadata=_metadata(
                        collection=collection,
                        spec=spec,
                        source_path=source_path,
                        source_hash=source_hash,
                        page_start=_page_for_position(page_starts, span.start, page_numbers),
                        page_end=_page_for_position(
                            page_starts, max(span.start, span.end - 1), page_numbers
                        ),
                        boundary=None,
                        content=span.text,
                        chunk_type="paragraph",
                        chunk_index=index,
                        domain=resolved_domain,
                    ),
                )
            )
    return chunks


def chunk_document(
    text: str,
    act_name: str,
    source_file: str,
    max_tokens: int = 650,
) -> list[Document]:
    """Compatibility wrapper for callers that have one unpaged text string."""

    from .config import DocumentSpec

    year_match = re.search(r"(?<!\d)(18|19|20|21)\d{2}(?!\d)", source_file)
    spec = DocumentSpec(
        source_file,
        act_name,
        "act",
        int(year_match.group(0)) if year_match else None,
    )
    page = ExtractedPage(1, text, True, "pymupdf")
    return chunk_pages(
        [page],
        collection="unknown",
        spec=spec,
        source_path=source_file,
        source_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        max_tokens=max_tokens,
    )

