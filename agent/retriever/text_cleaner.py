"""Conservative legal-text normalization that preserves bilingual content."""

from __future__ import annotations

import re


def clean_text(text: str, document_name: str = "") -> str:
    """Clean extraction artifacts while retaining legal structure and languages."""

    del document_name  # Kept in the signature for compatibility; no language is discarded.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")
    # Join words split by a line-wrap hyphen, but leave legal punctuation alone.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    # A line containing only a number is normally a printed page number. This
    # does not touch section numbers such as "12." or subsection markers.
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$\n?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Clean pages independently so page provenance remains available."""

    return [(page_number, clean_text(text)) for page_number, text in pages]

