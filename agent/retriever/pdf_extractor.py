"""Page-preserving PDF extraction with automatic mixed-PDF OCR detection."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from .ocr import run_ocrmypdf

LOGGER = logging.getLogger(__name__)


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be read or OCR cannot produce a usable copy."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    had_extractable_text: bool
    extraction_method: str


def _has_extractable_text(text: str, *, minimum_characters: int = 24) -> bool:
    """Detect meaningful text without relying on the PDF font table."""

    compact = "".join(text.split())
    if len(compact) < minimum_characters:
        return False
    return sum(char.isalpha() for char in compact) >= 8


def _read_pages(pdf_path: Path) -> list[str]:
    try:
        with fitz.open(str(pdf_path)) as pdf:
            return [page.get_text("text") for page in pdf]
    except Exception as exc:  # PyMuPDF exposes several exception types by version.
        raise PDFExtractionError(f"PyMuPDF could not read {pdf_path}: {exc}") from exc


def extract_pages(pdf_path: str | Path) -> list[ExtractedPage]:
    """Extract every page, OCRing only when at least one page needs it.

    Suspect pages are copied to a temporary PDF and force-OCRed, then merged
    back into the original page sequence in memory. Digital pages are never
    sent through OCR, including in mixed PDFs.
    """

    source = Path(pdf_path)
    original_text = _read_pages(source)
    if not original_text:
        raise PDFExtractionError(f"PDF has no pages: {source}")

    text_flags = [_has_extractable_text(text) for text in original_text]
    missing_pages = [index + 1 for index, present in enumerate(text_flags) if not present]
    LOGGER.info(
        "PDF %s: %d pages, %d pages with extractable text",
        source.name,
        len(original_text),
        len(original_text) - len(missing_pages),
    )

    if not missing_pages:
        return [
            ExtractedPage(index + 1, text, True, "pymupdf")
            for index, text in enumerate(original_text)
        ]

    LOGGER.info("PDF %s: OCR required for pages %s", source.name, missing_pages)
    with tempfile.TemporaryDirectory(prefix="legal_ocr_") as temp_dir:
        target = Path(temp_dir) / "ocr_targets.pdf"
        searchable = Path(temp_dir) / "searchable.pdf"
        try:
            # OCR only suspect pages. This is more precise than OCRmyPDF's
            # document-wide --skip-text behavior for mixed PDFs whose image
            # pages contain a misleading, tiny text layer.
            with fitz.open(str(source)) as original, fitz.open() as selected:
                for page_number in missing_pages:
                    selected.insert_pdf(
                        original,
                        from_page=page_number - 1,
                        to_page=page_number - 1,
                    )
                selected.save(str(target))
            run_ocrmypdf(input_path=target, output_path=searchable, force_ocr=True)
            ocr_text = _read_pages(searchable)
        except Exception as exc:
            if isinstance(exc, PDFExtractionError):
                raise
            raise PDFExtractionError(f"OCR failed for {source}: {exc}") from exc

    if len(ocr_text) != len(missing_pages):
        raise PDFExtractionError(
            f"OCR target page count changed for {source}: "
            f"expected {len(missing_pages)}, got {len(ocr_text)}"
        )

    pages: list[ExtractedPage] = []
    ocr_index = 0
    for index, original in enumerate(original_text):
        if text_flags[index]:
            pages.append(ExtractedPage(index + 1, original, True, "pymupdf"))
        else:
            pages.append(
                ExtractedPage(index + 1, ocr_text[ocr_index], False, "ocrmypdf+pymupdf")
            )
            ocr_index += 1
    return pages


def needs_ocr(pdf_path: str | Path) -> bool:
    """Return whether any page lacks meaningful extractable text."""

    return any(not _has_extractable_text(text) for text in _read_pages(Path(pdf_path)))


def extract_text_pymupdf(pdf_path: str | Path) -> str:
    """Compatibility helper for callers that only need a digital PDF string."""

    return "\n\n".join(_read_pages(Path(pdf_path)))


def extract_document_text(pdf_path: str | Path) -> tuple[str, bool]:
    """Compatibility helper returning text plus whether OCR was used."""

    pages = extract_pages(pdf_path)
    used_ocr = any(page.extraction_method.startswith("ocr") for page in pages)
    return "\n\n".join(page.text for page in pages), used_ocr

