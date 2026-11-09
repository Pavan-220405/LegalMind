"""OCRmyPDF integration used by the page-aware PDF extractor."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import settings


class OCRError(RuntimeError):
    """Raised when OCR is required but cannot be completed."""


def run_ocrmypdf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    force_ocr: bool = False,
) -> Path:
    """Create a searchable temporary PDF without modifying the source file.

    ``--skip-text`` is the default for a complete PDF. The extractor passes
    only suspect pages in a temporary PDF and uses ``force_ocr`` there, which
    also handles pages with a tiny or corrupt text layer.
    """

    configured = settings.ocrmypdf_command
    executable = shutil.which(configured)
    if executable is None and Path(configured).exists():
        executable = str(Path(configured))
    if executable is None and configured.lower() in {"ocrmypdf", "ocrmypdf.exe"}:
        sibling = Path(sys.executable).with_name("ocrmypdf.exe")
        if sibling.exists():
            executable = str(sibling)
    if executable is None:
        raise OCRError(
            "OCR is required, but OCRmyPDF was not found. Install OCRmyPDF and "
            "Tesseract, then ensure both commands are available on PATH."
        )

    destination = Path(output_path)
    command = [
        executable,
        "--force-ocr" if force_ocr else "--skip-text",
        "--deskew",
        "--rotate-pages",
        "--output-type",
        "pdf",
        str(input_path),
        str(destination),
    ]
    # Keep Hindi enabled by default so OCR does not silently lose Devanagari.
    # Install hin.traineddata or override OCR_LANGUAGE for English-only sources.
    language = settings.ocr_language
    if language:
        command[1:1] = ["--language", language]
    environment = os.environ.copy()
    tesseract_command = settings.tesseract_cmd or ""
    tesseract_candidates = [
        Path(tesseract_command) if tesseract_command else None,
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in tesseract_candidates:
        if candidate and candidate.exists():
            current_path = environment.get("PATH", "")
            environment["PATH"] = str(candidate.parent) + os.pathsep + current_path
            break

    if language:
        try:
            language_check = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise OCRError(
                "Tesseract was not found. Install Tesseract and add it to PATH."
            ) from exc
        if language_check.returncode == 0:
            available = {
                line.strip()
                for line in language_check.stdout.splitlines()
                if line.strip() and not line.lower().startswith("list of available")
            }
            requested = set(language.replace("+", " ").split())
            missing = sorted(requested - available)
            if missing:
                raise OCRError(
                    "Tesseract language data is missing: "
                    f"{', '.join(missing)}. Install the language packs or set "
                    "OCR_LANGUAGE to languages installed on this machine."
                )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0 or not destination.exists():
        detail = (result.stderr or result.stdout or "no OCRmyPDF output").strip()
        raise OCRError(f"OCRmyPDF failed for {input_path}: {detail[-1200:]}")
    return destination

