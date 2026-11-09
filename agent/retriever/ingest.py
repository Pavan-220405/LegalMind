"""Production PDF ingestion for the three legal RAG collections."""

from __future__ import annotations

import argparse
import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .chunker import chunk_pages, detect_section_boundaries
from .config import (
    COLLECTION_NAMES,
    DOMAIN_ALIASES,
    DOCUMENTS_BY_DOMAIN,
    DocumentSpec,
    resolve_document,
)
from .embeddings import get_embedding_model_name, get_embeddings
from .pdf_extractor import (
    PDFExtractionError,
    extract_document_text,
    extract_pages,
)
from .chroma import (
    delete_collection,
    ensure_collection,
    get_vector_store,
)
from .retrieval import get_bm25_retriever

LOGGER = logging.getLogger(__name__)


def _get_embeddings():
    """Compatibility alias retained for the existing retriever module."""

    return get_embeddings()


def extract_text(pdf_path: str | Path, pdf_name: str = "") -> str:
    """Compatibility wrapper returning complete text with automatic OCR."""

    del pdf_name
    text, _ = extract_document_text(pdf_path)
    return text


@dataclass(frozen=True)
class DocumentIngestionResult:
    source_path: str
    chunks: int
    changed: bool


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_vectorstore(collection_name: str):
    return ensure_collection(collection_name)


def _ensure_embedding_compatibility(vectorstore: Any, collection_name: str) -> Any:
    """Validate the model marker without deleting an existing collection."""

    current_model = get_embedding_model_name()
    metadatas = vectorstore.get(limit=1, include=["metadatas"]).get("metadatas", [])
    if not metadatas:
        return vectorstore
    stored_model = metadatas[0].get("embedding_model")
    if stored_model == current_model:
        return vectorstore

    raise RuntimeError(
        f"Collection {collection_name!r} uses embedding model "
        f"{stored_model or 'unknown/legacy'!r}, not {current_model!r}. "
        "Use the explicit --reset command to reindex this collection."
    )


def _existing_source_ids(vectorstore: Any, source_path: str) -> list[str]:
    """Read existing stable Chroma IDs without generating embeddings."""
    try:
        records = vectorstore.get(where={"source_path": source_path}, include=["metadatas"])
    except TypeError:  # small compatible test doubles may expose get without include
        records = vectorstore.get(where={"source_path": source_path})
    return [str(item) for item in records.get("ids", [])]


def _stable_ids(collection_name: str, source_path: str, chunks: list[Document]) -> list[str]:
    ids: list[str] = []
    for index, chunk in enumerate(chunks):
        payload = "\x1f".join(
            (
                collection_name,
                source_path,
                str(chunk.metadata.get("source_sha256", "")),
                str(index),
                str(chunk.metadata.get("page_start", "")),
                str(chunk.metadata.get("page_end", "")),
                str(chunk.metadata.get("embedding_model", "")),
                chunk.page_content,
            )
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        # Chroma accepts UUIDs; derive them deterministically from the full hash.
        ids.append(str(uuid.UUID(digest[:32])))
        chunk.metadata["chunk_id"] = ids[-1]
    return ids


def _replace_source(
    vectorstore: Any,
    *,
    collection_name: str,
    source_path: str,
    chunks: list[Document],
) -> bool:
    """Upsert one source, preserving old data if embedding/insertion fails."""

    old_ids = _existing_source_ids(vectorstore, source_path)
    new_ids = _stable_ids(collection_name, source_path, chunks)
    if old_ids and set(old_ids) == set(new_ids):
        LOGGER.info("Unchanged source %s; skipping embedding", source_path)
        return False

    old_id_set = set(old_ids)
    ids_to_add = [item for item in new_ids if item not in old_id_set]
    chunks_to_add = [
        chunk for item, chunk in zip(new_ids, chunks) if item not in old_id_set
    ]
    try:
        if chunks_to_add:
            vectorstore.add_documents(chunks_to_add, ids=ids_to_add)
        ids_to_remove = [item for item in old_ids if item not in set(new_ids)]
        if ids_to_remove:
            vectorstore.delete(ids=ids_to_remove)
    except Exception:
        try:
            if ids_to_add:
                vectorstore.delete(ids=ids_to_add)
        except Exception:
            LOGGER.exception("Could not clean partial insertion for %s", source_path)
        raise
    if ids_to_add or old_ids != new_ids:
        get_bm25_retriever.cache_clear()
    return True


def _reset_collection(collection_name: str) -> None:
    """Delete only one configured collection when explicitly requested."""

    delete_collection(collection_name)
    get_bm25_retriever.cache_clear()
    _make_vectorstore(collection_name)
    LOGGER.info("Reset collection %s", collection_name)


def ingest_document(
    *,
    collection_name: str,
    spec: DocumentSpec,
    vectorstore: Any,
) -> DocumentIngestionResult:
    path = resolve_document(spec)
    source_path = spec.relative_path.replace("\\", "/")
    if not path.exists():
        raise FileNotFoundError(f"Configured PDF not found: {path}")

    LOGGER.info("Processing %s -> %s", source_path, collection_name)
    digest = _source_hash(path)
    pages = extract_pages(path)
    page_texts = "\n\n".join(page.text for page in pages)
    boundaries = detect_section_boundaries(page_texts)
    chunks = chunk_pages(
        pages,
        collection=collection_name,
        spec=spec,
        source_path=source_path,
        source_hash=digest,
        domain=next(
            (domain for domain, specs in DOCUMENTS_BY_DOMAIN.items() if spec in specs),
            collection_name,
        ),
    )
    embedding_model = get_embedding_model_name()
    for chunk in chunks:
        chunk.metadata["embedding_model"] = embedding_model
    LOGGER.info(
        "%s: structure boundaries detected=%d, chunks generated=%d",
        source_path,
        len(boundaries),
        len(chunks),
    )
    if not chunks:
        raise PDFExtractionError(f"No usable text/chunks extracted from {path}")

    changed = _replace_source(
        vectorstore,
        collection_name=collection_name,
        source_path=source_path,
        chunks=chunks,
    )
    LOGGER.info(
        "%s: %s %d chunks in %s",
        source_path,
        "updated" if changed else "already current",
        len(chunks),
        collection_name,
    )
    return DocumentIngestionResult(source_path, len(chunks), changed)


def ingest_collection(domain: str, *, reset: bool = False) -> dict[str, Any]:
    """Ingest one legal domain and return a machine-readable summary."""

    from .config import canonical_domain

    domain = canonical_domain(domain)
    collection_name = COLLECTION_NAMES[domain]
    if reset:
        _reset_collection(collection_name)
    vectorstore = _make_vectorstore(collection_name)
    if not reset:
        vectorstore = _ensure_embedding_compatibility(vectorstore, collection_name)
    results: list[DocumentIngestionResult] = []
    failures: list[dict[str, str]] = []

    LOGGER.info("=== %s collection (%s) ===", collection_name, domain)
    for spec in DOCUMENTS_BY_DOMAIN[domain]:
        try:
            results.append(
                ingest_document(
                    collection_name=collection_name,
                    spec=spec,
                    vectorstore=vectorstore,
                )
            )
        except Exception as exc:
            LOGGER.exception("Failed to ingest %s", spec.relative_path)
            failures.append({"source_path": spec.relative_path, "error": str(exc)})

    return {
        "domain": domain,
        "collection": collection_name,
        "documents": len(results),
        "chunks": sum(item.chunks for item in results),
        "changed_documents": sum(item.changed for item in results),
        "failures": failures,
    }


def ingest_all(*, reset: bool = False) -> dict[str, dict[str, Any]]:
    """Ingest consumer, labour, and property independently."""

    summaries = {
        domain: ingest_collection(domain, reset=reset)
        for domain in DOCUMENTS_BY_DOMAIN
    }
    LOGGER.info("=== ingestion summary ===")
    for domain, summary in summaries.items():
        LOGGER.info(
            "%s Collection: documents=%d chunks=%d failures=%d",
            summary["collection"].title(),
            summary["documents"],
            summary["chunks"],
            len(summary["failures"]),
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest legal PDFs into persistent Chroma")
    parser.add_argument(
        "--collection",
        choices=sorted(
            set(DOCUMENTS_BY_DOMAIN)
            | set(DOMAIN_ALIASES)
            | set(COLLECTION_NAMES.values())
        ),
        help="ingest only one legal domain (default: all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="reset only the selected collection(s) before ingestion",
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    summaries = (
        {args.collection: ingest_collection(args.collection, reset=args.reset)}
        if args.collection
        else ingest_all(reset=args.reset)
    )
    return 1 if any(summary["failures"] for summary in summaries.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

