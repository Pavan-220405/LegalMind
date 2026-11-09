"""Central configuration and document inventory for legal PDF ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


DOCS_DIR = _project_path(os.getenv("LEGAL_DOCS_DIR", ""), PROJECT_ROOT / "docs")
CHROMA_PERSIST_DIRECTORY = _project_path(
    os.getenv("CHROMA_PERSIST_DIRECTORY", os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")),
    PROJECT_ROOT / "data" / "chroma",
)
# Backward-compatible name for older callers.
CHROMA_PERSIST_DIR = CHROMA_PERSIST_DIRECTORY
VECTOR_DB = os.getenv("VECTOR_DB", "chroma").strip().lower()
if VECTOR_DB != "chroma":
    raise ValueError(f"Unsupported VECTOR_DB {VECTOR_DB!r}; this build supports chroma")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
DENSE_CANDIDATES = _env_int("DENSE_CANDIDATES", 20)
BM25_CANDIDATES = _env_int("BM25_CANDIDATES", _env_int("SPARSE_CANDIDATES", 20))
SPARSE_CANDIDATES = BM25_CANDIDATES
RETRIEVAL_CANDIDATES = _env_int("RETRIEVAL_CANDIDATES", 20)
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
MULTI_QUERY_ENABLED = _env_bool("MULTI_QUERY_ENABLED", True)
MULTI_QUERY_COUNT = _env_int("MULTI_QUERY_COUNT", 3)
RERANKER_ENABLED = _env_bool("RERANKER_ENABLED", True)
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "voyage").strip().lower()
if RERANKER_PROVIDER not in {"cohere", "voyage"}:
    raise ValueError("RERANKER_PROVIDER must be 'cohere' or 'voyage'")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "rerank-v5.0-fast")
VOYAGE_RERANKER_MODEL = os.getenv("VOYAGE_RERANKER_MODEL", "rerank-3-lite")
RERANK_TOP_K = _env_int("RERANK_TOP_K", 6)
LLM_PROVIDER = os.getenv(
    "LEGAL_RAG_LLM_PROVIDER",
    "groq" if os.getenv("GROQ_API_KEY") else "google",
).strip().lower()


@dataclass(frozen=True)
class DocumentSpec:
    """A known legal source and the domain it belongs to."""

    relative_path: str
    document_name: str
    document_type: str
    act_year: int | None


DOCUMENTS_BY_DOMAIN: dict[str, tuple[DocumentSpec, ...]] = {
    "consumer": (
        DocumentSpec(
            "consumer/Consumer Protection Act 2019.pdf",
            "Consumer Protection Act, 2019",
            "act",
            2019,
        ),
        DocumentSpec(
            "consumer/consumer_general_rules_2020.pdf",
            "Consumer Protection (General) Rules, 2020",
            "rules",
            2020,
        ),
        DocumentSpec(
            "consumer/consumer_regulations_2020.pdf",
            "Consumer Protection (Consumer Commission) Regulations, 2020",
            "regulations",
            2020,
        ),
    ),
    "labour": (
        DocumentSpec(
            "labour/code_on_wages_2019.pdf",
            "Code on Wages, 2019",
            "act",
            2019,
        ),
        DocumentSpec(
            "labour/code_on_wages_2026.pdf",
            "Code on Wages (Amendment), 2026",
            "act",
            2026,
        ),
        # This file is physically under docs/consumer, but legally belongs here.
        DocumentSpec(
            "consumer/code on social security 2020.pdf",
            "Code on Social Security, 2020",
            "act",
            2020,
        ),
    ),
    "property": (
        DocumentSpec(
            "property/model tenancy act 2021.pdf",
            "Model Tenancy Act, 2021",
            "act",
            2021,
        ),
        DocumentSpec(
            "property/registration act 1908.pdf",
            "Registration Act, 1908",
            "act",
            1908,
        ),
        DocumentSpec(
            "property/transfer of property act 1882.pdf",
            "Transfer of Property Act, 1882",
            "act",
            1882,
        ),
    ),
}


COLLECTION_NAMES = {
    "consumer": os.getenv("CONSUMER_COLLECTION_NAME", "consumer_law"),
    "labour": os.getenv("LABOUR_COLLECTION_NAME", "wages_law"),
    "property": os.getenv("PROPERTY_COLLECTION_NAME", "rent_law"),
}

# Public compatibility mapping. Values are the actual configured paths, not
# inferred paths, so a misplaced Social Security PDF remains correctly routed.
COLLECTIONS: dict[str, list[str]] = {
    COLLECTION_NAMES[domain]: [spec.relative_path for spec in specs]
    for domain, specs in DOCUMENTS_BY_DOMAIN.items()
}


def resolve_document(spec: DocumentSpec) -> Path:
    """Resolve a configured source below the configured docs directory."""

    return DOCS_DIR / spec.relative_path


DOMAIN_ALIASES = {
    "rent": "property",
    "rent_tenancy": "property",
    "wages": "labour",
}


def canonical_domain(domain: str) -> str:
    """Normalize a public domain label to the configured document domain."""

    normalized = domain.strip().lower()
    configured_domain = next(
        (item for item, name in COLLECTION_NAMES.items() if name.lower() == normalized),
        None,
    )
    normalized = configured_domain or DOMAIN_ALIASES.get(normalized, normalized)
    if normalized not in COLLECTION_NAMES:
        raise ValueError(f"Unknown legal domain: {domain}")
    return normalized


def collection_name(domain: str) -> str:
    """Return a configured vector collection for a domain or collection name."""

    if domain in COLLECTION_NAMES.values():
        return domain
    return COLLECTION_NAMES[canonical_domain(domain)]

