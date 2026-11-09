"""One source of truth for application, retrieval, and model settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Validated environment-backed settings shared by all app modules."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    project_root: Path = PROJECT_ROOT
    docs_dir: Path = Field(default=PROJECT_ROOT / "docs", validation_alias="LEGAL_DOCS_DIR")
    chroma_persist_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma",
        validation_alias=AliasChoices("CHROMA_PERSIST_DIRECTORY", "CHROMA_PERSIST_DIR"),
    )
    vector_db: Literal["chroma"] = "chroma"

    consumer_collection_name: str = "consumer_law"
    labour_collection_name: str = "wages_law"
    property_collection_name: str = "rent_law"

    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_candidates: int = Field(default=20, ge=1)
    bm25_candidates: int = Field(
        default=20,
        ge=1,
        validation_alias=AliasChoices("BM25_CANDIDATES", "SPARSE_CANDIDATES"),
    )
    retrieval_candidates: int = Field(default=20, ge=1)
    dense_weight: float = Field(default=0.5, ge=0)
    bm25_weight: float = Field(default=0.5, ge=0)
    multi_query_enabled: bool = True
    multi_query_count: int = Field(default=3, ge=1)
    reranker_enabled: bool = True
    reranker_provider: Literal["cohere", "voyage"] = "voyage"
    reranker_model: str = "rerank-v5.0-fast"
    voyage_reranker_model: str = "rerank-3-lite"
    rerank_top_k: int = Field(default=6, ge=1)

    legal_rag_llm_provider: Literal["groq", "google"] = "groq"
    legal_model: str = Field(
        default_factory=lambda data: (
            "openai/gpt-oss-20b"
            if data["legal_rag_llm_provider"] == "groq"
            else "gemini-2.5-flash"
        )
    )
    small_model: str = Field(
        default_factory=lambda data: (
            "openai/gpt-oss-20b"
            if data["legal_rag_llm_provider"] == "groq"
            else "gemini-2.0-flash"
        )
    )
    legal_model_temperature: float = Field(default=0.0, ge=0, le=2)

    hf_embedding_model: str = "BAAI/bge-m3"
    hf_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    )
    hf_embedding_device: str = "cpu"
    hf_embedding_batch_size: int = Field(default=8, ge=1)

    database_url: str | None = Field(default=None, repr=False)
    google_api_key: str | None = Field(default=None, repr=False)
    groq_api_key: str | None = Field(default=None, repr=False)
    cohere_api_key: str | None = Field(default=None, repr=False)
    voyage_api_key: str | None = Field(default=None, repr=False)

    ocrmypdf_command: str = "ocrmypdf"
    ocr_language: str = "eng+hin"
    tesseract_cmd: str | None = Field(default=None, repr=False)
    conversation_context_messages: int = Field(default=16, ge=1)

    @model_validator(mode="before")
    @classmethod
    def choose_default_provider(cls, values: Any) -> Any:
        if isinstance(values, dict):
            values = dict(values)
            if not values.get("legal_rag_llm_provider") and not values.get(
                "LEGAL_RAG_LLM_PROVIDER"
            ):
                has_groq_key = values.get("groq_api_key") or values.get(
                    "GROQ_API_KEY"
                ) or os.getenv("GROQ_API_KEY")
                values["legal_rag_llm_provider"] = "groq" if has_groq_key else "google"
        return values

    @field_validator("docs_dir", "chroma_persist_directory", mode="after")
    @classmethod
    def resolve_project_paths(cls, value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator(
        "legal_rag_llm_provider",
        "reranker_provider",
        "retrieval_mode",
        "vector_db",
        mode="before",
    )
    @classmethod
    def normalize_choice(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def collection_names(self) -> dict[str, str]:
        return {
            "consumer": self.consumer_collection_name,
            "labour": self.labour_collection_name,
            "property": self.property_collection_name,
        }

    @classmethod
    def from_env(cls) -> Settings:
        """Compatibility constructor; BaseSettings reads env and .env directly."""

        return cls()


settings = Settings()


@dataclass(frozen=True)
class DocumentSpec:
    """A known legal source and the domain it belongs to."""

    relative_path: str
    document_name: str
    document_type: str
    act_year: int | None


DOCUMENTS_BY_DOMAIN: dict[str, tuple[DocumentSpec, ...]] = {
    "consumer": (
        DocumentSpec("consumer/Consumer Protection Act 2019.pdf", "Consumer Protection Act, 2019", "act", 2019),
        DocumentSpec("consumer/consumer_general_rules_2020.pdf", "Consumer Protection (General) Rules, 2020", "rules", 2020),
        DocumentSpec("consumer/consumer_regulations_2020.pdf", "Consumer Protection (Consumer Commission) Regulations, 2020", "regulations", 2020),
    ),
    "labour": (
        DocumentSpec("labour/code_on_wages_2019.pdf", "Code on Wages, 2019", "act", 2019),
        DocumentSpec("labour/code_on_wages_2026.pdf", "Code on Wages (Amendment), 2026", "act", 2026),
        DocumentSpec("consumer/code on social security 2020.pdf", "Code on Social Security, 2020", "act", 2020),
    ),
    "property": (
        DocumentSpec("property/model tenancy act 2021.pdf", "Model Tenancy Act, 2021", "act", 2021),
        DocumentSpec("property/registration act 1908.pdf", "Registration Act, 1908", "act", 1908),
        DocumentSpec("property/transfer of property act 1882.pdf", "Transfer of Property Act, 1882", "act", 1882),
    ),
}

COLLECTION_NAMES = settings.collection_names
COLLECTIONS: dict[str, list[str]] = {
    COLLECTION_NAMES[domain]: [spec.relative_path for spec in specs]
    for domain, specs in DOCUMENTS_BY_DOMAIN.items()
}
DOMAIN_ALIASES = {"rent": "property", "rent_tenancy": "property", "wages": "labour"}


def resolve_document(spec: DocumentSpec) -> Path:
    return settings.docs_dir / spec.relative_path


def canonical_domain(domain: str) -> str:
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
    if domain in COLLECTION_NAMES.values():
        return domain
    return COLLECTION_NAMES[canonical_domain(domain)]


# Compatibility names retained for existing ingestion/retrieval imports.
DOCS_DIR = settings.docs_dir
CHROMA_PERSIST_DIRECTORY = settings.chroma_persist_directory
CHROMA_PERSIST_DIR = CHROMA_PERSIST_DIRECTORY
VECTOR_DB = settings.vector_db
RETRIEVAL_MODE = settings.retrieval_mode
DENSE_CANDIDATES = settings.dense_candidates
BM25_CANDIDATES = settings.bm25_candidates
SPARSE_CANDIDATES = BM25_CANDIDATES
RETRIEVAL_CANDIDATES = settings.retrieval_candidates
DENSE_WEIGHT = settings.dense_weight
BM25_WEIGHT = settings.bm25_weight
MULTI_QUERY_ENABLED = settings.multi_query_enabled
MULTI_QUERY_COUNT = settings.multi_query_count
RERANKER_ENABLED = settings.reranker_enabled
RERANKER_PROVIDER = settings.reranker_provider
RERANKER_MODEL = settings.reranker_model
VOYAGE_RERANKER_MODEL = settings.voyage_reranker_model
RERANK_TOP_K = settings.rerank_top_k
LLM_PROVIDER = settings.legal_rag_llm_provider


__all__ = [
    "Settings", "settings", "PROJECT_ROOT", "DOCS_DIR",
    "CHROMA_PERSIST_DIRECTORY", "CHROMA_PERSIST_DIR", "VECTOR_DB",
    "RETRIEVAL_MODE", "DENSE_CANDIDATES", "BM25_CANDIDATES",
    "SPARSE_CANDIDATES", "RETRIEVAL_CANDIDATES", "DENSE_WEIGHT",
    "BM25_WEIGHT", "MULTI_QUERY_ENABLED", "MULTI_QUERY_COUNT",
    "RERANKER_ENABLED", "RERANKER_PROVIDER", "RERANKER_MODEL",
    "VOYAGE_RERANKER_MODEL", "RERANK_TOP_K", "LLM_PROVIDER",
    "DocumentSpec", "DOCUMENTS_BY_DOMAIN", "COLLECTION_NAMES",
    "COLLECTIONS", "DOMAIN_ALIASES", "resolve_document",
    "canonical_domain", "collection_name",
]
