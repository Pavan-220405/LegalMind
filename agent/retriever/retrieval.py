"""Reusable hybrid retrieval pipeline for persistent Chroma collections.

The pipeline is deliberately domain-scoped:

    original query -> optional MultiQuery expansion -> Chroma dense + BM25
    EnsembleRetriever with reciprocal-rank fusion -> stable-id deduplication -> optional BGE
    cross-encoder reranking -> final documents with provenance metadata

No legal reasoning or answer generation belongs in this module.
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import settings

try:
    from langchain_classic.retrievers.multi_query import (
        MultiQueryRetriever as ClassicMultiQueryRetriever,
    )
except ImportError:  # pragma: no cover - dependency is declared in requirements
    ClassicMultiQueryRetriever = None

from .config import (
    DENSE_CANDIDATES,
    BM25_CANDIDATES,
    DENSE_WEIGHT,
    BM25_WEIGHT,
    MULTI_QUERY_COUNT,
    MULTI_QUERY_ENABLED,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_PROVIDER,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_MODE,
    VOYAGE_RERANKER_MODEL,
    collection_name as configured_collection_name,
)
from .chroma import get_vector_store

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_bm25_retriever(collection_name: str, candidate_count: int = BM25_CANDIDATES):
    """Reconstruct a lexical index from persistent Chroma chunks on demand."""
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("BM25 retrieval requires langchain-community and rank-bm25") from exc
    store = get_vector_store(collection_name)
    records = store.get(include=["documents", "metadatas"])
    docs = []
    for chunk_id, content, metadata in zip(
        records.get("ids", []), records.get("documents", []), records.get("metadatas", [])
    ):
        if content is None:
            continue
        provenance = dict(metadata or {})
        provenance.setdefault("chunk_id", str(chunk_id))
        docs.append(Document(page_content=content, metadata=provenance))
    if not docs:
        # BM25's index requires at least one document; an empty corpus returns no hits.
        return EmptyBM25Retriever()
    return BM25Retriever.from_documents(docs, k=candidate_count)


class EmptyBM25Retriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return []


def _document_key(document: Document) -> str:
    """Return a stable deduplication key, with a safe legacy fallback."""

    metadata = document.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    source = "\x1f".join(
        str(metadata.get(field, ""))
        for field in ("source_path", "page_start", "page_end", "section_number")
    )
    return hashlib.sha256(f"{source}\x1f{document.page_content}".encode()).hexdigest()


class HybridRetriever:
    """Domain-scoped Chroma dense and BM25 retriever with weighted RRF fusion."""

    def __init__(
        self,
        collection_name: str,
        *,
        mode: str = RETRIEVAL_MODE,
        dense_candidates: int = DENSE_CANDIDATES,
        sparse_candidates: int = BM25_CANDIDATES,
        output_candidates: int = RETRIEVAL_CANDIDATES,
    ) -> None:
        self.collection_name = configured_collection_name(collection_name)
        self.mode = mode
        self.dense_candidates = dense_candidates
        self.sparse_candidates = sparse_candidates
        self.output_candidates = output_candidates
        self._store = None
        self._ensemble = None

    @property
    def store(self):
        if self._store is None:
            self._store = get_vector_store(self.collection_name)
        return self._store

    def _build_ensemble(self):
        if self._ensemble is None:
            try:
                from langchain_classic.retrievers.ensemble import EnsembleRetriever
            except ImportError:  # pragma: no cover
                from langchain.retrievers.ensemble import EnsembleRetriever
            dense = self.store.as_retriever(search_kwargs={"k": self.dense_candidates})
            bm25 = get_bm25_retriever(self.collection_name, self.sparse_candidates)
            self._ensemble = EnsembleRetriever(
                retrievers=[dense, bm25],
                weights=[DENSE_WEIGHT, BM25_WEIGHT],
                id_key="chunk_id",
            )
        return self._ensemble

    def dense_retrieve(self, query: str) -> list[Document]:
        return self.store.similarity_search(query, k=self.dense_candidates)

    def sparse_retrieve(self, query: str) -> list[Document]:
        return self._build_ensemble().retrievers[1].invoke(query)

    def hybrid_retrieve(self, query: str) -> list[Document]:
        return self._build_ensemble().invoke(query)[: self.output_candidates]

    def invoke(self, query: str) -> list[Document]:
        if not query or not query.strip():
            return []
        mode = self.mode.strip().lower()
        if mode == "dense":
            return self.dense_retrieve(query)
        if mode == "sparse":
            return self.sparse_retrieve(query)
        if mode == "hybrid":
            return self.hybrid_retrieve(query)
        raise ValueError(f"Unsupported retrieval mode: {self.mode!r}")

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Compatibility method for LangChain retriever-style callers."""

        return self.invoke(query)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _parse_alternatives(text: str, original: str, count: int) -> list[str]:
    alternatives: list[str] = []
    original_key = " ".join(original.lower().split())
    for raw_line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", raw_line).strip()
        line = line.strip(" \t\"'")
        if not line:
            continue
        if " ".join(line.lower().split()) == original_key:
            continue
        if line not in alternatives:
            alternatives.append(line)
        if len(alternatives) >= count:
            break
    return alternatives


def _protected_query_terms(query: str) -> list[str]:
    """Capture factual tokens that query expansion must retain verbatim."""
    patterns = (
        r"\b(?:section|sec\.?|rule|regulation|article|chapter)\s+\d+[A-Za-z]?(?:[-/]\d+)?\b",
        r"\b(?:\d{1,2}[/-]){1,2}\d{2,4}\b|\b(?:18|19|20|21)\d{2}\b",
        r"(?:₹|Rs\.?|INR|USD|\$)\s*[\d,]+(?:\.\d+)?",
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|days?|months?|years?)\b",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}\b",
    )
    terms: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, query):
            term = match.group(0).strip()
            # Ignore a capitalized first word that is only sentence casing.
            if term.split()[0] == query.strip().split()[0] and len(term.split()) == 1:
                continue
            if term.lower() not in {item.lower() for item in terms}:
                terms.append(term)
    return terms


def _preserves_query_terms(candidate: str, original: str) -> bool:
    lowered = " ".join(candidate.lower().split())
    return all(" ".join(term.lower().split()) in lowered for term in _protected_query_terms(original))


@lru_cache(maxsize=1)
def get_query_expansion_llm():
    """Build the configured LLM only when MultiQuery is actually enabled."""

    if settings.legal_rag_llm_provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install langchain-groq for Groq query expansion") from exc
        return ChatGroq(
            model=settings.small_model,
            api_key=settings.groq_api_key,
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install langchain-google-genai for Google query expansion"
        ) from exc
    return ChatGoogleGenerativeAI(
        model=settings.small_model,
        temperature=0,
        google_api_key=settings.google_api_key,
    )


def build_classic_multi_query_retriever(retriever: Any, llm: Any):
    """Build LangChain Classic's standard retriever for external integrations.

    The pipeline below uses ``LegalMultiQueryRetriever`` so stable chunk-ID
    deduplication and the exact query-count contract are always enforced.
    This adapter remains available for callers that already use a LangChain
    ``BaseRetriever`` and want the upstream implementation directly.
    """

    if ClassicMultiQueryRetriever is None:  # pragma: no cover
        raise RuntimeError("Install langchain-classic for MultiQueryRetriever")
    return ClassicMultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        include_original=True,
    )


class LegalMultiQueryRetriever:
    """Generate factual query variants and merge results by stable chunk ID.

    This follows the LangChain MultiQueryRetriever contract while retaining
    the original query explicitly. ``llm`` and ``query_generator`` are
    injectable so ordinary tests never need an API key or a model download.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        llm: Any | None = None,
        query_count: int = MULTI_QUERY_COUNT,
        query_generator: Callable[[str, int], Sequence[str]] | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.query_count = query_count
        self.query_generator = query_generator

    def generate_queries(self, original_query: str) -> list[str]:
        if self.query_count < 1:
            return [original_query]
        if self.query_generator is not None:
            generated = list(self.query_generator(original_query, self.query_count))
        else:
            llm = self.llm or get_query_expansion_llm()
            prompt = (
                "Generate exactly {count} alternative search queries for the legal "
                "retrieval request below. Preserve every name, date, monetary "
                "amount, act name, section number, and jurisdiction. Do not add "
                "facts, legal conclusions, or unrelated domains. Return one "
                "query per line and no commentary.\n\nRequest: {query}"
            ).format(count=self.query_count, query=original_query)
            generated = _parse_alternatives(_response_text(llm.invoke(prompt)), original_query, self.query_count)
        alternatives: list[str] = []
        original_key = " ".join(original_query.lower().split())
        for query in generated:
            cleaned = " ".join(str(query).split()).strip()
            if not _preserves_query_terms(cleaned, original_query):
                continue
            if cleaned and " ".join(cleaned.lower().split()) != original_key and cleaned not in alternatives:
                alternatives.append(cleaned)
            if len(alternatives) == self.query_count:
                break
        return [original_query, *alternatives]

    def invoke(self, query: str) -> list[Document]:
        if not query or not query.strip():
            return []
        merged: list[Document] = []
        seen: set[str] = set()
        for variant in self.generate_queries(query):
            for document in self.retriever.invoke(variant):
                key = _document_key(document)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(document)
        return merged

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)


class CohereReranker:
    """LangChain Cohere reranker adapter that retains source metadata."""

    def __init__(self, model_name: str = RERANKER_MODEL, *, compressor: Any | None = None):
        self.model_name = model_name
        self._compressor = compressor

    @property
    def compressor(self):
        if self._compressor is None:
            try:
                from langchain_cohere import CohereRerank
            except ImportError as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "Install langchain-cohere to use Cohere reranking"
                ) from exc
            if not settings.cohere_api_key:
                raise RuntimeError("Set COHERE_API_KEY to use the Cohere reranker")
            # Keep the provider client reusable. We request all candidate ranks;
            # the local top_k controls the documents returned to the caller.
            self._compressor = CohereRerank(
                model=self.model_name,
                top_n=None,
                cohere_api_key=settings.cohere_api_key,
            )
        return self._compressor

    def rank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int = RERANK_TOP_K,
    ) -> list[Document]:
        if not documents or top_k <= 0:
            return []
        ranked: list[Document] = []
        for document in self.compressor.compress_documents(documents, query)[:top_k]:
            document.metadata = dict(document.metadata or {})
            score = document.metadata.get("relevance_score")
            if score is not None:
                document.metadata["reranker_score"] = float(score)
            ranked.append(document)
        return ranked


class VoyageReranker:
    """Optional LangChain Voyage reranker adapter with preserved metadata."""

    def __init__(self, model_name: str = VOYAGE_RERANKER_MODEL, *, compressor: Any | None = None):
        self.model_name = model_name
        self._compressor = compressor

    @property
    def compressor(self):
        if self._compressor is None:
            try:
                from langchain_voyageai import VoyageAIRerank
            except ImportError as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "Install langchain-voyageai to use Voyage AI reranking"
                ) from exc
            if not settings.voyage_api_key:
                raise RuntimeError("Set VOYAGE_API_KEY to use the Voyage AI reranker")
            self._compressor = VoyageAIRerank(
                model=self.model_name,
                top_k=None,
                api_key=settings.voyage_api_key,
            )
        return self._compressor

    def rank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int = RERANK_TOP_K,
    ) -> list[Document]:
        if not documents or top_k <= 0:
            return []
        ranked: list[Document] = []
        for document in self.compressor.compress_documents(documents, query)[:top_k]:
            document.metadata = dict(document.metadata or {})
            score = document.metadata.get("relevance_score")
            if score is not None:
                document.metadata["reranker_score"] = float(score)
            ranked.append(document)
        return ranked


@lru_cache(maxsize=1)
def get_reranker():
    """Use Voyage first and fail over to Cohere for provider rate limits."""
    if RERANKER_PROVIDER == "voyage":
        return VoyageWithCohereFallback(VoyageReranker(), CohereReranker())
    return CohereReranker()


def _is_rate_limit_error(error: Exception) -> bool:
    """Recognize common provider/HTTP rate-limit exception shapes."""
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status = status or getattr(response, "status_code", None)
    if status == 429:
        return True
    code = str(getattr(error, "code", "")).lower()
    name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        "rate_limit" in code
        or "ratelimit" in name
        or "rate limit" in message
        or "too many requests" in message
        or "status code 429" in message
    )


class VoyageWithCohereFallback:
    """Rerank with Voyage; use Cohere only when Voyage is rate limited."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback

    def rank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int = RERANK_TOP_K,
    ) -> list[Document]:
        try:
            return self.primary.rank(query, documents, top_k=top_k)
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            logger.warning("Voyage reranker rate limited; falling back to Cohere")
            return self.fallback.rank(query, documents, top_k=top_k)


def _deduplicate(documents: Iterable[Document]) -> list[Document]:
    result: list[Document] = []
    seen: set[str] = set()
    for document in documents:
        key = _document_key(document)
        if key not in seen:
            seen.add(key)
            result.append(document)
    return result


def retrieve_documents(
    query: str,
    collection_name: str,
    *,
    multi_query_enabled: bool | None = None,
    reranker_enabled: bool | None = None,
    llm: Any | None = None,
    reranker: Any | None = None,
) -> list[Document]:
    """Return provenance-preserving legal chunks for one collection.

    ``collection_name`` may be a configured Chroma collection (for example
    ``rent_law``) or a domain alias (for example ``rent``). It never means
    “search all collections”.
    """

    if not query or not query.strip():
        return []
    base = HybridRetriever(configured_collection_name(collection_name))
    use_multi_query = MULTI_QUERY_ENABLED if multi_query_enabled is None else multi_query_enabled
    use_reranker = RERANKER_ENABLED if reranker_enabled is None else reranker_enabled
    candidates: list[Document]
    if use_multi_query:
        candidates = LegalMultiQueryRetriever(base, llm=llm).invoke(query)
    else:
        candidates = base.invoke(query)
    candidates = _deduplicate(candidates)
    if use_reranker:
        return (reranker or get_reranker()).rank(query, candidates, top_k=RERANK_TOP_K)
    return candidates


__all__ = [
    "HybridRetriever",
    "LegalMultiQueryRetriever",
    "CohereReranker",
    "VoyageReranker",
    "VoyageWithCohereFallback",
    "build_classic_multi_query_retriever",
    "get_query_expansion_llm",
    "get_reranker",
    "retrieve_documents",
]
