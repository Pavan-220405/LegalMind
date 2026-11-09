"""Backward-compatible facade over the Chroma legal retrieval pipeline."""

from __future__ import annotations

from .config import COLLECTION_NAMES, DOMAIN_ALIASES, collection_name
from .retrieval import HybridRetriever, retrieve_documents

DOMAIN_COLLECTIONS = {
    **{domain: name for domain, name in COLLECTION_NAMES.items()},
    **{alias: COLLECTION_NAMES[domain] for alias, domain in DOMAIN_ALIASES.items()},
}


class LazyRetriever:
    """Retriever facade that does not load embeddings until invoked."""

    def __init__(self, collection_name: str, k: int = 4):
        self.collection_name = collection_name
        self.k = k

    def invoke(self, query: str):
        return HybridRetriever(self.collection_name, output_candidates=self.k).invoke(query)

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


property_retriever = LazyRetriever(COLLECTION_NAMES["property"])
consumer_retriever = LazyRetriever(COLLECTION_NAMES["consumer"])
labour_retriever = LazyRetriever(COLLECTION_NAMES["labour"])
rent_retriever = property_retriever
wages_retriever = labour_retriever

DOMAIN_RETRIEVERS = {
    "rent": rent_retriever,
    "rent_tenancy": rent_retriever,
    "consumer": consumer_retriever,
    "wages": wages_retriever,
    "labour": labour_retriever,
    "property": property_retriever,
}


def retrieve(domain: str, query: str, *, k: int = 4) -> list[str]:
    """Legacy helper returning text only, with hybrid retrieval."""

    if not query.strip() or domain.strip().lower() not in DOMAIN_COLLECTIONS:
        return []
    documents = retrieve_documents(
        query,
        collection_name(domain),
        multi_query_enabled=False,
        reranker_enabled=False,
    )
    return [document.page_content for document in documents[:k]]


__all__ = [
    "DOMAIN_COLLECTIONS",
    "DOMAIN_RETRIEVERS",
    "HybridRetriever",
    "LazyRetriever",
    "consumer_retriever",
    "labour_retriever",
    "property_retriever",
    "rent_retriever",
    "retrieve",
    "retrieve_documents",
    "wages_retriever",
]
