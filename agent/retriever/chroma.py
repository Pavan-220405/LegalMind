"""Persistent Chroma collections for the legal document corpus."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAMES
from .embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_chroma_client():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("Persistent Chroma requires chromadb") from exc
    CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIRECTORY))


@lru_cache(maxsize=None)
def get_vector_store(collection_name: str):
    """Open or create a persistent LangChain Chroma collection."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("Chroma retrieval requires langchain-chroma and chromadb") from exc
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PERSIST_DIRECTORY),
        client=get_chroma_client(),
    )


def get_collection(collection_name: str):
    return get_vector_store(collection_name)


class LazyCollection:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def __getattr__(self, attribute: str) -> Any:
        return getattr(get_collection(self.collection_name), attribute)


property_store = LazyCollection(COLLECTION_NAMES["property"])
consumer_store = LazyCollection(COLLECTION_NAMES["consumer"])
labour_store = LazyCollection(COLLECTION_NAMES["labour"])
rent_store = property_store
wages_store = labour_store


def ensure_collection(collection_name: str):
    return get_vector_store(collection_name)


def collection_exists(collection_name: str) -> bool:
    names = get_chroma_client().list_collections()
    return collection_name in {
        item if isinstance(item, str) else item.name for item in names
    }


def collection_count(collection_name: str) -> int:
    if not collection_exists(collection_name):
        return 0
    return int(get_chroma_client().get_collection(collection_name).count())


def delete_collection(collection_name: str) -> None:
    if collection_exists(collection_name):
        get_chroma_client().delete_collection(collection_name)
    get_vector_store.cache_clear()


__all__ = [
    "LazyCollection", "collection_count", "collection_exists", "delete_collection", "get_chroma_client",
    "ensure_collection", "get_collection", "get_vector_store",
]
