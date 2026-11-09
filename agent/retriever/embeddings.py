"""Local Hugging Face embedding construction for legal retrieval."""

from __future__ import annotations

from functools import lru_cache

from config import settings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def get_embedding_model_name() -> str:
    """Return the model identifier used in IDs and migration metadata."""

    return settings.hf_embedding_model


@lru_cache(maxsize=1)
def get_embeddings():
    """Build the local Hugging Face embedder only when needed."""

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on environment setup
        raise RuntimeError(
            "Hugging Face embeddings require langchain-huggingface and "
            "sentence-transformers. Install requirements.txt and retry."
        ) from exc

    model_kwargs: dict[str, str] = {"device": settings.hf_embedding_device}
    if settings.hf_token:
        model_kwargs["token"] = settings.hf_token

    return HuggingFaceEmbeddings(
        model_name=get_embedding_model_name(),
        model_kwargs=model_kwargs,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.hf_embedding_batch_size,
        },
    )

