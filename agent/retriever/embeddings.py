"""Local Hugging Face embedding construction for legal retrieval."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def get_embedding_model_name() -> str:
    """Return the model identifier used in IDs and migration metadata."""

    return os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


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

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    device = os.getenv("HF_EMBEDDING_DEVICE", "cpu")
    batch_size = int(os.getenv("HF_EMBEDDING_BATCH_SIZE", "8"))
    model_kwargs: dict[str, str] = {"device": device}
    if token:
        model_kwargs["token"] = token

    return HuggingFaceEmbeddings(
        model_name=get_embedding_model_name(),
        model_kwargs=model_kwargs,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": batch_size,
        },
    )

