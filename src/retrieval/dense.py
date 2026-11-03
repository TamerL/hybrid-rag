"""Dense (embedding-based) retrieval with FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import AppConfig, ROOT_DIR
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


def build_embeddings(config: AppConfig) -> Embeddings:
    """Create a local sentence-transformers embedder."""
    cfg = config.embeddings
    logger.info("Loading embedding model: %s (device=%s)", cfg.model_name, cfg.device)
    return HuggingFaceEmbeddings(
        model_name=cfg.model_name,
        model_kwargs={"device": cfg.device},
        encode_kwargs={
            "normalize_embeddings": cfg.normalize,
            "batch_size": cfg.batch_size,
        },
    )


def build_dense_index(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_dir: Optional[Path] = None,
) -> FAISS:
    """Build (or load) a FAISS index from chunks."""
    if persist_dir and (persist_dir / "index.faiss").exists():
        logger.info("Loading existing FAISS index from %s", persist_dir)
        return FAISS.load_local(
            str(persist_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("Building FAISS index over %d chunks...", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if persist_dir:
        persist_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(persist_dir))
        logger.info("Saved FAISS index to %s", persist_dir)

    return vectorstore


def dense_retrieve(
    vectorstore: FAISS,
    query: str,
    top_k: int = 20,
) -> list[Document]:
    """Return top-k documents by dense similarity."""
    return vectorstore.similarity_search(query, k=top_k)
