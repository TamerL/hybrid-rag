"""Sparse (BM25) retrieval."""

from __future__ import annotations

from typing import Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.utils.config import AppConfig
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


def build_sparse_retriever(
    chunks: list[Document],
    config: AppConfig,
) -> BM25Retriever:
    """
    Build a BM25 retriever.

    Note: rank_bm25 is used under the hood by LangChain's BM25Retriever.
    Parameters k1/b are set via the underlying BM25Okapi if needed;
    LangChain exposes a simpler interface so we keep defaults here and
    document the intended values in config.
    """
    logger.info("Building BM25 index over %d chunks...", len(chunks))
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = config.retrieval.top_k_sparse
    return retriever


def sparse_retrieve(
    retriever: BM25Retriever,
    query: str,
    top_k: Optional[int] = None,
) -> list[Document]:
    if top_k is not None:
        retriever.k = top_k
    return retriever.invoke(query)
