"""Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from langchain_core.documents import Document

from src.utils.config import AppConfig
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = 60,
    top_n: int = 15,
) -> list[Document]:
    """
    Reciprocal Rank Fusion.

    score(d) = Σ 1 / (k + rank_i(d))
    Documents appearing in multiple lists receive higher scores.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            # Prefer chunk_id for uniqueness; fall back to content hash
            key = doc.metadata.get("chunk_id") or str(hash(doc.page_content))
            scores[key] += 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    # Sort by fused score descending
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = []
    for key in sorted_keys[:top_n]:
        doc = doc_map[key]
        # Attach fusion score for debugging / analysis
        doc.metadata = {**doc.metadata, "rrf_score": scores[key]}
        fused.append(doc)

    return fused


def hybrid_retrieve(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    config: AppConfig,
) -> list[Document]:
    """Fuse dense and sparse results via RRF."""
    cfg = config.retrieval
    if cfg.fusion_method != "rrf":
        logger.warning("Only RRF is implemented; falling back to RRF")

    fused = reciprocal_rank_fusion(
        [dense_docs, sparse_docs],
        k=cfg.rrf_k,
        top_n=cfg.top_k_hybrid,
    )
    logger.debug(
        "Hybrid fusion: dense=%d, sparse=%d → fused=%d",
        len(dense_docs),
        len(sparse_docs),
        len(fused),
    )
    return fused
