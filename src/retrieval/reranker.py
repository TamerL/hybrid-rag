"""Cross-encoder re-ranking."""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.utils.config import AppConfig
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


class CrossEncoderReranker:
    """Thin wrapper around sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info("Loading cross-encoder: %s", model_name)
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored = []
        for doc, score in zip(documents, scores):
            doc.metadata = {**doc.metadata, "rerank_score": float(score)}
            scored.append(doc)

        scored.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
        return scored[:top_k]


def build_reranker(config: AppConfig) -> Optional[CrossEncoderReranker]:
    if not config.reranking.enabled:
        return None
    cfg = config.reranking
    return CrossEncoderReranker(cfg.model_name, device=cfg.device)
