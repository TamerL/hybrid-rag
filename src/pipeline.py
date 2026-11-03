"""End-to-end RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document

from src.data.chunker import chunk_documents
from src.data.loader import load_documents
from src.generation.generator import build_llm, generate_answer
from src.retrieval.dense import build_dense_index, build_embeddings, dense_retrieve
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import CrossEncoderReranker, build_reranker
from src.retrieval.sparse import build_sparse_retriever, sparse_retrieve
from src.utils.config import AppConfig, ROOT_DIR, load_config
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


@dataclass
class RAGPipeline:
    """Hybrid RAG pipeline with dense + sparse retrieval and re-ranking."""

    config: AppConfig
    chunks: list[Document] = field(default_factory=list)
    embeddings: Any = None
    dense_index: Any = None
    sparse_retriever: Any = None
    reranker: Optional[CrossEncoderReranker] = None
    llm: Any = None
    _initialized: bool = False

    @classmethod
    def from_config(cls, config_path: Optional[str | Path] = None) -> "RAGPipeline":
        config = load_config(config_path)
        return cls(config=config)

    def initialize(self, force_rebuild: bool = False) -> None:
        """Load corpus, chunk, build indexes, and prepare retrieval models.

        Note: The LLM is loaded lazily only when generation is needed.
        This allows pure retrieval commands to run without an API key.
        """
        if self._initialized and not force_rebuild:
            return

        logger.info("Initializing RAG pipeline...")

        # 1. Load & chunk
        docs = load_documents(self.config)
        self.chunks = chunk_documents(docs, self.config)

        # 2. Dense
        self.embeddings = build_embeddings(self.config)
        persist = ROOT_DIR / self.config.data.processed_dir / "faiss_index"
        if force_rebuild and persist.exists():
            import shutil
            shutil.rmtree(persist)
        self.dense_index = build_dense_index(self.chunks, self.embeddings, persist)

        # 3. Sparse
        self.sparse_retriever = build_sparse_retriever(self.chunks, self.config)

        # 4. Reranker
        self.reranker = build_reranker(self.config)

        # LLM is intentionally NOT built here (lazy load in query())
        self.llm = None

        self._initialized = True
        logger.info("Pipeline ready. %d chunks indexed.", len(self.chunks))

    def retrieve(self, query: str) -> list[Document]:
        """Full retrieval stack: dense + sparse → RRF → optional re-rank."""
        if not self._initialized:
            self.initialize()

        cfg = self.config.retrieval
        dense_docs = dense_retrieve(self.dense_index, query, top_k=cfg.top_k_dense)
        sparse_docs = sparse_retrieve(
            self.sparse_retriever, query, top_k=cfg.top_k_sparse
        )
        hybrid_docs = hybrid_retrieve(dense_docs, sparse_docs, self.config)

        if self.reranker is not None:
            return self.reranker.rerank(
                query, hybrid_docs, top_k=self.config.reranking.top_k
            )
        return hybrid_docs[: self.config.reranking.top_k]

    def query(self, question: str) -> dict[str, Any]:
        """End-to-end: retrieve → generate → return answer + citations + contexts."""
        contexts = self.retrieve(question)

        # Lazy-load LLM only when generation is actually needed
        if self.llm is None:
            self.llm = build_llm(self.config)

        result = generate_answer(question, contexts, self.llm, self.config)
        result["question"] = question
        return result

    def get_stats(self) -> dict[str, Any]:
        return {
            "num_chunks": len(self.chunks),
            "embedding_model": self.config.embeddings.model_name,
            "reranker": self.config.reranking.model_name
            if self.config.reranking.enabled
            else None,
            "generation_model": self.config.generation.model,
            "initialized": self._initialized,
        }
