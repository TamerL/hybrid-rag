"""Document chunking with documented strategy."""

from __future__ import annotations

from typing import Callable

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import AppConfig, ChunkingConfig
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


def _tiktoken_len(text: str, encoding_name: str = "cl100k_base") -> int:
    """Token length using OpenAI-compatible encoding."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def build_length_function(cfg: ChunkingConfig) -> Callable[[str], int]:
    if cfg.length_function == "tiktoken":
        return lambda text: _tiktoken_len(text, cfg.encoding_name)
    # Fallback to character length
    return len


def chunk_documents(
    documents: list[Document],
    config: AppConfig,
) -> list[Document]:
    """
    Apply recursive character splitting.

    Design decisions (documented in README and config):
    - RecursiveCharacterTextSplitter prefers larger semantic units.
    - 512-token target balances context richness vs. noise.
    - 64-token overlap reduces information loss at boundaries.
    - Metadata is propagated; each chunk also gets chunk_index and total_chunks.
    """
    cfg = config.chunking
    length_fn = build_length_function(cfg)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
        length_function=length_fn,
        is_separator_regex=False,
    )

    chunks: list[Document] = []
    for doc in documents:
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            # Enrich metadata
            split.metadata = {
                **doc.metadata,
                "chunk_index": i,
                "total_chunks": len(splits),
                "chunk_id": f"{doc.metadata.get('file_id', 'unknown')}_{i:03d}",
            }
            chunks.append(split)

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        cfg.chunk_size,
        cfg.chunk_overlap,
    )
    return chunks
