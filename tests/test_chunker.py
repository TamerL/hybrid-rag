"""Unit tests for chunking."""

from langchain_core.documents import Document

from src.data.chunker import chunk_documents
from src.utils.config import load_config


def test_chunk_documents_produces_chunks():
    config = load_config()
    docs = [
        Document(
            page_content="This is a short test document. " * 50,
            metadata={"source": "test.md", "file_id": "abc123", "title": "Test"},
        )
    ]
    chunks = chunk_documents(docs, config)
    assert len(chunks) >= 1
    assert all("chunk_id" in c.metadata for c in chunks)
    assert all("chunk_index" in c.metadata for c in chunks)
