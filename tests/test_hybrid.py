"""Tests for RRF fusion."""

from langchain_core.documents import Document

from src.retrieval.hybrid import reciprocal_rank_fusion


def _doc(cid: str, content: str = "x") -> Document:
    return Document(page_content=content, metadata={"chunk_id": cid})


def test_rrf_prefers_docs_appearing_in_both_lists():
    list_a = [_doc("a"), _doc("b"), _doc("c")]
    list_b = [_doc("b"), _doc("d"), _doc("a")]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=4)
    ids = [d.metadata["chunk_id"] for d in fused]
    # "a" and "b" appear in both → should rank highest
    assert ids[0] in {"a", "b"}
    assert ids[1] in {"a", "b"}
    assert "c" in ids or "d" in ids
