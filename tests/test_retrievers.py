import pytest
from langchain_core.documents import Document

from researchops_copilot.retrieval.retrievers import (
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)


def create_document(chunk_id: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"chunk_id": chunk_id},
    )


def test_tokenize_for_bm25_normalizes_text() -> None:
    tokens = tokenize_for_bm25(
        "Scaled Dot-Product Attention uses Softmax."
    )

    assert tokens == [
        "scaled",
        "dot",
        "product",
        "attention",
        "uses",
        "softmax",
    ]


def test_rrf_prioritizes_document_found_by_both_retrievers() -> None:
    document_a = create_document("A", "Document A")
    document_b = create_document("B", "Document B")
    document_c = create_document("C", "Document C")

    results = reciprocal_rank_fusion(
        result_lists=[
            [document_a, document_b],
            [document_a, document_c],
        ],
        weights=[0.5, 0.5],
        limit=3,
    )

    assert results[0].metadata["chunk_id"] == "A"
    assert results[0].metadata["retrieval_method"] == "hybrid"
    assert "rrf_score" in results[0].metadata


def test_rrf_removes_duplicate_documents() -> None:
    document_a = create_document("A", "Document A")

    results = reciprocal_rank_fusion(
        result_lists=[
            [document_a],
            [document_a],
        ],
        limit=4,
    )

    assert len(results) == 1


def test_rrf_rejects_mismatched_weights() -> None:
    document_a = create_document("A", "Document A")

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            result_lists=[[document_a], [document_a]],
            weights=[1.0],
        )