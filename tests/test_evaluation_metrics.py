from langchain_core.documents import Document

from researchops_copilot.evaluation.metrics import (
    EvaluationCase,
    calculate_term_coverage,
    evaluate_case,
)
from researchops_copilot.rag.schemas import (
    Citation,
    RAGResponse,
)


def test_calculate_term_coverage() -> None:
    coverage, matched_terms = calculate_term_coverage(
        answer=(
            "Attention computes dot products and applies softmax."
        ),
        expected_terms=[
            "dot products",
            "softmax",
            "square root",
        ],
    )

    assert coverage == 2 / 3
    assert matched_terms == [
        "dot products",
        "softmax",
    ]


def test_answerable_case_passes() -> None:
    case = EvaluationCase(
        id="attention",
        question="What is attention?",
        answerable=True,
        expected_pages=[4],
        expected_terms=["softmax"],
    )

    document = Document(
        page_content="Attention applies softmax.",
        metadata={
            "page_number": 4,
            "chunk_id": "doc:page-4:chunk-0",
        },
    )

    response = RAGResponse(
        answer="Attention applies softmax [S1].",
        answered=True,
        citations=(
            Citation(
                source_id="S1",
                file_name="paper.pdf",
                page_number=4,
                chunk_id="doc:page-4:chunk-0",
            ),
        ),
        retrieved_documents=(document,),
    )

    result = evaluate_case(case, response)

    assert result["retrieval_hit_at_k"] is True
    assert result["citation_page_hit"] is True
    assert result["term_coverage"] == 1.0
    assert result["passed"] is True


def test_unanswerable_case_requires_abstention() -> None:
    case = EvaluationCase(
        id="favorite_food",
        question="What is the author's favorite food?",
        answerable=False,
    )

    response = RAGResponse(
        answer=(
            "The documents do not provide enough information."
        ),
        answered=False,
        citations=(),
        retrieved_documents=(),
    )

    result = evaluate_case(case, response)

    assert result["answerability_correct"] is True
    assert result["citation_page_hit"] is True
    assert result["passed"] is True