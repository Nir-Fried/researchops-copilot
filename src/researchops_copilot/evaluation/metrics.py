import re
from typing import Any

from pydantic import BaseModel, Field

from researchops_copilot.rag.schemas import RAGResponse


class EvaluationCase(BaseModel):
    """One manually curated RAG evaluation example."""

    id: str
    question: str
    answerable: bool

    expected_pages: list[int] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize text for simple case-insensitive term matching."""

    normalized = text.casefold()
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def calculate_term_coverage(
    answer: str,
    expected_terms: list[str],
) -> tuple[float, list[str]]:
    """Return the fraction of expected terms found in an answer."""

    if not expected_terms:
        return 1.0, []

    normalized_answer = normalize_text(answer)

    matched_terms = [
        term
        for term in expected_terms
        if normalize_text(term) in normalized_answer
    ]

    coverage = len(matched_terms) / len(expected_terms)

    return coverage, matched_terms


def evaluate_case(
    case: EvaluationCase,
    response: RAGResponse,
    minimum_term_coverage: float = 0.66,
) -> dict[str, Any]:
    """Evaluate one RAG response using deterministic checks."""

    expected_pages = {
        str(page)
        for page in case.expected_pages
    }

    retrieved_pages = {
        str(document.metadata.get("page_number", "unknown"))
        for document in response.retrieved_documents
    }

    cited_pages = {
        str(citation.page_number)
        for citation in response.citations
    }

    answerability_correct = response.answered == case.answerable

    term_coverage, matched_terms = calculate_term_coverage(
        answer=response.answer,
        expected_terms=case.expected_terms,
    )

    if case.answerable:
        retrieval_hit = bool(expected_pages & retrieved_pages)
        citation_page_hit = bool(expected_pages & cited_pages)

        passed = (
            answerability_correct
            and retrieval_hit
            and citation_page_hit
            and term_coverage >= minimum_term_coverage
        )
    else:
        retrieval_hit = None
        citation_page_hit = len(response.citations) == 0

        passed = (
            answerability_correct
            and citation_page_hit
        )

    return {
        "id": case.id,
        "question": case.question,
        "expected_answerable": case.answerable,
        "actual_answerable": response.answered,
        "answerability_correct": answerability_correct,
        "expected_pages": sorted(expected_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "cited_pages": sorted(cited_pages),
        "retrieval_hit_at_k": retrieval_hit,
        "citation_page_hit": citation_page_hit,
        "term_coverage": term_coverage,
        "matched_terms": matched_terms,
        "expected_terms": case.expected_terms,
        "answer": response.answer,
        "passed": passed,
    }