from langchain_core.documents import Document

from researchops_copilot.rag.chain import (
    build_citation_map,
    format_retrieved_context,
    validate_generated_answer,
)


def create_test_documents() -> list[Document]:
    return [
        Document(
            page_content="The Transformer uses self-attention.",
            metadata={
                "file_name": "paper.pdf",
                "page_number": 3,
                "chunk_id": "doc:page-3:chunk-0",
            },
        ),
        Document(
            page_content="The big model achieved 28.4 BLEU.",
            metadata={
                "file_name": "paper.pdf",
                "page_number": 8,
                "chunk_id": "doc:page-8:chunk-1",
            },
        ),
    ]


def test_format_retrieved_context_adds_source_ids() -> None:
    context = format_retrieved_context(create_test_documents())

    assert '<source id="S1"' in context
    assert '<source id="S2"' in context
    assert 'page="3"' in context
    assert 'page="8"' in context


def test_build_citation_map_uses_document_metadata() -> None:
    citation_map = build_citation_map(create_test_documents())

    assert citation_map["S1"].page_number == 3
    assert citation_map["S2"].page_number == 8
    assert citation_map["S2"].file_name == "paper.pdf"


def test_validate_generated_answer_removes_fake_citations() -> None:
    generated_output = {
        "answered": True,
        "answer": "The score was 28.4 BLEU [S2].",
        "citation_ids": ["S2", "S99"],
    }

    answer, citations = validate_generated_answer(
        generated_output,
        create_test_documents(),
    )

    assert answer.answered is True
    assert len(citations) == 1
    assert citations[0].source_id == "S2"


def test_grounded_answer_requires_valid_citation() -> None:
    generated_output = {
        "answered": True,
        "answer": "An unsupported answer.",
        "citation_ids": ["S99"],
    }

    answer, citations = validate_generated_answer(
        generated_output,
        create_test_documents(),
    )

    assert answer.answered is False
    assert citations == ()


def test_unanswerable_response_has_no_citations() -> None:
    generated_output = {
        "answered": False,
        "answer": "I do not know.",
        "citation_ids": [],
    }

    answer, citations = validate_generated_answer(
        generated_output,
        create_test_documents(),
    )

    assert answer.answered is False
    assert citations == ()