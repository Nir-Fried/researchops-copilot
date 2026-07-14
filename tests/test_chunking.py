from langchain_core.documents import Document

from researchops_copilot.ingestion.chunking import split_documents


def test_split_documents_preserves_metadata_and_adds_chunk_ids() -> None:
    document = Document(
        page_content="This is a sentence about machine learning. " * 100,
        metadata={
            "document_id": "test-document",
            "source": "test.pdf",
            "page_number": 1,
        },
    )

    chunks = split_documents(
        [document],
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) > 1

    chunk_ids = {chunk.metadata["chunk_id"] for chunk in chunks}

    assert len(chunk_ids) == len(chunks)

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page_number"] == 1
        assert chunk.metadata["document_id"] == "test-document"
        assert "start_index" in chunk.metadata
        assert chunk.metadata["character_count"] == len(chunk.page_content)