from collections.abc import Sequence

from langchain_core.documents import Document

ALLOWED_METADATA_TYPES = (str, int, float, bool)


def sanitize_document_metadata(
    documents: Sequence[Document],
) -> list[Document]:
    """Remove metadata values that cannot be safely stored in Chroma."""

    sanitized_documents: list[Document] = []

    for document in documents:
        sanitized_metadata = {
            key: value
            for key, value in document.metadata.items()
            if isinstance(value, ALLOWED_METADATA_TYPES)
        }

        sanitized_documents.append(
            Document(
                page_content=document.page_content,
                metadata=sanitized_metadata,
            )
        )

    return sanitized_documents