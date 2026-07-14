from collections import defaultdict
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents while preserving and enriching their metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_documents(list(documents))

    chunks: list[Document] = []
    chunks_per_page: dict[tuple[str, int], int] = defaultdict(int)

    for chunk in raw_chunks:
        metadata = dict(chunk.metadata)

        document_id = str(metadata.get("document_id", "unknown"))
        page_number = int(metadata.get("page_number", 0))

        page_key = (document_id, page_number)
        page_chunk_index = chunks_per_page[page_key]
        chunks_per_page[page_key] += 1

        metadata.update(
            {
                "chunk_index_on_page": page_chunk_index,
                "chunk_id": (
                    f"{document_id}:page-{page_number}:"
                    f"chunk-{page_chunk_index}"
                ),
                "character_count": len(chunk.page_content),
            }
        )

        chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    return chunks