from dataclasses import dataclass

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class GeneratedRAGAnswer(BaseModel):
    """Structured output produced by the language model."""

    answered: bool = Field(
        description=(
            "True only when the retrieved context contains enough "
            "information to answer the question."
        )
    )

    answer: str = Field(
        description=(
            "The grounded answer. Include inline source references such "
            "as [S1] after factual claims."
        )
    )

    citation_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Unique source identifiers used in the answer, such as "
            "['S1', 'S3']. Use only identifiers provided in the context."
        ),
    )


@dataclass(frozen=True)
class Citation:
    """Validated source information displayed to the user."""

    source_id: str
    file_name: str
    page_number: int | str
    chunk_id: str


@dataclass(frozen=True)
class RAGResponse:
    """Final application response."""

    answer: str
    answered: bool
    citations: tuple[Citation, ...]
    retrieved_documents: tuple[Document, ...]