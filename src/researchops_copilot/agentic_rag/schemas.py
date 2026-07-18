from pydantic import BaseModel, Field


class RetrievalGrade(BaseModel):
    """Assessment of the retrieved evidence."""

    relevant_source_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Source identifiers containing information relevant to the "
            "question, such as S1 and S3."
        ),
    )

    sufficient_context: bool = Field(
        description=(
            "True only when the relevant sources contain enough "
            "information to answer the user's question."
        )
    )

    reason: str = Field(
        description=(
            "A brief explanation of why the context is or is not "
            "sufficient."
        )
    )


class RewrittenQuery(BaseModel):
    """Improved search query produced after weak retrieval."""

    rewritten_query: str = Field(
        description=(
            "A concise standalone search query preserving the original "
            "user intent."
        )
    )