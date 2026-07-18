from typing import TypedDict

from langchain_core.documents import Document

from researchops_copilot.rag.schemas import RAGResponse
from researchops_copilot.retrieval.retrievers import RetrieverStrategy


class AgenticRAGState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # Original user request
    question: str

    # Current retrieval query, which may be rewritten
    query: str

    # Retrieval configuration
    retriever_strategy: RetrieverStrategy
    k: int

    # Retry configuration
    retry_count: int
    max_retries: int

    # Debugging and observability
    queries_tried: list[str]
    grade_reason: str

    # Retrieved evidence
    retrieved_documents: list[Document]
    relevant_documents: list[Document]
    sufficient_context: bool

    # Final result
    response: RAGResponse