import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from researchops_copilot.retrieval.vector_store import get_vector_store


RetrieverStrategy = Literal["dense", "bm25", "hybrid"]

DEFAULT_RRF_CONSTANT = 60
MAX_BM25_RESULTS = 50


def tokenize_for_bm25(text: str) -> list[str]:
    """Convert text into normalized tokens for BM25."""

    return re.findall(r"\b\w+\b", text.casefold())


def _document_key(document: Document) -> str:
    """Return a stable identifier used when merging retrieval results."""

    chunk_id = document.metadata.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    content_hash = hashlib.sha256(
        document.page_content.encode("utf-8")
    ).hexdigest()

    return content_hash


def load_indexed_documents() -> list[Document]:
    """Load all indexed chunks back from the Chroma collection."""

    vector_store = get_vector_store()

    stored_data = vector_store.get(
        include=["documents", "metadatas"],
    )

    ids = stored_data.get("ids") or []
    contents = stored_data.get("documents") or []
    metadatas = stored_data.get("metadatas") or []

    if not contents:
        raise ValueError(
            "The vector store is empty. Run scripts/build_index.py first."
        )

    documents: list[Document] = []

    for index, content in enumerate(contents):
        if content is None:
            continue

        metadata = dict(metadatas[index] or {})

        if index < len(ids):
            metadata.setdefault("chunk_id", ids[index])

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(
            "No valid documents could be loaded from the vector store."
        )

    return documents


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """Build and cache an in-memory BM25 index."""

    documents = load_indexed_documents()

    retriever = BM25Retriever.from_documents(
        documents=documents,
        preprocess_func=tokenize_for_bm25,
    )

    retriever.k = min(
        MAX_BM25_RESULTS,
        len(documents),
    )

    return retriever


def retrieve_dense(
    query: str,
    k: int,
) -> list[Document]:
    """Retrieve semantically similar chunks from Chroma."""

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query=query,
        k=k,
    )


def retrieve_bm25(
    query: str,
    k: int,
) -> list[Document]:
    """Retrieve chunks using BM25 keyword ranking."""

    retriever = get_bm25_retriever()
    results = retriever.invoke(query)

    return list(results[:k])


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Document]],
    weights: Sequence[float] | None = None,
    limit: int = 4,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
) -> list[Document]:
    """Fuse ranked result lists using weighted reciprocal-rank fusion."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    if rrf_constant < 0:
        raise ValueError("rrf_constant cannot be negative.")

    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError(
            "The number of weights must match the number of result lists."
        )

    if any(weight < 0 for weight in weights):
        raise ValueError("Fusion weights cannot be negative.")

    if sum(weights) == 0:
        raise ValueError("At least one fusion weight must be positive.")

    scores: dict[str, float] = defaultdict(float)
    documents_by_key: dict[str, Document] = {}
    first_seen_order: dict[str, int] = {}

    next_order = 0

    for weight, documents in zip(
        weights,
        result_lists,
        strict=True,
    ):
        for rank, document in enumerate(documents, start=1):
            key = _document_key(document)

            scores[key] += weight / (rrf_constant + rank)
            documents_by_key.setdefault(key, document)

            if key not in first_seen_order:
                first_seen_order[key] = next_order
                next_order += 1

    ranked_keys = sorted(
        scores,
        key=lambda key: (
            -scores[key],
            first_seen_order[key],
        ),
    )

    fused_documents: list[Document] = []

    for key in ranked_keys[:limit]:
        original_document = documents_by_key[key]
        metadata = dict(original_document.metadata)

        metadata.update(
            {
                "retrieval_method": "hybrid",
                "rrf_score": scores[key],
            }
        )

        fused_documents.append(
            Document(
                page_content=original_document.page_content,
                metadata=metadata,
            )
        )

    return fused_documents


def retrieve_hybrid(
    query: str,
    k: int,
) -> list[Document]:
    """Combine dense and BM25 rankings."""

    candidate_k = max(k * 2, 8)

    dense_results = retrieve_dense(
        query=query,
        k=candidate_k,
    )

    bm25_results = retrieve_bm25(
        query=query,
        k=candidate_k,
    )

    return reciprocal_rank_fusion(
        result_lists=[
            dense_results,
            bm25_results,
        ],
        weights=[
            0.5,
            0.5,
        ],
        limit=k,
    )


def retrieve_documents(
    query: str,
    k: int = 4,
    strategy: RetrieverStrategy = "dense",
) -> list[Document]:
    """Retrieve documents using the selected strategy."""

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if strategy == "dense":
        return retrieve_dense(cleaned_query, k)

    if strategy == "bm25":
        return retrieve_bm25(cleaned_query, k)

    if strategy == "hybrid":
        return retrieve_hybrid(cleaned_query, k)

    raise ValueError(f"Unknown retrieval strategy: {strategy}")