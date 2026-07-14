import math

from langchain_core.tracers.langchain import wait_for_all_tracers

from researchops_copilot.models import get_embeddings, get_llm


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate cosine similarity without an additional dependency."""

    numerator = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))

    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def test_embeddings() -> None:
    embeddings = get_embeddings()

    documents = [
        "K-center selects samples that cover the embedding space.",
        "Random sampling selects each training example independently.",
        "Whisper is an encoder-decoder model used for speech recognition.",
    ]

    query = "Which selection method attempts to cover the embedding space?"

    document_vectors = embeddings.embed_documents(documents)
    query_vector = embeddings.embed_query(query)

    scored_documents = [
        (cosine_similarity(query_vector, vector), document)
        for document, vector in zip(documents, document_vectors, strict=True)
    ]

    scored_documents.sort(reverse=True)

    print("\nEmbedding test")
    print(f"Embedding dimension: {len(query_vector)}")

    for score, document in scored_documents:
        print(f"{score:.4f} | {document}")


def test_llm() -> None:
    llm = get_llm()

    response = llm.invoke(
        "In one sentence, explain what retrieval-augmented generation is."
    )

    print("\nLLM test")
    print(response.content)


def main() -> None:
    try:
        test_embeddings()
        test_llm()
    finally:
        # Ensure LangSmith sends pending traces before the script exits.
        wait_for_all_tracers()


if __name__ == "__main__":
    main()