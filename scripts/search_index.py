import argparse

from researchops_copilot.retrieval.vector_store import get_vector_store


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search the persistent document vector store."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Question or semantic search query.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve.",
    )

    return parser


def main() -> None:
    args = create_argument_parser().parse_args()

    if args.k <= 0:
        raise ValueError("--k must be greater than zero.")

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query=args.query,
        k=args.k,
    )

    print(f"\nQuery: {args.query}")
    print(f"Retrieved chunks: {len(results)}")

    for rank, document in enumerate(results, start=1):
        metadata = document.metadata
        preview = " ".join(document.page_content.split())

        print(f"\n{'=' * 70}")
        print(f"Rank: {rank}")
        print(f"Source: {metadata.get('source', 'unknown')}")
        print(f"Page: {metadata.get('page_number', 'unknown')}")
        print(f"Chunk ID: {metadata.get('chunk_id', 'unknown')}")
        print("\nText:")
        print(preview[:900])


if __name__ == "__main__":
    main()