import argparse

from langchain_core.tracers.langchain import (
    wait_for_all_tracers,
)

from researchops_copilot.agentic_rag.graph import (
    get_agentic_rag_graph,
)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Answer questions using an agentic LangGraph "
            "RAG workflow."
        )
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to ask about the indexed documents.",
    )

    parser.add_argument(
        "--retriever",
        choices=[
            "dense",
            "bm25",
            "hybrid",
        ],
        default="hybrid",
        help="Retrieval strategy to use.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum number of query rewrites.",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display the final relevant chunks.",
    )

    return parser


def main() -> None:
    args = create_argument_parser().parse_args()

    if args.k <= 0:
        raise ValueError("--k must be greater than zero.")

    if args.max_retries < 0:
        raise ValueError(
            "--max-retries cannot be negative."
        )

    graph = get_agentic_rag_graph()

    try:
        result = graph.invoke(
            {
                "question": args.question,
                "query": args.question,
                "retriever_strategy": args.retriever,
                "k": args.k,
                "retry_count": 0,
                "max_retries": args.max_retries,
                "queries_tried": [],
            },
            config={
                "recursion_limit": 12,
            },
        )

        response = result["response"]

        print("\nAnswer")
        print("=" * 70)
        print(response.answer)

        print(
            "\nGrounded:",
            "yes" if response.answered else "no",
        )

        print("Retriever:", args.retriever)
        print(
            "Retrieval attempts:",
            len(result.get("queries_tried", [])),
        )

        print(
            "Query rewrites:",
            result.get("retry_count", 0),
        )

        print("\nQueries tried")

        for index, query in enumerate(
            result.get("queries_tried", []),
            start=1,
        ):
            print(f"{index}. {query}")

        grade_reason = result.get("grade_reason")

        if grade_reason:
            print("\nFinal evidence grade")
            print(grade_reason)

        if response.citations:
            print("\nSources")

            for citation in response.citations:
                print(
                    f"[{citation.source_id}] "
                    f"{citation.file_name}, "
                    f"page {citation.page_number}"
                )
                print(
                    f"    Chunk: {citation.chunk_id}"
                )

        if args.show_context:
            print("\nFinal relevant context")

            for index, document in enumerate(
                response.retrieved_documents,
                start=1,
            ):
                preview = " ".join(
                    document.page_content.split()
                )

                print(f"\n{'-' * 70}")
                print(f"[S{index}]")
                print(
                    "Source:",
                    document.metadata.get(
                        "source",
                        "unknown",
                    ),
                )
                print(
                    "Page:",
                    document.metadata.get(
                        "page_number",
                        "unknown",
                    ),
                )
                print(preview[:1000])

    finally:
        wait_for_all_tracers()


if __name__ == "__main__":
    main()