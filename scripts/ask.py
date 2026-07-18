import argparse

from langchain_core.tracers.langchain import wait_for_all_tracers

from researchops_copilot.rag.chain import answer_question


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Answer a question using citation-grounded document RAG."
        )
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to ask about the indexed documents.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of document chunks to retrieve.",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display every retrieved chunk after the answer.",
    )

    parser.add_argument(
    "--retriever",
    type=str,
    choices=["dense", "bm25", "hybrid"],
    default="dense",
    help="Retrieval strategy to use.",
    )

    return parser


def main() -> None:
    args = create_argument_parser().parse_args()

    try:
        response = answer_question(
        question=args.question,
        k=args.k,
        retriever_strategy=args.retriever,
        )

        print("\nAnswer")
        print("=" * 70)
        print(response.answer)

        print("\nGrounded:", "yes" if response.answered else "no")
        print("Retriever:", args.retriever)

        if response.citations:
            print("\nSources")

            for citation in response.citations:
                print(
                    f"[{citation.source_id}] "
                    f"{citation.file_name}, "
                    f"page {citation.page_number}"
                )
                print(f"    Chunk: {citation.chunk_id}")

        if args.show_context:
            print("\nRetrieved context")

            for index, document in enumerate(
                response.retrieved_documents,
                start=1,
            ):
                preview = " ".join(document.page_content.split())

                print(f"\n{'-' * 70}")
                print(f"[S{index}]")
                print(
                    "Source:",
                    document.metadata.get("source", "unknown"),
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
        # Give LangSmith time to send pending traces.
        wait_for_all_tracers()


if __name__ == "__main__":
    main()