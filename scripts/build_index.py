import argparse
from pathlib import Path

from researchops_copilot.ingestion.chunking import split_documents
from researchops_copilot.ingestion.metadata import sanitize_document_metadata
from researchops_copilot.ingestion.pdf_loader import load_pdf_pages
from researchops_copilot.retrieval.vector_store import get_vector_store


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a persistent Chroma index from PDF documents."
    )

    parser.add_argument(
        "document_path",
        type=Path,
        nargs="?",
        default=Path("data/documents"),
        help="A PDF file or directory containing PDF files.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Maximum chunk size in characters.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Target overlap between chunks in characters.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma collection before indexing.",
    )

    return parser


def find_pdf_files(document_path: Path) -> list[Path]:
    """Return PDF files from a file path or directory."""

    path = document_path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Document path not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, received: {path}")
        return [path]

    pdf_files = sorted(path.rglob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {path}")

    return pdf_files


def main() -> None:
    args = create_argument_parser().parse_args()

    pdf_files = find_pdf_files(args.document_path)
    all_chunks = []

    print(f"\nFound {len(pdf_files)} PDF file(s).")

    for pdf_path in pdf_files:
        pages = load_pdf_pages(pdf_path)

        chunks = split_documents(
            pages,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        chunks = sanitize_document_metadata(chunks)
        all_chunks.extend(chunks)

        print(
            f"Loaded {pdf_path.name}: "
            f"{len(pages)} pages, {len(chunks)} chunks"
        )

    vector_store = get_vector_store()

    if args.reset:
        print("\nResetting existing collection...")
        vector_store.reset_collection()

    chunk_ids = [
        str(chunk.metadata["chunk_id"])
        for chunk in all_chunks
    ]

    print(f"\nEmbedding and indexing {len(all_chunks)} chunks...")

    vector_store.add_documents(
        documents=all_chunks,
        ids=chunk_ids,
    )

    stored_data = vector_store.get()
    stored_count = len(stored_data["ids"])

    print("\nIndexing completed")
    print(f"Documents processed: {len(pdf_files)}")
    print(f"Chunks indexed: {len(all_chunks)}")
    print(f"Total chunks currently stored: {stored_count}")
    print("Vector store directory: data/vectorstore")


if __name__ == "__main__":
    main()