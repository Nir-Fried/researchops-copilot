import argparse
import json
from pathlib import Path

from researchops_copilot.ingestion.chunking import split_documents
from researchops_copilot.ingestion.pdf_loader import load_pdf_pages


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a PDF and inspect its extracted chunks."
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF file.",
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
        "--preview-count",
        type=int,
        default=3,
        help="Number of chunks to display.",
    )

    return parser


def main() -> None:
    args = create_argument_parser().parse_args()

    pages = load_pdf_pages(args.pdf_path)

    chunks = split_documents(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    total_characters = sum(len(page.page_content) for page in pages)
    average_chunk_size = (
        sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    )

    print("\nPDF ingestion summary")
    print(f"File: {args.pdf_path.name}")
    print(f"Pages with extracted text: {len(pages)}")
    print(f"Extracted characters: {total_characters:,}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Average chunk size: {average_chunk_size:.1f} characters")

    for index, chunk in enumerate(chunks[: args.preview_count], start=1):
        preview = " ".join(chunk.page_content.split())

        print(f"\n{'=' * 70}")
        print(f"Chunk preview {index}")
        print(json.dumps(chunk.metadata, indent=2, ensure_ascii=False))
        print("\nText:")
        print(preview[:700])


if __name__ == "__main__":
    main()