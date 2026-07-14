import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def _calculate_file_hash(file_path: Path) -> str:
    """Return a stable identifier based on the file contents."""

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        while block := file.read(1024 * 1024):
            hasher.update(block)

    return hasher.hexdigest()[:16]


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    """Load a PDF as one LangChain Document per non-empty page."""

    path = Path(pdf_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.suffix}")

    loader = PyPDFLoader(str(path))
    raw_pages = loader.load()

    document_id = _calculate_file_hash(path)
    pages: list[Document] = []

    for page_index, page in enumerate(raw_pages):
        # Null characters occasionally appear in extracted PDF text.
        content = page.page_content.replace("\x00", "").strip()

        if not content:
            continue

        loader_page_index = page.metadata.get("page", page_index)

        try:
            page_number = int(loader_page_index) + 1
        except (TypeError, ValueError):
            page_number = page_index + 1

        metadata = {
            **page.metadata,
            "source": path.name,
            "file_name": path.name,
            "document_id": document_id,
            "page_number": page_number,
        }

        pages.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    if not pages:
        raise ValueError(
            f"No text was extracted from {path.name}. "
            "The PDF may be scanned or image-based."
        )

    return pages