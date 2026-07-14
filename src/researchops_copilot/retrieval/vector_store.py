from pathlib import Path

from langchain_chroma import Chroma

from researchops_copilot.models import get_embeddings

DEFAULT_COLLECTION_NAME = "researchops_documents"
DEFAULT_PERSIST_DIRECTORY = Path("data/vectorstore")


def get_vector_store(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Open or create the project's persistent Chroma vector store."""

    directory = Path(persist_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(directory),
    )