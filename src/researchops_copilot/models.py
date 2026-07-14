from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()

CHAT_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"


def get_llm(
    *,
    temperature: float = 0.2,
) -> ChatGoogleGenerativeAI:
    """Create the project's default Gemini chat model."""

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL_NAME,
        temperature=temperature,
        max_retries=2,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the project's default Gemini embedding model."""

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )