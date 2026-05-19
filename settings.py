from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def apply_runtime_environment(settings: dict[str, str | None]) -> None:
    for key, value in settings.items():
        if value is not None and value != "":
            os.environ[key] = value


@dataclass(frozen=True)
class RuntimeSettings:
    model_provider: str = "ollama"
    model_id: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    search_provider: str = "tavily"
    tavily_api_key: str | None = None
    tavily_search_depth: str = "advanced"
    tavily_include_answer: bool = True
    scrapegraph_api_key: str | None = None
    scrapegraph_mode: str = "cloud"
    scrapegraph_llm_model: str = "ollama/llama3.2"
    scrapegraph_llm_base_url: str = "http://localhost:11434"
    scrapegraph_embedding_model: str = "ollama/nomic-embed-text"

    # RAG settings
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_host: str = "http://localhost:11434"
    rag_top_k: int = 5
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 100

    # RAG settings
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_host: str = "http://localhost:11434"
    rag_top_k: int = 5
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 100

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        return cls(
            model_provider=(_env("MODEL_PROVIDER", "ollama") or "ollama"),
            model_id=_env("MODEL_ID", "qwen2.5:7b") or "qwen2.5:7b",
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434",
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_base_url=_env("OPENAI_BASE_URL"),
            search_provider=(_env("SEARCH_PROVIDER", "tavily") or "tavily"),
            tavily_api_key=_env("TAVILY_API_KEY"),
            tavily_search_depth=_env("TAVILY_SEARCH_DEPTH", "advanced") or "advanced",
            tavily_include_answer=_truthy(_env("TAVILY_INCLUDE_ANSWER", "true")),
            scrapegraph_api_key=_env("SGAI_API_KEY"),
            scrapegraph_mode=(_env("SCRAPEGRAPH_MODE", "cloud") or "cloud"),
            scrapegraph_llm_model=_env("SCRAPEGRAPH_LLM_MODEL", "ollama/llama3.2") or "ollama/llama3.2",
            scrapegraph_llm_base_url=_env("SCRAPEGRAPH_LLM_BASE_URL", "http://localhost:11434") or "http://localhost:11434",
            scrapegraph_embedding_model=_env("SCRAPEGRAPH_EMBEDDING_MODEL", "ollama/nomic-embed-text") or "ollama/nomic-embed-text",
            embedding_provider=(_env("EMBEDDING_PROVIDER", "ollama") or "ollama"),
            embedding_model=_env("EMBEDDING_MODEL", "nomic-embed-text") or "nomic-embed-text",
            embedding_host=_env("EMBEDDING_HOST", "http://localhost:11434") or "http://localhost:11434",
            rag_top_k=int(_env("RAG_TOP_K", "5") or "5"),
            rag_chunk_size=int(_env("RAG_CHUNK_SIZE", "800") or "800"),
            rag_chunk_overlap=int(_env("RAG_CHUNK_OVERLAP", "100") or "100"),
        )
