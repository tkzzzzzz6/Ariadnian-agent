import os
from typing import Any

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat, OpenAILike
from agno.tools import Toolkit
from agno.tools.scrapegraph import ScrapeGraphTools
from agno.tools.tavily import TavilyTools
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


def build_model() -> Any:
    provider = (_env("MODEL_PROVIDER", "ollama") or "ollama").lower()
    model_id = _env("MODEL_ID", "qwen2.5:7b") or "qwen2.5:7b"

    if provider == "ollama":
        return Ollama(
            id=model_id,
            host=_env("OLLAMA_HOST", "http://localhost:11434"),
        )

    if provider in {"openai", "openai_chat", "openai-chat"}:
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided and not found in environment")
        return OpenAIChat(
            id=model_id,
            api_key=api_key,
            base_url=_env("OPENAI_BASE_URL"),
        )

    if provider in {"openai_like", "openai-compatible", "openai_compat", "custom"}:
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided and not found in environment")
        return OpenAILike(
            id=model_id,
            api_key=api_key,
            base_url=_env("OPENAI_BASE_URL"),
        )

    raise ValueError(
        f"Unsupported MODEL_PROVIDER={provider!r}. Use ollama, openai, or openai_like."
    )


class _LocalScrapeGraphToolkit(Toolkit):
    def __init__(self, llm_model: str, llm_base_url: str, embedding_model: str) -> None:
        super().__init__(name="local_scrapegraph")
        self.llm_config = {
            "model": llm_model,
            "temperature": 0,
            "base_url": llm_base_url,
        }
        self.embedding_config = {
            "model": embedding_model,
            "base_url": llm_base_url,
        }

    def search(self, query: str) -> str:
        """Search the web using DuckDuckGo and extract information with a local LLM."""
        from scrapegraphai.graphs import SearchGraph

        graph_config = {
            "llm": self.llm_config,
            "embeddings": self.embedding_config,
            "max_results": 5,
            "verbose": False,
        }

        search_graph = SearchGraph(prompt=query, config=graph_config)
        result = search_graph.run()
        return str(result)


def build_search_tools() -> list[Any]:
    provider = (_env("SEARCH_PROVIDER", "tavily") or "tavily").lower()

    if provider == "tavily":
        api_key = _env("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not provided and not found in environment")
        return [
            TavilyTools(
                api_key=api_key,
                search_depth=_env("TAVILY_SEARCH_DEPTH", "advanced") or "advanced",
                include_answer=_truthy(_env("TAVILY_INCLUDE_ANSWER", "true")),
            )
        ]

    if provider == "scrapegraph":
        mode = (_env("SCRAPEGRAPH_MODE", "cloud") or "cloud").lower()
        if mode == "cloud":
            api_key = _env("SGAI_API_KEY")
            if not api_key:
                raise ValueError("SGAI_API_KEY not provided and not found in environment")
            return [ScrapeGraphTools(api_key=api_key)]
        if mode == "local":
            return [
                _LocalScrapeGraphToolkit(
                    llm_model=_env("SCRAPEGRAPH_LLM_MODEL", "ollama/llama3.2") or "ollama/llama3.2",
                    llm_base_url=_env("SCRAPEGRAPH_LLM_BASE_URL", "http://localhost:11434") or "http://localhost:11434",
                    embedding_model=_env("SCRAPEGRAPH_EMBEDDING_MODEL", "ollama/nomic-embed-text") or "ollama/nomic-embed-text",
                )
            ]
        raise ValueError(
            f"Unsupported SCRAPEGRAPH_MODE={mode!r}. Use cloud or local."
        )

    raise ValueError(
        f"Unsupported SEARCH_PROVIDER={provider!r}. Use tavily or scrapegraph."
    )


def build_searcher_agent() -> Agent:
    return Agent(
        tools=build_search_tools(),
        model=build_model(),

        markdown=True,
        description=(
            "You are ResearchBot-X, an expert at finding and extracting high-quality, "
            "up-to-date information from the web. Your job is to gather comprehensive, "
            "reliable, and diverse sources on the given topic."
        ),
        instructions=(
            "1. Search for the most recent and authoritative and up-to-date sources (news, blogs, official docs, research papers, forums, etc.) on the topic.\n"
            "2. Extract key facts, statistics, and expert opinions.\n"
            "3. Cover multiple perspectives and highlight any disagreements or controversies.\n"
            "4. Include relevant statistics, data, and expert opinions where possible.\n"
            "5. Organize your findings in a clear, structured format (e.g., markdown table or sections by source type).\n"
            "6. If the topic is ambiguous, clarify with the user before proceeding.\n"
            "7. Be as comprehensive and verbose as possible—err on the side of including more detail.\n"
            "8. Mention the References & Sources of the Content. (It's Must)"
        ),
    )


def build_analyst_agent() -> Agent:
    return Agent(
        model=build_model(),
        markdown=True,
        description=(
            "You are AnalystBot-X, a critical thinker who synthesizes research findings "
            "into actionable insights. Your job is to analyze, compare, and interpret the "
            "information provided by the researcher."
        ),
        instructions=(
            "1. Identify key themes, trends, and contradictions in the research.\n"
            "2. Highlight the most important findings and their implications.\n"
            "3. Suggest areas for further investigation if gaps are found.\n"
            "4. Present your analysis in a structured, easy-to-read format.\n"
            "5. Extract and list ONLY the reference links or sources that were ACTUALLY found and provided by the researcher in their findings. Do NOT create, invent, or hallucinate any links.\n"
            "6. If no links were provided by the researcher, do not include a References section.\n"
            "7. Don't add hallucinations or make up information. Use ONLY the links that were explicitly passed to you by the researcher.\n"
            "8. Verify that each link you include was actually present in the researcher's findings before listing it.\n"
            "9. If there's no Link found from the previous agent then just say, No reference Found."
        ),
    )


def build_writer_agent() -> Agent:
    return Agent(
        model=build_model(),
        markdown=True,
        description=(
            "You are WriterBot-X, a professional technical writer. Your job is to craft "
            "a clear, engaging, and well-structured report based on the analyst's summary."
        ),
        instructions=(
            "1. Write an engaging introduction that sets the context.\n"
            "2. Organize the main findings into logical sections with headings.\n"
            "3. Use bullet points, tables, or lists for clarity where appropriate.\n"
            "4. Conclude with a summary and actionable recommendations.\n"
            "5. Include a References & Sources section ONLY if the analyst provided actual links from their analysis.\n"
            "6. Use ONLY the reference links that were explicitly provided by the analyst in their analysis. Do NOT create, invent, or hallucinate any links.\n"
            "7. If the analyst provided links, format them as clickable markdown links in the References section.\n"
            "8. If no links were provided by the analyst, do not include a References section at all.\n"
            "9. Never add fake or made-up links - only use links that were actually found and passed through the research chain."
        ),
    )