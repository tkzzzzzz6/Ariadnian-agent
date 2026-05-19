from typing import Any

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat, OpenAILike
from agno.tools import Toolkit
from agno.tools.scrapegraph import ScrapeGraphTools
from agno.tools.tavily import TavilyTools

from settings import RuntimeSettings


class _ArxivToolkit(Toolkit):
    def __init__(self, max_results: int = 10, delay_seconds: float = 3.0) -> None:
        super().__init__(name="arxiv")
        self.max_results = max_results
        import arxiv as _arxiv

        self.client = _arxiv.Client(
            page_size=max_results,
            delay_seconds=delay_seconds,
            num_retries=3,
        )

    def search_arxiv(self, query: str) -> str:
        """Search arXiv for academic papers related to the query. Returns titles, authors, summaries and links."""
        import arxiv as _arxiv

        search = _arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=_arxiv.SortCriterion.Relevance,
        )
        try:
            results = list(self.client.results(search))
            if not results:
                return "No papers found on arXiv for this query."
            lines = []
            for r in results:
                authors = ", ".join(str(a) for a in r.authors[:3])
                if len(r.authors) > 3:
                    authors += " et al."
                link = r.pdf_url or r.entry_id
                lines.append(
                    f"- **{r.title}** ({authors})\n  {r.summary[:300]}...\n  Link: {link}"
                )
            return "\n".join(lines)
        except Exception as e:
            return (
                f"arXiv search error: {e}. "
                "The arXiv API may be temporarily unavailable or rate-limited. "
                "Please try again later or switch to another search provider."
            )


class LocalScrapeGraphToolkit(Toolkit):
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


def build_model(settings: RuntimeSettings) -> Any:
    provider = settings.model_provider.lower()

    model_builders = {
        "ollama": lambda: Ollama(id=settings.model_id, host=settings.ollama_host),
        "openai": lambda: OpenAIChat(
            id=settings.model_id,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
        "openai_chat": lambda: OpenAIChat(
            id=settings.model_id,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
        "openai-chat": lambda: OpenAIChat(
            id=settings.model_id,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
        "openai_like": lambda: OpenAILike(
            id=settings.model_id,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
        "openai-compatible": lambda: OpenAILike(
            id=settings.model_id,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
        "openai_compat": lambda: OpenAILike(
            id=settings.model_id,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
        "custom": lambda: OpenAILike(
            id=settings.model_id,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
    }

    if provider not in model_builders:
        raise ValueError(
            f"Unsupported MODEL_PROVIDER={settings.model_provider!r}. Use ollama, openai, or openai_like."
        )

    if provider in {"openai", "openai_chat", "openai-chat", "openai_like", "openai-compatible", "openai_compat", "custom"} and not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not provided and not found in environment")

    return model_builders[provider]()


def build_search_tools(settings: RuntimeSettings) -> list[Any]:
    provider = settings.search_provider.lower()

    if provider == "tavily":
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not provided and not found in environment")
        return [
            TavilyTools(
                api_key=settings.tavily_api_key,
                search_depth=settings.tavily_search_depth,
                include_answer=settings.tavily_include_answer,
            )
        ]

    if provider == "scrapegraph":
        mode = (settings.scrapegraph_mode or "cloud").lower()
        if mode == "cloud":
            if not settings.scrapegraph_api_key:
                raise ValueError("SGAI_API_KEY not provided and not found in environment")
            return [ScrapeGraphTools(api_key=settings.scrapegraph_api_key)]
        if mode == "local":
            return [
                LocalScrapeGraphToolkit(
                    llm_model=settings.scrapegraph_llm_model,
                    llm_base_url=settings.scrapegraph_llm_base_url,
                    embedding_model=settings.scrapegraph_embedding_model,
                )
            ]
        raise ValueError(
            f"Unsupported SCRAPEGRAPH_MODE={mode!r}. Use cloud or local."
        )

    if provider == "arxiv":
        return [_ArxivToolkit()]

    raise ValueError(
        f"Unsupported SEARCH_PROVIDER={settings.search_provider!r}. Use tavily, scrapegraph, or arxiv."
    )


def build_searcher_agent(settings: RuntimeSettings) -> Agent:
    return Agent(
        tools=build_search_tools(settings),
        model=build_model(settings),

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


def build_analyst_agent(settings: RuntimeSettings) -> Agent:
    return Agent(
        model=build_model(settings),
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


def build_writer_agent(settings: RuntimeSettings) -> Agent:
    return Agent(
        model=build_model(settings),
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