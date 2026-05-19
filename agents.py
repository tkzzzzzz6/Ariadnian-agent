from typing import Any, Iterator

from agno.agent import Agent
from agno.utils.log import logger
from providers import (
    build_analyst_agent,
    build_searcher_agent,
    build_writer_agent,
)
from settings import RuntimeSettings


class DeepResearcherAgent:
    """
    A multi-stage research workflow that:
    1. Gathers information from the web using advanced scraping tools.
    2. Analyzes and synthesizes the findings.
    3. Produces a clear, well-structured report.
    """

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings or RuntimeSettings.from_environment()
        self.searcher: Agent = build_searcher_agent(self.settings)
        self.analyst: Agent = build_analyst_agent(self.settings)
        self.writer: Agent = build_writer_agent(self.settings)

    def run(self, topic: str) -> Iterator[Any]:
        """
        Orchestrates the research, analysis, and report writing process for a given topic.
        """
        logger.info(f"Running deep researcher agent for topic: {topic}")

        # Step 1: Research
        research_content = self.searcher.run(topic)
        # logger.info(f"Searcher content: {research_content.content}")

        logger.info("Analysis started")
        # Step 2: Analysis
        analysis = self.analyst.run(research_content.content)
        # logger.info(f"Analyst analysis: {analysis.content}")

        logger.info("Report Writing Started")
        # Step 3: Report Writing
        report = self.writer.run(analysis.content, stream=True)
        yield from report


def run_research(query: str) -> str:
    agent = DeepResearcherAgent()
    final_report_iterator = agent.run(
        topic=query,
    )
    logger.info("Report Generated")

    # Collect all streaming content into a single string
    full_report = ""
    for chunk in final_report_iterator:
        if chunk.content:
            full_report += chunk.content

    return full_report


if __name__ == "__main__":
    topic = "Extract information about the latest advancements in artificial intelligence, including key breakthroughs, applications, and industry impact."
    response = run_research(topic)
    print(response)
