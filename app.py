import os

os.environ["STREAMLIT_TELEMETRY_OPT_OUT"] = "1"

import streamlit as st
from agents import DeepResearcherAgent
from settings import apply_runtime_environment, RuntimeSettings
import time
import base64
import re

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔎",
)

with open("./assets/scrapegraph.png", "rb") as scrapegraph_file:
    scrapegraph_base64 = base64.b64encode(scrapegraph_file.read()).decode()

    # Create title with embedded images
    title_html = f"""
    <div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 32px 0 24px 0;">
        <h1 style="margin: 0; padding: 0; font-size: 2.5rem; font-weight: bold;">
            <span style="font-size:2.5rem;">🔎</span> Agentic Deep Searcher with 
            <span style="color: #fb542c;">Agno</span> & 
            <span style="color: #8564ff;">Scrapegraph</span>
            <img src="data:image/png;base64,{scrapegraph_base64}" style="height: 60px; margin-left: 12px; vertical-align: middle;"/>
        </h1>
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Model provider")
    model_provider = st.selectbox(
        "Choose model backend",
        ["ollama", "openai_like", "openai"],
        index=0,
    )
    def _default_model_id(provider: str) -> str:
        if provider == "ollama":
            return "llama3.2"
        if provider == "openai":
            return "gpt-4o-mini"
        return "mimo-v2.5-pro"

    model_id = st.text_input(
        "Model name / id",
        value=_default_model_id(model_provider),
    )
    ollama_host = st.text_input("Ollama host", value="http://localhost:11434")
    openai_api_key = st.text_input("OpenAI API key", type="password")
    openai_base_url = st.text_input("OpenAI base URL / route", value="")
    st.divider()

    st.subheader("Search provider")
    search_provider = st.selectbox(
        "Choose search backend",
        ["tavily", "scrapegraph"],
        index=0,
    )
    tavily_api_key = st.text_input("Tavily API key", type="password")
    scrapegraph_api_key = ""
    scrapegraph_mode = "cloud"
    scrapegraph_llm_model = "ollama/llama3.2"
    scrapegraph_llm_base_url = "http://localhost:11434"
    scrapegraph_embedding_model = "ollama/nomic-embed-text"
    if search_provider == "scrapegraph":
        scrapegraph_mode = st.radio("Scrapegraph mode", ["cloud", "local"], index=0)
        if scrapegraph_mode == "cloud":
            scrapegraph_api_key = st.text_input("Scrapegraph API key", type="password")
        else:
            scrapegraph_llm_model = st.text_input("Local LLM model", value="ollama/llama3.2")
            scrapegraph_llm_base_url = st.text_input("Ollama base URL", value="http://localhost:11434")
            scrapegraph_embedding_model = st.text_input("Embedding model", value="ollama/nomic-embed-text")
    else:
        scrapegraph_api_key = st.text_input("Scrapegraph API key", type="password")
    tavily_search_depth = st.selectbox("Tavily search depth", ["basic", "advanced"], index=1)
    tavily_include_answer = st.checkbox("Include Tavily answer", value=True)
    st.divider()

    st.header("About")
    st.markdown(
        """
    This application is powered by a `DeepResearcherAgent` which leverages multiple AI agents for a comprehensive research process:
    - **Searcher**: Finds and extracts information from the web.
    - **Analyst**: Synthesizes and interprets the research findings.
    - **Writer**: Produces a final, polished report.
    """
    )

apply_runtime_environment(
    {
        "MODEL_PROVIDER": model_provider,
        "MODEL_ID": model_id,
        "OLLAMA_HOST": ollama_host,
        "SEARCH_PROVIDER": search_provider,
        "TAVILY_SEARCH_DEPTH": tavily_search_depth,
        "TAVILY_INCLUDE_ANSWER": "true" if tavily_include_answer else "false",
        "OPENAI_API_KEY": openai_api_key,
        "OPENAI_BASE_URL": openai_base_url,
        "TAVILY_API_KEY": tavily_api_key,
        "SGAI_API_KEY": scrapegraph_api_key,
        "SCRAPEGRAPH_MODE": scrapegraph_mode,
        "SCRAPEGRAPH_LLM_MODEL": scrapegraph_llm_model,
        "SCRAPEGRAPH_LLM_BASE_URL": scrapegraph_llm_base_url,
        "SCRAPEGRAPH_EMBEDDING_MODEL": scrapegraph_embedding_model,
    }
)

# Chat input at the bottom
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    try:
        settings = RuntimeSettings.from_environment()
        agent = DeepResearcherAgent(settings=settings)
        with st.status("Executing research plan...", expanded=True) as status:
            # PHASE 1: Researching
            phase1_msg = "🧠 **Phase 1: Researching** - Finding and extracting relevant information from the web..."
            status.write(phase1_msg)
            research_content = agent.searcher.run(user_input)

            # PHASE 2: Analyzing
            phase2_msg = "🔬 **Phase 2: Analyzing** - Synthesizing and interpreting the research findings..."
            status.write(phase2_msg)
            analysis = agent.analyst.run(research_content.content)

            # PHASE 3: Writing Report
            phase3_msg = (
                "✍️ **Phase 3: Writing Report** - Producing a final, polished report..."
            )
            status.write(phase3_msg)
            report_iterator = agent.writer.run(analysis.content, stream=True)

        # Move report display outside of status block
        full_report = ""
        report_container = st.empty()
        for chunk in report_iterator:
            if chunk.content:
                full_report += chunk.content
                cleaned_report = re.sub(r"^```(?:[a-zA-Z]*)?\n?", "", full_report)
                cleaned_report = re.sub(r"\n?```$", "", cleaned_report)
                report_container.markdown(cleaned_report, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
