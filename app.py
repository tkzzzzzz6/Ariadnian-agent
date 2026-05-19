import os

os.environ["STREAMLIT_TELEMETRY_OPT_OUT"] = "1"

import streamlit as st
from agents import DeepResearcherAgent, RAGResearcherAgent
from settings import apply_runtime_environment, RuntimeSettings
import time
import base64
import re

from rag.retriever import RAGRetriever
from rag.embedder import Embedder
from rag.vector_store import VectorStore

st.set_page_config(
    page_title="Ariadnian-agent",
    page_icon="🔎",
)

with open("./assets/ariadne.png", "rb") as ariadne_file:
    ariadne_base64 = base64.b64encode(ariadne_file.read()).decode()

    title_html = f"<div style=\"display: flex; justify-content: center; align-items: center; width: 100%; padding: 32px 0 24px 0;\"><h1 style=\"margin: 0; padding: 0; font-size: 2.5rem; font-weight: bold;\"><img src=\"data:image/png;base64,{ariadne_base64}\" style=\"height: 60px; margin-left: 12px; vertical-align: middle;\" /> Agentic Deep Searcher with <span style=\"color: #fb542c;\">Agno</span> &amp; <span style=\"color: #8564ff;\">Scrapegraph</span></h1></div>"
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

    st.subheader("📄 RAG (Document QA)")
    use_rag = st.toggle("Enable RAG mode", value=False, help="Upload documents and ask questions based on their content")
    if use_rag:
        embedding_provider = st.selectbox("Embedding provider", ["ollama", "openai"], index=0)
        embedding_model = st.text_input("Embedding model", value="nomic-embed-text")
        embedding_host = st.text_input("Embedding host (Ollama)", value="http://localhost:11434")
        top_k = st.slider("Retrieval top-k", min_value=1, max_value=20, value=5)
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

if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None
if "rag_docs_ingested" not in st.session_state:
    st.session_state.rag_docs_ingested = 0

if use_rag:
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, or Markdown files",
        type=["pdf", "txt", "docx", "md", "markdown"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        settings = RuntimeSettings.from_environment()
        embedder = Embedder(
            provider=embedding_provider,
            model=embedding_model,
            host=embedding_host,
            api_key=openai_api_key if embedding_provider == "openai" else None,
            base_url=openai_base_url if embedding_provider == "openai" else None,
        )
        retriever = RAGRetriever(
            vector_store=VectorStore(),
            embedder=embedder,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

        if st.button("🚀 Process Documents"):
            progress_bar = st.progress(0)
            total_chunks = 0
            for i, uploaded_file in enumerate(uploaded_files):
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    chunks = retriever.ingest(temp_path)
                    total_chunks += chunks
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    st.success(f"✅ {uploaded_file.name}: {chunks} chunks ingested")
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            st.session_state.rag_retriever = retriever
            st.session_state.rag_docs_ingested = total_chunks
            st.info(f"📚 Total chunks in knowledge base: {total_chunks}")

    if st.session_state.rag_docs_ingested > 0:
        st.success(f"📚 Knowledge base ready: {st.session_state.rag_docs_ingested} chunks")
        if st.button("🗑️ Clear Knowledge Base"):
            if st.session_state.rag_retriever:
                st.session_state.rag_retriever.reset()
            st.session_state.rag_retriever = None
            st.session_state.rag_docs_ingested = 0
            st.rerun()

placeholder = "Ask a question about your documents..." if use_rag else "Enter a research topic..."
user_input = st.chat_input(placeholder)

if user_input:
    try:
        settings = RuntimeSettings.from_environment()

        if use_rag and st.session_state.rag_retriever:
            agent = RAGResearcherAgent(
                settings=settings,
                retriever=st.session_state.rag_retriever,
            )
            with st.status("Executing document QA plan...", expanded=True) as status:
                phase1_msg = "🔍 **Phase 1: Retrieving** - Finding relevant document chunks..."
                status.write(phase1_msg)

                phase2_msg = "🔬 **Phase 2: Analyzing** - Synthesizing document context..."
                status.write(phase2_msg)
                context = "\n\n---\n\n".join(agent.retriever.retrieve(user_input))
                analysis = agent.analyst.run(f"Question: {user_input}\n\nContext:\n{context}")

                phase3_msg = "✍️ **Phase 3: Writing Report** - Producing answer based on documents..."
                status.write(phase3_msg)
                report_iterator = agent.writer.run(analysis.content, stream=True)
        else:
            agent = DeepResearcherAgent(settings=settings)
            with st.status("Executing research plan...", expanded=True) as status:
                phase1_msg = "🧠 **Phase 1: Researching** - Finding and extracting relevant information from the web..."
                status.write(phase1_msg)
                research_content = agent.searcher.run(user_input)

                phase2_msg = "🔬 **Phase 2: Analyzing** - Synthesizing and interpreting the research findings..."
                status.write(phase2_msg)
                analysis = agent.analyst.run(research_content.content)

                phase3_msg = "✍️ **Phase 3: Writing Report** - Producing a final, polished report..."
                status.write(phase3_msg)
                report_iterator = agent.writer.run(analysis.content, stream=True)

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
