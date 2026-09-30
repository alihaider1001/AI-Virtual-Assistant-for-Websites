import os

import requests
import streamlit as st


st.set_page_config(
    page_title="Website Assistant",
    page_icon=":material/forum:",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.markdown(
    """
    <style>
    :root { --ink: #f4f0e6; --muted: #9fa49f; --line: #303733; --accent: #d9ff57; --panel: #151a18; }
    .stApp { background: #0b0f0e; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(11, 15, 14, 0.85); }
    [data-testid="stSidebar"] { background: #111614; border-right: 1px solid var(--line); }
    .brand { font-size: 0.78rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
    .hero { padding: 2.4rem 0 1.5rem; border-bottom: 1px solid var(--line); }
    .hero h1 { font-size: clamp(2rem, 4vw, 4.4rem); line-height: 0.98; letter-spacing: -0.04em; margin: 0.4rem 0 1rem; color: var(--ink); }
    .hero p { max-width: 700px; color: var(--muted); font-size: 1.05rem; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 0.9rem 1rem; min-height: 88px; }
    .metric-label { color: var(--muted); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { color: var(--ink); font-size: 1.15rem; font-weight: 700; margin-top: 0.45rem; word-break: break-word; }
    .section-label { color: var(--accent); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700; margin: 1.4rem 0 0.7rem; }
    .source { border-left: 2px solid var(--accent); padding: 0.35rem 0.7rem; margin-top: 0.5rem; color: var(--muted); font-size: 0.82rem; }
    div[data-testid="stChatMessage"] { border-bottom: 1px solid rgba(48, 55, 51, 0.7); padding-bottom: 1rem; }
    div[data-testid="stChatInput"] { border-color: var(--accent); }
    .empty-state { border: 1px dashed var(--line); border-radius: 8px; padding: 1.5rem; margin-top: 1.5rem; color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_status():
    try:
        response = requests.get(f"{API_URL}/status", timeout=4)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as error:
        return None, str(error)


status, status_error = fetch_status()

with st.sidebar:
    st.markdown('<div class="brand">Local RAG system</div>', unsafe_allow_html=True)
    st.markdown("### System status")
    if status:
        if status["index_ready"]:
            st.success("Knowledge index: Ready")
        else:
            st.warning("Knowledge index: Not built")
        if status["ollama_online"]:
            st.success("Ollama: Online")
        else:
            st.error("Ollama: Offline")
        if status["ollama_online"] and not status["generation_model_available"]:
            st.warning(f"Configured model unavailable: {status['generation_model']}")

        st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
        st.write(f"{status['chunk_count']:,} indexed chunks")
        st.write(f"{status['embedding_dimensions']} embedding dimensions")
        st.write(f"Retrieves {status['top_k']} chunks per question")

        st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)
        st.caption("Embedding")
        st.code(status["embedding_model"], language=None)
        st.caption("Answer generation")
        st.code(status["generation_model"], language=None)

        with st.expander("Ollama models detected"):
            if status["ollama_models"]:
                for model in status["ollama_models"]:
                    st.write(model)
            else:
                st.caption("No models reported by Ollama.")
    else:
        st.error("Backend unavailable")
        st.caption(status_error)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Answers use only the indexed website and Wikipedia content.")

st.markdown(
    '<div class="hero"><div class="brand">Research workspace</div><h1>Website<br>Assistant</h1><p>Ask grounded questions about the indexed Shopify and Notion material. Every response is generated locally through Ollama and includes the sources used for retrieval.</p></div>',
    unsafe_allow_html=True,
)

if status:
    metric_columns = st.columns(4)
    metrics = [
        ("Index", "Ready" if status["index_ready"] else "Not built"),
        ("Knowledge", f"{status['chunk_count']:,} chunks"),
        ("Generator", status["generation_model"]),
        (
            "Runtime",
            "Model ready"
            if status["generation_model_available"]
            else "Model unavailable",
        ),
    ]
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown('<div class="empty-state"><strong>Start with a focused question.</strong><br>Try one of these prompts below, or write your own in the message box.</div>', unsafe_allow_html=True)
    prompt_columns = st.columns(3)
    prompts = [
        "What is Shopify used for?",
        "How can Shopify orders be managed?",
        "What is Notion used for?",
    ]
    for column, prompt in zip(prompt_columns, prompts):
        with column:
            if st.button(prompt, use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown('<div class="section-label">Retrieved sources</div>', unsafe_allow_html=True)
            for source in message["sources"]:
                source_text = source.get("source") or source.get("title") or "Indexed document"
                st.markdown(f'<div class="source">{source_text}</div>', unsafe_allow_html=True)

query = st.chat_input("Ask about Shopify, Notion, or the indexed documentation")
query = query or st.session_state.pop("pending_prompt", None)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and asking Ollama..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"query": query},
                    timeout=300,
                )
                response.raise_for_status()
                payload = response.json()
                answer = payload.get("answer", "No answer was returned.")
                sources = payload.get("sources", [])
                st.markdown(answer)
                if sources:
                    st.markdown('<div class="section-label">Retrieved sources</div>', unsafe_allow_html=True)
                    for source in sources:
                        source_text = source.get("source") or source.get("title") or "Indexed document"
                        st.markdown(f'<div class="source">{source_text}</div>', unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except requests.HTTPError as error:
                detail = error.response.json().get("detail", "The backend returned an error.")
                st.error(detail)
            except requests.RequestException as error:
                st.error(f"The backend could not be reached: {error}")