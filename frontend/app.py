import os

import streamlit as st

from backend.gemini_chain import answer_query_with_gemini
from backend.utils import load_local_index


st.set_page_config(
    page_title="Website Assistant",
    page_icon=":material/forum:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        value = None
    return str(value or os.getenv(name, default))


@st.cache_data(show_spinner=False)
def get_index_status() -> tuple[bool, int, int, str | None]:
    try:
        embeddings, records = load_local_index()
        return True, len(records), int(embeddings.shape[1]), None
    except (RuntimeError, OSError, ValueError) as error:
        return False, 0, 0, str(error)


def render_sources(sources: list[dict]) -> None:
    if sources:
        st.markdown(
            '<div class="section-label">Retrieved sources</div>',
            unsafe_allow_html=True,
        )
        for source in sources:
            source_text = (
                source.get("source") or source.get("title") or "Indexed document"
            )
            st.markdown(
                f'<div class="source">{source_text}</div>',
                unsafe_allow_html=True,
            )


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

gemini_api_key = get_setting("GEMINI_API_KEY")
gemini_model = get_setting("GEMINI_MODEL", "gemini-2.5-flash")
index_ready, chunk_count, embedding_dimensions, index_error = get_index_status()

with st.sidebar:
    st.markdown('<div class="brand">Gemini RAG system</div>', unsafe_allow_html=True)
    st.markdown("### System status")
    if index_ready:
        st.success("Knowledge index: Ready")
    else:
        st.error("Knowledge index: Unavailable")
        st.caption(index_error)
    if gemini_api_key:
        st.success("Gemini: Configured")
    else:
        st.error("Gemini: Not configured")
        st.caption("Add GEMINI_API_KEY in Streamlit secrets.")

    st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
    st.write(f"{chunk_count:,} indexed chunks")
    st.write(f"{embedding_dimensions} embedding dimensions")
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    st.code(gemini_model, language=None)
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Answers use only the indexed website and Wikipedia content.")

st.markdown(
    '<div class="hero"><div class="brand">Research workspace</div><h1>Website<br>Assistant</h1><p>Ask grounded questions about the indexed Shopify and Notion material. Every response is generated with Google Gemini and includes the sources used for retrieval.</p></div>',
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metrics = [
    ("Index", "Ready" if index_ready else "Unavailable"),
    ("Knowledge", f"{chunk_count:,} chunks"),
    ("Generator", gemini_model),
    ("Runtime", "Ready" if gemini_api_key and index_ready else "Needs setup"),
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
    st.markdown(
        '<div class="empty-state"><strong>Start with a focused question.</strong><br>Try one of these prompts below, or write your own in the message box.</div>',
        unsafe_allow_html=True,
    )
    prompt_columns = st.columns(3)
    for column, prompt in zip(
        prompt_columns,
        [
            "What is Shopify used for?",
            "How can Shopify orders be managed?",
            "What is Notion used for?",
        ],
    ):
        with column:
            if st.button(prompt, use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

query = st.chat_input("Ask about Shopify, Notion, or the indexed documentation")
query = query or st.session_state.pop("pending_prompt", None)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and asking Gemini..."):
            try:
                payload = answer_query_with_gemini(query, gemini_api_key, gemini_model)
                answer = payload["answer"]
                sources = payload.get("sources", [])
                st.markdown(answer)
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except (RuntimeError, ValueError) as error:
                st.error(str(error))
