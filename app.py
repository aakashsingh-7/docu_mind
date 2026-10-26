import os

import streamlit as st
from dotenv import load_dotenv

from config import (
    PROVIDERS,
    env_api_key,
    env_provider,
    fetch_model_ids,
    filter_chat_models,
    pick_best_chat_model,
)
from embedders import LOCAL_EMBEDDING_MODEL, LocalEmbedder
from rag_engine import RAGEngine
from utils import validate_pdf_upload

load_dotenv()

st.set_page_config(page_title="DocuMind RAG", page_icon="📚", layout="wide")
st.title("📚 DocuMind — AI-Powered Document Intelligence")
st.caption("Ask questions about your PDFs. Answers are grounded in retrieved passages and include source references.")


@st.cache_resource(show_spinner="Loading local embedding model (the first run downloads it)…")
def load_local_embedder(model_name: str) -> LocalEmbedder:
    return LocalEmbedder(model_name)


def render_sources(sources, expanded=False):
    with st.expander("Sources used", expanded=expanded):
        for src in sources:
            sim = src.get("dense_similarity")
            sim_text = f" · similarity {sim:.2f}" if sim is not None else ""
            st.markdown(f"**[{src['citation']}] {src['source']} — page {src['page']}**{sim_text}")
            st.caption(src["text"])


with st.sidebar:
    st.header("Connection")
    provider_keys = list(PROVIDERS)
    provider = st.selectbox(
        "Provider",
        provider_keys,
        index=provider_keys.index(env_provider()),
        format_func=lambda k: PROVIDERS[k]["label"],
    )
    preset = PROVIDERS[provider]
    api_key_input = st.text_input("API key", type="password", value=env_api_key())
    base_url_input = st.text_input(
        "Base URL (custom provider only)",
        value=os.getenv("LLM_BASE_URL", ""),
        disabled=provider != "custom",
    )
    effective_base_url = (base_url_input.strip() or None) if provider == "custom" else preset["base_url"]

    if st.button("Load models from provider", help="Asks the provider which models your key can use and picks the best one."):
        if not api_key_input.strip():
            st.error("Enter an API key first.")
        else:
            try:
                ids = filter_chat_models(fetch_model_ids(api_key_input.strip(), effective_base_url))
                if not ids:
                    st.warning("The provider returned no chat models.")
                else:
                    st.session_state[f"models_{provider}"] = ids
                    st.session_state[f"chat_model_{provider}"] = pick_best_chat_model(provider, ids, preset["chat_model"])
            except Exception as exc:
                st.error(f"Could not load models: {exc}")

    loaded_models = st.session_state.get(f"models_{provider}")
    if loaded_models:
        chat_model = st.selectbox("Chat model", loaded_models, key=f"chat_model_{provider}")
    else:
        chat_model = st.text_input(
            "Chat model",
            value=os.getenv("CHAT_MODEL") or preset["chat_model"],
            key=f"chat_model_{provider}",
        )

    backend_options = ["api", "local"]
    embedding_backend = st.selectbox(
        "Embeddings (applied when the index is built)",
        backend_options,
        index=backend_options.index(preset["embedding_backend"]),
        format_func=lambda b: "Provider API" if b == "api" else "Local, free (no key)",
        key=f"embedding_backend_{provider}",
    )
    if embedding_backend == "api":
        if provider == "groq":
            st.warning("Groq has no embeddings endpoint. Choose 'Local, free' for embeddings.")
        embedding_model = st.text_input(
            "Embedding model (API)",
            value=os.getenv("EMBEDDING_MODEL") or preset["embedding_model"],
            key=f"embedding_model_{provider}",
        )
    else:
        embedding_model = st.text_input(
            "Local embedding model",
            value=os.getenv("LOCAL_EMBEDDING_MODEL") or LOCAL_EMBEDDING_MODEL,
            key="local_embedding_model",
            help="Needs: pip install -r requirements-local.txt. Downloaded once, then runs on your computer.",
        )

    st.header("Indexing (applied on rebuild)")
    chunk_size = st.slider("Chunk size (characters)", 500, 1800, 1000, 100)
    overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 25)

    st.header("Retrieval (applied live)")
    top_k = st.slider("Retrieved passages", 2, 10, 5)
    dense_weight = st.slider("Dense retrieval weight", 0.0, 1.0, 0.65, 0.05)
    min_similarity = st.slider(
        "Minimum similarity", 0.0, 0.8, 0.2, 0.05,
        help="If the best semantic match is below this value and no keyword matches, the app answers "
             "'not in the documents' without calling the chat model. Tune it with your own test questions; "
             "a good value depends on the embedding model.",
    )
    use_memory = st.checkbox("Understand follow-up questions", value=True,
                             help="Rewrites follow-ups like 'and the second one?' into standalone questions. "
                                  "Costs one extra short model call per follow-up.")
    st.markdown("---")
    st.caption("Privacy note: uploaded PDFs are processed in this app session and sent to the configured API provider "
               "for embeddings and answers. Do not upload confidential documents unless permitted.")

if "engine" not in st.session_state:
    st.session_state.engine = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="PDF text must be extractable; scanned image-only PDFs require OCR, which is not included in this version.",
)

col_a, col_b = st.columns([1, 4])
with col_a:
    index_button = st.button("Build / rebuild index", type="primary", use_container_width=True)
with col_b:
    st.write("Indexing creates embeddings for the extracted text. This may incur API usage.")

if index_button:
    if not api_key_input.strip():
        st.error("Enter an API key in the sidebar or set LLM_API_KEY in your .env file.")
    elif not uploaded_files:
        st.error("Upload at least one PDF first.")
    elif not chat_model.strip() or not embedding_model.strip():
        st.error("Enter both a chat model and an embedding model name.")
    else:
        try:
            docs = []
            progress = st.progress(0, text="Reading PDFs…")
            for i, uploaded in enumerate(uploaded_files):
                validate_pdf_upload(uploaded)
                docs.append((uploaded.name, uploaded.getvalue()))
                progress.progress((i + 1) / len(uploaded_files), text=f"Loaded {uploaded.name}")
            embedder = load_local_embedder(embedding_model.strip()) if embedding_backend == "local" else None
            engine = RAGEngine(
                embedder=embedder,
                api_key=api_key_input.strip(),
                model=chat_model.strip(),
                embedding_model=embedding_model.strip(),
                base_url=effective_base_url,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                dense_weight=dense_weight,
                min_similarity=min_similarity,
            )
            with st.spinner("Extracting text and creating search index…"):
                engine.index_pdfs(docs)
            st.session_state.engine = engine
            st.session_state.doc_names = [name for name, _ in docs]
            st.session_state.messages = []
            st.success(f"Index ready: {len(docs)} PDF(s), {len(engine.chunks)} text chunks.")
        except Exception as exc:
            st.error(f"Could not build the index: {exc}")
            with st.expander("Technical details"):
                st.exception(exc)

if st.session_state.engine:
    st.success("Documents indexed: " + ", ".join(st.session_state.doc_names))
    if st.button("Clear indexed documents"):
        st.session_state.engine = None
        st.session_state.doc_names = []
        st.session_state.messages = []
        st.rerun()

    st.subheader("Ask your documents")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("searched_as"):
                st.caption(f"Searched for: {message['searched_as']}")
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask a question about the indexed PDFs…")
    if question:
        history = list(st.session_state.messages) if use_memory else None
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating an answer…"):
                try:
                    engine = st.session_state.engine
                    # Retrieval and chat settings take effect immediately; indexing settings need a rebuild.
                    engine.dense_weight = dense_weight
                    engine.min_similarity = min_similarity
                    engine.model = chat_model.strip()
                    result = engine.ask(question, top_k=top_k, history=history)
                    answer = result["answer"]
                    st.markdown(answer)
                    searched_as = result["standalone_question"] if result["standalone_question"] != question else None
                    if searched_as:
                        st.caption(f"Searched for: {searched_as}")
                    if result["sources"]:
                        render_sources(result["sources"], expanded=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result["sources"],
                        "searched_as": searched_as,
                    })
                except Exception as exc:
                    st.error(f"Question failed: {exc}")
                    with st.expander("Technical details"):
                        st.exception(exc)
else:
    st.info("Upload PDFs and click **Build / rebuild index** to begin.")
