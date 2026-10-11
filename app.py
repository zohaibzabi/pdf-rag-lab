"""Streamlit interface for the PDF RAG teaching lab."""

from __future__ import annotations

import os

import streamlit as st
from groq import APIConnectionError, APIStatusError, RateLimitError

from rag_engine import answer_question, build_index

st.set_page_config(
    page_title="PDF RAG Lab",
    page_icon="📚",
    layout="wide",
)

DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit Secrets or an environment variable."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or os.getenv(name, default)).strip()


st.title("📚 PDF Question Answering — RAG Lab")
st.caption(
    "Upload a PDF, build a searchable index, and ask questions grounded in the document. "
    "Designed for classroom demonstrations."
)

with st.sidebar:
    st.header("Lab settings")
    model_name = get_secret("GROQ_MODEL", DEFAULT_MODEL)
    top_k = st.slider("Retrieved chunks per question", min_value=1, max_value=8, value=4)
    st.markdown("**Pipeline**")
    st.markdown("1. Read PDF text\n2. Split into chunks\n3. Retrieve with TF-IDF\n4. Generate a grounded answer")
    st.caption("TF-IDF is a lightweight lexical retrieval method, not a neural embedding model.")
    st.divider()
    st.markdown("**Model**")
    st.code(model_name, language="text")
    st.caption("The hosted model needs a Groq API key. Free-tier limits apply.")

api_key = get_secret("GROQ_API_KEY")
if not api_key:
    st.info(
        "To enable answers, create a Groq API key and add it as `GROQ_API_KEY` in "
        "Streamlit Secrets (cloud) or your environment (local). Never commit your key to GitHub."
    )

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "doc_signature" not in st.session_state:
    st.session_state.doc_signature = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("1. Upload and index a PDF")
uploaded_pdf = st.file_uploader("Choose a text-based PDF", type=["pdf"])

col1, col2 = st.columns([1, 3])
with col1:
    build_button = st.button("Build RAG index", type="primary", use_container_width=True)
with col2:
    st.caption("Indexing processes the document; it does not fine-tune or train the language model.")

if build_button:
    if uploaded_pdf is None:
        st.warning("Please upload a PDF first.")
    else:
        pdf_bytes = uploaded_pdf.getvalue()
        signature = (uploaded_pdf.name, len(pdf_bytes))
        try:
            with st.spinner("Reading PDF and building the index..."):
                index = build_index(pdf_bytes, document_name=uploaded_pdf.name)
            st.session_state.rag_index = index
            st.session_state.doc_signature = signature
            st.session_state.messages = []
            st.success(
                f"Indexed **{index.document_name}**: {index.page_count} PDF pages, "
                f"{len(index.chunks)} text chunks."
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

index = st.session_state.rag_index
if index is not None:
    st.divider()
    st.subheader("2. Ask questions about your PDF")
    st.write(f"**Indexed document:** {index.document_name} · **Chunks:** {len(index.chunks)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved evidence"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**Page {source['page']} · Chunk {source['chunk']} · "
                            f"Score {source['score']:.3f}**\n\n{source['text']}"
                        )

    question = st.chat_input("Ask a question answered by the uploaded PDF...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not api_key:
                response = "Add your Groq API key in Streamlit Secrets or the local environment to generate answers."
                st.warning(response)
                sources = []
            else:
                try:
                    with st.spinner("Retrieving evidence and generating an answer..."):
                        response, sources = answer_question(
                            index=index,
                            question=question,
                            api_key=api_key,
                            model=model_name,
                            top_k=top_k,
                        )
                    st.markdown(response)
                    if sources:
                        with st.expander("Retrieved evidence"):
                            for source in sources:
                                st.markdown(
                                    f"**Page {source['page']} · Chunk {source['chunk']} · "
                                    f"Score {source['score']:.3f}**\n\n{source['text']}"
                                )
                except RateLimitError:
                    response = (
                        "The model provider's free-tier rate limit was reached. Wait for the "
                        "limit to reset, or try again later."
                    )
                    sources = []
                    st.warning(response)
                except APIConnectionError:
                    response = "Could not connect to the model service. Check your connection and try again."
                    sources = []
                    st.error(response)
                except APIStatusError as exc:
                    response = f"The model API returned an error (HTTP {exc.status_code}). Check the model name and API key."
                    sources = []
                    st.error(response)
                except Exception as exc:
                    response = f"Answer generation failed: {exc}"
                    sources = []
                    st.error(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
else:
    st.info("Upload a PDF and click **Build RAG index** to begin.")

with st.expander("What students should observe"):
    st.markdown(
        "- **Ingestion:** selectable text is extracted from PDF pages.\n"
        "- **Chunking:** pages are split into overlapping word chunks.\n"
        "- **Retrieval:** TF-IDF ranks passages related to the question.\n"
        "- **Augmentation:** retrieved passages are added to the model prompt.\n"
        "- **Generation:** the model answers from the supplied context and should cite page numbers.\n\n"
        "This demo does not fine-tune a model. It creates a temporary index for the uploaded PDF. "
        "Scanned PDFs need OCR before upload."
    )
