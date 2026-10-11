"""Small, teaching-friendly Retrieval-Augmented Generation (RAG) engine.

This example uses TF-IDF retrieval so students can inspect the retrieval stage
without downloading a large embedding model. Generation uses a hosted Groq API
model; the PDF itself is parsed and indexed inside the Streamlit app.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_CHUNK_WORDS = 180
DEFAULT_OVERLAP_WORDS = 35


@dataclass
class RAGIndex:
    """A searchable collection of text chunks from one PDF."""

    chunks: list[dict[str, Any]]
    vectorizer: TfidfVectorizer
    matrix: Any
    page_count: int
    document_name: str


def extract_pdf_pages(pdf_bytes: bytes) -> tuple[list[dict[str, Any]], int]:
    """Extract selectable text from a PDF, preserving page numbers (1-based)."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:  # pypdf can raise several parsing exceptions
        raise ValueError("This PDF could not be opened. Please try another PDF.") from exc

    if reader.is_encrypted:
        try:
            decrypt_status = reader.decrypt("")
        except Exception as exc:
            raise ValueError("This PDF is password-protected. Upload an unlocked PDF.") from exc
        if decrypt_status == 0:
            raise ValueError("This PDF is password-protected. Upload an unlocked PDF.")

    pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append({"page": page_number, "text": text})

    page_count = len(reader.pages)
    if not pages:
        raise ValueError(
            "No selectable text was found. If this is a scanned/image-only PDF, "
            "run OCR on it first, then upload the searchable PDF."
        )
    return pages, page_count


def _chunk_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_words < 20:
        raise ValueError("Chunk size must be at least 20 words.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Overlap must be zero or more and smaller than chunk size.")

    chunks: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        segment = words[start : start + chunk_words]
        if segment:
            chunks.append(" ".join(segment))
        if start + chunk_words >= len(words):
            break
    return chunks


def build_index(
    pdf_bytes: bytes,
    document_name: str = "uploaded.pdf",
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> RAGIndex:
    """Extract, chunk, and index the PDF. This is indexing, not model training."""
    pages, page_count = extract_pdf_pages(pdf_bytes)
    chunks: list[dict[str, Any]] = []

    for page in pages:
        for chunk_number, text in enumerate(
            _chunk_words(page["text"], chunk_words, overlap_words), start=1
        ):
            chunks.append(
                {
                    "text": text,
                    "page": page["page"],
                    "chunk": chunk_number,
                }
            )

    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    except ValueError as exc:
        raise ValueError(
            "The PDF text is too short or does not contain searchable words."
        ) from exc

    return RAGIndex(
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
        page_count=page_count,
        document_name=document_name,
    )


def retrieve(index: RAGIndex, question: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Retrieve the most relevant text chunks using TF-IDF cosine similarity."""
    if not question.strip():
        return []

    query_vector = index.vectorizer.transform([question.strip()])
    # TfidfVectorizer normalizes rows by default; the dot product is cosine similarity.
    scores = (index.matrix @ query_vector.T).toarray().ravel()
    best_indices = np.argsort(scores)[::-1][: max(1, min(top_k, len(index.chunks)))]

    results: list[dict[str, Any]] = []
    for idx in best_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        item = dict(index.chunks[int(idx)])
        item["score"] = score
        results.append(item)
    return results


def answer_question(
    index: RAGIndex,
    question: str,
    api_key: str,
    model: str = "openai/gpt-oss-20b",
    top_k: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve PDF evidence and ask a Groq-hosted open-weight model to answer."""
    if not api_key:
        raise ValueError(
            "A Groq API key is required for answer generation. Add GROQ_API_KEY "
            "to Streamlit Secrets or your local environment."
        )

    from groq import Groq

    retrieved = retrieve(index, question, top_k=top_k)
    if not retrieved:
        return (
            "I could not find a relevant passage in this PDF for that question. "
            "Try using different keywords or asking a more specific question.",
            [],
        )

    context_parts = []
    for item in retrieved:
        context_parts.append(
            f"[PDF page {item['page']}, chunk {item['chunk']}]\n{item['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful PDF question-answering assistant for an AI lab. "
                    "Answer using only the supplied PDF context. Do not invent facts or "
                    "use outside knowledge to fill gaps. If the context does not answer "
                    "the question, say that the PDF does not provide enough information. "
                    "Cite supporting pages in the format [Page 2]. Be clear and concise."
                ),
            },
            {
                "role": "user",
                "content": f"PDF CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            },
        ],
        temperature=0.2,
        max_tokens=700,
    )
    answer = completion.choices[0].message.content or "The model returned an empty response."
    return answer, retrieved
