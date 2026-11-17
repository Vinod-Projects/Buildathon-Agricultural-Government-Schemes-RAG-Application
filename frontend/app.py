import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
_UPLOAD_TIMEOUT_SECONDS = 600
_ASK_TIMEOUT_SECONDS = 180


class BackendError(RuntimeError):
    pass


def request_backend(
    backend_url: str,
    method: str,
    endpoint: str,
    *,
    timeout: int,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        response = requests.request(
            method,
            f"{backend_url}{endpoint}",
            timeout=timeout,
            **kwargs,
        )
    except requests.Timeout as exc:
        raise BackendError(
            "The backend request timed out. Check that the backend is running, "
            "then try again."
        ) from exc
    except requests.ConnectionError as exc:
        raise BackendError(
            "Cannot connect to the backend. Start FastAPI and verify its URL."
        ) from exc
    except requests.RequestException as exc:
        raise BackendError(f"Backend request failed: {exc}") from exc

    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise BackendError("The backend returned a response that was not valid JSON") from exc

    if response.status_code >= 400:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise BackendError(f"Backend returned HTTP {response.status_code}: {detail}")
    if not isinstance(payload, dict):
        raise BackendError("The backend returned an unexpected response")
    return payload


def show_upload_tab(backend_url: str) -> None:
    st.subheader("Upload scheme PDFs")
    st.write(
        "Upload one or more PDF documents. New PDFs are added to the existing "
        "local index; uploading identical file contents again will not create duplicates."
    )
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="The backend allows up to 10 files per request and 20 MB per file.",
        key="scheme_pdf_upload",
    )

    if st.button(
        "Process PDFs",
        type="primary",
        disabled=not uploaded_files,
        key="process_pdfs",
    ):
        files = [
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/pdf",
                ),
            )
            for uploaded_file in uploaded_files
        ]
        try:
            with st.spinner("Extracting, chunking, and indexing the PDFs..."):
                result = request_backend(
                    backend_url,
                    "POST",
                    "/upload-pdf",
                    timeout=_UPLOAD_TIMEOUT_SECONDS,
                    files=files,
                )
        except BackendError as exc:
            st.error(str(exc))
            return

        for item in result.get("results", []):
            message = (
                f"**{item.get('filename', 'PDF')}** — {item.get('message', '')} "
                f"({item.get('page_count', 0)} pages, "
                f"{item.get('chunk_count', 0)} chunks)"
            )
            if item.get("status") == "processed":
                st.success(message)
            elif item.get("status") == "duplicate":
                st.info(message)
            else:
                st.error(message)

        first, second = st.columns(2)
        first.metric("Indexed documents", result.get("total_documents", 0))
        second.metric("Searchable chunks", result.get("total_chunks", 0))


def show_question_tab(backend_url: str) -> None:
    st.subheader("Ask about the uploaded documents")
    st.caption(
        "Answers are generated from retrieved PDF excerpts. Eligibility, subsidy "
        "amounts, and dates should be verified against current official guidance."
    )

    with st.form("question_form"):
        question = st.text_area(
            "Your question",
            placeholder="For example: What does the uploaded document say about this scheme?",
            max_chars=2000,
        )
        submitted = st.form_submit_button("Get an answer", type="primary")

    if not submitted:
        return
    if not question.strip():
        st.error("Enter a question before submitting.")
        return

    try:
        with st.spinner("Searching the documents and asking Ollama..."):
            result = request_backend(
                backend_url,
                "POST",
                "/ask",
                timeout=_ASK_TIMEOUT_SECONDS,
                json={"question": question.strip()},
            )
    except BackendError as exc:
        st.error(str(exc))
        return

    st.markdown("### Answer")
    st.markdown(result.get("answer", "The backend returned no answer."))

    sources = result.get("sources", [])
    st.markdown(f"### Sources ({len(sources)})")
    if not sources:
        st.info("The backend did not return source excerpts for this answer.")
        return

    for index, source in enumerate(sources, start=1):
        page = source.get("page_number")
        page_label = f"page {page}" if page is not None else "page unknown"
        title = (
            f"{index}. {source.get('filename', 'Unknown PDF')} — {page_label} "
            f"(distance {source.get('distance', 0):.3f})"
        )
        with st.expander(title):
            st.text(source.get("text", ""))


def show_documents_tab(backend_url: str) -> None:
    st.subheader("Indexed documents")
    try:
        result = request_backend(
            backend_url,
            "GET",
            "/documents",
            timeout=10,
        )
    except BackendError as exc:
        st.error(str(exc))
        return

    documents = result.get("documents", [])
    first, second = st.columns(2)
    first.metric("Indexed documents", result.get("total_documents", 0))
    second.metric("Searchable chunks", result.get("total_chunks", 0))
    if not documents:
        st.info("No PDFs have been indexed yet.")
        return

    st.dataframe(
        [
            {
                "Filename": document.get("filename"),
                "Pages": document.get("page_count"),
                "Chunks": document.get("chunk_count"),
                "Size (MB)": round(document.get("size_bytes", 0) / (1024 * 1024), 2),
                "Uploaded (UTC)": document.get("uploaded_at"),
            }
            for document in documents
        ],
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Agricultural Schemes RAG",
        page_icon="🌱",
        layout="wide",
    )
    st.title("Agricultural Government Schemes")
    st.caption("Ask questions grounded in your uploaded PDF documents.")

    with st.sidebar:
        st.header("Backend")
        backend_url = st.text_input(
            "FastAPI URL",
            value=_DEFAULT_BACKEND_URL,
            help="The URL where the FastAPI backend is running.",
        ).strip().rstrip("/")
        if not backend_url:
            st.error("Enter the URL for the FastAPI backend.")
        else:
            try:
                health = request_backend(
                    backend_url,
                    "GET",
                    "/health",
                    timeout=5,
                )
                st.success("Backend connected")
                st.metric("Indexed PDFs", health.get("total_documents", 0))
                st.metric("Indexed chunks", health.get("total_chunks", 0))
                st.caption(f"Embeddings: {health.get('embedding_model', 'unknown')}")
            except BackendError as exc:
                st.error(str(exc))

        st.divider()
        st.caption(
            "Use uploaded documents as the source of truth. Verify time-sensitive "
            "scheme details with current official guidance."
        )

    if not backend_url:
        st.warning("Set the FastAPI URL in the sidebar to use the application.")
        return

    upload_tab, question_tab, documents_tab = st.tabs(
        ["Upload PDFs", "Ask a question", "Indexed documents"]
    )
    with upload_tab:
        show_upload_tab(backend_url)
    with question_tab:
        show_question_tab(backend_url)
    with documents_tab:
        show_documents_tab(backend_url)


if __name__ == "__main__":
    main()
