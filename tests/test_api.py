from unittest.mock import Mock

from fastapi.testclient import TestClient

from backend import main
from backend.ingestion import InvalidPDFError
from backend.llm import OllamaUnavailableError
from backend.store import SearchResult


def test_upload_endpoint_reports_success_and_invalid_file(
    monkeypatch,
) -> None:
    def process_pdf(filename: str, content: bytes) -> dict:
        if content == b"invalid":
            raise InvalidPDFError("The uploaded file does not have a valid PDF header")
        document = {
            "document_id": "a" * 64,
            "filename": filename,
            "page_count": 1,
            "chunk_count": 2,
        }
        return {
            "status": "processed",
            "message": "PDF processed",
            "document": document,
        }

    monkeypatch.setattr(main, "process_pdf", process_pdf)
    monkeypatch.setattr(main.store, "counts", lambda: (1, 2))
    client = TestClient(main.app)

    response = client.post(
        "/upload-pdf",
        files=[
            ("files", ("valid.pdf", b"valid", "application/pdf")),
            ("files", ("invalid.pdf", b"invalid", "application/pdf")),
        ],
    )

    assert response.status_code == 207
    body = response.json()
    assert [item["status"] for item in body["results"]] == ["processed", "failed"]
    assert body["total_documents"] == 1
    assert body["total_chunks"] == 2


def test_ask_returns_grounded_answer_and_source(monkeypatch) -> None:
    result = SearchResult(
        text="The document describes a farmer training programme.",
        metadata={"source": "scheme.pdf", "page_number": 2},
        distance=0.1,
    )
    monkeypatch.setattr(main, "retrieve", Mock(return_value=[result]))
    monkeypatch.setattr(
        main,
        "generate_answer",
        Mock(
            return_value=(
                "The document describes farmer training [scheme.pdf, page 2]."
            )
        ),
    )
    client = TestClient(main.app)

    response = client.post("/ask", json={"question": "What training is described?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("The document describes farmer training")
    assert body["used_chunks"] == 1
    assert body["sources"][0]["filename"] == "scheme.pdf"
    assert body["sources"][0]["page_number"] == 2


def test_ask_without_searchable_documents_returns_conflict(monkeypatch) -> None:
    monkeypatch.setattr(main, "retrieve", Mock(return_value=[]))
    client = TestClient(main.app)

    response = client.post("/ask", json={"question": "Who is eligible?"})

    assert response.status_code == 409
    assert "Upload a PDF" in response.json()["detail"]


def test_ask_reports_ollama_unavailability(monkeypatch) -> None:
    result = SearchResult(
        text="Relevant excerpt",
        metadata={"source": "scheme.pdf", "page_number": 1},
        distance=0.2,
    )
    monkeypatch.setattr(main, "retrieve", Mock(return_value=[result]))
    monkeypatch.setattr(
        main,
        "generate_answer",
        Mock(
            side_effect=OllamaUnavailableError(
                "Start Ollama before asking a question"
            )
        ),
    )
    client = TestClient(main.app)

    response = client.post("/ask", json={"question": "What does this document say?"})

    assert response.status_code == 503
    assert "Start Ollama" in response.json()["detail"]
