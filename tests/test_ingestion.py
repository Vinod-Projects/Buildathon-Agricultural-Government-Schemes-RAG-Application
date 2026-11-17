from io import BytesIO

import numpy as np
import pytest
from reportlab.pdfgen import canvas

from backend import ingestion
from backend.store import FaissStore


def make_pdf(text: str | None = "Agricultural machinery training") -> bytes:
    buffer = BytesIO()
    page = canvas.Canvas(buffer)
    if text:
        page.drawString(72, 720, text)
    page.showPage()
    page.save()
    return buffer.getvalue()


@pytest.fixture
def isolated_ingestion(tmp_path, monkeypatch):
    index = FaissStore(tmp_path / "index")
    monkeypatch.setattr(ingestion, "store", index)
    monkeypatch.setattr(ingestion, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        ingestion,
        "embed_texts",
        lambda texts: np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        ),
    )
    return index


def test_valid_pdf_is_indexed_and_identical_upload_is_deduplicated(
    isolated_ingestion,
) -> None:
    pdf_bytes = make_pdf()

    first = ingestion.process_pdf("scheme.pdf", pdf_bytes)
    second = ingestion.process_pdf("another-name.pdf", pdf_bytes)

    assert first["status"] == "processed"
    assert first["document"]["page_count"] == 1
    assert first["document"]["chunk_count"] > 0
    assert second["status"] == "duplicate"
    assert isolated_ingestion.counts() == (1, first["document"]["chunk_count"])


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("scheme.pdf", b"", "empty"),
        ("scheme.pdf", b"not a PDF", "header"),
        ("notes.txt", b"%PDF-1.4", "filename"),
    ],
)
def test_invalid_or_empty_uploads_are_rejected(
    isolated_ingestion,
    filename: str,
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(ingestion.InvalidPDFError, match=message):
        ingestion.process_pdf(filename, content)

    assert isolated_ingestion.counts() == (0, 0)


def test_pdf_without_extractable_text_is_rejected(isolated_ingestion) -> None:
    with pytest.raises(ingestion.InvalidPDFError, match="extractable text"):
        ingestion.process_pdf("scanned.pdf", make_pdf(text=None))

    assert isolated_ingestion.counts() == (0, 0)
