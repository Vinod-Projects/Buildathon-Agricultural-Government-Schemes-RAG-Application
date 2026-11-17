import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_UPLOAD_BYTES,
    UPLOAD_DIR,
)
from backend.embeddings import embed_texts
from backend.store import store

logger = logging.getLogger(__name__)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class InvalidPDFError(ValueError):
    pass


def _safe_filename(filename: str) -> str:
    cleaned = _CONTROL_CHARACTERS.sub("", filename.replace("\\", "/").rsplit("/", 1)[-1])
    cleaned = cleaned.strip().strip(".")
    if not cleaned or not cleaned.lower().endswith(".pdf"):
        raise InvalidPDFError("The uploaded file must have a .pdf filename")
    return cleaned


def process_pdf(filename: str, content: bytes) -> dict[str, Any]:
    safe_filename = _safe_filename(filename)
    if not content:
        raise InvalidPDFError("The uploaded PDF is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise InvalidPDFError(
            f"The PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit"
        )
    if b"%PDF-" not in content[:1024]:
        raise InvalidPDFError("The uploaded file does not have a valid PDF header")

    document_id = hashlib.sha256(content).hexdigest()
    existing_documents = store.list_documents()
    for existing in existing_documents:
        if existing["document_id"] == document_id:
            return {
                "status": "duplicate",
                "message": "This PDF's contents are already indexed",
                "document": existing,
            }

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise InvalidPDFError("Password-protected PDFs are not supported")
        page_count = len(reader.pages)
        if page_count == 0:
            raise InvalidPDFError("The PDF contains no pages")
    except (PdfReadError, OSError, ValueError) as exc:
        raise InvalidPDFError(f"Unable to read this PDF: {exc}") from exc

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = UPLOAD_DIR / f".{document_id}-{uuid.uuid4().hex}.pdf"
    saved_path = UPLOAD_DIR / f"{document_id}.pdf"
    try:
        temporary_path.write_bytes(content)
        try:
            pages = PyPDFLoader(str(temporary_path)).load()
        except (PdfReadError, OSError, ValueError) as exc:
            raise InvalidPDFError(f"Unable to extract PDF text: {exc}") from exc

        if not pages or not any(page.page_content.strip() for page in pages):
            raise InvalidPDFError("The PDF does not contain extractable text")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        split_pages = splitter.split_documents(pages)
        if not split_pages:
            raise InvalidPDFError("The PDF did not produce any searchable text chunks")

        chunks: list[dict[str, Any]] = []
        for page in split_pages:
            page_number = page.metadata.get("page")
            metadata = {
                "document_id": document_id,
                "source": safe_filename,
                "page_number": page_number + 1 if isinstance(page_number, int) else None,
                "start_index": page.metadata.get("start_index"),
            }
            chunks.append({"text": page.page_content, "metadata": metadata})

        vectors = embed_texts([chunk["text"] for chunk in chunks])
        document = {
            "document_id": document_id,
            "filename": safe_filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "size_bytes": len(content),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        if not saved_path.exists():
            temporary_path.replace(saved_path)
        was_added = store.add_document(document, chunks, vectors)
        if not was_added:
            duplicate = next(
                item for item in store.list_documents()
                if item["document_id"] == document_id
            )
            return {
                "status": "duplicate",
                "message": "This PDF's contents are already indexed",
                "document": duplicate,
            }

        logger.info(
            "Processed %s into %d chunks", safe_filename, len(chunks)
        )
        return {
            "status": "processed",
            "message": "PDF processed and added to the persistent index",
            "document": document,
        }
    finally:
        temporary_path.unlink(missing_ok=True)
