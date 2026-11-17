import logging

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status

from backend.config import (
    EMBEDDING_MODEL,
    MAX_FILES_PER_UPLOAD,
    MAX_UPLOAD_BYTES,
)
from backend.ingestion import InvalidPDFError, process_pdf
from backend.llm import OllamaUnavailableError, generate_answer
from backend.retrieval import retrieve, source_chunks
from backend.schemas import (
    AskRequest,
    AskResponse,
    DocumentSummary,
    DocumentsResponse,
    HealthResponse,
    UploadItemResult,
    UploadResponse,
)
from backend.store import store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agricultural Schemes RAG API",
    description="Upload agricultural scheme PDFs and ask questions grounded in them.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    total_documents, total_chunks = store.counts()
    return HealthResponse(
        status="ok",
        total_documents=total_documents,
        total_chunks=total_chunks,
        embedding_model=EMBEDDING_MODEL,
    )


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdfs(
    response: Response,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload at least one PDF file",
        )
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload no more than {MAX_FILES_PER_UPLOAD} files at once",
        )

    results: list[UploadItemResult] = []
    for uploaded_file in files:
        filename = uploaded_file.filename or "unnamed.pdf"
        try:
            content = await uploaded_file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise InvalidPDFError(
                    f"The PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit"
                )
            processed = process_pdf(filename, content)
            document = processed["document"]
            results.append(
                UploadItemResult(
                    filename=document["filename"],
                    status=processed["status"],
                    message=processed["message"],
                    document_id=document["document_id"],
                    page_count=document["page_count"],
                    chunk_count=document["chunk_count"],
                )
            )
        except InvalidPDFError as exc:
            logger.warning("Rejected uploaded file %r: %s", filename, exc)
            results.append(
                UploadItemResult(
                    filename=filename,
                    status="failed",
                    message=str(exc),
                )
            )
        finally:
            await uploaded_file.close()

    total_documents, total_chunks = store.counts()
    if any(item.status == "failed" for item in results):
        response.status_code = status.HTTP_207_MULTI_STATUS
    return UploadResponse(
        results=results,
        total_documents=total_documents,
        total_chunks=total_chunks,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be blank",
        )

    results = retrieve(question)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No searchable PDFs are indexed. Upload a PDF before asking a question.",
        )

    try:
        answer = generate_answer(question, results)
    except OllamaUnavailableError as exc:
        logger.warning("Question answering failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    sources = source_chunks(results)
    return AskResponse(answer=answer, sources=sources, used_chunks=len(sources))


@app.get("/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    indexed_documents = store.list_documents()
    total_documents, total_chunks = store.counts()
    return DocumentsResponse(
        documents=[DocumentSummary(**document) for document in indexed_documents],
        total_documents=total_documents,
        total_chunks=total_chunks,
    )
