# Agricultural Government Schemes RAG Application

A beginner-friendly Retrieval-Augmented Generation (RAG) application for answering questions from agricultural government scheme PDFs uploaded by the user. Answers will be grounded in those documents, with source filenames and page information where available.

> **Build status:** Phases 1–7 are documented and implemented. The 14-test suite passes; tests mock the embedding model and Ollama responses.

## Phase 1 — Understand the project

### What is RAG?

Retrieval-Augmented Generation combines search with a language model. Rather than asking a model to answer from memory, the application first finds relevant passages in the uploaded documents and then asks the model to answer using those passages. This helps keep answers tied to the source material. If the documents do not contain enough information, the application should say so instead of guessing.

### The two pipelines

- **Ingestion** prepares documents for search: validate uploaded PDFs, extract text and page metadata, split text into overlapping chunks, embed each chunk, and save the vectors and metadata to a local FAISS index. Later uploads are added without discarding the existing index.
- **Retrieval and question answering** embeds the user's question with the same model, retrieves the four most relevant chunks, and sends those chunks and the question to Ollama. The response includes source details and clearly states when the evidence is insufficient.

### Components in plain language

| Component | Purpose |
| --- | --- |
| **Streamlit** | Provides the browser interface for uploading PDFs, checking processing status, asking questions, and reading answers and sources. |
| **FastAPI** | Provides the backend API used by the interface to process documents and answer questions. |
| **Embeddings** | Turn text into numeric vectors so passages with similar meaning can be compared. The initial model is `sentence-transformers/all-MiniLM-L6-v2`; indexing and querying must use the same model. |
| **Chunking** | Divides long PDF text into smaller, overlapping passages so relevant evidence can be retrieved without sending an entire document to the language model. The initial settings are 650 characters per chunk and 120 characters of overlap. |
| **FAISS** | Stores and searches the passage vectors locally. The initial index uses `IndexFlatL2` and returns the four closest chunks. |
| **Vector similarity search** | Compares the question vector with stored passage vectors and ranks passages by closeness. Similarity indicates relevance, not that a claim is true. |
| **Ollama and Qwen 2.5** | Run the answer-generating language model locally. The starter model is `qwen2.5:0.5b`. |
| **LangChain** | Supplies PDF loading and text splitting; FAISS is used directly for local vector storage and search. |

Scheme eligibility, subsidy amounts, and dates can change. Answers should be checked against current official government guidance. The TNAU agricultural engineering schemes page is a domain reference, not a substitute for uploaded source documents: <https://agritech.tnau.ac.in/agricultural_engineering/agriengg_govt_schemes.html>.

## Phase 2 — Set up the environment

### Prerequisites

- Python 3.11–3.13 (Python 3.13 is recommended and was used to install the dependencies and run the test suite on Windows).
- Git and a terminal.
- Ollama installed and running locally: <https://ollama.com/download>.
- Enough disk space for the embedding model, Ollama model, uploaded PDFs, and local index.

FAISS packaging varies by operating system. If installing `faiss-cpu` fails on native Windows, use a supported Linux environment such as WSL2 or an appropriate Conda environment; do not silently substitute a different vector database.

### Create the project folders

From the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force frontend, backend, data\uploads, data\faiss_index
```

The folders are included in source control using `.gitkeep` files. User PDFs and generated index data are excluded.

### Create and install into a virtual environment

For example, create a virtual environment with Python 3.11:

```powershell
py -3.13 -m venv .venv
```

On Windows, you can run commands through the environment's Python without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt -r frontend\requirements.txt
```

Alternatively, activate it in PowerShell for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The process-scoped policy change ends when that PowerShell window closes. In Command Prompt, use `.venv\Scripts\activate.bat` instead. Avoid changing the machine-wide execution policy just to activate a project environment.

### Download the starter language model

After installing and starting Ollama:

```powershell
ollama pull qwen2.5:0.5b
ollama list
```

The sentence-transformers embedding model is downloaded when it is first needed by the application. An internet connection may be required for that initial download.

### Local configuration

Copy `.env.example` to `.env` if you need to change local settings. `.env` is ignored by Git; do not put credentials or private data in tracked files.

### Project layout

```text
.
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI endpoints
│   ├── config.py               # Local settings and paths
│   ├── schemas.py              # API request and response models
│   ├── ingestion.py            # PDF validation, extraction, chunking, and indexing
│   ├── embeddings.py           # Shared embedding model
│   ├── store.py                # Persistent FAISS index and document metadata
│   ├── retrieval.py            # Similarity search and source metadata
│   ├── llm.py                  # Grounded Ollama responses
│   └── requirements.txt
├── frontend/
│   ├── app.py                  # Streamlit interface
│   └── requirements.txt
├── tests/
│   ├── test_api.py
│   ├── test_ingestion.py
│   ├── test_llm.py
│   └── test_store.py
├── data/
│   ├── uploads/                # Local uploaded PDFs (not committed)
│   └── faiss_index/            # Persistent FAISS data (not committed)
├── .env.example
├── .gitignore
├── requirements-dev.txt
└── README.md
```

## Phase 3 — Backend API and ingestion

Start the API from the repository root after installing the dependencies and downloading the Ollama model:

```powershell
uvicorn backend.main:app --reload
```

Interactive API documentation is available at <http://127.0.0.1:8000/docs>. The backend provides:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check backend status. |
| `POST` | `/upload-pdf` | Validate and process one or more PDF files. |
| `POST` | `/ask` | Retrieve relevant evidence and generate a grounded answer. |
| `GET` | `/documents` | List indexed documents. |

PDF uploads are limited to 20 MB each and 10 files per request by default. Identical file contents are detected by SHA-256 and are not indexed twice; uploads with the same filename but different contents remain separate documents. Failed files in a mixed batch are reported individually, while valid PDFs are still processed.

The FAISS index uses versioned local generations so an interrupted update does not replace the last complete index. It stores FAISS binary files and JSON metadata rather than Python pickle files. Keep the local index directory trusted and do not load index files from untrusted sources. The embedding model identifier is saved with each generation; changing it requires re-ingesting the PDFs.

## Phase 4 — Streamlit interface

The interface has separate tabs for PDF uploads, question answering, and the indexed-document list. It displays processing results and chunk counts, shows retrieved source excerpts, and reports backend/Ollama errors instead of hiding them.

Start the frontend from a second terminal while the backend is running:

```powershell
streamlit run frontend/app.py
```

The interface will be available at <http://localhost:8501>.

The frontend reads `BACKEND_URL` from `.env` (default `http://127.0.0.1:8000`). You can also change the FastAPI URL in the sidebar.

## Phase 5 — Run the integrated application

1. Install the project dependencies and download the Ollama model using the Phase 2 instructions.
2. In the first terminal, from the repository root, start the API:

   ```powershell
   uvicorn backend.main:app --reload
   ```

   Check that <http://127.0.0.1:8000/docs> opens and that `GET /health` reports `status: ok`.
3. In a second terminal, from the same repository root, start Streamlit:

   ```powershell
   streamlit run frontend/app.py
   ```

   Open <http://localhost:8501>. The sidebar defaults to `http://127.0.0.1:8000`; change it or set `BACKEND_URL` in `.env` if the API is listening elsewhere.
4. Open **Upload PDFs**, select one or more scheme PDFs, and choose **Process PDFs**. Review each file's result and the indexed-document and chunk counts. Existing documents remain indexed when new PDFs are uploaded.
5. Open **Ask a question**, submit a question covered by the uploaded material, and review both the answer and its source excerpts. For missing information, the answer should state that the uploaded documents are insufficient rather than inventing eligibility, subsidy, or date details.

The backend and frontend are connected through `/health`, `/upload-pdf`, `/ask`, and `/documents`. Keep both terminal processes running while using the app.

## Phase 6 — Test the application

Install the application and test dependencies into the project virtual environment, then run the test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover valid, invalid, empty, and textless PDFs; duplicate and multiple uploads; FAISS persistence and nearest-vector ordering; grounded-answer instructions and source metadata; empty-index questions; and Ollama connection failures. The tests mock the embedding model and Ollama service, so they do not need model downloads or a running Ollama server.

## Phase 7 — Documentation and presentation walkthrough

### Suggested presentation flow

1. **Explain the goal:** answer agricultural-scheme questions from user-provided PDFs, not from hardcoded examples or unverified model memory.
2. **Show ingestion:** upload a scheme PDF in Streamlit and explain PDF validation, page-aware extraction, chunking, embeddings, and the persistent FAISS index. Point out the processed document and chunk counts.
3. **Show retrieval:** ask a question that the PDF answers. Explain that the question uses the same embedding model as the document chunks and that FAISS returns the four nearest passages.
4. **Show grounding:** read the answer and expand its source excerpts to verify the filename and page. Explain that the answer is generated by local Ollama using those excerpts as evidence.
5. **Show uncertainty handling:** ask a question that the uploaded PDF does not answer. The response should explicitly say the documents do not contain sufficient information instead of inventing eligibility, subsidy, or date details.
6. **Close with limitations:** answers depend on PDF text that can be extracted (scanned-image OCR is not included), Ollama must be running for answers, and time-sensitive scheme details must be checked against current official government guidance.

The test suite can be run with the commands in Phase 6. The live smoke check confirms that FastAPI `/health` and the Streamlit page respond; full model generation requires a locally running Ollama instance and its downloaded Qwen model.
