import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _project_path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_TIMEOUT_SECONDS = _positive_int("OLLAMA_TIMEOUT_SECONDS", 120)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
UPLOAD_DIR = _project_path("UPLOAD_DIR", "data/uploads")
FAISS_INDEX_DIR = _project_path("FAISS_INDEX_DIR", "data/faiss_index")
MAX_UPLOAD_BYTES = _positive_int("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
MAX_FILES_PER_UPLOAD = _positive_int("MAX_FILES_PER_UPLOAD", 10)
RETRIEVAL_TOP_K = _positive_int("RETRIEVAL_TOP_K", 4)
CHUNK_SIZE = _positive_int("CHUNK_SIZE", 650)
CHUNK_OVERLAP = _positive_int("CHUNK_OVERLAP", 120)

if not 3 <= RETRIEVAL_TOP_K <= 5:
    raise ValueError("RETRIEVAL_TOP_K must be between 3 and 5")
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
