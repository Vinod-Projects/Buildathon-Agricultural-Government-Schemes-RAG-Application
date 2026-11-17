import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.config import EMBEDDING_MODEL, FAISS_INDEX_DIR

logger = logging.getLogger(__name__)
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class SearchResult:
    text: str
    metadata: dict[str, Any]
    distance: float


class FaissStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.generations_directory = directory / "generations"
        self.current_file = directory / "CURRENT"
        self._lock = threading.RLock()

    def contains_document(self, document_id: str) -> bool:
        with self._lock:
            _, _, documents = self._load_state()
            return any(
                document.get("document_id") == document_id for document in documents
            )

    def add_document(
        self,
        document: dict[str, Any],
        chunks: list[dict[str, Any]],
        vectors: np.ndarray,
    ) -> bool:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not len(chunks):
            raise ValueError("Chunk records and embedding vectors must have matching rows")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding vectors must contain only finite values")

        with self._lock:
            index, existing_chunks, documents = self._load_state()
            document_id = document["document_id"]
            if any(item.get("document_id") == document_id for item in documents):
                return False

            if index is None:
                index = faiss.IndexFlatL2(vectors.shape[1])
            elif index.d != vectors.shape[1]:
                raise RuntimeError(
                    "Embedding dimensions do not match the saved FAISS index"
                )

            if not np.allclose(
                np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3
            ):
                raise ValueError("Embeddings must be normalized before indexing")

            index.add(vectors)
            self._write_state(
                index,
                existing_chunks + chunks,
                documents + [document],
            )
            logger.info(
                "Indexed document %s with %d chunks", document_id, len(chunks)
            )
            return True

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("A query must be represented by exactly one vector")

        with self._lock:
            index, chunks, _ = self._load_state()
            if index is None or index.ntotal == 0:
                return []
            if query_vector.shape[1] != index.d:
                raise RuntimeError(
                    "Query embedding dimensions do not match the saved FAISS index"
                )

            distances, positions = index.search(
                query_vector, min(top_k, int(index.ntotal))
            )
            results: list[SearchResult] = []
            for distance, position in zip(distances[0], positions[0]):
                if position < 0:
                    continue
                chunk = chunks[int(position)]
                results.append(
                    SearchResult(
                        text=chunk["text"],
                        metadata=chunk["metadata"],
                        distance=float(distance),
                    )
                )
            return results

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            _, _, documents = self._load_state()
            return sorted(
                documents,
                key=lambda item: item["uploaded_at"],
                reverse=True,
            )

    def counts(self) -> tuple[int, int]:
        with self._lock:
            _, chunks, documents = self._load_state()
            return len(documents), len(chunks)

    def _load_state(
        self,
    ) -> tuple[faiss.Index | None, list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.current_file.exists():
            return None, [], []

        generation = self.current_file.read_text(encoding="utf-8").strip()
        if not _GENERATION_PATTERN.fullmatch(generation):
            raise RuntimeError("The FAISS index CURRENT file is invalid")

        generation_directory = self.generations_directory / generation
        index_file = generation_directory / "index.faiss"
        metadata_file = generation_directory / "metadata.json"
        if not index_file.is_file() or not metadata_file.is_file():
            raise RuntimeError("The active FAISS index generation is incomplete")

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("embedding_model") != EMBEDDING_MODEL:
            raise RuntimeError(
                "The saved index uses a different embedding model. "
                "Use the original model or re-ingest the PDFs."
            )
        index = faiss.read_index(str(index_file))
        chunks = metadata.get("chunks")
        documents = metadata.get("documents")
        if not isinstance(chunks, list) or not isinstance(documents, list):
            raise RuntimeError("The FAISS index metadata is invalid")
        if index.ntotal != len(chunks) or index.d != metadata.get("dimension"):
            raise RuntimeError("The FAISS index and its metadata do not match")
        return index, chunks, documents

    def _write_state(
        self,
        index: faiss.Index,
        chunks: list[dict[str, Any]],
        documents: list[dict[str, Any]],
    ) -> None:
        generation = uuid.uuid4().hex
        self.generations_directory.mkdir(parents=True, exist_ok=True)
        temporary_directory = self.generations_directory / f".tmp-{generation}"
        final_directory = self.generations_directory / generation
        temporary_directory.mkdir()

        faiss.write_index(index, str(temporary_directory / "index.faiss"))
        metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "dimension": int(index.d),
            "chunks": chunks,
            "documents": documents,
        }
        (temporary_directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_directory.replace(final_directory)

        temporary_current = self.directory / f".CURRENT-{generation}"
        temporary_current.write_text(generation, encoding="utf-8")
        temporary_current.replace(self.current_file)


store = FaissStore(FAISS_INDEX_DIR)
