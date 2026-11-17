from backend.config import RETRIEVAL_TOP_K
from backend.embeddings import embed_texts
from backend.schemas import SourceChunk
from backend.store import SearchResult, store


def retrieve(question: str) -> list[SearchResult]:
    query_vector = embed_texts([question])
    return store.search(query_vector, RETRIEVAL_TOP_K)


def source_chunks(results: list[SearchResult]) -> list[SourceChunk]:
    return [
        SourceChunk(
            filename=result.metadata["source"],
            page_number=result.metadata.get("page_number"),
            text=result.text,
            distance=result.distance,
        )
        for result in results
    ]
