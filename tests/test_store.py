import numpy as np

from backend.store import FaissStore


def document(document_id: str, filename: str) -> dict:
    return {
        "document_id": document_id,
        "filename": filename,
        "page_count": 1,
        "chunk_count": 1,
        "size_bytes": 100,
        "uploaded_at": "2026-01-01T00:00:00+00:00",
    }


def chunk(document_id: str, filename: str) -> dict:
    return {
        "text": f"Text from {filename}",
        "metadata": {
            "document_id": document_id,
            "source": filename,
            "page_number": 1,
            "start_index": 0,
        },
    }


def test_index_persists_and_appends_documents(tmp_path) -> None:
    first_store = FaissStore(tmp_path)
    first_store.add_document(
        document("first", "first.pdf"),
        [chunk("first", "first.pdf")],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    reloaded_store = FaissStore(tmp_path)
    reloaded_store.add_document(
        document("second", "second.pdf"),
        [chunk("second", "second.pdf")],
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )

    persisted_store = FaissStore(tmp_path)
    matches = persisted_store.search(
        np.asarray([[0.99, 0.01]], dtype=np.float32),
        top_k=4,
    )

    assert persisted_store.counts() == (2, 2)
    assert {item["filename"] for item in persisted_store.list_documents()} == {
        "first.pdf",
        "second.pdf",
    }
    assert matches[0].metadata["source"] == "first.pdf"


def test_duplicate_document_is_not_added_twice(tmp_path) -> None:
    store = FaissStore(tmp_path)
    entry = document("same", "same.pdf")
    records = [chunk("same", "same.pdf")]
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)

    assert store.add_document(entry, records, vectors)
    assert not store.add_document(entry, records, vectors)
    assert store.counts() == (1, 1)


def test_vector_search_ranks_relevant_and_irrelevant_chunks(tmp_path) -> None:
    store = FaissStore(tmp_path)
    store.add_document(
        document("relevant", "mechanisation.pdf"),
        [chunk("relevant", "mechanisation.pdf")],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    store.add_document(
        document("irrelevant", "training.pdf"),
        [chunk("irrelevant", "training.pdf")],
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )

    relevant_query = store.search(np.asarray([[0.99, 0.01]], dtype=np.float32), 2)
    unrelated_query = store.search(np.asarray([[0.01, 0.99]], dtype=np.float32), 2)

    assert relevant_query[0].metadata["source"] == "mechanisation.pdf"
    assert unrelated_query[0].metadata["source"] == "training.pdf"
