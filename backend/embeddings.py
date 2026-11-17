from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from backend.config import EMBEDDING_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("At least one text is required to generate embeddings")

    vectors = get_embedding_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise RuntimeError("The embedding model returned invalid vectors")
    return result
