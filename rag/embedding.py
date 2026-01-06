from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.utils import chunked


def load_model() -> SentenceTransformer:
    model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
    vectors = []
    for batch in chunked(list(texts), batch_size):
        embeddings = model.encode(batch, normalize_embeddings=True)
        vectors.append(np.asarray(embeddings, dtype="float32"))
    if not vectors:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return np.vstack(vectors)
