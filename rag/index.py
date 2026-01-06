from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rag.config import (
    INDEX_BM25_PATH,
    INDEX_EMBED_PATH,
    INDEX_FAISS_PATH,
    INDEX_ITEMS_PATH,
    INDEX_META_PATH,
    ITEMS_PATH,
    ensure_data_dirs,
)
from rag.embedding import embed_texts, load_model
from rag.io import read_jsonl, write_jsonl
from rag.models import NormalizedRecord
from rag.utils import tokenize


def build_index() -> int:
    ensure_data_dirs()
    rows = read_jsonl(ITEMS_PATH)
    records = [NormalizedRecord.model_validate(row) for row in rows]
    texts = [rec.text for rec in records]
    tokens = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokens)

    model = load_model()
    embeddings = embed_texts(model, texts)
    dim = embeddings.shape[1] if embeddings.size else model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    if embeddings.size:
        index.add(embeddings)

    INDEX_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FAISS_PATH))
    np.save(INDEX_EMBED_PATH, embeddings)
    with INDEX_BM25_PATH.open("wb") as handle:
        pickle.dump(bm25, handle)

    write_jsonl(INDEX_ITEMS_PATH, [rec.model_dump() for rec in records], mode="w")
    meta = {
        "embed_model": os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
        "bm25_weight": float(os.getenv("BM25_WEIGHT", "0.35")),
        "dense_weight": float(os.getenv("DENSE_WEIGHT", "0.65")),
        "item_count": len(records),
    }
    INDEX_META_PATH.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    return len(records)
