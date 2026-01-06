from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = PROCESSED_DIR / "index"

ITEMS_PATH = INTERIM_DIR / "items.jsonl"
STYLECARD_PATH = PROCESSED_DIR / "stylecard.json"
DRAFTS_PATH = PROCESSED_DIR / "drafts.jsonl"
ACCEPTED_PATH = PROCESSED_DIR / "accepted.jsonl"

INDEX_FAISS_PATH = INDEX_DIR / "faiss.index"
INDEX_BM25_PATH = INDEX_DIR / "bm25.pkl"
INDEX_EMBED_PATH = INDEX_DIR / "embeddings.npy"
INDEX_ITEMS_PATH = INDEX_DIR / "items.jsonl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
