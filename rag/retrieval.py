from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from rag.config import INDEX_BM25_PATH, INDEX_EMBED_PATH, INDEX_ITEMS_PATH, INDEX_META_PATH
from rag.embedding import load_model
from rag.io import read_jsonl
from rag.models import EvidenceItem, NormalizedRecord
from rag.utils import haversine_km, tokenize


@dataclass
class IndexBundle:
    items: list[NormalizedRecord]
    embeddings: np.ndarray
    bm25: object
    meta: dict


def load_index() -> IndexBundle:
    rows = read_jsonl(INDEX_ITEMS_PATH)
    items = [NormalizedRecord.model_validate(row) for row in rows]
    embeddings = np.load(INDEX_EMBED_PATH)
    with INDEX_BM25_PATH.open("rb") as handle:
        bm25 = pickle.load(handle)
    meta = json.loads(Path(INDEX_META_PATH).read_text(encoding="utf-8"))
    return IndexBundle(items=items, embeddings=embeddings, bm25=bm25, meta=meta)


def _parse_date(target: str) -> date:
    return datetime.strptime(target, "%Y-%m-%d").date()


def filter_indices(
    items: list[NormalizedRecord],
    target_date: str,
    days: int,
    lat: float | None,
    lon: float | None,
    km: float | None,
) -> list[int]:
    target = _parse_date(target_date)
    indices: list[int] = []
    for idx, item in enumerate(items):
        day_diff = abs((item.utc_datetime.date() - target).days)
        if day_diff > days:
            continue
        if lat is not None and lon is not None and km is not None:
            if item.geo is None:
                continue
            distance = haversine_km(lat, lon, item.geo.lat, item.geo.lon)
            if distance > km:
                continue
        indices.append(idx)
    return indices


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val < 1e-6:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def _query_text(target_date: str, lat: float | None, lon: float | None) -> str:
    parts = [f"journal entry {target_date}"]
    if lat is not None and lon is not None:
        parts.append(f"location {lat:.4f},{lon:.4f}")
    return " ".join(parts)


def _diversify(results: list[EvidenceItem], max_per_source: int, k: int) -> list[EvidenceItem]:
    buckets: dict[str, list[EvidenceItem]] = {}
    for item in results:
        buckets.setdefault(item.source, []).append(item)
    for items in buckets.values():
        items.sort(key=lambda x: x.score, reverse=True)
    ordered: list[EvidenceItem] = []
    while len(ordered) < k:
        progressed = False
        for source in list(buckets.keys()):
            if not buckets[source]:
                continue
            taken = sum(1 for item in ordered if item.source == source)
            if taken >= max_per_source:
                continue
            ordered.append(buckets[source].pop(0))
            progressed = True
            if len(ordered) >= k:
                break
        if not progressed:
            break
    return ordered


def retrieve(
    target_date: str,
    days: int = 3,
    lat: float | None = None,
    lon: float | None = None,
    km: float | None = None,
    top_k: int = 40,
) -> list[EvidenceItem]:
    bundle = load_index()
    indices = filter_indices(bundle.items, target_date, days, lat, lon, km)
    if not indices:
        return []

    query = _query_text(target_date, lat, lon)
    tokens = tokenize(query)
    bm25_scores = np.asarray(bundle.bm25.get_scores(tokens), dtype="float32")

    model = load_model()
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")[0]
    dense_scores = bundle.embeddings @ query_vec

    mask = np.zeros_like(dense_scores, dtype=bool)
    mask[indices] = True
    dense_scores = np.where(mask, dense_scores, 0.0)
    bm25_scores = np.where(mask, bm25_scores, 0.0)

    dense_norm = _normalize(dense_scores)
    sparse_norm = _normalize(bm25_scores)
    w_dense = float(os.getenv("DENSE_WEIGHT", "0.65"))
    w_sparse = float(os.getenv("BM25_WEIGHT", "0.35"))
    fused = (w_dense * dense_norm) + (w_sparse * sparse_norm)

    sorted_idx = np.argsort(fused)[::-1]
    results = []
    for idx in sorted_idx[: max(top_k * 3, top_k)]:
        if fused[idx] <= 0:
            continue
        item = bundle.items[idx]
        results.append(
            EvidenceItem(
                id=item.id,
                source=item.source,
                utc_datetime=item.utc_datetime,
                text=item.text,
                geo=item.geo,
                score=float(fused[idx]),
            )
        )

    return _diversify(results, max_per_source=8, k=top_k)
