from __future__ import annotations

import hashlib
from datetime import datetime

from rag.config import ITEMS_PATH, RAW_DIR, ensure_data_dirs
from rag.io import read_jsonl, write_jsonl
from rag.models import NormalizedRecord


def _stable_id(source: str, text: str, utc_dt: datetime) -> str:
    payload = f"{source}|{utc_dt.isoformat()}|{text}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{source}-{digest}"


def normalize() -> int:
    ensure_data_dirs()
    records: list[NormalizedRecord] = []
    for raw_path in RAW_DIR.glob("*.jsonl"):
        for row in read_jsonl(raw_path):
            try:
                record = NormalizedRecord.model_validate(row)
            except Exception:
                continue
            if not record.id:
                record.id = _stable_id(record.source, record.text, record.utc_datetime)
            records.append(record)

    dedup: dict[str, NormalizedRecord] = {}
    for record in records:
        key = f"{record.source}:{record.id}"
        if key not in dedup:
            dedup[key] = record

    write_jsonl(ITEMS_PATH, [r.model_dump() for r in dedup.values()], mode="w")
    return len(dedup)
