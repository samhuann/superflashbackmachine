from __future__ import annotations

import re
import uuid
from datetime import datetime, time, timezone
from pathlib import Path

from rag.config import ACCEPTED_PATH
from rag.io import read_jsonl, write_json


CITATION_RE = re.compile(r"\s*〔[^〕]+〕")


def _strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def export_entries(out_path: Path) -> int:
    rows = read_jsonl(ACCEPTED_PATH)
    entries = []
    for row in rows:
        date_str = row.get("date")
        text = row.get("text", "")
        meta = row.get("meta", {})
        if not date_str:
            continue
        try:
            day = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        creation_dt = datetime.combine(day, time(hour=12), tzinfo=timezone.utc)
        entry = {
            "uuid": str(uuid.uuid4()),
            "creationDate": creation_dt.isoformat(),
            "text": _strip_citations(text),
            "meta": {
                "citations": meta.get("citations", []),
                "confidence": row.get("confidence"),
            },
        }
        entries.append(entry)

    payload = {"entries": entries}
    write_json(out_path, payload)
    return len(entries)
