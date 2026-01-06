from __future__ import annotations

from datetime import date

from rag.config import ACCEPTED_PATH, DRAFTS_PATH, ITEMS_PATH
from rag.io import read_jsonl
from rag.models import DraftEntry, NormalizedRecord


def _date_key(dt) -> str:
    return dt.strftime("%Y-%m-%d")


def missing_dates() -> list[str]:
    rows = read_jsonl(ITEMS_PATH)
    records = [NormalizedRecord.model_validate(row) for row in rows]
    stubs = set()
    for record in records:
        if record.source != "dayone":
            continue
        if record.text.strip():
            continue
        if record.meta.get("is_stub") is False:
            continue
        stubs.add(_date_key(record.utc_datetime.date()))

    accepted_rows = read_jsonl(ACCEPTED_PATH)
    accepted_dates = {row.get("date") for row in accepted_rows if row.get("date")}
    return sorted(list(stubs - accepted_dates))


def load_drafts() -> dict[str, DraftEntry]:
    drafts = {}
    for row in read_jsonl(DRAFTS_PATH):
        try:
            draft = DraftEntry.model_validate(row)
            drafts[draft.date] = draft
        except Exception:
            continue
    return drafts
