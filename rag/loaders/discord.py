from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.models import NormalizedRecord
from rag.utils import parse_datetime


TS_KEYS = ["timestamp", "Timestamp", "timestampEdited", "TimestampEdited", "created_at"]


def _extract_messages(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [m for m in payload if isinstance(m, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            return [m for m in payload["messages"] if isinstance(m, dict)]
        if isinstance(payload.get("Messages"), list):
            return [m for m in payload["Messages"] if isinstance(m, dict)]
    return []


def _author_name(message: dict[str, Any]) -> str | None:
    author = message.get("author") or message.get("Author")
    if isinstance(author, dict):
        return author.get("name") or author.get("username") or author.get("displayName")
    if isinstance(author, str):
        return author
    return message.get("authorName") or message.get("username")


def load_discord(path: Path) -> list[NormalizedRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = _extract_messages(payload)
    channel = None
    if isinstance(payload, dict):
        channel = payload.get("channel", {}).get("name") or payload.get("channel", {}).get("id")
    records: list[NormalizedRecord] = []
    for message in messages:
        ts = None
        for key in TS_KEYS:
            if key in message:
                ts = message[key]
                break
        utc_dt = parse_datetime(ts)
        if utc_dt is None:
            continue
        content = message.get("content") or message.get("Content") or ""
        attachments = message.get("attachments") or message.get("Attachments") or []
        if isinstance(attachments, list) and attachments:
            attachment_names = []
            for att in attachments:
                if isinstance(att, dict):
                    name = att.get("filename") or att.get("name") or att.get("url")
                    if name:
                        attachment_names.append(str(name))
            if attachment_names:
                content = content + " [attachments: " + ", ".join(attachment_names) + "]"
        author = _author_name(message)
        message_id = message.get("id") or message.get("messageId") or f"discord-{utc_dt.timestamp()}"
        meta = {}
        if channel:
            meta["channel"] = channel
        records.append(
            NormalizedRecord(
                id=str(message_id),
                source="discord",
                utc_datetime=utc_dt,
                local_tz=None,
                text=str(content),
                geo=None,
                participants=[author] if author else [],
                meta=meta,
            )
        )
    return records
