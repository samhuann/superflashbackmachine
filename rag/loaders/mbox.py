from __future__ import annotations

import mailbox
from email.message import Message
from pathlib import Path

from rag.models import NormalizedRecord
from rag.utils import parse_datetime, snippet


def _get_body(message: Message) -> str:
    if message.is_multipart():
        parts = []
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    parts.append(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore"))
                except Exception:
                    continue
        return "\n".join(parts)
    try:
        return message.get_payload(decode=True).decode(message.get_content_charset() or "utf-8", errors="ignore")
    except Exception:
        return ""


def load_mbox(path: Path) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    mbox = mailbox.mbox(path)
    for msg in mbox:
        date_raw = msg.get("Date")
        utc_dt = parse_datetime(date_raw)
        if utc_dt is None:
            continue
        subject = msg.get("Subject") or "(no subject)"
        sender = msg.get("From") or ""
        body = _get_body(msg)
        text = f"Email from {sender}: {subject}. {snippet(body)}"
        message_id = msg.get("Message-Id") or msg.get("Message-ID") or f"mbox-{int(utc_dt.timestamp())}"
        records.append(
            NormalizedRecord(
                id=str(message_id),
                source="email",
                utc_datetime=utc_dt,
                local_tz=None,
                text=text,
                geo=None,
                participants=[sender] if sender else [],
                meta={"subject": subject, "from": sender, "to": msg.get("To")},
            )
        )
    return records
