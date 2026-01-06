from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.models import Geo, NormalizedRecord
from rag.utils import parse_datetime


DATE_KEYS = ["creationDate", "Creation Date", "creation_date", "date", "Date"]
TEXT_KEYS = ["text", "Text", "body", "Body"]
UUID_KEYS = ["uuid", "UUID", "id", "Id"]
TZ_KEYS = ["timeZone", "Time Zone", "timeZoneName", "Time Zone Name"]


def _first_key(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _parse_geo(location: dict[str, Any] | None) -> Geo | None:
    if not location:
        return None
    lat = location.get("latitude") or location.get("Latitude")
    lon = location.get("longitude") or location.get("Longitude")
    if lat is None or lon is None:
        return None
    try:
        return Geo(lat=float(lat), lon=float(lon))
    except (TypeError, ValueError):
        return None


def load_dayone(path: Path) -> list[NormalizedRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = None
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("Entries") or payload.get("data")
    if entries is None:
        return []

    records: list[NormalizedRecord] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        date_raw = _first_key(entry, DATE_KEYS)
        utc_dt = parse_datetime(date_raw)
        if utc_dt is None:
            continue
        text = _first_key(entry, TEXT_KEYS) or ""
        entry_id = _first_key(entry, UUID_KEYS) or f"dayone-{utc_dt.timestamp()}"
        tz_name = _first_key(entry, TZ_KEYS)
        location = entry.get("location") or entry.get("Location") or {}
        geo = _parse_geo(location)
        meta = {
            "is_stub": not bool(text.strip()),
            "place_name": location.get("placeName") or location.get("Place Name"),
            "tags": entry.get("tags") or entry.get("Tags"),
            "starred": entry.get("starred") or entry.get("Starred"),
        }
        records.append(
            NormalizedRecord(
                id=str(entry_id),
                source="dayone",
                utc_datetime=utc_dt,
                local_tz=str(tz_name) if tz_name else None,
                text=str(text),
                geo=geo,
                participants=[],
                meta={k: v for k, v in meta.items() if v is not None},
            )
        )
    return records
