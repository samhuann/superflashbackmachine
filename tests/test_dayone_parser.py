from __future__ import annotations

import json
from pathlib import Path

from rag.loaders.dayone import load_dayone


def test_dayone_parser_variants(tmp_path: Path) -> None:
    payload = {
        "Entries": [
            {
                "UUID": "abc-123",
                "Creation Date": "2024-01-02T10:00:00-05:00",
                "Text": "Went hiking.",
                "Location": {"Latitude": 40.0, "Longitude": -70.0, "Place Name": "Trail"},
            },
            {
                "uuid": "stub-1",
                "creationDate": "2024-01-03T12:00:00Z",
                "text": "",
            },
        ]
    }
    path = tmp_path / "dayone.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_dayone(path)
    assert len(records) == 2
    assert records[0].id == "abc-123"
    assert records[0].geo is not None
    assert records[1].meta.get("is_stub") is True
