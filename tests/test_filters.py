from __future__ import annotations

from datetime import datetime, timezone

from rag.models import Geo, NormalizedRecord
from rag.retrieval import filter_indices


def test_filter_indices_time_and_geo() -> None:
    items = [
        NormalizedRecord(
            id="1",
            source="dayone",
            utc_datetime=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            local_tz=None,
            text="A",
            geo=Geo(lat=40.0, lon=-70.0),
            participants=[],
            meta={},
        ),
        NormalizedRecord(
            id="2",
            source="dayone",
            utc_datetime=datetime(2024, 1, 10, 12, tzinfo=timezone.utc),
            local_tz=None,
            text="B",
            geo=Geo(lat=41.0, lon=-71.0),
            participants=[],
            meta={},
        ),
        NormalizedRecord(
            id="3",
            source="discord",
            utc_datetime=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            local_tz=None,
            text="C",
            geo=None,
            participants=[],
            meta={},
        ),
    ]

    indices = filter_indices(items, "2024-01-01", days=2, lat=40.0, lon=-70.0, km=50)
    assert indices == [0]

    indices = filter_indices(items, "2024-01-01", days=2, lat=None, lon=None, km=None)
    assert indices == [0, 2]
