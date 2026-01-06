from __future__ import annotations

from rag.models import EvidenceItem
from rag.retrieval import retrieve


def build_evidence_packet(
    target_date: str,
    days: int = 3,
    lat: float | None = None,
    lon: float | None = None,
    km: float | None = None,
    top_k: int = 40,
) -> dict:
    evidence = retrieve(target_date, days=days, lat=lat, lon=lon, km=km, top_k=top_k)
    return {
        "date": target_date,
        "count": len(evidence),
        "sources": sorted({item.source for item in evidence}),
        "evidence": [item.model_dump() for item in evidence],
    }


def packet_to_items(packet: dict) -> list[EvidenceItem]:
    return [EvidenceItem.model_validate(item) for item in packet.get("evidence", [])]
