from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from rag.config import DRAFTS_PATH, STYLECARD_PATH
from rag.io import read_json, write_jsonl
from rag.llm import generate_text, llm_available
from rag.models import DraftEntry, EvidenceItem, StyleCard
from rag.summarization import build_evidence_packet, packet_to_items
from rag.utils import snippet


CITATION_RE = re.compile(r"〔([^:]+):([^〕]+)〕")


def _load_stylecard() -> StyleCard:
    try:
        payload = read_json(STYLECARD_PATH)
        return StyleCard.model_validate(payload)
    except Exception:
        return StyleCard(
            avg_sentence_length=12.0,
            frequent_phrases=[],
            structure_hints=["Short, factual sentences."],
            privacy_knobs={"redact_names": False, "redact_locations": False, "redact_contact_info": False},
        )


def _citation_marker(item: EvidenceItem) -> str:
    return f"〔{item.source}:{item.id}〕"


def _detect_geo_conflict(evidence: list[EvidenceItem], km_threshold: float = 100.0) -> str | None:
    geos = [(item, item.geo) for item in evidence if item.geo]
    if len(geos) < 2:
        return None
    for i in range(len(geos)):
        for j in range(i + 1, len(geos)):
            a, geo_a = geos[i]
            b, geo_b = geos[j]
            if not geo_a or not geo_b:
                continue
            dist = ((geo_a.lat - geo_b.lat) ** 2 + (geo_a.lon - geo_b.lon) ** 2) ** 0.5 * 111
            if dist >= km_threshold:
                return (
                    f"[[UNCERTAIN: Evidence suggests multiple locations ({geo_a.lat:.2f},{geo_a.lon:.2f} vs "
                    f"{geo_b.lat:.2f},{geo_b.lon:.2f}).]]{_citation_marker(a)}{_citation_marker(b)}"
                )
    return None


def _deterministic_draft(target_date: str, evidence: list[EvidenceItem], style: StyleCard) -> str:
    if not evidence:
        return (
            f"Sparse evidence for {target_date}. [[UNCERTAIN: No strong signals found.]]"
        )

    lines = []
    for item in evidence[:5]:
        text = snippet(item.text, 180)
        sentence = text
        if not sentence and style.frequent_phrases:
            sentence = style.frequent_phrases[0].capitalize()
        if not sentence.endswith(('.', '!', '?')):
            sentence += "."
        sentence += _citation_marker(item)
        lines.append(sentence)

    conflict = _detect_geo_conflict(evidence)
    if conflict:
        lines.append(conflict)

    if len(evidence) < 3:
        lines.append("Open questions: Where was I? Who was I with?")

    return " ".join(lines)


def _build_prompt(target_date: str, evidence: list[EvidenceItem], style: StyleCard) -> str:
    evidence_lines = []
    for item in evidence:
        evidence_lines.append(
            f"- {item.utc_datetime.isoformat()} {item.text} {_citation_marker(item)}"
        )
    hints = "\n".join(f"- {hint}" for hint in style.structure_hints)
    return (
        "You are drafting a personal journal entry. Use the evidence list only. "
        "Every factual sentence MUST include a citation marker like 〔source:id〕. "
        "If there are conflicts, add [[UNCERTAIN: ...]] and include both citations. "
        f"Date: {target_date}\n"
        f"Style hints:\n{hints}\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
    )


def _extract_citations(text: str) -> list[dict[str, Any]]:
    citations = []
    for match in CITATION_RE.finditer(text):
        citations.append({"source": match.group(1), "id": match.group(2)})
    return citations


def _confidence(evidence: list[EvidenceItem]) -> float:
    if not evidence:
        return 0.1
    density = min(len(evidence) / 10.0, 1.0)
    sources = len({item.source for item in evidence})
    source_score = min(sources / 3.0, 1.0)
    geo_anchor = 1.0 if any(item.geo for item in evidence) else 0.0
    time_anchor = 1.0 if len(evidence) >= 3 else 0.0
    anchor_score = 0.5 * geo_anchor + 0.5 * time_anchor
    return round((0.4 * density + 0.3 * source_score + 0.3 * anchor_score), 3)


def draft_entry(
    target_date: str,
    lat: float | None = None,
    lon: float | None = None,
    days: int = 3,
    km: float | None = None,
) -> DraftEntry:
    packet = build_evidence_packet(target_date, days=days, lat=lat, lon=lon, km=km)
    evidence = packet_to_items(packet)
    style = _load_stylecard()
    if llm_available():
        prompt = _build_prompt(target_date, evidence, style)
        text = generate_text(prompt)
        if not text:
            text = _deterministic_draft(target_date, evidence, style)
    else:
        text = _deterministic_draft(target_date, evidence, style)

    draft = DraftEntry(
        date=target_date,
        text=text,
        confidence=_confidence(evidence),
        meta={
            "citations": _extract_citations(text),
            "evidence_count": len(evidence),
            "sources": packet.get("sources", []),
        },
    )
    return draft


def save_draft(draft: DraftEntry) -> None:
    write_jsonl(DRAFTS_PATH, [draft.model_dump()], mode="a")
