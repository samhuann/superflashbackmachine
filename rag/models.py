from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Geo(BaseModel):
    lat: float
    lon: float


class NormalizedRecord(BaseModel):
    id: str
    source: str
    utc_datetime: datetime
    local_tz: str | None = None
    text: str
    geo: Geo | None = None
    participants: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class StyleCard(BaseModel):
    avg_sentence_length: float
    frequent_phrases: list[str]
    structure_hints: list[str]
    privacy_knobs: dict[str, Any]


class EvidenceItem(BaseModel):
    id: str
    source: str
    utc_datetime: datetime
    text: str
    geo: Geo | None = None
    score: float = 0.0


class DraftEntry(BaseModel):
    date: str
    text: str
    confidence: float
    meta: dict[str, Any] = Field(default_factory=dict)
