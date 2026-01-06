from __future__ import annotations

import re
from collections import Counter

from rag.config import ITEMS_PATH, STYLECARD_PATH
from rag.io import read_jsonl, write_json
from rag.models import NormalizedRecord, StyleCard
from rag.utils import tokenize


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def _sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
    return sentences


def build_stylecard() -> StyleCard:
    rows = read_jsonl(ITEMS_PATH)
    records = [NormalizedRecord.model_validate(row) for row in rows]
    entries = [r for r in records if r.source == "dayone" and r.text.strip()]
    if not entries:
        card = StyleCard(
            avg_sentence_length=12.0,
            frequent_phrases=[],
            structure_hints=["Short, factual sentences."],
            privacy_knobs={"redact_names": False, "redact_locations": False, "redact_contact_info": False},
        )
        write_json(STYLECARD_PATH, card.model_dump())
        return card

    sentence_lengths = []
    phrase_counter: Counter[str] = Counter()
    for entry in entries:
        sentences = _sentences(entry.text)
        for sentence in sentences:
            tokens = tokenize(sentence)
            if tokens:
                sentence_lengths.append(len(tokens))
            for n in (2, 3):
                for i in range(len(tokens) - n + 1):
                    phrase_counter[" ".join(tokens[i : i + n])] += 1

    avg_len = sum(sentence_lengths) / max(len(sentence_lengths), 1)
    frequent_phrases = [phrase for phrase, _ in phrase_counter.most_common(15)]

    hints = []
    if avg_len < 12:
        hints.append("Prefers short sentences.")
    else:
        hints.append("Prefers longer, reflective sentences.")
    if any("morning" in p or "evening" in p for p in frequent_phrases):
        hints.append("Often anchors entries to time of day.")
    if any("felt" in p or "feel" in p for p in frequent_phrases):
        hints.append("Uses emotional descriptors.")

    card = StyleCard(
        avg_sentence_length=avg_len,
        frequent_phrases=frequent_phrases,
        structure_hints=hints,
        privacy_knobs={"redact_names": False, "redact_locations": False, "redact_contact_info": False},
    )
    write_json(STYLECARD_PATH, card.model_dump())
    return card
