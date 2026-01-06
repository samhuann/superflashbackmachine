# SuperFlashbackMachine

Local-first journal backfill tool that reconstructs missing entries with hybrid retrieval-augmented generation. It ingests chat logs, photos, emails, and Day One exports, builds a hybrid BM25 + vector index, and drafts grounded entries in your voice with citations and conflict flags. Review in a terminal TUI, then export clean JSON back to your journal app.

## Features
- Offline by default: embeddings via `sentence-transformers`, deterministic drafting when no LLM is configured.
- Modular pipeline: loaders → normalize → style card → index → retrieve → draft → review → export.
- Hybrid retrieval with metadata filters (time window + geo radius).
- Fact-level citations `〔source:id〕` and conflict warnings `[[UNCERTAIN: ...]]`.
- Human-in-the-loop review queue with accept/edit/regenerate/flag.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Optional LLM config (still works without):

```bash
cp .env.example .env
# set OPENAI_API_KEY or LOCAL_LLM_ENDPOINT
```

## Data Inputs (MVP)

- **Day One JSON**: supports common export variants (`entries` vs `Entries`, `uuid` vs `UUID`, etc.)
- **Discord JSON**: common export formats with `messages` list.
- **Photos**: reads EXIF timestamp + GPS, and Google Takeout sidecars (`.jpg.json`).
- **MBOX**: parses headers and text/plain body snippet.

## CLI Usage

```bash
# ingest
sfbm ingest dayone path/to/DayOne.json
sfbm ingest discord path/to/messages.json
sfbm ingest photos path/to/photo/folder
sfbm ingest mbox path/to/mailbox.mbox

# normalize → style card → index
sfbm normalize
sfbm stylecard build
sfbm index build

# draft for a date
sfbm draft 2024-02-10 --days 3 --lat 37.77 --lon -122.42 --km 25

# review queue (Day One stubs)
sfbm review

# export
sfbm export --out data/processed/export.json
```

## Output Files

- `data/raw/*.jsonl` — ingested source records (normalized schema per source).
- `data/interim/items.jsonl` — normalized records used for indexing.
- `data/processed/stylecard.json` — learned style card.
- `data/processed/index/*` — hybrid index artifacts.
- `data/processed/drafts.jsonl` — draft queue.
- `data/processed/accepted.jsonl` — accepted drafts.

## Privacy Notes

- All processing runs locally by default.
- If you set `OPENAI_API_KEY` or `LOCAL_LLM_ENDPOINT`, drafting prompts will be sent to the configured LLM.
- Data directories are git-ignored by default.

## Confidence Scoring

Confidence is a 0–1 score combining:
- evidence density (more evidence → higher)
- multi-source corroboration
- presence of time/geo anchors

## Assumptions

- Day One “stubs” are entries with empty text.
- Target date filters are evaluated in UTC date boundaries for consistency.
- Drafting without an LLM is extractive and conservative, still with citations.

## Testing

```bash
pytest
```

## TODO (nice-to-have)

- More robust Day One import/export mapping (tags, photos, weather).
- Better conflict detection using time-of-day clusters.
- Optional vector store backends (Chroma, SQLite)
- Per-source weight tuning and stop-word filtering.
