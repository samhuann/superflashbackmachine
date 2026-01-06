from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from rag.config import ACCEPTED_PATH, RAW_DIR
from rag.drafting import draft_entry, save_draft
from rag.export import export_entries
from rag.io import write_jsonl
from rag.loaders.dayone import load_dayone
from rag.loaders.discord import load_discord
from rag.loaders.mbox import load_mbox
from rag.loaders.photos import load_photos
from rag.normalization import normalize
from rag.review import load_drafts, missing_dates
from rag.stylecard import build_stylecard
from rag.index import build_index

app = typer.Typer(help="SuperFlashbackMachine CLI")
ingest_app = typer.Typer(help="Ingest raw sources")
stylecard_app = typer.Typer(help="Style card operations")
index_app = typer.Typer(help="Index operations")
app.add_typer(ingest_app, name="ingest")
app.add_typer(stylecard_app, name="stylecard")
app.add_typer(index_app, name="index")
console = Console()


@ingest_app.command("dayone")
def ingest_dayone(path: Path) -> None:
    records = load_dayone(path)
    out_path = RAW_DIR / "dayone.jsonl"
    write_jsonl(out_path, [r.model_dump() for r in records])
    console.print(f"Ingested {len(records)} Day One entries -> {out_path}")


@ingest_app.command("discord")
def ingest_discord(path: Path) -> None:
    records = load_discord(path)
    out_path = RAW_DIR / "discord.jsonl"
    write_jsonl(out_path, [r.model_dump() for r in records])
    console.print(f"Ingested {len(records)} Discord messages -> {out_path}")


@ingest_app.command("photos")
def ingest_photos(folder: Path) -> None:
    records = load_photos(folder)
    out_path = RAW_DIR / "photos.jsonl"
    write_jsonl(out_path, [r.model_dump() for r in records])
    console.print(f"Ingested {len(records)} photos -> {out_path}")


@ingest_app.command("mbox")
def ingest_mbox(path: Path) -> None:
    records = load_mbox(path)
    out_path = RAW_DIR / "mbox.jsonl"
    write_jsonl(out_path, [r.model_dump() for r in records])
    console.print(f"Ingested {len(records)} emails -> {out_path}")


@app.command("normalize")
def normalize_cmd() -> None:
    count = normalize()
    console.print(f"Normalized {count} records -> data/interim/items.jsonl")


@stylecard_app.command("build")
def stylecard_cmd() -> None:
    card = build_stylecard()
    console.print("Style card built.")
    console.print(card.model_dump_json(indent=2))


@index_app.command("build")
def index_cmd() -> None:
    count = build_index()
    console.print(f"Index built with {count} items.")


@app.command("draft")
def draft_cmd(
    date: str,
    lat: float | None = typer.Option(None),
    lon: float | None = typer.Option(None),
    days: int = typer.Option(3),
    km: float | None = typer.Option(None),
) -> None:
    draft = draft_entry(date, lat=lat, lon=lon, days=days, km=km)
    save_draft(draft)
    console.print(Panel(draft.text, title=f"Draft {date} ({draft.confidence})"))


@app.command("review")
def review_cmd() -> None:
    dates = missing_dates()
    drafts = load_drafts()
    if not dates:
        console.print("No missing dates found.")
        raise typer.Exit()

    editor = os.getenv("EDITOR") or ("notepad" if os.name == "nt" else "nano")

    for target_date in dates:
        draft = drafts.get(target_date) or draft_entry(target_date)
        console.print(Panel(draft.text, title=f"{target_date} (confidence {draft.confidence})"))
        action = Prompt.ask("Action", choices=["a", "e", "r", "f", "s", "q"], default="a")
        if action == "q":
            break
        if action == "s":
            continue
        if action == "r":
            draft = draft_entry(target_date)
            console.print(Panel(draft.text, title=f"{target_date} (confidence {draft.confidence})"))
        if action == "e":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as handle:
                handle.write(draft.text)
                temp_path = handle.name
            subprocess.call([editor, temp_path])
            draft_text = Path(temp_path).read_text(encoding="utf-8")
            draft.text = draft_text
        if action in {"a", "e", "r", "f"}:
            meta = draft.meta.copy()
            if action == "f":
                meta["flagged_uncertain"] = True
            accepted = draft.model_copy(update={"meta": meta})
            write_jsonl(ACCEPTED_PATH, [accepted.model_dump()], mode="a")
            console.print(f"Saved accepted draft for {target_date}.")


@app.command("export")
def export_cmd(out: Path = typer.Option(..., "--out")) -> None:
    count = export_entries(out)
    console.print(f"Exported {count} entries -> {out}")


if __name__ == "__main__":
    app()
