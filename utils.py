"""Small shared helpers for reporting project and model status."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


def require_numpy():
    """Import NumPy only when numerical processing is requested."""
    try:
        import numpy
    except ImportError as error:
        raise RuntimeError("NumPy is required. Install dependencies with `pip install -r requirements.txt`.") from error
    return numpy


def save_midi_uploads(uploaded_files: list[Any], destination: Path) -> tuple[list[str], list[str]]:
    """Validate and save uploaded MIDI files without overwriting existing data."""
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    messages: list[str] = []

    for uploaded_file in uploaded_files:
        original_name = str(getattr(uploaded_file, "name", ""))
        file_name = Path(original_name.replace("\\", "/")).name
        safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", file_name).strip(" .")
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in {".mid", ".midi"}:
            messages.append(f"Rejected {original_name or 'unnamed file'}: choose a .mid or .midi file.")
            continue

        try:
            content = uploaded_file.getvalue()
        except Exception as error:
            messages.append(f"Could not read {safe_name}: {error}")
            continue

        header_length = int.from_bytes(content[4:8], byteorder="big") if len(content) >= 8 else 0
        if len(content) < 14 or content[:4] != b"MThd" or header_length < 6 or 8 + header_length > len(content):
            messages.append(f"Rejected {safe_name}: the file does not have a valid MIDI header.")
            continue

        candidate = destination / safe_name
        if candidate.is_file() and candidate.read_bytes() == content:
            messages.append(f"Skipped {safe_name}: that exact file is already in the dataset.")
            continue
        if candidate.exists():
            candidate = destination / f"{candidate.stem}_{uuid4().hex[:8]}{candidate.suffix}"

        try:
            with candidate.open("xb") as midi_file:
                midi_file.write(content)
        except OSError as error:
            messages.append(f"Could not save {safe_name}: {error}")
            continue
        saved.append(candidate.name)

    return saved, messages


def get_model_status(model_path: Path, vocabulary_path: Path) -> dict[str, Any]:
    """Return model availability and the metadata needed by the interface."""
    status: dict[str, Any] = {
        "available": False,
        "model_exists": model_path.is_file(),
        "vocabulary_exists": vocabulary_path.is_file(),
        "sequence_length": None,
        "vocabulary_size": None,
        "midi_file_count": None,
        "model_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "error": None,
    }
    if not status["model_exists"] or not status["vocabulary_exists"]:
        return status

    try:
        with vocabulary_path.open(encoding="utf-8") as vocabulary_file:
            metadata = json.load(vocabulary_file)
        if not isinstance(metadata, dict):
            raise ValueError("Vocabulary metadata must be a JSON object.")
        events = metadata.get("events")
        sequence_length = metadata.get("sequence_length")
        if (
            not isinstance(events, list)
            or not events
            or not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length < 1
        ):
            raise ValueError("Vocabulary metadata is missing its events or sequence length.")
        status.update(
            available=True,
            sequence_length=sequence_length,
            vocabulary_size=len(events),
            midi_file_count=metadata.get("midi_file_count"),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        status["error"] = str(error)
    return status


def format_file_size(size_bytes: int | None) -> str:
    """Format a file size for compact display."""
    if size_bytes is None:
        return "Not available"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"