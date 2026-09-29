"""Parse MIDI files and prepare compact event sequences for LSTM training."""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from utils import require_numpy


PROJECT_DIR = Path(__file__).resolve().parent
MIDI_DIR = PROJECT_DIR / "dataset" / "midi"
PROCESSED_DATA_PATH = PROJECT_DIR / "dataset" / "processed_data.npz"
DEFAULT_SEQUENCE_LENGTH = 100
DURATION_QUANTUM = 0.25


def _format_duration(quarter_length: float) -> str:
    """Quantize duration to sixteenth-note increments to limit vocabulary size."""
    if not math.isfinite(quarter_length) or quarter_length <= 0:
        quarter_length = DURATION_QUANTUM
    quantized = max(DURATION_QUANTUM, round(quarter_length / DURATION_QUANTUM) * DURATION_QUANTUM)
    return f"{quantized:.2f}".rstrip("0").rstrip(".")


def _event_from_element(element: object, note_type: type, chord_type: type) -> str | None:
    """Convert a Music21 Note or Chord into a stable textual event."""
    duration = _format_duration(float(element.duration.quarterLength))
    if isinstance(element, note_type):
        return f"N|{int(element.pitch.midi)}|{duration}"
    if isinstance(element, chord_type):
        pitches = ",".join(str(int(pitch.midi)) for pitch in sorted(element.pitches, key=lambda item: item.midi))
        return f"C|{pitches}|{duration}"
    return None


def read_midi_events(midi_path: Path) -> list[str]:
    """Read note and chord events from one MIDI file using Music21."""
    try:
        from music21 import chord, converter, note
    except ImportError as error:
        raise RuntimeError("Music21 is required. Install dependencies with `pip install -r requirements.txt`.") from error

    score = converter.parse(str(midi_path), format="midi")
    events: list[str] = []
    for element in score.flatten().notes:
        event = _event_from_element(element, note.Note, chord.Chord)
        if event is not None:
            events.append(event)
    return events


def find_midi_files(dataset_dir: Path = MIDI_DIR) -> list[Path]:
    """Find MIDI files recursively in a dataset directory."""
    if not dataset_dir.exists():
        return []
    return sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )


def preprocess_dataset(
    dataset_dir: Path = MIDI_DIR,
    output_path: Path = PROCESSED_DATA_PATH,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, int | list[str]]:
    """Parse MIDI files and atomically save encoded events and training starts."""
    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")

    midi_files = find_midi_files(dataset_dir)
    if not midi_files:
        raise FileNotFoundError(f"No .mid or .midi files were found in {dataset_dir}.")
    np = require_numpy()

    event_streams: list[list[str]] = []
    skipped_files: list[str] = []
    for index, midi_path in enumerate(midi_files, start=1):
        if progress_callback:
            progress_callback(f"Reading {index}/{len(midi_files)}: {midi_path.name}")
        try:
            events = read_midi_events(midi_path)
            if events:
                event_streams.append(events)
            else:
                skipped_files.append(f"{midi_path.name}: no note or chord events")
        except Exception as error:
            skipped_files.append(f"{midi_path.name}: {error}")

    vocabulary = sorted({event for stream in event_streams for event in stream})
    if not vocabulary:
        details = "; ".join(skipped_files[:3])
        suffix = f" Details: {details}" if details else ""
        raise ValueError(f"No usable note or chord events were found.{suffix}")

    event_to_id = {event: index for index, event in enumerate(vocabulary)}
    encoded_streams = [np.asarray([event_to_id[event] for event in stream], dtype=np.int32) for stream in event_streams]
    sequence_starts: list[np.ndarray] = []
    offset = 0
    for stream in encoded_streams:
        valid_count = len(stream) - sequence_length
        if valid_count > 0:
            sequence_starts.append(np.arange(offset, offset + valid_count, dtype=np.int64))
        offset += len(stream)

    if not sequence_starts:
        longest = max((len(stream) for stream in encoded_streams), default=0)
        raise ValueError(
            f"No MIDI file has more than {sequence_length} events (longest has {longest}). "
            "Add longer MIDI files or use a shorter sequence length."
        )

    all_events = np.concatenate(encoded_streams)
    all_starts = np.concatenate(sequence_starts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        np.savez_compressed(
            temporary_path,
            events=all_events,
            sequence_starts=all_starts,
            vocabulary_json=np.asarray(json.dumps(vocabulary)),
            sequence_length=np.asarray(sequence_length, dtype=np.int32),
            midi_file_count=np.asarray(len(event_streams), dtype=np.int32),
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "midi_file_count": len(event_streams),
        "event_count": len(all_events),
        "vocabulary_size": len(vocabulary),
        "sequence_count": len(all_starts),
        "skipped_files": skipped_files,
    }


def load_processed_data(processed_path: Path = PROCESSED_DATA_PATH) -> dict[str, object]:
    """Load preprocessed data without enabling NumPy object deserialization."""
    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Run preprocessing first.")
    np = require_numpy()
    with np.load(processed_path, allow_pickle=False) as data:
        return {
            "events": data["events"].copy(),
            "sequence_starts": data["sequence_starts"].copy(),
            "vocabulary": json.loads(str(data["vocabulary_json"].item())),
            "sequence_length": int(data["sequence_length"].item()),
            "midi_file_count": int(data["midi_file_count"].item()),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MIDI event sequences for model training.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Number of previous events in each training input (default: 100).",
    )
    args = parser.parse_args()
    try:
        summary = preprocess_dataset(sequence_length=args.sequence_length)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(
        "Preprocessing complete: "
        f"{summary['midi_file_count']} MIDI files, {summary['event_count']} events, "
        f"{summary['vocabulary_size']} unique events, {summary['sequence_count']} sequences."
    )
    for warning in summary["skipped_files"]:
        print(f"Skipped: {warning}")