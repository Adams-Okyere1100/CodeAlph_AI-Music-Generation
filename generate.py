"""Generate new event sequences and export them as MIDI files."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from model import MODEL_PATH, VOCABULARY_PATH
from preprocess import PROCESSED_DATA_PATH, load_processed_data
from utils import require_numpy


PROJECT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = PROJECT_DIR / "generated"


def _parse_event(event: str) -> tuple[str, list[int], float]:
    """Decode one note or chord token and validate its musical values."""
    try:
        kind, pitches_text, duration_text = event.split("|")
        pitches = [int(pitch) for pitch in pitches_text.split(",")]
        duration = float(duration_text)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid musical event in vocabulary: {event!r}.") from error
    if kind not in {"N", "C"} or not pitches or any(pitch < 0 or pitch > 127 for pitch in pitches):
        raise ValueError(f"Invalid musical event in vocabulary: {event!r}.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid musical event in vocabulary: {event!r}.")
    if kind == "N" and len(pitches) != 1:
        raise ValueError(f"Invalid note event in vocabulary: {event!r}.")
    return kind, pitches, duration


def write_events_to_midi(events: list[str], output_path: Path) -> Path:
    """Write generated note/chord tokens as a sequential MIDI composition."""
    try:
        from music21 import chord, note, stream
    except ImportError as error:
        raise RuntimeError("Music21 is required for MIDI export. Install dependencies with `pip install -r requirements.txt`.") from error

    if not events:
        raise ValueError("There are no generated events to write.")
    composition = stream.Stream()
    offset = 0.0
    for event in events:
        kind, pitches, duration = _parse_event(event)
        if kind == "N":
            music_event = note.Note(pitches[0])
        else:
            music_event = chord.Chord(pitches)
        music_event.duration.quarterLength = duration
        composition.insert(offset, music_event)
        offset += duration

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composition.write("midi", fp=str(output_path))
    return output_path


def _select_seed(
    metadata: dict[str, object],
    event_to_id: dict[str, int],
    sequence_length: int,
    rng: Any,
) -> Any:
    """Choose a random training window, falling back to the saved seed."""
    np = require_numpy()
    if PROCESSED_DATA_PATH.is_file():
        data = load_processed_data(PROCESSED_DATA_PATH)
        if data["vocabulary"] == metadata["events"] and int(data["sequence_length"]) == sequence_length:
            starts = np.asarray(data["sequence_starts"], dtype=np.int64)
            events = np.asarray(data["events"], dtype=np.int32)
            if len(starts):
                start = int(rng.choice(starts))
                return events[start:start + sequence_length].copy()

    saved_seed = metadata.get("seed_events")
    if isinstance(saved_seed, list) and len(saved_seed) == sequence_length:
        try:
            return np.asarray([event_to_id[str(event)] for event in saved_seed], dtype=np.int32)
        except KeyError:
            pass
    raise FileNotFoundError(
        f"A starting sequence is unavailable. Restore the processed dataset at {PROCESSED_DATA_PATH} "
        "or preprocess and train the model again."
    )


def generate_music(
    number_of_events: int = 200,
    temperature: float = 0.8,
    model_path: Path = MODEL_PATH,
    vocabulary_path: Path = VOCABULARY_PATH,
    output_dir: Path = GENERATED_DIR,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Generate a requested number of events and save them to a MIDI file."""
    np = require_numpy()
    if number_of_events < 1:
        raise ValueError("Number of events must be at least 1.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite number greater than 0.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train a model first.")
    if not vocabulary_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found at {vocabulary_path}. Train a model first.")

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow is required for generation. Install dependencies with `pip install -r requirements.txt`.") from error

    with vocabulary_path.open(encoding="utf-8") as vocabulary_file:
        metadata = json.load(vocabulary_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Vocabulary file is invalid: {vocabulary_path}.")
    vocabulary = metadata.get("events")
    sequence_length = metadata.get("sequence_length")
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or not all(isinstance(event, str) for event in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
        or not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < 1
    ):
        raise ValueError(f"Vocabulary file is invalid: {vocabulary_path}.")

    event_to_id = {event: index for index, event in enumerate(vocabulary)}
    rng = np.random.default_rng(random_seed)
    current_sequence = _select_seed(metadata, event_to_id, sequence_length, rng)
    model = tf.keras.models.load_model(model_path)
    if int(model.output_shape[-1]) != len(vocabulary):
        raise ValueError("Model output size does not match the saved vocabulary. Retrain the model.")

    generated_events: list[str] = []
    for _ in range(number_of_events):
        probabilities = np.asarray(model.predict(current_sequence[np.newaxis, :], verbose=0)[0], dtype=np.float64)
        if probabilities.shape != (len(vocabulary),) or not np.all(np.isfinite(probabilities)):
            raise ValueError("The model returned invalid next-event probabilities.")
        logits = np.log(np.maximum(probabilities, 1e-12)) / temperature
        logits -= np.max(logits)
        sampling_probabilities = np.exp(logits)
        sampling_probabilities /= sampling_probabilities.sum()
        next_event_id = int(rng.choice(len(vocabulary), p=sampling_probabilities))
        generated_events.append(str(vocabulary[next_event_id]))
        current_sequence = np.concatenate((current_sequence[1:], np.asarray([next_event_id], dtype=np.int32)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"generated_{timestamp}_{uuid4().hex[:8]}.mid"
    write_events_to_midi(generated_events, output_path)
    return {
        "path": output_path,
        "events": generated_events,
        "event_count": len(generated_events),
        "temperature": float(temperature),
        "seed_length": sequence_length,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a MIDI file using the trained LSTM model.")
    parser.add_argument("--events", type=int, default=200, help="Number of new musical events to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature; higher values are more random.")
    args = parser.parse_args()
    try:
        result = generate_music(args.events, args.temperature)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Generated {result['event_count']} events: {result['path']}")