"""Train and save the next-event LSTM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

from model import MODEL_PATH, VOCABULARY_PATH, build_model
from preprocess import PROCESSED_DATA_PATH, load_processed_data
from utils import require_numpy


def train_model(
    epochs: int = 20,
    batch_size: int = 64,
    processed_path: Path = PROCESSED_DATA_PATH,
    model_path: Path = MODEL_PATH,
    vocabulary_path: Path = VOCABULARY_PATH,
    callbacks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Train the model and persist it with the matching event vocabulary."""
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    data = load_processed_data(processed_path)
    np = require_numpy()
    events = np.asarray(data["events"], dtype=np.int32)
    sequence_starts = np.asarray(data["sequence_starts"], dtype=np.int64)
    vocabulary = list(data["vocabulary"])
    sequence_length = int(data["sequence_length"])
    if not len(sequence_starts):
        raise ValueError("The processed dataset contains no training sequences. Run preprocessing again.")

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow is required. Install dependencies with `pip install -r requirements.txt`.") from error

    model = build_model(len(vocabulary), sequence_length=sequence_length)
    event_tensor = tf.convert_to_tensor(events, dtype=tf.int32)
    start_dataset = tf.data.Dataset.from_tensor_slices(sequence_starts)
    start_dataset = start_dataset.shuffle(
        buffer_size=min(len(sequence_starts), 10_000),
        seed=42,
        reshuffle_each_iteration=True,
    )

    def gather_training_pair(start: Any):
        positions = tf.range(start, start + sequence_length + 1, dtype=tf.int64)
        window = tf.gather(event_tensor, positions)
        return window[:-1], window[-1]

    training_dataset = (
        start_dataset
        .map(gather_training_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    history = model.fit(training_dataset, epochs=epochs, callbacks=list(callbacks or []), verbose=1, shuffle=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_metadata = {
        "events": vocabulary,
        "sequence_length": sequence_length,
        "midi_file_count": int(data["midi_file_count"]),
        "event_count": len(events),
        "seed_events": [
            vocabulary[int(event_id)]
            for event_id in events[int(sequence_starts[0]):int(sequence_starts[0]) + sequence_length]
        ],
    }
    with NamedTemporaryFile("w", encoding="utf-8", dir=vocabulary_path.parent, suffix=".json", delete=False) as temporary_file:
        json.dump(vocabulary_metadata, temporary_file, ensure_ascii=True, indent=2)
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(vocabulary_path)

    return {
        "model_path": model_path,
        "vocabulary_path": vocabulary_path,
        "sequence_length": sequence_length,
        "vocabulary_size": len(vocabulary),
        "sequence_count": len(sequence_starts),
        "midi_file_count": int(data["midi_file_count"]),
        "history": history.history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM music-generation model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    args = parser.parse_args()
    try:
        result = train_model(epochs=args.epochs, batch_size=args.batch_size)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"Training complete. Model saved to {result['model_path']} "
        f"with {result['vocabulary_size']} musical events."
    )