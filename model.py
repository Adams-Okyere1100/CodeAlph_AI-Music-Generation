"""TensorFlow/Keras architecture for next-event music prediction."""

from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "music_generator.keras"
VOCABULARY_PATH = PROJECT_DIR / "models" / "vocabulary.json"
DEFAULT_SEQUENCE_LENGTH = 100


def build_model(
    vocabulary_size: int,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    embedding_dimension: int = 96,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
):
    """Build and compile a next-event classifier using two LSTM layers."""
    if vocabulary_size < 2:
        raise ValueError("The vocabulary must contain at least two musical events.")
    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("Dropout rate must be between 0 (inclusive) and 1 (exclusive).")

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow is required. Install dependencies with `pip install -r requirements.txt`.") from error

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length,), dtype="int32"),
            tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dimension),
            tf.keras.layers.LSTM(lstm_units, return_sequences=True),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(vocabulary_size),
            tf.keras.layers.Softmax(),
        ],
        name="music_event_lstm",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model