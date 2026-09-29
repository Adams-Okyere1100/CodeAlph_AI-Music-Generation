"""Streamlit interface for the AI Music Generator."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from generate import GENERATED_DIR, generate_music
from model import MODEL_PATH, VOCABULARY_PATH
from preprocess import MIDI_DIR, PROCESSED_DATA_PATH, find_midi_files, preprocess_dataset
from train import train_model
from utils import format_file_size, get_model_status, save_midi_uploads


st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="wide")


def main() -> None:
    st.title("🎵 AI Music Generator")
    st.caption("Train an LSTM on MIDI music, then generate a new MIDI sequence.")

    model_status = get_model_status(MODEL_PATH, VOCABULARY_PATH)
    midi_files = find_midi_files(MIDI_DIR)

    st.header("Dataset")
    uploaded_files = st.file_uploader(
        "Add MIDI files",
        type=["mid", "midi"],
        accept_multiple_files=True,
        help="Select one or more MIDI files from your computer to use as training data.",
    )
    if st.button("Add selected files", disabled=not uploaded_files, use_container_width=True):
        saved_files, upload_messages = save_midi_uploads(uploaded_files, MIDI_DIR)
        midi_files = find_midi_files(MIDI_DIR)
        if saved_files:
            st.success(f"Added {len(saved_files)} MIDI file(s): {', '.join(saved_files)}")
        for upload_message in upload_messages:
            if upload_message.startswith("Skipped "):
                st.info(upload_message)
            else:
                st.error(upload_message)

    dataset_columns = st.columns(3)
    dataset_columns[0].metric("MIDI files", len(midi_files))
    dataset_columns[1].metric("Dataset status", "Ready" if midi_files else "No MIDI files")
    dataset_columns[2].metric("Preprocessed data", "Available" if PROCESSED_DATA_PATH.is_file() else "Not prepared")
    if not midi_files:
        st.info("Add .mid or .midi files to the dataset/midi folder to get started.")

    sequence_length = st.number_input(
        "Sequence length",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Number of previous musical events used to predict the next event.",
    )
    if st.button("Preprocess MIDI files", disabled=not midi_files, use_container_width=True):
        progress_message = st.empty()
        try:
            with st.spinner("Reading MIDI files and preparing event sequences..."):
                result = preprocess_dataset(
                    sequence_length=int(sequence_length),
                    progress_callback=progress_message.info,
                )
            st.success(
                f"Prepared {result['sequence_count']:,} sequences from {result['midi_file_count']} MIDI files "
                f"({result['vocabulary_size']:,} unique events)."
            )
            for skipped_file in result["skipped_files"]:
                st.warning(f"Skipped {skipped_file}")
        except Exception as error:
            st.error(f"Preprocessing could not finish: {error}")

    st.divider()
    st.header("Model Training")
    training_columns = st.columns(2)
    epochs = training_columns[0].number_input("Epochs", min_value=1, max_value=500, value=20, step=1)
    batch_size = training_columns[1].number_input("Batch size", min_value=1, max_value=512, value=64, step=8)
    if not PROCESSED_DATA_PATH.is_file():
        st.caption("Preprocess the MIDI dataset before training.")
    if st.button("Train LSTM model", disabled=not PROCESSED_DATA_PATH.is_file(), use_container_width=True):
        try:
            import tensorflow as tf

            progress = st.progress(0, text="Preparing training...")
            training_status = st.empty()

            class TrainingProgress(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
                    current_epoch = epoch + 1
                    accuracy = (logs or {}).get("sparse_categorical_accuracy")
                    detail = f"Epoch {current_epoch}/{int(epochs)}"
                    if accuracy is not None:
                        detail += f" | accuracy {accuracy:.3f}"
                    progress.progress(current_epoch / int(epochs), text=detail)
                    training_status.info(detail)

            with st.spinner("Training the LSTM model. This may take a while on a CPU."):
                result = train_model(epochs=int(epochs), batch_size=int(batch_size), callbacks=[TrainingProgress()])
            model_status = get_model_status(MODEL_PATH, VOCABULARY_PATH)
            st.success(f"Model saved to {result['model_path']}.")
        except Exception as error:
            st.error(f"Training could not finish: {error}")

    st.divider()
    st.header("Music Generation")
    generation_columns = st.columns(2)
    number_of_events = generation_columns[0].number_input(
        "New events",
        min_value=1,
        max_value=2000,
        value=200,
        step=10,
        help="How many new note or chord events to generate.",
    )
    temperature = generation_columns[1].slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Lower values are more predictable; higher values add variety.",
    )
    if st.button("Generate music", disabled=not model_status["available"], use_container_width=True):
        try:
            with st.spinner("Generating events and writing the MIDI file..."):
                result = generate_music(number_of_events=int(number_of_events), temperature=float(temperature))
            st.session_state["generated_midi_path"] = str(result["path"])
            st.session_state["generated_event_count"] = int(result["event_count"])
            st.session_state["generated_temperature"] = float(result["temperature"])
            st.success(f"Generated {result['event_count']} events at temperature {result['temperature']:.1f}.")
        except Exception as error:
            st.error(f"Music generation could not finish: {error}")

    generated_path = st.session_state.get("generated_midi_path")
    if generated_path:
        midi_path = Path(generated_path)
        if midi_path.is_file() and midi_path.parent.resolve() == GENERATED_DIR.resolve():
            st.write(f"Generated MIDI: `{midi_path.name}` ({st.session_state.get('generated_event_count', 0)} events)")
            st.download_button(
                "Download MIDI",
                data=midi_path.read_bytes(),
                file_name=midi_path.name,
                mime="audio/midi",
                use_container_width=True,
            )

    st.divider()
    st.header("Model Status")
    if model_status["available"]:
        st.success("A trained model is ready to generate music.")
        info_columns = st.columns(4)
        info_columns[0].metric("Sequence length", model_status["sequence_length"])
        info_columns[1].metric("Musical events", f"{model_status['vocabulary_size']:,}")
        info_columns[2].metric("Training MIDI files", model_status["midi_file_count"])
        info_columns[3].metric("Model size", format_file_size(model_status["model_size_bytes"]))
    elif model_status["error"]:
        st.warning(f"Model metadata could not be read: {model_status['error']}")
    elif model_status["model_exists"] or model_status["vocabulary_exists"]:
        st.warning("The model and vocabulary are incomplete. Train the model again to restore both files.")
    else:
        st.info("No trained model is available yet.")


if __name__ == "__main__":
    main()