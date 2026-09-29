# AI Music Generator

A student-focused project for training an LSTM neural network to learn patterns in MIDI music and generate new musical sequences. Music21 reads MIDI files, TensorFlow/Keras trains the model, and Streamlit provides a beginner-friendly interface.

## Features

- Read note and chord events from MIDI files with Music21.
- Build a reusable event vocabulary and compact training sequences.
- Train a configurable two-layer LSTM model with TensorFlow/Keras.
- Save and reuse the trained model and its vocabulary.
- Generate sequences with configurable length and temperature.
- Export generated sequences as MIDI files that can be downloaded from Streamlit.
- View dataset and model status and start project tasks in Streamlit.

## Technologies

- Python
- Music21 for MIDI parsing and musical event handling
- TensorFlow/Keras for the LSTM model
- NumPy for numerical data
- Streamlit for the web interface

## Project Structure

```text
AI-Music-Generator/
├── app.py                 # Streamlit interface
├── preprocess.py          # MIDI parsing and sequence preparation
├── model.py               # LSTM architecture
├── train.py               # Model training
├── generate.py            # Sequence generation and MIDI export
├── utils.py               # Shared status and display helpers
├── dataset/
│   ├── midi/                # Place source .mid/.midi files here
│   └── processed_data.npz   # Generated training cache (created by preprocessing)
├── generated/              # Generated MIDI output
├── models/
│   ├── music_generator.keras  # Saved model
│   └── vocabulary.json        # Event vocabulary and generation seed
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Use Python 3.10–3.12 for the broadest TensorFlow compatibility. TensorFlow wheel availability varies by Python version, operating system, and processor architecture.

Create and activate a virtual environment from the project directory:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Add MIDI Files

Add `.mid` or `.midi` files with the **Add MIDI files** uploader in the Dataset section, or copy them into `dataset/midi/`. Uploaded files are checked for a MIDI header, duplicate uploads are skipped, and existing files are not overwritten. Use files you have permission to use, and keep the collection varied enough to give the model examples of the musical style you want it to learn. Source MIDI files are intentionally excluded from Git.

## Preprocess the Dataset

Preprocessing reads `.mid` and `.midi` files recursively from `dataset/midi/`, extracts notes and chords, quantizes durations, and saves integer event data and valid sequence starts to `dataset/processed_data.npz`. The default sequence length is 100 events.

```bash
python preprocess.py
python preprocess.py --sequence-length 80
```

Files that cannot be parsed are skipped with a message. If no valid training sequences can be made, use longer MIDI files or reduce the sequence length.

## Train the Model

Training reads the preprocessed cache, creates shuffled batches without materializing every overlapping window in memory, and saves the model to `models/music_generator.keras`. The matching vocabulary and a fallback generation seed are saved to `models/vocabulary.json`.

```bash
python train.py
python train.py --epochs 30 --batch-size 32
```

Training can be CPU-intensive. Reduce the batch size or number of epochs if the device is slow or short on memory.

## Generate Music

Generation loads the saved model and vocabulary, samples events with a configurable temperature, and writes a MIDI file into `generated/`. The requested event count is the number of new events in the output MIDI.

```bash
python generate.py --events 200 --temperature 0.8
```

Lower temperatures make sampling more predictable; higher temperatures add variety. Temperature must be greater than zero.

## Run Streamlit

Launch the interface from the project directory with:

```bash
streamlit run app.py
```

The interface shows the number of MIDI files, preprocessing status, training controls and progress, model metadata, and generation controls. Generated MIDI files can be downloaded in the browser.

## Troubleshooting

- **TensorFlow will not install:** Confirm that your Python version, operating system, and processor architecture are supported by TensorFlow. Python 3.10–3.12 is a good starting point; create a fresh virtual environment with a compatible release and retry.
- **No MIDI files are found:** Check that files ending in `.mid` or `.midi` are directly inside `dataset/midi/` and that they are valid MIDI files.
- **No sequences are created:** Make sure at least one MIDI file contains more events than the selected sequence length. Try `python preprocess.py --sequence-length 50`.
- **Preprocessing or training runs out of memory:** Start with fewer MIDI files, shorter sequences, a smaller batch size, or fewer epochs.
- **Generated output is missing:** Confirm preprocessing and training have completed successfully and that the expected saved model and vocabulary are present.
- **Audio playback is unavailable:** This project exports MIDI and does not require an audio synthesizer. Open the downloaded MIDI in a compatible music application to listen to it.

## Future Improvements

- Optional MIDI-to-audio playback when a compatible synthesizer is available.
- More expressive event representations, including timing and duration.
- Validation metrics, checkpointing, and selectable musical styles.
- Improved support for larger datasets and resource-constrained devices.