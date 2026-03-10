import os
import pandas as pd
import librosa
import cv2
import numpy as np
from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "Final_5Class_Dataset"

SPEECH_DIR = DATASET_DIR / "speech"
METADATA_PATH = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = BASE_DIR / "Extracted_Features"
SPEC_DIR = OUTPUT_DIR / "Spectrograms"

SR = 16000

# Spectrogram parameters
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# ==========================================


# ================= DATASET STATS =================
def print_dataset_stats(df):

    print("\n===== DATASET OVERVIEW =====\n")

    print("Total rows:", len(df))
    print("Unique speakers:", df["speaker_id"].nunique())
    print("Unique recordings:", df["recording_id"].nunique())

    print("\n----- Gender distribution -----")
    print(df["gender"].value_counts())

    print("\n----- Class distribution -----")
    print(df["label"].value_counts())

    print("\n----- Disease distribution -----")
    print(df["disease"].value_counts())

    print("\n----- Train/Test split -----")
    print(df["split"].value_counts())

    print("\nTrain class distribution:")
    print(df[df["split"] == "train"]["label"].value_counts())

    print("\nTest class distribution:")
    print(df[df["split"] == "test"]["label"].value_counts())

    print("\n=============================\n")


# ================= LOG MEL SPECTROGRAM =================
def save_log_mel_spectrogram(y, sr, out_path):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # clip dynamic range
    mel_db = np.clip(mel_db, -80, 0)

    # scale to image
    mel_img = ((mel_db + 80) / 80 * 255).astype(np.uint8)

    cv2.imwrite(str(out_path), mel_img)


# ================= MAIN =================

metadata = pd.read_csv(METADATA_PATH)

if "split" not in metadata.columns:
    print("ERROR: split column missing")
    exit()

print_dataset_stats(metadata)

# create output dir
(SPEC_DIR / "speech").mkdir(parents=True, exist_ok=True)

processed = 0
skipped = 0

print("Starting spectrogram extraction...\n")

for _, row in metadata.iterrows():

    recording_id = row["recording_id"]
    label = int(row["label"])
    disease = row["disease"]

    class_folder = f"C{label}_" + disease.replace(" ", "_")

    speech_path = SPEECH_DIR / class_folder / f"{recording_id}.wav"

    if not speech_path.exists():
        skipped += 1
        continue

    output_class_dir = SPEC_DIR / "speech" / class_folder
    output_class_dir.mkdir(parents=True, exist_ok=True)

    try:

        y, _ = librosa.load(speech_path, sr=SR)

        # optional silence trimming
        y, _ = librosa.effects.trim(y)

        save_log_mel_spectrogram(
            y,
            SR,
            output_class_dir / f"{recording_id}.png"
        )

        processed += 1

    except Exception as e:
        print(f"Error processing {recording_id}: {e}")
        skipped += 1


print("\n===== PROCESS SUMMARY =====")
print("Processed:", processed)
print("Skipped:", skipped)
print("Output folder:", SPEC_DIR)
print("===========================")

print("\nLog-Mel spectrogram extraction completed.")