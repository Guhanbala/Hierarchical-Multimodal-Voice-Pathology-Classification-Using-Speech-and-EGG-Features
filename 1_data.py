import os
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# ================= USER CONFIG =================
DATASET_FOLDER_NAME = "dataset"
BASE_DIR = Path(__file__).parent.resolve()
ROOT_DATA_FOLDER = BASE_DIR / DATASET_FOLDER_NAME
OUTPUT_FOLDER = BASE_DIR / "Final_5Class_Dataset"
RANDOM_STATE = 42

# ================= CLASS LABELS =================
CLASS_MAP = {
    "healthy": 0,
    "Laryngitis": 1,
    "Hyperfunktionelle Dysphonie": 2,
    "Kontaktpachydermie": 3,
    "Rekurrensparese": 4
}

CLASS_NAME_MAP = {
    0: "C0_healthy",
    1: "C1_Laryngitis",
    2: "C2_Hyperfunktionelle_Dysphonie",
    3: "C3_Kontaktpachydermie",
    4: "C4_Rekurrensparese"
}

# ================= WAV CONVERSION =================
def convert_to_wav(input_path, output_path):
    try:
        cmd = ['ffmpeg', '-y', '-i', str(input_path), str(output_path), '-v', 'error']
        subprocess.run(cmd, check=True)
        return output_path.exists()
    except Exception:
        return False


# ================= AGE =================
def calculate_age(birth_date, recording_date):
    try:
        birth = datetime.strptime(str(birth_date).strip(), "%Y-%m-%d")
        rec = datetime.strptime(str(recording_date).strip(), "%Y-%m-%d")

        return rec.year - birth.year - (
            (rec.month, rec.day) < (birth.month, birth.day)
        )
    except:
        return None


# ================= READ METADATA =================
def read_overview(csv_path):

    try:
        df = pd.read_csv(csv_path, encoding="latin-1")

        if "AufnahmeID" not in df.columns:
            df = pd.read_csv(csv_path, sep=";", encoding="latin-1")

        info = {}

        for _, row in df.iterrows():

            rec_id = str(row["AufnahmeID"])
            speaker_id = str(row["SprecherID"])
            gender = str(row["Geschlecht"]).strip()

            age = None
            if pd.notna(row["Geburtsdatum"]) and pd.notna(row["AufnahmeDatum"]):
                age = calculate_age(row["Geburtsdatum"], row["AufnahmeDatum"])

            info[rec_id] = {
                "speaker_id": speaker_id,
                "gender": gender,
                "age": age
            }

        return info

    except:
        return {}


# ================= PROCESS DISEASE =================
def process_disease_folder(disease_folder, DEST_SPEECH, DEST_EGG):

    disease_name = disease_folder.name

    if disease_name not in CLASS_MAP:
        return []

    label = CLASS_MAP[disease_name]
    class_folder = CLASS_NAME_MAP[label]

    csv_path = disease_folder / "overview.csv"

    if not csv_path.exists():
        return []

    print(f"Processing {disease_name}")

    overview = read_overview(csv_path)

    metadata = []

    for root, _, files in os.walk(disease_folder):

        for file in files:

            if "a_n" not in file:
                continue

            if "egg" in file.lower():
                continue

            if "phrase" in file:
                continue

            rec_id = file.split("-")[0]

            if rec_id not in overview:
                continue

            speech_file = Path(root) / file

            egg_file = None

            for f in os.listdir(root):

                if rec_id in f and "egg" in f.lower() and "a_n" in f:
                    egg_file = Path(root) / f
                    break

            if egg_file is None:
                continue

            wav_name = f"{rec_id}.wav"

            speech_out = DEST_SPEECH / class_folder / wav_name
            egg_out = DEST_EGG / class_folder / wav_name

            speech_ok = convert_to_wav(speech_file, speech_out)
            egg_ok = convert_to_wav(egg_file, egg_out)

            if speech_ok and egg_ok:

                info = overview[rec_id]

                metadata.append({

                    "recording_id": rec_id,
                    "speaker_id": info["speaker_id"],
                    "gender": info["gender"],
                    "age": info["age"],
                    "disease": disease_name,
                    "label": label

                })

    return metadata


# ================= MAIN =================
def main():

    DEST_SPEECH = OUTPUT_FOLDER / "speech"
    DEST_EGG = OUTPUT_FOLDER / "egg"

    DEST_SPEECH.mkdir(parents=True, exist_ok=True)
    DEST_EGG.mkdir(parents=True, exist_ok=True)

    for c in CLASS_NAME_MAP.values():

        (DEST_SPEECH / c).mkdir(parents=True, exist_ok=True)
        (DEST_EGG / c).mkdir(parents=True, exist_ok=True)

    all_metadata = []

    for disease_dir in ROOT_DATA_FOLDER.iterdir():

        if disease_dir.is_dir():

            all_metadata.extend(
                process_disease_folder(
                    disease_dir,
                    DEST_SPEECH,
                    DEST_EGG
                )
            )

    if not all_metadata:
        print("No data processed")
        return

    df = pd.DataFrame(all_metadata)

    # ================= SPEAKER LEVEL DEDUP =================
    df = df.drop_duplicates(subset=["speaker_id"])

    print("Total speakers:", len(df))

    # ================= SPEAKER LEVEL SPLIT =================
    train_spk, test_spk = train_test_split(

        df["speaker_id"],
        test_size=0.2,
        stratify=df["label"],
        random_state=RANDOM_STATE

    )

    df["split"] = "train"
    df.loc[df["speaker_id"].isin(test_spk), "split"] = "test"

    df.to_csv(OUTPUT_FOLDER / "metadata.csv", index=False)

    print("\nDataset created")
    print(df["label"].value_counts())
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()