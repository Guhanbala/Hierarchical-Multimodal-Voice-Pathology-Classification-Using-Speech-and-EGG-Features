import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import random
from collections import Counter

# ================= REPRODUCIBILITY =================

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================= CONFIG =================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.0005
PATIENCE = 10

print("Using device:", DEVICE)

# ================= LABEL MAPPING =================

def map_binary(lbl):
    return 0 if lbl == 0 else 1

def map_diagnosis(lbl):
    if lbl == 0:
        return -1
    elif lbl in [1, 3]:
        return 1
    elif lbl == 2:
        return 0
    elif lbl == 4:
        return 2
    else:
        return -1

# ================= DATASET =================

class SpeechDataset(Dataset):

    def __init__(self, metadata_df, spectrogram_root):
        self.df = metadata_df.reset_index(drop=True)
        self.root = Path(spectrogram_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        rid = row["recording_id"]
        label = row["label"]
        disease = row["disease"]

        class_folder = f"C{label}_" + disease.replace(" ", "_")

        img_path = self.root / class_folder / f"{rid}.png"

        img = Image.open(img_path).convert("L")
        img = img.resize((128,128))

        img = np.array(img) / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img, map_binary(label), map_diagnosis(label)

# ================= MODEL =================

class HierarchicalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.stage1_head = nn.Linear(128,2)
        self.stage2_head = nn.Linear(128,3)

    def forward(self,x):

        x = self.features(x)
        x = self.global_pool(x)

        emb = self.embedding(x)

        out1 = self.stage1_head(emb)
        out2 = self.stage2_head(emb)

        return emb,out1,out2

# ================= DATASET STATISTICS =================

def print_dataset_stats(metadata):

    print("\n================ DATASET STATISTICS ================")

    total_samples = len(metadata)
    print(f"\nTotal Samples: {total_samples}")

    print("\n--- Samples per class ---")

    class_counts = metadata["label"].value_counts().sort_index()

    for lbl,count in class_counts.items():
        print(f"Class {lbl}: {count}")

    print("====================================================\n")

# ================= TRAIN FUNCTION =================

def train_and_test(metadata_df):

    print("\n==============================")
    print("Training model on COMBINED dataset")
    print("==============================")

    train_df = metadata_df[metadata_df["split"] == "train"]
    test_df = metadata_df[metadata_df["split"] == "test"]

    train_dataset = SpeechDataset(train_df,"Extracted_Features/Spectrograms/speech")
    test_dataset = SpeechDataset(test_df,"Extracted_Features/Spectrograms/speech")

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    labels = train_dataset.df["label"].values
    binary_labels = [map_binary(l) for l in labels]
    diag_labels = [map_diagnosis(l) for l in labels if l!=0]

    bin_count = Counter(binary_labels)
    diag_count = Counter(diag_labels)

    weights_stage1 = torch.tensor(
        [len(binary_labels)/bin_count[0],len(binary_labels)/bin_count[1]],
        dtype=torch.float32).to(DEVICE)

    weights_stage2 = torch.tensor(
        [len(diag_labels)/diag_count[0],
         len(diag_labels)/diag_count[1],
         len(diag_labels)/diag_count[2]],
        dtype=torch.float32).to(DEVICE)

    model = HierarchicalCNN().to(DEVICE)

    criterion_stage1 = nn.CrossEntropyLoss(weight=weights_stage1)
    criterion_stage2 = nn.CrossEntropyLoss(weight=weights_stage2)

    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # ================= TRAINING =================

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        total_samples = 0

        correct_stage1 = 0
        total_stage1 = 0

        correct_stage2 = 0
        total_stage2 = 0

        for x,y_bin,y_diag in train_loader:

            x = x.to(DEVICE)
            y_bin = y_bin.to(DEVICE)
            y_diag = y_diag.to(DEVICE)

            optimizer.zero_grad()

            emb,out1,out2 = model(x)

            loss1 = criterion_stage1(out1,y_bin)

            mask = y_bin==1

            if mask.sum()>0:
                loss2 = criterion_stage2(out2[mask],y_diag[mask])
            else:
                loss2 = torch.tensor(0.0,device=DEVICE)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*x.size(0)
            total_samples += x.size(0)

            preds1 = torch.argmax(out1,dim=1)

            correct_stage1 += (preds1==y_bin).sum().item()
            total_stage1 += y_bin.size(0)

            if mask.sum()>0:

                preds2 = torch.argmax(out2[mask],dim=1)

                correct_stage2 += (preds2==y_diag[mask]).sum().item()
                total_stage2 += mask.sum().item()

        scheduler.step()

        avg_loss = total_loss / total_samples

        acc_stage1 = correct_stage1 / total_stage1
        acc_stage2 = correct_stage2 / total_stage2 if total_stage2>0 else 0

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Stage1 Acc: {acc_stage1:.4f} | "
            f"Stage2 Acc: {acc_stage2:.4f}"
        )

        if avg_loss < best_loss:

            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()

        else:

            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ================= FINAL TEST =================

    model.eval()

    correct_stage1 = 0
    total_stage1 = 0
    correct_stage2 = 0
    total_stage2 = 0

    with torch.no_grad():

        for x,y_bin,y_diag in test_loader:

            x = x.to(DEVICE)
            y_bin = y_bin.to(DEVICE)
            y_diag = y_diag.to(DEVICE)

            emb,out1,out2 = model(x)

            preds1 = torch.argmax(out1,dim=1)

            correct_stage1 += (preds1==y_bin).sum().item()
            total_stage1 += y_bin.size(0)

            mask = y_bin==1

            if mask.sum()>0:

                preds2 = torch.argmax(out2[mask],dim=1)

                correct_stage2 += (preds2==y_diag[mask]).sum().item()
                total_stage2 += mask.sum().item()

    print("\n===== FINAL TEST RESULTS =====")

    print(f"Stage-1 Accuracy: {100 * correct_stage1 / total_stage1:.2f}%")

    if total_stage2>0:
        print(f"Stage-2 Accuracy: {100 * correct_stage2 / total_stage2:.2f}%")

# ================= MAIN =================

metadata = pd.read_csv("Final_5Class_Dataset/metadata.csv")

# Print dataset stats
print_dataset_stats(metadata)

# Train single combined model
train_and_test(metadata)