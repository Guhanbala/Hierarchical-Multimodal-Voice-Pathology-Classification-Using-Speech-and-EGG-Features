import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ================= FILE PATHS =================

SPEECH_CSV = r"D:\speech_project\Extracted_Features\speech_cnn_128_features.csv"
EGG_CSV = r"D:\speech_project\Extracted_Features\eeegg_glottal_192_features.csv"
METADATA = r"Final_5Class_Dataset\metadata.csv"

# ================= LOAD DATA =================

speech_df = pd.read_csv(SPEECH_CSV)
egg_df = pd.read_csv(EGG_CSV)
metadata = pd.read_csv(METADATA)

print("Speech features:", speech_df.shape)
print("EGG features:", egg_df.shape)

# ================= MERGE FEATURES =================

df = pd.merge(speech_df, egg_df, on="sample_index")

# ================= ADD SPLIT INFORMATION =================

df["split"] = metadata["split"]

# ================= FEATURE SELECTION =================

speech_features = [c for c in df.columns if c.startswith("f")]
egg_features = [c for c in df.columns if c.startswith("g")]

# ================= SPLIT DATA =================

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

X_train_s = train_df[speech_features]
X_test_s = test_df[speech_features]

X_train_e = train_df[egg_features]
X_test_e = test_df[egg_features]

y_train_stage1 = train_df["stage1_label"]
y_test_stage1 = test_df["stage1_label"]

y_train_stage2 = train_df["stage2_label"]
y_test_stage2 = test_df["stage2_label"]

# ================= NORMALIZATION =================

scaler_s = StandardScaler()
scaler_e = StandardScaler()

X_train_s = scaler_s.fit_transform(X_train_s)
X_test_s = scaler_s.transform(X_test_s)

X_train_e = scaler_e.fit_transform(X_train_e)
X_test_e = scaler_e.transform(X_test_e)

# ================= MODELS =================

svm_model = SVC(probability=True, kernel="rbf", random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# ================= TRAIN =================

print("\nTraining SVM (Speech features)...")
svm_model.fit(X_train_s, y_train_stage1)

print("Training Random Forest (EGG features)...")
rf_model.fit(X_train_e, y_train_stage1)

# ================= PREDICTIONS =================

speech_probs = svm_model.predict_proba(X_test_s)
egg_probs = rf_model.predict_proba(X_test_e)

# ================= LATE FUSION (AVERAGE) =================

fused_probs = (speech_probs + egg_probs) / 2

stage1_preds = np.argmax(fused_probs, axis=1)

# ================= STAGE 1 RESULTS =================

acc_stage1 = accuracy_score(y_test_stage1, stage1_preds)

print("\n===== STAGE 1 RESULTS =====")
print(f"Stage-1 Accuracy: {acc_stage1*100:.2f}%")

cm1 = confusion_matrix(y_test_stage1, stage1_preds)

disp1 = ConfusionMatrixDisplay(
    confusion_matrix=cm1,
    display_labels=["Healthy","Pathological"]
)

disp1.plot(cmap="Blues")
plt.title("Stage-1 Confusion Matrix (Healthy vs Pathological)")
plt.show()

# ================= STAGE 2 =================

mask = y_test_stage1 == 1

speech_probs2 = speech_probs[mask]
egg_probs2 = egg_probs[mask]

fused_probs2 = (speech_probs2 + egg_probs2) / 2

stage2_preds = np.argmax(fused_probs2, axis=1)

y_test_stage2_filtered = y_test_stage2[mask]

acc_stage2 = accuracy_score(y_test_stage2_filtered, stage2_preds)

print("\n===== STAGE 2 RESULTS =====")
print(f"Stage-2 Accuracy: {acc_stage2*100:.2f}%")

cm2 = confusion_matrix(y_test_stage2_filtered, stage2_preds)

disp2 = ConfusionMatrixDisplay(
    confusion_matrix=cm2,
    display_labels=[
        "Structural",
        "Hyperfunctional",
        "Neurological"
    ]
)

disp2.plot(cmap="Blues")
plt.title("Stage-2 Confusion Matrix (Diagnosis)")
plt.show()
