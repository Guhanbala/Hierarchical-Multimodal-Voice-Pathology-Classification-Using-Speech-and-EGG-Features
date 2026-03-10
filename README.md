# Hierarchical Multimodal Voice Pathology Classification Using Speech and EGG Features

> A two-stage machine learning framework for voice disorder screening and diagnosis, integrating deep speech representations with handcrafted EGG-based glottal features via late fusion.

---

## 📄 Paper

**"Hierarchical Multimodal Voice Pathology Classification Using Speech and EGG Features"**  
Gowtham S.D., Guhan K.B., Vaishnavi Gupta, Vyshnav Kumar  
Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India

---

## 🧠 Overview

Voice disorders impair communication and quality of life across a wide population. Clinical assessment traditionally depends on subjective perceptual evaluation and specialist availability. This project proposes an **automated, objective, non-invasive** system to support clinical decision-making.

The framework operates in **two hierarchical stages**, mirroring real clinical workflows:

| Stage | Task | Modality Used | Model |
|-------|------|--------------|-------|
| **Stage 1** | Binary detection: Healthy vs Pathological | Speech (CNN) + EGG (RF) via Late Fusion | Binary Classifier |
| **Stage 2** | Diagnosis: Structural/Inflammatory vs Hyperfunctional vs Neurological | Fused predictions from both modalities | Multi-class Classifier |

---

## 🏗️ System Architecture

```
Raw Dataset (Speech + EGG Signals)
         │
         ├──────────────────────┬──────────────────────────────┐
         ▼                      ▼                              │
  SPEECH PIPELINE          EGG PIPELINE                       │
  ─────────────────        ────────────────                    │
  Audio Preprocessing      Signal Preprocessing               │
  (Resample 16kHz,         (Filtering + Normalization)        │
   Normalize)                      │                          │
         │               Glottal Feature Extraction            │
         ▼               (192 Handcrafted Features via MATLAB) │
  Log-Mel Spectrogram              │                          │
         │               Random Forest Classifier              │
         ▼                         │                          │
  CNN Feature Learning             │                          │
  (Conv + BN + MaxPool +           │                          │
   Global Avg Pooling)             │                          │
         │                         │                          │
         └──────────┬──────────────┘                          │
                    ▼                                         │
              LATE FUSION                                     │
          (Combine Probabilities)                             │
                    │                                         │
          ┌─────────┴─────────┐                              │
          ▼                   ▼                              │
     Stage 1:             Stage 2:                           │
     Detection            Diagnosis                          │
  (Healthy vs Path.)  (Structural / Inflammatory /           │
                       Hyperfunctional / Neurological)       │
```

---

## 📁 Repository Structure

```
├── 1_data.py            # Dataset loading, preprocessing, patient-independent train-test split
├── 2_spectrogram.py     # Log-Mel spectrogram generation from speech signals
├── 3_speech_cnn.py      # CNN model definition, training, and speech-based prediction
├── 4_late_fussion.py    # Late fusion of CNN + Random Forest predictions, hierarchical classification
└── README.md
```

> **Note:** EGG feature extraction is performed separately in **MATLAB**.  
> Reference MATLAB code for glottal feature extraction:  
## 🔬 Methodology

### 1. Dataset Preparation (`1_data.py`)
- Synchronized **speech + EGG recordings** from healthy and voice-disordered subjects
- All audio resampled to **16 kHz**
- **Patient-independent split**: ~80% training, ~20% testing (no subject appears in both sets to prevent data leakage)

### 2. Speech Feature Extraction (`2_spectrogram.py`)
- Convert raw speech → **Log-Mel spectrogram** (via STFT + Mel filterbank projection)
- Log-scale transformation + normalization
- Resized to fixed resolution as CNN input

### 3. CNN-based Speech Classification (`3_speech_cnn.py`)
- Stacked **convolutional layers** with Batch Normalization and MaxPooling
- **Global Average Pooling** → compact embeddings
- Outputs class probability estimates

### 4. EGG Feature Extraction (MATLAB)
- Extracts **192 handcrafted glottal features** per recording
- Features span: glottal cycle parameters, time-domain characteristics, frequency-domain descriptors
- Captures vocal fold vibration and phonatory dynamics not visible in acoustic signal alone

### 5. EGG-based Random Forest Classifier
- Trained on the 192-dimensional EGG feature vectors
- Well-suited for high-dimensional handcrafted features
- Outputs class probability estimates independently

### 6. Late Fusion + Hierarchical Classification (`4_late_fussion.py`)
- Each modality produces **independent probability estimates**
- Predictions are **combined at decision level** (late fusion)
- **Stage 1**: Fused prediction → Healthy or Pathological
- **Stage 2**: Pathological samples only → Structural/Inflammatory, Hyperfunctional, or Neurological

---

## 📊 Results

### Stage 1 — Pathology Detection (Healthy vs Pathological)

| Metric | Value |
|--------|-------|
| **Accuracy** | **82.93%** |
| Correctly classified Healthy | 100 / 120 |
| Correctly classified Pathological | 66 / 80 |
| False Negatives (Path. → Healthy) | 14 |
| False Positives (Healthy → Path.) | 20 |

### Stage 2 — Pathological Diagnosis (3-class)

| Metric | Value |
|--------|-------|
| **Accuracy** | **67.24%** |
| Structural/Inflammatory (correct) | 25 / 35 |
| Hyperfunctional (correct) | 30 / 45 |
| Neurological (correct) | 12 / 20 |

> **Key insight:** Confusion between Structural and Hyperfunctional disorders is clinically expected — both conditions share overlapping symptoms (hoarseness, vocal strain, spectral irregularities).

---

## ⚙️ Installation & Requirements

### Python Dependencies
```bash
pip install numpy librosa scikit-learn tensorflow torch torchaudio matplotlib seaborn
```

### MATLAB
- Required for EGG glottal feature extraction
- Reference code available via the Amrita SharePoint link above

---

## 🚀 Running the Pipeline

Run the scripts in order:

```bash
# Step 1: Prepare and preprocess the dataset
python 1_data.py

# Step 2: Generate log-Mel spectrograms from speech signals
python 2_spectrogram.py

# Step 3: Train CNN on spectrograms and generate speech model predictions
python 3_speech_cnn.py

# Step 4 (MATLAB): Extract 192 EGG features and train Random Forest
# → Run the MATLAB glottal feature extraction scripts separately

# Step 5: Fuse predictions and run hierarchical classification
python 4_late_fussion.py
```

