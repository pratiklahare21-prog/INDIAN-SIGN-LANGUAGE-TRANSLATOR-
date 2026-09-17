# ISL Sign2Speech

Real-time Indian Sign Language (ISL) recognition pipeline that translates hand signs into sentences and speaks them aloud.
Runs entirely **local and offline** on your Windows PC — built with MediaPipe Hands, a Word LSTM + Alphabet MLP, and a Streamlit web UI.

---

## 🎯 Features

- **A–Z Fingerspelling Recognition** — static hand poses for every letter via a trained MLP classifier.
- **60 Word Signs** — dynamic 2-hand gesture recognition via a sequence-trained Bidirectional LSTM.
- **Sentence Builder** — tap to append recognized words/letters, edit on the fly, auto-space tokens.
- **Text-to-Speech (TTS)** — one-click `pyttsx3` (SAPI5 on Windows) reads the full sentence aloud.
- **Live Webcam Feed** — low-latency OpenCV capture with MediaPipe landmark overlay drawn in-browser.
- **Prediction History** — last N predictions with per-class confidence logged to the sidebar.
- **Confidence Thresholds** — adjustable `WORD_THRESHOLD` and `ALPHABET_THRESHOLD` sliders to suppress jitter.
- **Offline / Privacy-Focused** — no cloud calls, no telemetry; video frames never leave your machine.

---

## 🧱 Architecture

```
                     ┌──────────────────────────────┐
  Webcam / Video  ──►│  MediaPipe Hands (per frame) ├──► 63-d landmark vector (x,y,z × 21)
                     └───────────┬──────────────────┘
                                 │
              ┌──────────────────┴────────────────────┐
              │                                       │
              ▼                                       ▼
   ┌──────────────────────┐              ┌──────────────────────────┐
   │  Alphabet MLP        │              │  Word LSTM (sequence)    │
   │  (single-frame,      │              │  (60-frame window,       │
   │   26 classes)        │              │   bidirectional, 60 cls) │
   └──────────┬───────────┘              └────────────┬─────────────┘
              │                                       │
              └──────────────────┬────────────────────┘
                                 ▼
                     ┌──────────────────────┐
                     │   SentenceBuilder    │  (debounce, dedupe,
                     └──────────┬───────────┘   confidence gating,
                                │               history buffer)
                                ▼
                     ┌──────────────────────┐
                     │  pyttsx3 Speech      │  (SAPI5 voices,
                     └──────────────────────┘   rate/volume tunable)
```

**Input pipeline detail:**
- 21 hand landmarks × 3 coordinates (x, y, z) = **63 floats per frame**.
- Alphabet model consumes a single 63-d vector (static image).
- Word model consumes a fixed-length sequence of `SEQUENCE_LENGTH=60` frames × 63-d.

---

## 📋 Requirements

| Item | Required |
|---|---|
| **OS** | Windows 10 or Windows 11 (64-bit) |
| **Python** | 3.10, 3.11, or 3.12 — **NOT 3.13+** (TensorFlow 2.16 has no 3.13 wheels) |
| **Hardware** | Working webcam (internal USB or external) |
| **Audio** | Speakers / headphones for TTS output |
| **Disk** | ~4 GB free for the Kaggle word dataset + ~500 MB for PyPI packages |
| **RAM** | 8 GB minimum (16 GB recommended for training) |

---

## 🚀 Step-by-Step Setup

### Step 1 — Clone / navigate to the repo

```bat
git clone <your-repo-url>
cd d:\ISL-Sign2Speech
```

If you already have the folder locally just `cd d:\ISL-Sign2Speech`.

### Step 2 — Create and activate a virtual environment

```bat
python -m venv venv
venv\Scripts\activate
```

Your prompt should now start with `(venv)`.

### Step 3 — Install Python dependencies

```bat
pip install --upgrade pip
pip install -r requirements.txt
```

This installs TensorFlow 2.16, MediaPipe 0.10.14, OpenCV, Streamlit, pyttsx3, sklearn, tqdm, matplotlib, and friends.

### Step 4 — Download the word (video) dataset

Grab the 60-class ISL video dataset from Kaggle:

👉 **https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset** (~3.48 GB)

Extract the ZIP so that each class lives in its own folder. The final layout **must** be:

```
dataset\
  words\
    <ClassName1>\
      001.mp4
      002.mp4
      ...
    <ClassName2>\
      001.mp4
      ...
```

**Class names rule:** the subfolder names become class labels — keep them as-is. (If the Kaggle zip contains a top-level folder like `Indian Sign Language Video Dataset/`, move the per-class folders up so they are direct children of `dataset\words\`.)

Expected ~60 class subfolders total. You do **not** need to list or rename them.

### Step 5 — Prepare the alphabet (image) dataset

Put **at least 10 JPG or PNG images per letter** under letter-named folders:

```
dataset\
  alphabet\
    A\  img1.jpg  img2.png  ...
    B\  img1.jpg  ...
    ...
    Z\  ...
```

**Three ways to get alphabet data (pick one):**
1. **Shoot your own** — hold up each ISL letter under normal room lighting, ~30+ images per letter works best.
2. **Reuse the word dataset** — if the Kaggle word download already contains per-letter A–Z folders, copy/move them under `dataset\alphabet\`.
3. **Download a public ISL alphabet image dataset** — search Kaggle for "Indian Sign Language alphabet images", download, and drop the 26 class folders into `dataset\alphabet\`.

### Step 6 — Preprocess data and train models

Run each command in order from the repo root (with venv active):

```bat
python src\preprocess_alphabet.py
python src\train_alphabet.py

python src\preprocess_words.py
python src\train_words.py
```

**What each step produces:**

| Step | Outputs |
|---|---|
| `preprocess_alphabet.py` | `data\landmarks\alphabet_X.npy`, `alphabet_y.npy`, `alphabet_X_train.npy`, `alphabet_X_val.npy`, etc. + `models\labels_alphabet.json` |
| `train_alphabet.py`     | `models\alphabet_model.keras`, `models\alphabet_model_best.keras`, `models\scaler_alphabet.json`, `data\training_log_alphabet.csv`, `models\alphabet_training_curves.png` |
| `preprocess_words.py`   | `data\landmarks\words_X.npy`, `words_y.npy`, `words_X_train.npy`, `words_X_val.npy`, etc. + `models\labels_word.json` |
| `train_words.py`        | `models\word_model.keras`, `models\word_model_best.keras`, `models\scaler_words.json`, `data\training_log_words.csv`, `models\word_training_curves.png` |

**Smoke (quick) training flag.** For a fast end-to-end check you can pass `--quick` to the trainers. It caps samples and runs only 1 epoch — great for verifying the pipeline before a full overnight train:

```bat
python src\train_alphabet.py --quick
python src\train_words.py --quick
```

You can also override epochs directly, e.g. `python src\train_words.py --epochs 120`.

### Step 7 — Launch the app

Either (easy mode):

```bat
run.bat
```

Or manually (venv must be active):

```bat
streamlit run app.py
```

A browser tab should open at `http://localhost:8501/` showing the webcam UI.

---

## ▶️ What run.bat does (one-click launcher)

Double-clicking `run.bat` or running it from cmd performs **all of these in order**:

1. **Checks Python on PATH** — bails with a clear error if `python.exe` is missing, and warns if the version is outside 3.10–3.12.
2. **Auto-creates `venv\`** — runs `python -m venv venv` only if `venv\Scripts\python.exe` is absent.
3. **Activates the venv** — calls `venv\Scripts\activate.bat` for the rest of the script.
4. **Upgrades pip + installs packages** — `pip install -r requirements.txt` (idempotent; skips if all pinned versions already installed).
5. **Runs `setup\check_env.py`** — prints a colourful environment / dataset / model summary.
6. **Auto-trains alphabet if needed** — if `models\alphabet_model.keras` is missing **and** `dataset\alphabet\` exists, runs `preprocess_alphabet.py` then `train_alphabet.py --quick`.
7. **Auto-trains words if needed** — if `models\word_model.keras` is missing **and** `dataset\words\` exists, runs `preprocess_words.py` then `train_words.py --quick`.
8. **Launches Streamlit** — `streamlit run app.py`, then `pause` at the end so you can read any errors.

> **Tip:** auto-train uses `--quick`. When you are ready for a real model, run the train commands manually **without** `--quick`.

---

## 🔧 CLI Cheatsheet

| Script | Purpose | Common flags |
|---|---|---|
| `python src\preprocess_alphabet.py` | Reads A–Z images from `dataset\alphabet\`, runs MediaPipe Hands per image, splits and saves 63-d landmark `.npy` arrays + `labels_alphabet.json`. | `--data-dir` (override alphabet root), `--out-dir` (override landmark folder), `--labels-path` (override label JSON path) |
| `python src\preprocess_words.py`   | Reads per-class MP4s from `dataset\words\`, samples 60 frames/video, runs MediaPipe, saves 60×63 landmark sequences + `labels_word.json`. | `--data-dir`, `--out-dir`, `--labels-path`, `--seq-len N` (change sequence length, default 60) |
| `python src\train_alphabet.py`     | Loads precomputed alphabet landmarks, trains an MLP classifier, saves `.keras` model + StandardScaler JSON + training curves PNG + CSV log. | `--epochs N` (override max epochs, default 50), `--quick` (1 epoch / 200 train / 50 val), `--landmarks-dir`, `--labels-path`, `--model-out` |
| `python src\train_words.py`        | Loads precomputed word landmark sequences, trains a Bidirectional LSTM, saves `.keras` model + scaler + curves + CSV log. | `--epochs N` (default 80), `--quick` (1 epoch / 100 train / 25 val), `--landmarks-dir`, `--labels-path`, `--model-out` |
| `python setup\check_env.py`        | No-op diagnostic. Verifies Python version (3.10–3.12), required packages, creates missing folders, counts alphabet/word dataset classes, checks for 4 model files, then prints recommended "Next steps". | (no flags) |
| `streamlit run app.py`             | Starts the Streamlit web UI at `http://localhost:8501/`. Loads both trained models, the HandTracker, SentenceBuilder, and pyttsx3 engine. | Streamlit flags: `--server.port 8501`, `--server.headless true`, `--server.address 127.0.0.1` |

---

## 🤔 Troubleshooting

- **TensorFlow install fails on Windows**
  → Use Python 3.10–3.12 (NOT 3.13+). Install the [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). If still failing: try `pip install tensorflow-cpu==2.16.1`, or install TensorFlow via Conda (`conda install tensorflow=2.16`).

- **OpenCV VideoCapture returns None / webcam blank in the UI**
  → Change `WEBCAM_INDEX` in `config.py` from 0 to 1 or 2. Grant camera permission to the terminal/IDE (Windows Settings → Privacy & security → Camera). Close any app holding the webcam (Zoom, Teams, OBS, Chrome tabs using camera).

- **MediaPipe not finding hands (no landmarks drawn, predictions all 0)**
  → Ensure a plain background, even lighting, and your palm is facing the camera. Raise `min_detection_confidence` / `min_tracking_confidence` in `src\hand_tracker.py`'s `HandTracker` constructor if needed (defaults are usually fine).

- **pyttsx3 fails / no audio on Windows**
  → Make sure SAPI5 voices exist: Settings → Time & Language → Speech → Manage voices, install at least one. Reinstall pyttsx3 cleanly: `pip uninstall -y pyttsx3 && pip install pyttsx3`. Run `python -c "import pyttsx3; e=pyttsx3.init(); e.say('hello'); e.runAndWait()"` as a quick test.

- **Low word-recognition accuracy**
  → Remove the `--quick` flag and train for full epochs. Ensure each word class has at least 20 videos. Raise the `WORD_THRESHOLD` slider in the UI to ignore shaky predictions. Open `models\labels_word.json` and double-check the class index matches the sign you are performing (most confusion happens between visually similar pairs).

- **Low alphabet (A–Z) accuracy**
  → Add more images per letter (target 30+, include varied backgrounds, different hands, different lighting). Re-run `preprocess_alphabet.py` then `train_alphabet.py` (no `--quick`).

- **Streamlit opens a blank tab / CORS / connection refused**
  → Always access via `http://localhost:8501/` (not `http://0.0.0.0:8501/`). Ensure Windows Firewall permits Python for localhost (it usually auto-prompts the first run). If you changed the port manually, add `--server.port 8501`.

- **`check_env.py` shows packages MISSING right after `pip install -r requirements.txt`**
  → Almost certainly you forgot to activate the venv. Run `venv\Scripts\activate` and re-run both commands. Confirm with `where python` — the first result should be `d:\ISL-Sign2Speech\venv\Scripts\python.exe`.

- **(Bonus) Training runs but val_acc plateaus very low**
  → Check landmark shapes visually via the curves PNG under `models\`. If training loss ≫ val loss, increase dropout in `config.py` (`ALPHABET_MODEL_KWARGS` / `WORD_MODEL_KWARGS`). If both are high, add more data or double `--epochs`.

---

## 📁 Project Tree

```
ISL-Sign2Speech\
├── app.py                          # Streamlit web UI entry point
├── config.py                       # All tunables: paths, model sizes, thresholds, SEED
├── requirements.txt                # Pinned deps (TF 2.16, MP 0.10.14, …)
├── run.bat                         # One-click Windows launcher
├── README.md                       # This file
│
├── dataset\                        # (created by check_env / you)
│   ├── alphabet\
│   │   ├── A\  *.jpg/*.png         # ≥10 images per A–Z folder
│   │   ├── B\
│   │   └── ... Z\
│   └── words\
│       ├── <WordClass1>\  *.mp4    # 60 class folders, each with N videos
│       ├── <WordClass2>\
│       └── ...
│
├── data\
│   ├── landmarks\                  # Preprocessed .npy arrays produced by preprocess_*.py
│   │   ├── alphabet_X_train.npy
│   │   ├── alphabet_y_train.npy
│   │   ├── alphabet_X_val.npy
│   │   ├── alphabet_y_val.npy
│   │   ├── alphabet_X.npy
│   │   ├── alphabet_y.npy
│   │   ├── words_X_train.npy
│   │   ├── words_y_train.npy
│   │   ├── words_X_val.npy
│   │   ├── words_y_val.npy
│   │   ├── words_X.npy
│   │   └── words_y.npy
│   ├── training_log_alphabet.csv
│   └── training_log_words.csv
│
├── models\
│   ├── alphabet_model.keras        # Trained MLP
│   ├── alphabet_model_best.keras
│   ├── word_model.keras            # Trained LSTM
│   ├── word_model_best.keras
│   ├── labels_alphabet.json        # label↔idx map for 26 letters
│   ├── labels_word.json            # label↔idx map for ~60 word classes
│   ├── scaler_alphabet.json        # sklearn StandardScaler state
│   ├── scaler_words.json
│   ├── alphabet_training_curves.png
│   └── word_training_curves.png
│
├── setup\
│   └── check_env.py                # Diagnostic script + next-step planner
│
├── src\
│   ├── __init__.py
│   ├── hand_tracker.py             # MediaPipe Hands wrapper → 63-d vec/frame
│   ├── alphabet_recognizer.py      # Loads + runs alphabet_model.keras
│   ├── word_recognizer.py          # Loads + runs word_model.keras over seq windows
│   ├── sentence_builder.py         # Debounce, dedupe, token buffering, history
│   ├── speech.py                   # pyttsx3 thin wrapper
│   ├── preprocess_alphabet.py      # A–Z images → landmark .npy + labels JSON
│   ├── preprocess_words.py         # Word videos → seq landmark .npy + labels JSON
│   ├── train_alphabet.py           # Train alphabet MLP (--quick / --epochs)
│   └── train_words.py              # Train word BiLSTM (--quick / --epochs)
│
├── tests\                          # (you create this when adding tests)
│   └── smoke_e2e.py                # End-to-end smoke: preproc → train tiny → predict
│
└── venv\                           # Virtual environment (auto-created)
```

---

## 🧪 Smoke Testing

Before or after a full train, run these to confirm every component loads:

1. **Environment sanity check** — packages, folders, dataset counts, model presence:
   ```bat
   python setup\check_env.py
   ```

2. **Full end-to-end smoke test** (drops small synthetic data, preprocesses, quick-trains both models, runs a single predict):
   ```bat
   python tests\smoke_e2e.py
   ```

3. **Per-module self-tests** — each `src\` module can be run standalone to exercise its code path:
   ```bat
   python src\hand_tracker.py
   python src\alphabet_recognizer.py
   python src\word_recognizer.py
   python src\sentence_builder.py
   python src\speech.py
   ```

4. **Streamlit UI smoke** — after models are present, just launch and confirm the page renders:
   ```bat
   streamlit run app.py --server.headless true
   ```
   Then Ctrl+C once you see `localhost:8501` with no errors.

---

## License / Credits

- **Code:** MIT — see `LICENSE` file if present in the repo root.
- **Datasets:** © their respective owners. Word video dataset from [prasadshet on Kaggle](https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset) under that dataset's license.
- **MediaPipe Hands:** © Google LLC, released under the Apache 2.0 license.
"# INDIAN-SIGN-LANGUAGE-TRANSLATOR-" 
