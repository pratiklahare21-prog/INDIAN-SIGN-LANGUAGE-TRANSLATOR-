# ISL Sign2Speech - Product Requirements Document

## Overview
- **Summary**: A ready-to-run Indian Sign Language (ISL) recognition system that detects hand signs via live webcam, recognizes both A–Z alphabet gestures and 60 word-level signs (from Kaggle dataset), builds sentences, and speaks them aloud using text-to-speech — all wrapped in a simple Streamlit UI.
- **Purpose**: Enable hearing-impaired users to translate ISL hand signs into spoken English sentences using an offline, privacy-focused, local Python pipeline.
- **Target Users**: Developers, students, and end-users who need a local ISL → speech converter; users comfortable with "download dataset → run scripts → launch UI" workflow.

## Goals
- Recognize 26 ISL alphabet signs (A–Z) from static hand poses.
- Recognize 60 ISL word signs from short video clips (using Kaggle word dataset).
- Merge both recognizers into a single live webcam session.
- Build sentences by appending recognized tokens; expose Clear / Space / Delete controls.
- Convert sentence text to offline speech output (pyttsx3 / SAPI5 on Windows).
- Provide a single Streamlit UI page with: webcam preview, live predicted label + confidence, token history, sentence box, and Speak button.
- Provide end-to-end scripts: dataset structure check → preprocessing (MediaPipe landmarks) → training both models → launch UI.
- Provide `run.bat` one-click launcher and a numbered step-by-step run guide.

## Non-Goals
- Continuous/fluent sentence-level sign recognition (frame-by-frame sentence sequences).
- Web/cloud deployment; strictly local (no API keys, no Internet at runtime).
- Sign production (speech → avatar hands); only sign → speech direction.
- Custom handcrafted ISL dataset; existing Kaggle datasets drive training.
- Real-time fps optimization beyond usable real-time (>= 10 fps on typical laptop CPU).

## Background & Context
- Repository `d:\ISL-Sign2Speech` is freshly scaffolded with a `venv\` and `requirements.txt.txt` (contains tensorflow / opencv-python / mediapipe / numpy / pandas / scikit-learn / streamlit / streamlit-webrtc / av / pyttsx3 / Pillow).
- ChatGPT design memo proposes MediaPipe hand landmarks as the shared feature representation:
  - A–Z model: Dense/CNN over 21×3 landmarks per frame.
  - Word model: LSTM over sequences (60 frames × 21×3 landmarks) per video.
- Kaggle word dataset reference: `Indian Sign Language Video Dataset` by prasadshet (60 classes, videos, ~3.48 GB, MIT license).
- A–Z ISL static images can be bootstrapped from an existing public ISL alphabet image dataset or generated synthetically during preprocessing if the user drops images into `dataset/alphabet/`.

## Functional Requirements
- **FR-1**: Dataset loader auto-discovers classes from folder names under `dataset/words/` (subfolder = class name) and `dataset/alphabet/` (subfolder = class name, A–Z).
- **FR-2**: Preprocessing script extracts MediaPipe hand landmarks for every frame of every video (words) and every image (alphabet), saving numpy arrays to `data/landmarks/`.
- **FR-3**: Training script trains and saves `models/alphabet_model.keras` + `models/labels_alphabet.json` and `models/word_model.keras` + `models/labels_word.json`.
- **FR-4**: A `HandTracker` module wraps MediaPipe Hands and returns normalized 63-dim landmark vector (or zeros if no hand).
- **FR-5**: `AlphabetRecognizer` loads alphabet model + labels and predicts class + confidence from a single landmark vector (minimum confidence threshold configurable).
- **FR-6**: `WordRecognizer` loads word model + labels and predicts class + confidence from a sliding window sequence (default 60 frames) of landmark vectors.
- **FR-7**: `SentenceBuilder` buffers recognized tokens, debounces duplicates (same token within N frames ignored), supports Space / Backspace / Clear, and exposes a current sentence string.
- **FR-8**: `SpeechEngine` speaks the current sentence (or custom text) offline using pyttsx3; speak/stop/repeat controls.
- **FR-9**: Streamlit UI (`app.py`) renders:
  - Sidebar: Model mode toggle (Alphabet / Words / Both), confidence threshold sliders, sliding-window length, debounce threshold.
  - Main area: Live webcam canvas with hand-landmark overlay.
  - Live prediction panel: Current token + confidence % (Alphabet and Words separately).
  - Sentence panel: Editable sentence text area + Space / Delete word / Backspace char / Clear buttons + Speak button.
  - History panel: Scrolling list of last 20 recognized tokens with timestamp and confidence.
- **FR-10**: A `setup/` batch script (and Python fallback) validates prerequisites: Python version, required folders exist, and suggests next steps when models or dataset are missing.
- **FR-11**: `run.bat` entry point: 1) install deps if needed, 2) check dataset + models, 3) launch `app.py`. Graceful error messages when dataset/models absent.

## Non-Functional Requirements
- **NFR-1**: Offline-first — no runtime network call required after dependency install and dataset download.
- **NFR-2**: Windows-first — must run on Windows 10/11 with default Python 3.10–3.12; include Linux/macOS notes where feasible.
- **NFR-3**: Reproducible training — fixed random seeds, deterministic preprocessing order, and a `config.py` holding all tunable parameters.
- **NFR-4**: Fail-soft UI — missing models degrade gracefully with setup instructions instead of crashing.
- **NFR-5**: Simple code layout — flat module structure with clear file responsibilities (no deep package nesting).

## Constraints
- **Technical**:
  - MediaPipe Hands for hand keypoints (no custom pose).
  - TensorFlow/Keras (.keras format) for both models (no PyTorch).
  - Streamlit + streamlit-webrtc/OpenCV for webcam UI.
  - pyttsx3 (SAPI5 on Windows) for TTS (no cloud TTS).
- **Business**:
  - No API keys or paid services.
  - Dataset downloaded separately by user; we don't re-host.
- **Dependencies**:
  - Python 3.10–3.12 (avoid 3.13+ because TensorFlow wheels lag on Windows).
  - Dependencies enumerated in `requirements.txt`.

## Assumptions
- User has a working webcam and audio output.
- User can obtain Kaggle word dataset and place it in `dataset/words/<ClassName>/` folder structure.
- User can obtain an ISL A–Z image dataset and place it in `dataset/alphabet/<Letter>/` (or generate from video dataset frames for A–Z classes if present).
- Training runs on user machine (CPU acceptable for A–Z; GPU recommended for word LSTM but not required).

## Open Questions
- [ ] Confirm the exact Kaggle dataset the user will use (current plan: prasadshet `Indian Sign Language Video Dataset`). If the user picks ISL-CSLTR instead, folder mapping and preprocessing logic will be different.
- [ ] Confirm whether an A–Z ISL dataset is available, or if the user wants us to ship a synthetic/minimal fallback that trains from a few sample frames per letter (reduced accuracy for demo).

## Acceptance Criteria

### AC-1: End-to-end launch without crash
- **Type**: `rule`
- **Given**: Dependencies installed (`pip install -r requirements.txt`), Python 3.10–3.12 on Windows, valid webcam.
- **When**: User runs `run.bat`.
- **Then**: Streamlit opens in the browser; if models/dataset missing, UI shows setup steps without traceback; if models present, webcam feed starts.
- **Pass Condition**: `run.bat` exits 0, Streamlit process stays alive > 10 seconds, logs contain no Python Exception-level stack trace.
- **Evidence**: Screenshot of running UI + console output of `run.bat`.

### AC-2: Alphabet model trains and saves artifacts
- **Type**: `rule`
- **Given**: `dataset/alphabet/A/..` through `dataset/alphabet/Z/..` populated with >= 10 images per letter.
- **When**: Training script invoked: `python src/train_alphabet.py`.
- **Then**: `models/alphabet_model.keras` and `models/labels_alphabet.json` exist; training log reports val accuracy on last epoch.
- **Pass Condition**: Both files exist, `labels_alphabet.json` contains exactly 26 keys ("A".."Z"), training script exit code 0.
- **Evidence**: `ls models/` listing + tail of training log.

### AC-3: Word model trains and saves artifacts
- **Type**: `rule`
- **Given**: `dataset/words/<Class>/` populated with >= 5 videos per class.
- **When**: `python src/train_words.py` runs.
- **Then**: `models/word_model.keras` and `models/labels_word.json` exist; preprocessing step creates `data/landmarks/words_X.npy` and `data/landmarks/words_y.npy`.
- **Pass Condition**: Both `.keras` + `.json` artifacts present, number of classes in labels file equals number of word subfolders in dataset.
- **Evidence**: File listing + script exit code 0.

### AC-4: Live A–Z recognition works
- **Type**: `rule`
- **Given**: Alphabet model loaded, UI running, user shows hand for letter "A".
- **When**: Hand is inside webcam frame for >= 1 second with confidence above threshold.
- **Then**: Live prediction panel shows "A" with confidence >= threshold; SentenceBuilder adds "A" to sentence once (debounced).
- **Pass Condition**: UI panel displays predicted letter within 1.5 seconds of stable hand pose; sentence contains "A" and history records it.
- **Evidence**: Screenshot of UI showing prediction + sentence + history.

### AC-5: Live word recognition works with sliding window
- **Type**: `rule`
- **Given**: Word model loaded, UI set to Words mode, user performs a word sign (e.g. "Hello") for >= 2 seconds.
- **When**: 60-frame buffer fills and sign motion is captured.
- **Then**: Word prediction fires with label and confidence, token is appended to sentence with duplicate debouncing.
- **Pass Condition**: Correct (or top-K plausible) word label shown and spoken when Speak pressed.
- **Evidence**: Screen recording or screenshot showing word token + confidence.

### AC-6: Speak button produces audio
- **Type**: `rule`
- **Given**: Sentence contains text and pyttsx3 engine initialized.
- **When**: User clicks Speak button.
- **Then**: Audio plays the sentence text; console logs "Speaking: <text>".
- **Pass Condition**: Audio audible on machine; Speak completes without exception.
- **Evidence**: Console output line containing "Speaking:" and user confirmation / audio log.

### AC-7: Clear / Delete / Space / Backspace controls mutate sentence
- **Type**: `rule`
- **Given**: Sentence "Hello world" is present.
- **When**: Click Delete → sentence becomes "Hello"; click Space → "Hello "; type "X" → "Hello X"; click Clear → "".
- **Then**: Each action updates the editable sentence box immediately.
- **Pass Condition**: All four mutations produce expected sentence state; visible in UI.
- **Evidence**: Screenshot sequence or UI state capture.

### AC-8: Dataset preprocessing is deterministic and reproducible
- **Type**: `rubric`
- **Dimension**: Reproducibility of preprocessing pipeline
- **Scale**: 1–5
- **Anchors**:
  - 1 = Non-deterministic ordering / random splits not seeded → different output per run.
  - 3 = Seeds set but preprocessing order depends on OS file-system traversal.
  - 5 = Seeds explicitly set (numpy, TF, random), class order sorted, dataset shuffled with fixed seed, identical `.npy` hashes on repeated runs.
- **Pass Threshold**: >= 4
- **Evidence**: `config.py` shows SEED constants; comparing two preprocessing runs produces same array checksum (md5 of saved `.npy`).

### AC-9: UI usability — controls discoverable in under 30 seconds
- **Type**: `rubric`
- **Dimension**: Streamlit UI clarity and self-service
- **Scale**: 1–5
- **Anchors**:
  - 1 = Buttons hidden in expanders, layout cluttered, unclear which mode is active.
  - 3 = Controls present but ungrouped; mode state visible only in sidebar.
  - 5 = Clear section headings (Webcam / Prediction / Sentence / History); active mode highlighted; first-time user can start webcam and press Speak within 30 seconds.
- **Pass Threshold**: >= 4
- **Evidence**: UI screenshots + heuristic walkthrough by implementer (list of clicks required → count ≤ 4).

### AC-10: Documentation — run steps are complete and ordered
- **Type**: `rubric`
- **Dimension**: Completeness of `run.bat` / step-by-step guide
- **Scale**: 1–5
- **Anchors**:
  - 1 = Missing dependency install step; no mention of dataset download link.
  - 3 = Steps exist but skip venv creation / model training trigger; dataset link present but folder-layout expectation vague.
  - 5 = Numbered steps cover venv creation, pip install, exact Kaggle dataset link + folder-layout mapping, training commands, `run.bat` usage, and troubleshooting bullets for common failures (no webcam, no model, TensorFlow install error on Windows).
- **Pass Threshold**: >= 4
- **Evidence**: README content / `run.bat` comments / UI help text.
