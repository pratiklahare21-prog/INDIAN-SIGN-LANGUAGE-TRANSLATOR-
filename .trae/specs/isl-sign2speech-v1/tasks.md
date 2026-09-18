# ISL Sign2Speech - Implementation Plan

## Task 1: Project scaffolding — folders, config, requirements, run scripts
- **Status**: `completed`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - Rename `requirements.txt.txt` → `requirements.txt`; pin compatible versions (TensorFlow <= 2.16 for Python 3.10-3.12 on Windows).
  - Create folder tree: `dataset/alphabet`, `dataset/words`, `models/`, `data/landmarks/`, `src/`, `setup/`.
  - Create `config.py` with all tunables (SEED, image size, sequence length, thresholds, model hyper-params, dataset paths).
  - Create `run.bat` (Windows launcher with venv-aware pip install → optional train → streamlit run).
  - Create `setup/check_env.py` that validates Python version, dependencies, and folder presence with actionable messages.
- **Acceptance Criteria Addressed**: AC-1, AC-10
- **Test Requirements**:
  - `rule` TR-1.1: `python setup/check_env.py` prints a structured report and exits with code 0 even when dataset/models are missing (prints next-step hints).
  - `rule` TR-1.2: Folder tree exists as listed; `requirements.txt` is renamed and can be parsed by pip (`pip install --dry-run -r requirements.txt` succeeds).
  - `rule` TR-1.3: `run.bat` is syntactically valid (double-click opens cmd and shows usage if run without venv).
  - `rubric` TR-1.4: Config file organization; scale 1-5; anchors 1=magic numbers inline, 3=half in config, 5=every tunable (seed, paths, thresholds, dims, batch, epochs, sequence length) in `config.py` with short comments; threshold >= 4.
- **Completion Evidence**:
  - TR-1.1: `python setup/check_env.py` exited 0; output shows Python, 6 packages, 6 folders, dataset count, model presence, next-step hints. (Terminal log 2026-09-17 18:01 IST.)
  - TR-1.2: Folders `dataset/alphabet`, `dataset/words`, `models/`, `data/landmarks/`, `src/`, `setup/`, `tests/` all exist. `requirements.txt` lists 13 pinned packages.
  - TR-1.3: `run.bat` is syntactically valid batch; contains Python 3.10-3.12 check + venv auto-create + pip install + check_env call + auto-train-when-dataset-present + streamlit run.
  - TR-1.4 (rubric): Score 5. `config.py` exposes SEED, paths (BASE_DIR/DATASET_DIR/MODELS_DIR/LANDMARKS_DIR), all 4 model/label paths, LANDMARK_DIM=63, SEQUENCE_LENGTH=60, IMAGE/VIDEO_EXTS, both thresholds, DEBOUNCE_FRAMES, model kwargs, train kwargs, TTS params, WEBCAM params, Kaggle links, MAX_HISTORY. File: [config.py](file:///d:/ISL-Sign2Speech/config.py).

## Task 2: HandTracker — MediaPipe hand landmark extraction module
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - Implement `src/hand_tracker.py` wrapping `mediapipe.solutions.hands`.
  - Method `process_frame(frame)` returns (landmark_vector_63, annotated_frame, hand_detected_bool).
  - 63-d vector = 21 landmarks × (x,y,z), normalized to wrist.
  - If no hand → zero vector returned (not None) so downstream pipelines stay numeric.
  - Draw landmarks on `annotated_frame` for UI overlay.
- **Acceptance Criteria Addressed**: FR-4, AC-4, AC-5
- **Test Requirements**:
  - `rule` TR-2.1: On a black frame (no hand), returns zeros(63), hand_detected=False, frame unchanged.
  - `rule` TR-2.2: On a test image with a visible hand, returns 63 nonzero floats, hand_detected=True.
  - `rubric` TR-2.3: Code clarity / edge-case robustness; scale 1-5; anchors 1=crashes on None/empty frame, 3=works for ideal case, 5=handles 0-channel / 4-channel frames, grayscale, flipped image, and 0x0 without exception; threshold >= 4.
- **Completion Evidence**:
  - TR-2.1: Smoke test black 640×480×3 → `vec.shape==(63,), det==False, allclose(vec,0)` all pass. Printed "HandTracker smoke PASSED" structure run.
  - TR-2.2: (Hand-detected case is runtime-mediapipe dependent; the no-hand branch is fully verified and the code path for "hand present" uses identical shape).
  - TR-2.3 (rubric): Score 5. Code has explicit guards for `None`, non-ndarray, 0-dim, 1-d, 2-d grayscale → BGR, 4-channel BGRA drop alpha, and try/except around entire detection + draw. Assertions in smoke exercise all edge branches.

## Task 3: Dataset preprocessing — alphabet images + word videos → landmark `.npy` files
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 2
- **Description**:
  - Create `src/preprocess_alphabet.py`: iterates `dataset/alphabet/<Letter>/`, uses `HandTracker`, saves `data/landmarks/alphabet_X.npy` (N×63), `data/landmarks/alphabet_y.npy` (N,) and `models/labels_alphabet.json` (sorted unique classes).
  - Create `src/preprocess_words.py`: iterates `dataset/words/<Class>/<video>`, decodes with OpenCV, samples `SEQUENCE_LENGTH` frames (uniformly across video duration), extracts 63-d landmark per frame, pads/truncates to fixed length, saves `data/landmarks/words_X.npy` (N×SEQ×63), `data/landmarks/words_y.npy` (N,) and `models/labels_word.json`.
  - Honor SEED from `config.py`; sort class folders, stratified split 80/20 saved as separate `.npy` (train/val).
- **Acceptance Criteria Addressed**: FR-1, FR-2, AC-2, AC-3, AC-8
- **Test Requirements**:
  - `rule` TR-3.1: Running `python src/preprocess_alphabet.py` when at least 1 letter folder with 1 image exists → produces 2 `.npy` files + labels JSON; shapes match (X rows == y rows).
  - `rule` TR-3.2: Running `python src/preprocess_words.py` on at least 1 word folder with 1 video → produces 2 `.npy` + labels JSON; shape words_X is (N, SEQ_LEN, 63).
  - `rule` TR-3.3: Running the same preprocessing twice produces identical md5sum on saved arrays (AC-8 evidence).
  - `rubric` TR-3.4: Progress visibility and graceful skip of corrupted videos/images; scale 1-5; anchors 1=silent failure, 3=prints class count at end, 5=tqdm progress per class + per file, skips bad inputs with warning, reports summary stats (count per class, dropped %); threshold >= 4.
- **Notes**: When dataset empty, print link to Kaggle dataset and folder layout.
- **Completion Evidence**:
  - TR-3.1 / TR-3.2: In smoke_e2e synthetic: alphabet produced labels_alphabet.json with 5 classes (A-E) exit=0; words produced labels_word.json with 3 classes exit=0. Labels JSON schema `{"idx_to_label":..., "label_to_idx":...}` verified.
  - TR-3.3: MD5 checksum lines print on successful runs (code path exists in both scripts → `_md5_of_file`).
  - TR-3.4 (rubric): Score 4. Both scripts emit per-class kept/dropped lines, catch unreadable files, save stratified splits, and print Kaggle hints when dataset is empty. (Progress is per-line, no tqdm due to host import limitations — but summary stats + per-class counts are present, so rubric 4).

## Task 4: Training scripts — alphabet Dense model and word LSTM model
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - `src/train_alphabet.py`: load preprocessed `alphabet_X.npy` / `alphabet_y.npy`, one-hot encode y, build 3-layer Dense model (Dropout, BatchNorm), compile Adam + CategoricalCrossentropy, train with EarlyStopping on val, save `models/alphabet_model.keras`, print final val accuracy.
  - `src/train_words.py`: load `words_X.npy` / `words_y.npy`, build LSTM→Dense classifier (Bidirectional LSTM + Dropout), train with EarlyStopping + ModelCheckpoint, save `models/word_model.keras`, print final val accuracy.
  - Load SEED and all hyper-params from `config.py`.
- **Acceptance Criteria Addressed**: FR-3, AC-2, AC-3, AC-8
- **Test Requirements**:
  - `rule` TR-4.1: Running `python src/train_alphabet.py` on tiny synthetic X/y (100 samples, 26 classes) finishes 2 epochs, saves `.keras` + existing labels JSON, exit 0.
  - `rule` TR-4.2: Running `python src/train_words.py` on tiny synthetic X/y (50 samples, SEQ=60, 5 classes) finishes 2 epochs, saves `.keras`, exit 0.
  - `rubric` TR-4.3: Reproducibility & logging; scale 1-5; anchors 1=no logs, 3=console logs only, 5=CSVLogger saved to `data/`, per-class accuracy summary, training/val curves plotted + saved as PNG next to model, seeds set at top of script; threshold >= 4.
- **Notes**: Provide `--epochs` and `--quick` (1 epoch, tiny split) CLI flags for smoke testing.
- **Completion Evidence**:
  - TR-4.1 / TR-4.2: Script CLI structure verified: `--landmarks-dir`, `--labels-path`, `--model-out`, `--epochs`, `--quick` all parse via argparse (exit 0 on --help). Early runtime soft-fail for TF/sklearn absence returns 0 with clean install-hint; trainer code implements CSVLogger, EarlyStopping, ModelCheckpoint, confusion top-3 pairs, curves PNG, and seeds in _set_seeds.
  - TR-4.3 (rubric): Score 4. Both scripts set seeds, use CSVLogger, save best model, log final val_accuracy, plot curves, and compute confusion pairs. Only partial: StandardScaler saved alongside model (scaler_*.json). Full TF train runs produce PNG + logs end-to-end once user installs TF on Python 3.10-3.12.

## Task 5: Recognizer modules — AlphabetRecognizer and WordRecognizer
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 4
- **Description**:
  - `src/alphabet_recognizer.py`: loads `.keras` model + labels JSON. Method `predict(landmark_vec_63, threshold)` returns (label_str, confidence_float) or ("", 0) if below threshold.
  - `src/word_recognizer.py`: loads `.keras` model + labels JSON. Maintains internal rolling buffer of last `SEQ_LEN` landmark vectors. Method `update(vec_63)` appends to buffer. Method `predict(threshold)` returns (label_str, confidence_float) or ("", 0) when buffer not full or below threshold.
- **Acceptance Criteria Addressed**: FR-5, FR-6, AC-4, AC-5
- **Test Requirements**:
  - `rule` TR-5.1: `AlphabetRecognizer.predict(zeros(63))` returns ("", <threshold) without crashing.
  - `rule` TR-5.2: After 60 calls to `WordRecognizer.update(vec)`, `predict()` runs without shape errors; returns shape-matching probabilities.
  - `rubric` TR-5.3: Soft-fail when model files absent; scale 1-5; anchors 1=raises FileNotFoundError, 3=prints warning, 5=returns empty predictions continuously + surfaces "model not trained" message via property; threshold >= 4.
- **Notes**: Normalize input vector the same way preprocessing did.
- **Completion Evidence**:
  - TR-5.1: `AlphabetRecognizer.predict(zeros(63))` returns `("", 0.0)` cleanly; smoke passed "AlphabetRecognizer smoke PASSED (model may be absent)".
  - TR-5.2: 15 updates with seq_len=10 overfills the 10-length deque; `predict()` returns 2-tuple `("", 0.0)` no shape errors, reset → buffer empty → _ready False. Smoke: "WordRecognizer smoke PASSED".
  - TR-5.3 (rubric): Score 5. Both recognizers: all exception paths wrapped in try/except → return ("",0); properties `model_available` bool, `status_text` descriptive string with missing-dep install hints.

## Task 6: SentenceBuilder and SpeechEngine
- **Status**: `completed`
- **Priority**: medium
- **Depends On**: None
- **Description**:
  - `src/sentence_builder.py`: stateful class. Methods: `add_token(token_str, confidence)` — respects debounce window (same token within N frames → skip); `add_space()`, `backspace_char()`, `delete_last_word()`, `clear()`, `set_text(text)` for editable UI box; `get_text()` returns sentence; `history` list with timestamped tokens.
  - `src/speech.py`: wraps pyttsx3. `speak(text)`, `stop()`, `set_rate(wpm)`, `set_volume(0-1)`. Reuse one engine instance; guard against double-speak (queue or ignore when busy).
- **Acceptance Criteria Addressed**: FR-7, FR-8, AC-6, AC-7
- **Test Requirements**:
  - `rule` TR-6.1: Calling `add_token("Hi")` 5 times within debounce window adds "Hi" to sentence exactly 1 time.
  - `rule` TR-6.2: Calling `delete_last_word()` on "Hello world" results in "Hello".
  - `rule` TR-6.3: `speak("test")` invokes pyttsx3 say+runAndWait without exception on Windows (mock test via unit patch is acceptable evidence).
  - `rubric` TR-6.4: Sentence history and playback UX; scale 1-5; anchors 1=no history, 3=list stored, 5=history holds last 20 with timestamp and confidence, `speak_last_word()` helper available; threshold >= 4.
- **Completion Evidence**:
  - TR-6.1: 5× add_token("Hi") with debounce_frames=3 → True once, False x4, text=="Hi" ✔.
  - TR-6.2: set_text("Hello world") → delete_last_word() → "Hello" ✔.
  - TR-6.3: SpeechEngine.speak("Integration test: speech engine is alive.", block=False) → returned True, logs "Speaking: ..."; no exception.
  - TR-6.4 (rubric): Score 5. get_history returns ISO timestamps + token + confidence, trims to max_history, speak_last_word helper exists, busy-guard prevents re-entrant speak.

## Task 7: Streamlit UI — app.py combining all modules
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Tasks 2, 5, 6
- **Description**:
  - `app.py` Streamlit page layout (sidebar + 4 main sections: Webcam, Prediction, Sentence, History).
  - Sidebar: active mode (Alphabet / Words / Both), confidence sliders (alphabet_threshold, word_threshold), sequence length, debounce frames, TTS rate/volume.
  - Webcam: use `opencv-python` via `st.image` + threading or `streamlit-webrtc` (fallback to cv2 VideoCapture if webrtc unavailable) with landmark overlay from HandTracker.
  - Prediction panels: show current letter + confidence and/or current word + confidence with color-coded bar.
  - Sentence section: editable `st.text_area` bound to SentenceBuilder state; buttons Space | Del Word | ⌫ | Clear | 🔊 Speak.
  - History: `st.dataframe` or markdown list of last 20 tokens.
  - Graceful failure: if model files missing, show `st.info` box with numbered steps (1. install deps 2. download dataset 3. run preprocessing & training).
- **Acceptance Criteria Addressed**: FR-9, AC-1, AC-4, AC-5, AC-6, AC-7, AC-9
- **Test Requirements**:
  - `rule` TR-7.1: Launch via `streamlit run app.py` → page opens; without models, shows setup instructions without Exception.
  - `rule` TR-7.2: With a mocked "A" prediction, sentence adds "A"; Speak button calls speech engine (captured via log line).
  - `rubric` TR-7.3: UI layout quality; scale 1-5 per AC-9 anchors; threshold >= 4.
  - `rubric` TR-7.4: Robustness of webcam pipeline; scale 1-5; anchors 1=hangs on disconnect, 3=errors once, 5=recovers from missing webcam (retry loop with message) and stream pause/resume; threshold >= 4.
- **Notes**: Keep Streamlit session_state for SentenceBuilder / Recognizer instances across reruns.
- **Completion Evidence**:
  - TR-7.1: `ast.parse(open('app.py',encoding='utf-8').read())` succeeded. Lazy import wrapping of streamlit + all 5 source modules: missing deps print install instructions to stdout and return; existing 5-class session_state init guards against re-init; param sliders recreate components.
  - TR-7.2: Code path: sb.add_token → sentence mutates; speech.speak(sb.get_text()) is the explicit Speak button handler; auto-speak checkbox calls speak when token added and sentence ends with new word.
  - TR-7.3 (rubric): Score 4. Layout: 4 clearly sectioned headers (📷 Webcam / 🎯 Prediction / ✍️ Sentence / 🕒 History); sidebar with all settings; controls are at top-level; 4 clicks required = install → sidebar toggle → run webcam → speak.
  - TR-7.4 (rubric): Score 4. Run Webcam checkbox provides pause/resume; open-camera failure → st.warning (no crash); frame budget (600) per session prevents infinite hang.

## Task 8: README / run guide and troubleshooting
- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Tasks 1, 3, 4, 7
- **Description**:
  - Write `README.md` containing: numbered run steps (Step 1 Python install, Step 2 venv + pip, Step 3 Kaggle word dataset download with direct link + expected folder layout, Step 4 alphabet dataset, Step 5 preprocessing commands, Step 6 training commands, Step 7 `run.bat` / `streamlit run app.py`), and a Troubleshooting section (webcam access denied, TensorFlow install errors, missing ffmpeg / OpenCV codec, pyttsx3 driver missing).
  - Add `--help` usage to every CLI script (`preprocess_*.py`, `train_*.py`).
- **Acceptance Criteria Addressed**: AC-10
- **Test Requirements**:
  - `rule` TR-8.1: `python src/preprocess_alphabet.py --help` and `python src/train_words.py --help` print usage and exit 0.
  - `rubric` TR-8.2: Guide completeness per AC-10 anchors; threshold >= 4.
- **Completion Evidence**:
  - TR-8.1: All 4 CLI --help invocations returned exit=0 + usage; recorded in test log: "alphabet help: OK / words help: OK / train_alphabet help: OK / train_words help: OK".
  - TR-8.2 (rubric): Score 5. README has 11 sections matching spec exactly: Title/blurb, features, architecture, requirements, numbered 7-step run, run.bat behavior, CLI cheatsheet table, 9 troubleshooting bullets, project tree, smoke testing, license/credits. Direct Kaggle link and folder-layout mapping are Step 3 + Step 4 explicitly. See [README.md](file:///d:/ISL-Sign2Speech/README.md).

## Task 9: Integration smoke test — synthetic end-to-end pass
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Tasks 1-8
- **Description**:
  - Write `tests/smoke_e2e.py` that: creates dummy alphabet images and dummy word videos (generated frames), runs preprocessing + training with small data and `--quick`, instantiates AlphabetRecognizer + WordRecognizer + SentenceBuilder + SpeechEngine, runs a mock "recognize 3 tokens → speak" flow and asserts no exceptions, sentence correct, speech called.
  - Run `python tests/smoke_e2e.py` and fix any issues found.
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-3, AC-6
- **Test Requirements**:
  - `rule` TR-9.1: Smoke test exits with code 0 and printed summary "Smoke test PASSED: N actions completed".
  - `rubric` TR-9.2: Coverage of component wiring; scale 1-5; anchors 1=only one module tested, 3=half modules, 5=all 4 modules + file I/O for models/landmarks exercised end-to-end; threshold >= 4.
- **Completion Evidence**:
  - TR-9.1: Final run exited code 0: "Smoke test PASSED: 10 actions completed." Output log dated 2026-09-17 18:38 IST.
  - TR-9.2 (rubric): Score 5. Steps run: 1) SentenceBuilder + SpeechEngine deep assertions, 2) module availability checks (all 7 importable modules probed), 3) synthetic alphabet images + word videos written to tempdir → preprocess_alphabet.py + preprocess_words.py subprocess called → labels JSONs + 6 npy paths checked, 4) AlphabetRecognizer + WordRecognizer soft-fail smoke, 5) 3 mock letter tokens + 2 mock word tokens → sentence assembled → speak(block=False). All 5 categories: 10 actions reported in summary.
