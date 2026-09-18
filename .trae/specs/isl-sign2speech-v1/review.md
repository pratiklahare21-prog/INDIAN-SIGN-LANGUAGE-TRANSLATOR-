# ISL Sign2Speech - Independent Review

- [x] CP-R1: End-to-end run.bat launches or degrades gracefully
  - **Type**: `rule`
  - **Covers**: AC-1, FR-10, FR-11, Task 1
  - **Evidence**: run.bat has Python PATH check + 3.10-3.12 version warning + venv auto-create + pip install -r requirements + check_env + auto-train alphabet/words + streamlit run. setup/check_env.py exits 0 with numbered next-step hints (1.install deps, 2.download Kaggle link, 3.alphabet folder layout). app.py AST parse: PASSED (ast.parse succeeded).
  - **Check**: Run `run.bat` (or simulate its logic by running check_env.py then validating that app.py parses and imports). Expect no Python exception traceback; next-step hints printed when deps/dataset/models missing.

- [x] CP-R2: All module-level smoke scripts exit 0 with "PASSED" summary line
  - **Type**: `rule`
  - **Covers**: Tasks 2, 5, 6, FR-4, FR-5, FR-6, FR-7, FR-8, AC-4, AC-5, AC-6, AC-7
  - **Evidence**: All 5 smokes exit=0 with PASSED lines. hand_tracker.py: soft-skip "HandTracker smoke (structure only, no mediapipe): PASSED"; sentence_builder.py: "SentenceBuilder smoke PASSED"; speech.py: "SpeechEngine smoke PASSED"; alphabet_recognizer.py: "AlphabetRecognizer smoke PASSED (model may be absent)"; word_recognizer.py: "WordRecognizer smoke PASSED".
  - **Check**: Execute sequentially: `python src/hand_tracker.py`; `python src/sentence_builder.py`; `python src/speech.py`; `python src/alphabet_recognizer.py`; `python src/word_recognizer.py`. Each must print a PASSED line and exit 0 (mediapipe/tensorflow absent → soft-skip PASSED variant also accepted).

- [x] CP-R3: CLI --help works for all 4 scripts
  - **Type**: `rule`
  - **Covers**: Task 8, FR-1, FR-2, FR-3
  - **Evidence**: 4/4 `--help` exit=0 with argparse usage printed: preprocess_alphabet (data-dir/out-dir/labels-path); preprocess_words (data-dir/out-dir/labels-path/seq-len); train_alphabet (landmarks-dir/labels-path/model-out/epochs/quick); train_words (landmarks-dir/labels-path/model-out/epochs/quick).
  - **Check**: `python src/preprocess_alphabet.py --help`; `python src/preprocess_words.py --help`; `python src/train_alphabet.py --help`; `python src/train_words.py --help`. Each prints usage to stdout and exits 0.

- [x] CP-R4: Integration smoke_e2e exits 0 with "Smoke test PASSED: N actions completed"
  - **Type**: `rule`
  - **Covers**: Task 9, AC-1, AC-2, AC-3, AC-6, AC-7
  - **Evidence**: PYTHONIOENCODING=utf-8; python tests/smoke_e2e.py → exit 0, "Smoke test PASSED: 10 actions completed." (N=10 ≥ 6). Step 5 E2E mock confirmed sentence='H E Y Hello world' and speak(block=False) returned True.
  - **Check**: `$env:PYTHONIOENCODING='utf-8'; python tests/smoke_e2e.py`. Exit code 0. Summary line "Smoke test PASSED: N actions completed" with N >= 6 present in stdout.

- [x] CP-R5: Pipeline consistency — label JSON schema shared between preprocessors and recognizers
  - **Type**: `rule`
  - **Covers**: AC-2, AC-3, FR-1, FR-3, FR-5, FR-6
  - **Evidence**: All 4 files write/read `{"idx_to_label": {str(i): name}, "label_to_idx": {name: i}}` exactly. Writers: preprocess_alphabet.py lines 93-96; preprocess_words.py lines 141-144. Readers: alphabet_recognizer.py lines 68-72; word_recognizer.py lines 72-76. Prediction lookups consistently use `idx_to_label.get(str(max_idx), "")` (alphabet_recognizer.py:148, word_recognizer.py:176).
  - **Check**: Read preprocess_alphabet.py, preprocess_words.py, alphabet_recognizer.py, word_recognizer.py. All must write/read `{"idx_to_label": {str(i): name}, "label_to_idx": {name: i}}`. Verify recognizer code walks this schema exactly; no alternative keys used without fallback.

- [x] CP-U1: Workflow fidelity / implementation completeness vs tasks
  - **Type**: `rubric`
  - **Covers**: all 9 Tasks, all ACs/TRs
  - **Scale**: 1-5
  - **Anchors**: 1 = 4 or more pending items with no rationale; 3 = every task has edits but some TRs not independently re-run; 5 = every task Status=completed with explicit rule+rubric evidence in tasks.md AND reviewer can reproduce the rule checks locally.
  - **Pass Threshold**: >= 4
  - **Evidence**: Score 5/5. All 9 tasks Status=completed with explicit rule TR evidence (exit codes, md5 checksums, stdout quotes, timestamps) + rubric numeric scores (ranged 4-5). Reviewer independently reproduced 7 rule TRs (5 module smokes, 4 --help, smoke_e2e, app.py AST, check_env) — all green.

- [x] CP-U2: UI + Docs usability combo
  - **Type**: `rubric`
  - **Covers**: AC-9, AC-10, FR-9, FR-11
  - **Scale**: 1-5
  - **Anchors**: 1 = app.py UI has no section headings, README skips dataset download step; 3 = UI sections exist but layout order is confusing; README has steps in wrong order; 5 = app.py uses 📷/🎯/✍️/🕒 section headings + sidebar ⚙️ Settings, numbered 1..7 steps in README with exact Kaggle link + folder layout, run.bat auto-trains when dataset present, Troubleshooting >= 8 bullets.
  - **Pass Threshold**: >= 4
  - **Evidence**: Score 5/5. app.py: sidebar ⚙️ Settings + 📷 Webcam + 🎯 Live Prediction + ✍️ Sentence + 🕒 History. README: numbered Steps 1-7 (Clone→venv→pip→Kaggle link→alphabet layout→preprocess+train→run.bat); exact Kaggle URL (README line 102); folder layout mapping for words and alphabet; run.bat auto-train explainer; 9 troubleshooting bullets (TF install/webcam/MediaPipe/pyttsx3/word acc/alphabet acc/CORS/check_env/plateau val_acc).

## Review History

### Review R1
- **Result**: **pass**
- **Evidence**: All 7 checkpoints passed independently. Rule CPs (R1-R5): 5/5 verified with subprocess exit codes + file read line quotes. Rubric CPs (U1, U2): both scored 5/5 (≥4 threshold). Findings: 0. Recommended Issues: 0.
- **Blocked By**: N/A
- **Resume When**: N/A
