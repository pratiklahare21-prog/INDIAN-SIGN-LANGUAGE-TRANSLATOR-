import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import json
import numpy as np
from pathlib import Path

def section(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)

section("1. DATASET + EXTRACTED LANDMARKS VALIDITY")
print(f"Dataset words dir: {config.DATASET_WORDS_DIR}  exists={config.DATASET_WORDS_DIR.exists()}")
print(f"Landmarks dir: {config.LANDMARKS_DIR}  exists={config.LANDMARKS_DIR.exists()}")

Xw = np.load(config.LANDMARKS_DIR / "words_X.npy")
yw = np.load(config.LANDMARKS_DIR / "words_y.npy")
Xwt = np.load(config.LANDMARKS_DIR / "words_X_train.npy")
ywt = np.load(config.LANDMARKS_DIR / "words_y_train.npy")
Xwv = np.load(config.LANDMARKS_DIR / "words_X_val.npy")
ywv = np.load(config.LANDMARKS_DIR / "words_y_val.npy")
Xs = np.load(config.LANDMARKS_DIR / "sentences_X.npy")
ys = np.load(config.LANDMARKS_DIR / "sentences_y.npy")
Xst = np.load(config.LANDMARKS_DIR / "sentences_X_train.npy")
yst = np.load(config.LANDMARKS_DIR / "sentences_y_train.npy")
Xsv = np.load(config.LANDMARKS_DIR / "sentences_X_val.npy")
ysv = np.load(config.LANDMARKS_DIR / "sentences_y_val.npy")

print(f"\nWORDS LANDMARKS:")
print(f"  X:       {Xw.shape}  dtype={Xw.dtype}")
print(f"  y:       {yw.shape}  unique={len(np.unique(yw))}  min={yw.min()} max={yw.max()}")
print(f"  X_train: {Xwt.shape}  y_train: {ywt.shape}  uniq={len(np.unique(ywt))}")
print(f"  X_val:   {Xwv.shape}  y_val:   {ywv.shape}  uniq={len(np.unique(ywv))}")

print(f"\nSENTENCES LANDMARKS:")
print(f"  X:       {Xs.shape}  dtype={Xs.dtype}")
print(f"  y:       {ys.shape}  unique={len(np.unique(ys))}  min={ys.min()} max={ys.max()}")
print(f"  X_train: {Xst.shape}  y_train: {yst.shape}  uniq={len(np.unique(yst))}")
print(f"  X_val:   {Xsv.shape}  y_val:   {ysv.shape}  uniq={len(np.unique(ysv))}")

# Check for NaN/Inf/all-zeros
for name, arr in [("words_X", Xw), ("words_X_train", Xwt), ("words_X_val", Xwv),
                  ("sentences_X", Xs), ("sentences_X_train", Xst), ("sentences_X_val", Xsv)]:
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    n_allzero_sample = sum(1 for s in arr if np.allclose(s, 0))
    mean_nonzero = np.mean(np.abs(arr))
    print(f"\n  {name:22s} nan={n_nan:6d}  inf={n_inf:6d}  all-zero-samples={n_allzero_sample:4d}  mean(|x|)={mean_nonzero:.5f}")

# Check train/test split consistency: are all val classes present in train?
missing_w = set(ywv.tolist()) - set(ywt.tolist())
missing_s = set(ysv.tolist()) - set(yst.tolist())
print(f"\n  Word classes in val but NOT in train: {len(missing_w)}  -> {sorted(list(missing_w))[:10]}")
print(f"  Sentence classes in val but NOT in train: {len(missing_s)}  -> {sorted(list(missing_s))[:10]}")

# Per-class sample counts
w_counts = np.bincount(yw.astype(int), minlength=max(yw)+1)
s_counts = np.bincount(ys.astype(int), minlength=max(ys)+1)
print(f"\n  Word samples per class: min={w_counts.min()}  max={w_counts.max()}  mean={w_counts.mean():.1f}  classes_with_<2_samples={(w_counts<2).sum()}")
print(f"  Sentence samples per class: min={s_counts.min()}  max={s_counts.max()}  mean={s_counts.mean():.1f}  classes_with_<2_samples={(s_counts<2).sum()}")

section("2. LABELS CONSISTENCY CHECK")
lw = json.load(open(config.WORD_LABELS_PATH, encoding="utf-8"))
ls = json.load(open(config.SENTENCE_LABELS_PATH, encoding="utf-8"))
info = json.load(open(config.MODELS_DIR / "dataset_info.json", encoding="utf-8"))

print(f"labels_word.json:  idx_to_label entries={len(lw['idx_to_label'])}  label_to_idx entries={len(lw['label_to_idx'])}")
print(f"labels_sentences.json: idx_to_label entries={len(ls['idx_to_label'])}  label_to_idx entries={len(ls['label_to_idx'])}")
print(f"dataset_info.json: expected words={info['word_classes_count']}  sentences={info['sentence_classes_count']}")
print(f"word y unique: {len(np.unique(yw))}  expected from labels_word: {len(lw['idx_to_label'])}")
print(f"sentence y unique: {len(np.unique(ys))}  expected from labels_sentences: {len(ls['idx_to_label'])}")

# Cross-check: do label names match dataset_info class names?
label_words = sorted([v.upper() for v in lw['idx_to_label'].values()])
info_words = sorted([v.upper() for v in info['word_classes']])
missing_in_labels = [w for w in info_words if w not in label_words]
extra_in_labels = [w for w in label_words if w not in info_words]
print(f"\nWord classes in dataset_info but MISSING from labels_word.json: {len(missing_in_labels)}")
if missing_in_labels: print(f"  -> {missing_in_labels[:15]}")
print(f"Word classes EXTRA in labels_word.json (not in dataset_info): {len(extra_in_labels)}")
if extra_in_labels: print(f"  -> {extra_in_labels[:15]}")

label_sents = sorted([v.strip().lower() for v in ls['idx_to_label'].values()])
info_sents = sorted([v.strip().lower() for v in info['sentence_classes']])
missing_s = [s for s in info_sents if s not in label_sents]
extra_s = [s for s in label_sents if s not in info_sents]
print(f"\nSentences in dataset_info but MISSING from labels_sentences.json: {len(missing_s)}")
if missing_s: print(f"  -> {missing_s[:10]}")
print(f"Sentences EXTRA in labels_sentences.json (not in dataset_info): {len(extra_s)}")
if extra_s: print(f"  -> {extra_s[:10]}")

# Check word model accuracy on TRAINING SET (inference directly via joblib + scaler)
section("3. MODEL INTEGRITY + INFERENCE ON TRAIN/VAL DATA")
import joblib
from sklearn.metrics import accuracy_score
from src.feature_extraction import batch_extract_features

w_model_path = config.MODELS_DIR / "word_model.joblib"
s_model_path = config.MODELS_DIR / "sentence_model.joblib"
w_scaler_path = config.MODELS_DIR / "scaler_words.json"
s_scaler_path = config.MODELS_DIR / "scaler_sentences.json"

for name, mp, sp in [("WORD", w_model_path, w_scaler_path), ("SENTENCE", s_model_path, s_scaler_path)]:
    print(f"\n{name} MODEL:")
    print(f"  model file exists={mp.exists()}  size={mp.stat().st_size if mp.exists() else 0}")
    print(f"  scaler file exists={sp.exists()}  size={sp.stat().st_size if sp.exists() else 0}")

# Test Word model directly
print("\n--- WORD model direct inference ---")
w_clf = joblib.load(w_model_path)
w_scaler = json.load(open(w_scaler_path))
w_mean = np.asarray(w_scaler["mean"], dtype=np.float32)
w_scale = np.asarray(w_scaler["scale"], dtype=np.float32)

# Sample: train batch
sample_N = min(200, len(Xwt))
idxs = np.arange(sample_N)
F_tr = batch_extract_features(Xwt[idxs])
F_tr_s = (F_tr - w_mean) / np.maximum(w_scale, 1e-7)
pred_tr = w_clf.predict(F_tr_s)
true_tr = ywt[idxs]
train_acc_sample = accuracy_score(true_tr, pred_tr)
print(f"  Random {sample_N} TRAIN samples: acc = {train_acc_sample*100:.2f}%  (labels match = {(pred_tr==true_tr).sum()}/{sample_N})")

# Val batch
F_v = batch_extract_features(Xwv)
F_v_s = (F_v - w_mean) / np.maximum(w_scale, 1e-7)
pred_v = w_clf.predict(F_v_s)
val_acc = accuracy_score(ywv, pred_v)
if hasattr(w_clf, "predict_proba"):
    proba_v = w_clf.predict_proba(F_v_s)
    top_conf = np.max(proba_v, axis=1)
    print(f"  WORD VAL FULL ({len(Xwv)} samples): acc={val_acc*100:.2f}%  avg_top_conf={top_conf.mean():.3f}  med_top_conf={np.median(top_conf):.3f}  min_top_conf={top_conf.min():.3f}")
    # Show top-5 correctly recognized with highest confidence
    correct_mask = (pred_v == ywv)
    if correct_mask.sum() > 0:
        confs_c = top_conf[correct_mask]
        idxs_c = np.where(correct_mask)[0]
        order = np.argsort(-confs_c)[:5]
        print(f"  Top 5 correct word predictions (val set):")
        for rank, i in enumerate(order):
            real_idx = idxs_c[i]
            lbl = lw['idx_to_label'][str(int(ywv[real_idx]))]
            print(f"    #{rank+1} true={lbl}  conf={confs_c[i]:.3f}")
else:
    print(f"  WORD VAL FULL ({len(Xwv)} samples): acc={val_acc*100:.2f}%")

# Test Sentence model directly
print("\n--- SENTENCE model direct inference ---")
s_clf = joblib.load(s_model_path)
s_scaler = json.load(open(s_scaler_path))
s_mean = np.asarray(s_scaler["mean"], dtype=np.float32)
s_scale = np.asarray(s_scaler["scale"], dtype=np.float32)

F_st = batch_extract_features(Xst)
F_st_s = (F_st - s_mean) / np.maximum(s_scale, 1e-7)
pred_st = s_clf.predict(F_st_s)
st_acc = accuracy_score(yst, pred_st)

F_sv = batch_extract_features(Xsv)
F_sv_s = (F_sv - s_mean) / np.maximum(s_scale, 1e-7)
pred_sv = s_clf.predict(F_sv_s)
sv_acc = accuracy_score(ysv, pred_sv)
if hasattr(s_clf, "predict_proba"):
    proba_sv = s_clf.predict_proba(F_sv_s)
    top_conf_s = np.max(proba_sv, axis=1)
    print(f"  SENTENCE TRAIN acc={st_acc*100:.2f}%  VAL acc={sv_acc*100:.2f}%  avg_top_val_conf={top_conf_s.mean():.3f}")
else:
    print(f"  SENTENCE TRAIN acc={st_acc*100:.2f}%  VAL acc={sv_acc*100:.2f}%")

section("4. RECOGNIZERS (WordRecognizer + SentenceRecognizer) LOAD + INFERENCE")
from src.word_recognizer import WordRecognizer
from src.sentence_recognizer import SentenceRecognizer

wr = WordRecognizer(seq_len=config.SEQUENCE_LENGTH)
sr = SentenceRecognizer(seq_len=config.SEQUENCE_LENGTH)
print(f"WordRecognizer: available={wr.model_available}  status={wr.status_text}  seq_len={wr._seq_len}  buffer_maxlen={wr._buffer.maxlen}")
print(f"SentenceRecognizer: available={sr.model_available}  status={sr.status_text}  seq_len={sr._seq_len}  buffer_maxlen={sr._buffer.maxlen}")

# Critical check: Feed actual training sequences into recognizer and verify it predicts same class
# WordRecognizer uses rolling buffer of SEQUENCE_LENGTH. Feed each frame one-by-one then predict.
N_Test = min(10, len(Xwt))
ok_w = 0
ok_s = 0
for i in range(N_Test):
    wr.reset(); sr.reset()
    # Feed each frame of sequence i, one-by-one
    for t in range(Xwt.shape[1]):
        wr.update(Xwt[i, t, :])
    pred_wr, conf_wr = wr.predict(threshold=0.0)
    true_lbl_w = lw['idx_to_label'][str(int(ywt[i]))]
    match_w = pred_wr.upper() == true_lbl_w.upper()
    if match_w: ok_w += 1
    print(f"  WR seq#{i}: true={true_lbl_w}  pred={pred_wr}  conf={conf_wr:.3f}  match={match_w}")

for i in range(min(5, len(Xst))):
    sr.reset()
    for t in range(Xst.shape[1]):
        sr.update(Xst[i, t, :])
    pred_sr, conf_sr = sr.predict(threshold=0.0)
    true_lbl_s = ls['idx_to_label'][str(int(yst[i]))]
    match_s = pred_sr.strip().lower() == true_lbl_s.strip().lower()
    if match_s: ok_s += 1
    print(f"  SR seq#{i}: true={true_lbl_s[:50]}  pred={pred_sr[:50]}  conf={conf_sr:.3f}  match={match_s}")

print(f"\nRecognizer end-to-end matches (TRAIN sequences): WORDS {ok_w}/{N_Test}   SENTENCES {ok_s}/5")

# Critical: Is there a seq_len mismatch between preprocessed data shape and recognizer's _seq_len?
X_seqlen = Xwt.shape[1]
print(f"\nCRITICAL SEQ_LEN ALIGNMENT CHECK:")
print(f"  Preprocessed data seq_len (words_X_train.shape[1]) = {X_seqlen}")
print(f"  config.SEQUENCE_LENGTH = {config.SEQUENCE_LENGTH}")
print(f"  WordRecognizer._seq_len = {wr._seq_len}")
print(f"  SentenceRecognizer._seq_len = {sr._seq_len}")
print(f"  MATCH: {X_seqlen == config.SEQUENCE_LENGTH == wr._seq_len == sr._seq_len}")

section("5. WordRecognizer PREDICT vs DIRECT joblib PREDICT MATCH")
# Take a word val sequence. Feed it to WordRecognizer AND to direct joblib predict. Compare output label.
mis_wr = 0
for i in range(min(20, len(Xwv))):
    wr.reset()
    for t in range(Xwv.shape[1]):
        wr.update(Xwv[i, t, :])
    lbl_wr, conf_wr = wr.predict(threshold=0.0)
    # direct
    feq = batch_extract_features(Xwv[i:i+1])
    fs = (feq - w_mean) / np.maximum(w_scale, 1e-7)
    d_idx = w_clf.predict(fs)[0]
    lbl_direct = lw['idx_to_label'][str(int(d_idx))]
    same = lbl_wr.upper() == lbl_direct.upper()
    if not same:
        mis_wr += 1
        if mis_wr <= 5:
            print(f"  MISMATCH seq#{i}: WR={lbl_wr}  direct={lbl_direct}  WR_conf={conf_wr:.3f}")
print(f"  WR vs direct-joblib mismatches (first 20 val): {mis_wr}/20")

section("6. SENTENCE BUILDER + GLOSS MAPPING")
from src.sentence_builder import SentenceBuilder
gmpath = config.MODELS_DIR / "gloss_to_sentence.json"
print(f"gloss_to_sentence.json: exists={gmpath.exists()}")
if gmpath.exists():
    gm = json.load(open(gmpath, encoding="utf-8"))
    print(f"  gloss entries: {len(gm)}")
    # Show first 5 mappings
    for i,(k,v) in enumerate(gm.items()):
        if i<5: print(f"  gloss[{k}] => {v}")
        else: break

# Test known ISL sentence gloss sequences
test_cases = [
    (["HELLO_HI", "HOW", "YOU"], "how are you"),
    (["YOU", "FREE", "TODAY"], "free today"),
    (["HELP", "ME"], "help me"),
    (["I", "NEED", "WATER"], "need water"),
    (["NICE", "MEET", "YOU"], "nice to meet you"),
]
sb = SentenceBuilder(debounce_frames=1)
print("\nSentenceBuilder output for known ISL sequences:")
for tokens, key_substr in test_cases:
    sb.clear()
    for t in tokens:
        sb.add_token(t, 0.9)
    out = sb.get_text()
    matched = key_substr.lower() in out.lower()
    print(f"  tokens={tokens}  =>  \"{out}\"  (contains '{key_substr}' = {matched})")

section("7. SPEECH ENGINE (pyttsx3)")
from src.speech import SpeechEngine
sp = SpeechEngine(rate=150, volume=0.5)
print(f"Speech engine: available={sp.is_available()}")
if sp.is_available():
    # Try a silent speak (short string, non-blocking). We expect no exceptions.
    try:
        result = sp.speak("Test.", block=False)
        print(f"  speak('Test.', block=False) returned: {result}  (no exception = PASS)")
    except Exception as e:
        print(f"  SPEAK FAILED: {type(e).__name__}: {e}")

section("8. HAND TRACKER + FULL LIVE PIPELINE (simulated)")
from src.hand_tracker import HandTracker
ht = HandTracker(static_image_mode=True, max_num_hands=2)
print(f"HandTracker instantiated OK. last_num_hands={ht.last_num_hands}")

# Full pipeline simulation: fake frame -> tracker -> vec -> wr -> sb -> TTS available
print("\nSimulating 60 frames of continuous gestures:")
rng = np.random.RandomState(config.SEED)
wr.reset(); sb.clear()
confirmed = 0
for frame in range(60):
    # Use an actual training word sequence, frame-by-frame, to simulate real recognition
    word_i = frame % len(Xwt)
    time_t = frame % Xwt.shape[1]
    vec = Xwt[word_i, time_t, :]
    # Simulate a slight noise to mimic real webcam
    vec = vec + rng.randn(*vec.shape).astype(np.float32) * 0.002
    wr.update(vec)
    lbl, conf = wr.predict(threshold=0.3)
    if lbl:
        added = sb.add_token(lbl, conf)
        if added:
            confirmed += 1
            print(f"  frame#{frame:3d}: CONFIRMED word={lbl} conf={conf:.2f}  sentence=\"{sb.get_text()}\"")
print(f"\nAfter 60 simulated frames: {confirmed} tokens confirmed  sentence=\"{sb.get_text()}\"")

print()
print("="*72)
print("  DIAGNOSTIC RUN COMPLETE.")
print("="*72)
