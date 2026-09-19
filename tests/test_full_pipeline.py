import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import json
import numpy as np
from collections import deque
from sklearn.metrics import accuracy_score

def section(title):
    print()
    print("=" * 70)
    print("  " + title)
    print("=" * 70)

# ============================================================
# TEST 1: VALIDATE RECOGNIZER PREDICTIONS ON TRAIN DATA WITH
# ACTUAL CONFIG THRESHOLDS (this is what webcam will actually use).
# ============================================================
section("Test 1: Word recognizer on 20 TRAIN sequences (actual config thresholds)")
from src.word_recognizer import WordRecognizer
from src.sentence_recognizer import SentenceRecognizer

Xwt = np.load(config.LANDMARKS_DIR / "words_X_train.npy").astype(np.float32)
ywt = np.load(config.LANDMARKS_DIR / "words_y_train.npy").astype(np.int32)
lw = json.load(open(config.WORD_LABELS_PATH, encoding="utf-8"))

wr = WordRecognizer(seq_len=config.SEQUENCE_LENGTH)
print(f"  Config WORD_THRESHOLD = {config.WORD_THRESHOLD}   DEBOUNCE_FRAMES = {config.DEBOUNCE_FRAMES}")

ok_strict = 0
ok_within_top3 = 0
N = min(40, len(Xwt))
predictions_made = 0
for i in range(N):
    wr.reset()
    true_label = lw["idx_to_label"][str(int(ywt[i]))]
    # Feed each frame, then predict using the actual recognizer (streak_threshold=3 as default)
    # Since streak requires 3 consistent updates, feed each frame AND predict after each.
    last_predictions = deque(maxlen=7)
    for t in range(Xwt.shape[1]):
        wr.update(Xwt[i, t, :])
        # call predict() with actual default threshold (will be "" if below threshold or streak < 3)
        lbl, conf = wr.predict()  # default threshold + streak=3
        if lbl:
            last_predictions.append((lbl, conf))
    # After whole sequence: take last non-empty prediction if any
    final_lbl, final_conf = ("", 0.0)
    if len(last_predictions) > 0:
        final_lbl, final_conf = last_predictions[-1]
        predictions_made += 1
        match = final_lbl.upper() == true_label.upper()
        if match:
            ok_strict += 1
    else:
        # Check with return_all=True (bypass threshold, see top-1)
        wr.reset()
        for t in range(Xwt.shape[1]):
            wr.update(Xwt[i, t, :])
        ra_lbl, ra_conf = wr.predict(threshold=0.0, streak_threshold=1, return_all=True)
        match_ra = ra_lbl.upper() == true_label.upper()
        if match_ra:
            ok_within_top3 += 1
        if i < 10:
            print(f"    #{i:2d}: true={true_label:22s} no-pred-thresh-pass   top1={ra_lbl} conf={ra_conf:.3f}  match={match_ra}")
        continue
    if i < 10:
        print(f"    #{i:2d}: true={true_label:22s} pred={final_lbl:22s} conf={final_conf:.3f}  match={match}")

print(f"\n  Summary {N} train sequences:")
print(f"    Predictions made (passed thresh+streak): {predictions_made}/{N}  ({100*predictions_made/N:.1f}%)")
print(f"    Of those made, correct strict match    : {ok_strict}/{predictions_made}  ({100*ok_strict/max(1,predictions_made):.1f}%)")
print(f"    Sequences w/ no prediction, but top-1 correct: {ok_within_top3}/{N - predictions_made}  (these need confidence tuning)")

# ============================================================
# TEST 2: SENTENCE RECOGNIZER with config thresholds.
# ============================================================
section("Test 2: Sentence recognizer on 15 TRAIN sequences")
Xst = np.load(config.LANDMARKS_DIR / "sentences_X_train.npy").astype(np.float32)
yst = np.load(config.LANDMARKS_DIR / "sentences_y_train.npy").astype(np.int32)
ls = json.load(open(config.SENTENCE_LABELS_PATH, encoding="utf-8"))
sr = SentenceRecognizer(seq_len=config.SEQUENCE_LENGTH)
print(f"  Config SENTENCE_THRESHOLD = {config.SENTENCE_THRESHOLD}")
N = min(15, len(Xst))
sent_ok = 0
sent_made = 0
for i in range(N):
    sr.reset()
    true_sent = ls["idx_to_label"][str(int(yst[i]))]
    last_pred = ("", 0.0)
    for t in range(Xst.shape[1]):
        sr.update(Xst[i, t, :])
        lbl, conf = sr.predict()
        if lbl:
            last_pred = (lbl, conf)
    if last_pred[0]:
        sent_made += 1
        if last_pred[0].strip().lower() == true_sent.strip().lower():
            sent_ok += 1
        flag = "MATCH" if last_pred[0].strip().lower() == true_sent.strip().lower() else "DIFF"
        print(f"    #{i:2d} [{flag:5s}] conf={last_pred[1]:.3f}  true=\"{true_sent[:60]}\"  pred=\"{last_pred[0][:60]}\"")
    else:
        sr.reset()
        for t in range(Xst.shape[1]):
            sr.update(Xst[i, t, :])
        r_lbl, r_conf = sr.predict(threshold=0.0, streak_threshold=1, return_all=True)
        print(f"    #{i:2d} [NONE ] top1 conf={r_conf:.3f}  true=\"{true_sent[:60]}\"  top1=\"{r_lbl[:60]}\"")
print(f"\n  Summary {N} sentence train sequences:")
print(f"    Predictions made: {sent_made}/{N}  of those correct: {sent_ok}/{sent_made}  ({100*sent_ok/max(1,sent_made):.1f}%)")

# ============================================================
# TEST 3: FULL LIVE SIMULATION WITH REAL SIGN SEQUENCES.
# Simulate: USER SIGNS 4 distinct signs in sequence.
# ============================================================
section("Test 3: Full live simulation — sign 4 distinct words, expect sentence formation")
# Pick 4 well-known words from the training set with distinct labels:
word_names = ["HELLO_HI", "HOW", "YOU", "THANK"]
word_indices = []
for name in word_names:
    candidates = [k for k, v in lw["label_to_idx"].items() if name.upper() in k.upper() or k.upper() == name.upper()]
    if candidates:
        idx = lw["label_to_idx"][candidates[0]]
        word_indices.append((name, int(idx)))
print(f"  Picked {len(word_indices)} distinct signs to simulate:")
for nm, idx in word_indices:
    samples = np.where(ywt == idx)[0]
    print(f"    {nm:12s} (idx={idx:3d})  {len(samples)} training samples")

wr.reset()
from src.sentence_builder import SentenceBuilder
sb = SentenceBuilder(debounce_frames=config.DEBOUNCE_FRAMES)
from src.speech import SpeechEngine
sp = SpeechEngine(rate=config.PYTTSX_RATE, volume=config.PYTTSX_VOLUME)
print(f"  Speech engine available: {sp.is_available()}")

# Stream: sign word 1 fully (30 frames), then pause 2 frames, sign word 2 fully, then pause, ...
stream_frames = []  # list of (vec, signame)
for signame, sigidx in word_indices:
    samples = np.where(ywt == sigidx)[0]
    if len(samples) == 0: continue
    use_seq_idx = samples[0]
    for f in range(30):
        stream_frames.append((np.copy(Xwt[use_seq_idx, f, :]), signame))
    # 3-frame "pause" of zeros (no hand / resting gesture) to break streak
    for _ in range(3):
        stream_frames.append((np.zeros(config.LANDMARK_DIM, dtype=np.float32) + 1e-6, "REST"))

signals_confirmed = []
all_sentences = []
frame_n = 0
for vec, signame in stream_frames:
    # Only update recognizer if this looks like a real hand (not all zeros / REST)
    if np.abs(vec).mean() > 0.005:
        wr.update(vec)
    else:
        # Rest frame: don't update recognizer. It will keep its recent buffer (old probs still apply).
        pass
    lbl, conf = wr.predict()  # use default config threshold + streak
    if lbl:
        added = sb.add_token(lbl, conf)
        if added:
            signals_confirmed.append((frame_n, lbl, conf, sb.get_text()))
            sentence_now = sb.get_text()
            if sp.is_available():
                try:
                    sp.speak(lbl.replace("_", " "), block=False)
                except Exception:
                    pass
            print(f"  Frame#{frame_n:3d}: CONFIRMED token=\"{lbl}\" conf={conf:.2f}  -> SENTENCE=\"{sentence_now}\"")
    frame_n += 1

final_sentence = sb.get_text()
tokens_confirmed = sb.get_raw_tokens()
print(f"\n  Live simulation complete. Total frames: {frame_n}")
print(f"  Confirmed tokens: {tokens_confirmed}")
print(f"  Total tokens confirmed: {len(signals_confirmed)}")
print(f"  Final sentence: \"{final_sentence}\"")
print(f"  Speech engine used during simulation (labels spoken): {sp.is_available()}")

# ============================================================
# TEST 4: Confidence vs threshold analysis (suggest tuning)
# ============================================================
section("Test 4: VALIDATION confidence distribution vs current thresholds")
from src.feature_extraction import batch_extract_features
import joblib
w_clf = joblib.load(config.MODELS_DIR / "word_model.joblib")
w_scaler_d = json.load(open(config.MODELS_DIR / "scaler_words.json", encoding="utf-8"))
w_mean = np.asarray(w_scaler_d["mean"], dtype=np.float32)
w_scale = np.asarray(w_scaler_d["scale"], dtype=np.float32)

Xwv = np.load(config.LANDMARKS_DIR / "words_X_val.npy").astype(np.float32)
ywv = np.load(config.LANDMARKS_DIR / "words_y_val.npy").astype(np.int32)
Fv = batch_extract_features(Xwv)
Fv_s = (Fv - w_mean) / np.maximum(w_scale, 1e-7)
probv = w_clf.predict_proba(Fv_s)
topc = np.max(probv, axis=1)

print(f"  Word VAL samples: {len(Xwv)}")
print(f"  Top-1 confidence: mean={topc.mean():.3f}  p25={np.quantile(topc,0.25):.3f}  p50={np.median(topc):.3f}  p75={np.quantile(topc,0.75):.3f}")
print(f"  Current WORD_THRESHOLD={config.WORD_THRESHOLD}  => {int((topc >= config.WORD_THRESHOLD).sum())}/{len(topc)} samples pass threshold")
# Accuracy at threshold
mask = topc >= config.WORD_THRESHOLD
if mask.sum() > 0:
    preds = w_clf.predict(Fv_s[mask])
    acc = accuracy_score(ywv[mask], preds)
    print(f"  Accuracy on samples passing threshold: {acc*100:.1f}%  ({mask.sum()} samples)")
else:
    print("  WARNING: zero validation samples pass the current threshold!")

# Show top-2 acc (indicator of how often correct answer is in top-2 predictions)
top2_idx = np.argsort(-probv, axis=1)[:, :2]
t2acc = sum(1 for i in range(len(ywv)) if ywv[i] in top2_idx[i]) / len(ywv)
print(f"  Top-2 accuracy (any conf): {t2acc*100:.1f}%   (if user corrects wrong prediction, model is likely to be in top-2)")

# Final summary
print()
print("=" * 70)
print("  ALL REGRESSION TESTS COMPLETE.")
print("=" * 70)
