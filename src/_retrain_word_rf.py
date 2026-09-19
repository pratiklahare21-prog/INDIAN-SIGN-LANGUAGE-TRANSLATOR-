import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import config
from src.feature_extraction import batch_extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib

Xwt = np.load(config.LANDMARKS_DIR / "words_X_train.npy").astype(np.float32)
ywt = np.load(config.LANDMARKS_DIR / "words_y_train.npy").astype(np.int32)
Xwv = np.load(config.LANDMARKS_DIR / "words_X_val.npy").astype(np.float32)
ywv = np.load(config.LANDMARKS_DIR / "words_y_val.npy").astype(np.int32)
print("Loaded train", Xwt.shape, "val", Xwv.shape, flush=True)

def aug(s, n=3):
    seqs = [s]
    for _ in range(n):
        a = s + np.random.randn(*s.shape).astype(np.float32) * 0.004
        sc = np.random.uniform(0.97, 1.03)
        wr = a[:, :3].copy()
        ce = a.reshape(30, 21, 3) - wr[:, None, :]
        ce *= sc
        a = (ce + wr[:, None, :]).reshape(30, 63)
        seqs.append(a)
    return seqs

counts = np.bincount(ywt.astype(int), minlength=max(ywt) + 1)
seqs_all = []
lbls_all = []
for i in range(len(Xwt)):
    n_aug = 6 if counts[int(ywt[i])] < 5 else 3
    for s in aug(Xwt[i], n_aug):
        seqs_all.append(s)
        lbls_all.append(int(ywt[i]))
print("Augmented sequences:", len(seqs_all), flush=True)

X_train_aug = np.stack(seqs_all, axis=0).astype(np.float32)
print("Extracting features train...", flush=True)
F_tr = batch_extract_features(X_train_aug)
y_tr = np.asarray(lbls_all, dtype=np.int32)
print("F_tr:", F_tr.shape, "y_tr:", y_tr.shape, flush=True)

print("Extracting val features...", flush=True)
F_v = batch_extract_features(Xwv)

scaler = StandardScaler()
F_tr_s = scaler.fit_transform(F_tr)
F_v_s = scaler.transform(F_v)
with open(config.MODELS_DIR / "scaler_words.json", "w", encoding="utf-8") as f:
    json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)
print("Scaler saved.", flush=True)

sw = compute_sample_weight("balanced", y_tr)
print("Training RandomForest (regularized, n_est=250, md=18, ss=3)...", flush=True)
clf = RandomForestClassifier(
    n_estimators=250,
    max_depth=18,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features=0.4,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=config.SEED,
    verbose=1,
)
clf.fit(F_tr_s, y_tr, sample_weight=sw)
print(f"OOB score: {clf.oob_score_*100:.2f}%", flush=True)

tr_acc = accuracy_score(y_tr, clf.predict(F_tr_s), sample_weight=sw)
v_acc = accuracy_score(ywv, clf.predict(F_v_s))
probv = clf.predict_proba(F_v_s)
topc = np.max(probv, axis=1)
top2 = np.argsort(-probv, axis=1)[:, :2]
t2acc = sum(1 for i in range(len(ywv)) if ywv[i] in top2[i]) / len(ywv)
print(f"\nWORD RESULTS: Train(wtd)={tr_acc*100:.2f}% Val={v_acc*100:.2f}% Top-2={t2acc*100:.2f}%", flush=True)
print(f"Val top conf avg/max/min: {topc.mean():.3f}/{topc.max():.3f}/{topc.min():.3f}", flush=True)
for t in [0.18, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50]:
    m = topc >= t
    if m.sum() == 0:
        print(f"  T{t}: pass=0/{len(ywv)}", flush=True)
        continue
    a = accuracy_score(ywv[m], clf.predict(F_v_s[m]))
    print(f"  T{t}: pass={m.sum()}/{len(ywv)} acc@pass={a*100:.1f}%", flush=True)

out = config.MODELS_DIR / "word_model.joblib"
joblib.dump(clf, out)
sz = os.path.getsize(out) / (1024 * 1024)
print(f"\nWORD MODEL SAVED: {out}  ({sz:.1f} MB)", flush=True)
