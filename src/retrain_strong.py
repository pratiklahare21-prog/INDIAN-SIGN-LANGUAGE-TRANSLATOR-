import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.feature_extraction import batch_extract_features


def augment_sequence(seq: np.ndarray, strength: float = 0.004) -> np.ndarray:
    """Lightweight temporal/geometric augmentation for 30x63 sign sequences without breaking the gesture shape."""
    seq = seq.astype(np.float32).copy()
    seq_len, _ = seq.shape

    # Subtle per-frame Gaussian jitter (landmark-level noise).
    jitter = np.random.randn(*seq.shape).astype(np.float32) * strength

    # Slow random translation drift (smooth across frames, wrist-relative so it simulates slight hand position shift).
    t = np.linspace(0, 1, seq_len, dtype=np.float32)
    drift_x = (np.random.randn(1).astype(np.float32)[0] * 0.003) * np.sin(2 * np.pi * t + np.random.rand() * np.pi)
    drift_y = (np.random.randn(1).astype(np.float32)[0] * 0.003) * np.sin(2 * np.pi * 0.7 * t + np.random.rand() * np.pi)
    drift_z = (np.random.randn(1).astype(np.float32)[0] * 0.0015) * np.sin(2 * np.pi * 1.3 * t + np.random.rand() * np.pi)
    drift = np.stack([drift_x, drift_y, drift_z], axis=-1)  # (T, 3)
    # Apply drift evenly to all 21 landmarks in each frame.
    drift_rep = np.tile(drift[:, None, :], (1, 21, 1)).reshape(seq_len, 63)

    # Time-warp by very slight temporal resample (±5%), keeping seq_len fixed via linear interp
    warp_factor = np.random.uniform(0.95, 1.05)
    src_axes = np.linspace(0, seq_len - 1, seq_len, dtype=np.float32)
    if warp_factor != 1.0:
        new_axes = np.linspace(0, seq_len - 1, int(round(seq_len * warp_factor)), dtype=np.float32)
        if len(new_axes) == seq_len:
            # same length, skip
            warped = seq + jitter + drift_rep
        else:
            from scipy.interpolate import interp1d
            warped_interp = np.zeros_like(seq)
            try:
                for dim in range(seq.shape[1]):
                    f = interp1d(src_axes, seq[:, dim], kind="linear", fill_value="extrapolate")
                    warped_interp[:, dim] = f(np.linspace(0, seq_len - 1, seq_len))
            except Exception:
                warped_interp = seq
            # Blend with jitter+drift (applied to warped sequence, not original).
            warped = warped_interp + jitter * 0.7 + drift_rep
    else:
        warped = seq + jitter + drift_rep

    # Slight uniform scale perturbation (±3%) to simulate distance-to-camera change.
    scale = np.random.uniform(0.97, 1.03)
    # Center around wrist (landmark 0), then scale.
    wrist = warped[:, :3].copy()
    centered = warped.reshape(seq_len, 21, 3) - wrist[:, None, :]
    centered *= scale
    warped = (centered + wrist[:, None, :]).reshape(seq_len, 63)

    return warped.astype(np.float32)


def augmented_features(X: np.ndarray, y: np.ndarray, aug_per_sample: int = 3, rare_aug: int = 6):
    """Generate augmented feature vectors; heavily augment rare classes (<5 samples/class)."""
    counts = np.bincount(y.astype(int), minlength=max(y) + 1)
    all_feats = []
    all_lbls = []
    for i in range(len(X)):
        label = int(y[i])
        all_feats.append(batch_extract_features(X[i:i + 1])[0])
        all_lbls.append(label)
        n_aug = rare_aug if counts[label] < 5 else aug_per_sample
        for _ in range(n_aug):
            s = augment_sequence(X[i])
            f = batch_extract_features(s[None, ...])[0]
            all_feats.append(f)
            all_lbls.append(label)
    return np.asarray(all_feats, dtype=np.float32), np.asarray(all_lbls, dtype=np.int32)


def train_word_model():
    print("=" * 60)
    print("TRAINING WORD MODEL (GradientBoosting + augmented features, class weights)")
    print("=" * 60)
    Xwt = np.load(config.LANDMARKS_DIR / "words_X_train.npy").astype(np.float32)
    ywt = np.load(config.LANDMARKS_DIR / "words_y_train.npy").astype(np.int32)
    Xwv = np.load(config.LANDMARKS_DIR / "words_X_val.npy").astype(np.float32)
    ywv = np.load(config.LANDMARKS_DIR / "words_y_val.npy").astype(np.int32)
    print(f"Train: {Xwt.shape}  Val: {Xwv.shape}")
    with open(config.WORD_LABELS_PATH, "r", encoding="utf-8") as f:
        lw = json.load(f)
    n_classes = len(lw["idx_to_label"])
    print(f"Num classes: {n_classes}")

    print("Extracting features + augmentation on TRAIN set (rare classes extra-augmented)...")
    F_tr, y_tr = augmented_features(Xwt, ywt, aug_per_sample=3, rare_aug=7)
    print(f"After aug: train features {F_tr.shape}  labels {y_tr.shape}")

    print("Extracting VAL features (no aug)...")
    F_v = batch_extract_features(Xwv)
    print(f"Val features: {F_v.shape}")

    scaler = StandardScaler()
    F_tr_s = scaler.fit_transform(F_tr)
    F_v_s = scaler.transform(F_v)

    scaler_dict = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    with open(config.MODELS_DIR / "scaler_words.json", "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f)
    print("Scaler saved: scaler_words.json")

    # GradientBoostingClassifier works better than RandomForest here when paired with stronger features:
    # - subsample=0.8 bagging fraction reduces overfit
    # - low learning_rate with many estimators + early stopping improves generalization
    # - max_depth 5/6 balances expressiveness without memorizing
    print("Training GradientBoostingClassifier (subsample=0.8, max_depth=5, class-weighted)...")
    sample_w = compute_sample_weight("balanced", y_tr)
    clf = GradientBoostingClassifier(
        n_estimators=320,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=config.SEED,
        verbose=1,
        n_iter_no_change=20,
        tol=1e-4,
        validation_fraction=0.12,
    )
    clf.fit(F_tr_s, y_tr, sample_weight=sample_w)

    if hasattr(clf, "best_iteration_") and clf.best_iteration_ is not None:
        print(f"  best_iteration_ (early stop): {clf.best_iteration_}")

    print("Computing train/val accuracies...")
    tr_acc = accuracy_score(y_tr, clf.predict(F_tr_s), sample_weight=sample_w)
    v_acc = accuracy_score(ywv, clf.predict(F_v_s))
    proba_v = clf.predict_proba(F_v_s)
    top_conf_v = np.max(proba_v, axis=1)

    # Accuracy when taking top-2 predictions (useful since threshold is low, user confirms)
    top2_idx = np.argsort(-proba_v, axis=1)[:, :2]
    top2_acc = sum(1 for i in range(len(ywv)) if ywv[i] in top2_idx[i]) / len(ywv)

    print(f"\nWORD MODEL TRAINING RESULTS:")
    print(f"  Wtd Train Acc: {tr_acc * 100:.2f}%   Val Acc: {v_acc * 100:.2f}%")
    print(f"  Top-2 Val Acc : {top2_acc * 100:.2f}%")
    print(f"  Val conf (avg/med/min): {top_conf_v.mean():.3f} / {np.median(top_conf_v):.3f} / {top_conf_v.min():.3f}")

    # Confidence distribution breakdown (how many val samples would pass our 0.22 threshold?)
    for t in [0.15, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50]:
        n_pass = int((top_conf_v >= t).sum())
        acc_pass = accuracy_score(ywv[top_conf_v >= t], clf.predict(F_v_s[top_conf_v >= t])) if n_pass > 0 else float('nan')
        print(f"  Thresh {t:.2f}: pass={n_pass}/{len(ywv)}  accuracy@pass={acc_pass*100:.1f}%" if n_pass > 0 else f"  Thresh {t:.2f}: pass=0/{len(ywv)}")

    out_path = config.MODELS_DIR / "word_model.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"\nWord model saved: {out_path}")
    return v_acc


def train_sentence_model():
    print()
    print("=" * 60)
    print("TRAINING SENTENCE MODEL (GradientBoosting + heavy augmentation)")
    print("=" * 60)
    Xst = np.load(config.LANDMARKS_DIR / "sentences_X_train.npy").astype(np.float32)
    yst = np.load(config.LANDMARKS_DIR / "sentences_y_train.npy").astype(np.int32)
    Xsv = np.load(config.LANDMARKS_DIR / "sentences_X_val.npy").astype(np.float32)
    ysv = np.load(config.LANDMARKS_DIR / "sentences_y_val.npy").astype(np.int32)
    print(f"Train: {Xst.shape}  Val: {Xsv.shape}")
    with open(config.SENTENCE_LABELS_PATH, "r", encoding="utf-8") as f:
        ls = json.load(f)
    n_classes = len(ls["idx_to_label"])
    print(f"Num classes: {n_classes}")

    print("Heavy augmentation on TRAIN set (sentence classes have few samples)...")
    F_tr, y_tr = augmented_features(Xst, yst, aug_per_sample=6, rare_aug=12)
    print(f"After aug: train features {F_tr.shape}  labels {y_tr.shape}")

    print("Extracting VAL features...")
    F_v = batch_extract_features(Xsv)

    scaler = StandardScaler()
    F_tr_s = scaler.fit_transform(F_tr)
    F_v_s = scaler.transform(F_v)

    scaler_dict = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    with open(config.MODELS_DIR / "scaler_sentences.json", "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f)
    print("Scaler saved: scaler_sentences.json")

    sample_w = compute_sample_weight("balanced", y_tr)
    print("Training GradientBoostingClassifier (subsample=0.85, max_depth=5)...")
    clf = GradientBoostingClassifier(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=config.SEED,
        verbose=1,
        n_iter_no_change=20,
        tol=1e-4,
        validation_fraction=0.15,
    )
    clf.fit(F_tr_s, y_tr, sample_weight=sample_w)

    if hasattr(clf, "best_iteration_") and clf.best_iteration_ is not None:
        print(f"  best_iteration_ (early stop): {clf.best_iteration_}")

    tr_acc = accuracy_score(y_tr, clf.predict(F_tr_s), sample_weight=sample_w)
    v_acc = accuracy_score(ysv, clf.predict(F_v_s)) if len(ysv) > 0 else 0.0
    proba_v = clf.predict_proba(F_v_s) if len(ysv) > 0 else None
    if proba_v is not None:
        top2_idx = np.argsort(-proba_v, axis=1)[:, :2]
        top2_acc = sum(1 for i in range(len(ysv)) if ysv[i] in top2_idx[i]) / len(ysv)
        top_conf_v = np.max(proba_v, axis=1)
    else:
        top2_acc = 0.0
        top_conf_v = np.array([0.0])

    print(f"\nSENTENCE MODEL TRAINING RESULTS:")
    print(f"  Wtd Train Acc: {tr_acc * 100:.2f}%   Val Acc: {v_acc * 100:.2f}%")
    print(f"  Top-2 Val Acc : {top2_acc * 100:.2f}%")
    print(f"  Val conf (avg/med): {top_conf_v.mean():.3f} / {np.median(top_conf_v):.3f}")
    if proba_v is not None:
        for t in [0.25, 0.30, 0.40, 0.50]:
            mask = top_conf_v >= t
            n_pass = int(mask.sum())
            acc_p = accuracy_score(ysv[mask], clf.predict(F_v_s[mask])) if n_pass > 0 else float('nan')
            print(f"  Thresh {t:.2f}: pass={n_pass}/{len(ysv)}  acc@pass={acc_p*100:.1f}%" if n_pass > 0 else f"  Thresh {t:.2f}: pass=0/{len(ysv)}")

    out_path = config.MODELS_DIR / "sentence_model.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"\nSentence model saved: {out_path}")
    return v_acc


if __name__ == "__main__":
    v1 = train_word_model()
    v2 = train_sentence_model()
    print()
    print("=" * 60)
    print(f"FINAL SUMMARY: Word Val Acc={v1*100:.1f}%  Sentence Val Acc={v2*100:.1f}%")
    print("=" * 60)
