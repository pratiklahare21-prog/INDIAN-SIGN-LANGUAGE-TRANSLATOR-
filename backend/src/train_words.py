import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.feature_extraction import batch_extract_features


def parse_args():
    p = argparse.ArgumentParser(description="Train ISL word sequence classifier.")
    p.add_argument("--landmarks-dir", type=Path, default=config.LANDMARKS_DIR)
    p.add_argument("--labels-path", type=Path, default=config.WORD_LABELS_PATH)
    p.add_argument("--model-out", type=Path, default=config.MODELS_DIR / "word_model.joblib")
    p.add_argument("--scaler-out", type=Path, default=config.MODELS_DIR / "scaler_words.json")
    p.add_argument("--epochs", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    landmarks_dir = args.landmarks_dir
    labels_path = args.labels_path
    model_out = args.model_out
    scaler_out = args.scaler_out

    x_train_path = landmarks_dir / "words_X_train.npy"
    if not x_train_path.exists():
        print("[train_words] Landmarks not found. Running src/preprocess_words.py first...")
        from src.preprocess_words import main as run_preprocess
        run_preprocess()

    if not x_train_path.exists():
        print(f"[train_words] Error: {x_train_path} missing.")
        return 1

    X_train_raw = np.load(landmarks_dir / "words_X_train.npy").astype(np.float32)
    y_train = np.load(landmarks_dir / "words_y_train.npy").astype(np.int32)
    X_val_raw = np.load(landmarks_dir / "words_X_val.npy").astype(np.float32)
    y_val = np.load(landmarks_dir / "words_y_val.npy").astype(np.int32)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    idx_to_label = labels_data["idx_to_label"]
    num_classes = len(idx_to_label)

    print(f"[train_words] Training on {len(X_train_raw)} train, {len(X_val_raw)} val samples across {num_classes} classes.")

    print("[train_words] Extracting normalized sequence features...")
    X_train = batch_extract_features(X_train_raw)
    X_val = batch_extract_features(X_val_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    scaler_dict = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(scaler_out, "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f)

    # Train Classifier with high regularization and deep capacity
    print("[train_words] Training sequence classifier...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=2,
        random_state=config.SEED,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, clf.predict(X_val_scaled))

    print(f"[train_words] Train Accuracy: {train_acc * 100:.2f}% | Validation Accuracy: {val_acc * 100:.2f}%")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"[train_words] Model saved successfully -> {model_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
