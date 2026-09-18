import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.feature_extraction import batch_extract_features

try:
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
    from tensorflow.keras.utils import to_categorical
    _TF_OK = True
except Exception:
    _TF_OK = False


def parse_args():
    p = argparse.ArgumentParser(description="Train ISL sentence sequence classifier.")
    p.add_argument("--landmarks-dir", type=Path, default=config.LANDMARKS_DIR)
    p.add_argument("--labels-path", type=Path, default=config.SENTENCE_LABELS_PATH)
    p.add_argument("--model-out", type=Path, default=config.MODELS_DIR / "sentence_model.joblib")
    p.add_argument("--scaler-out", type=Path, default=config.MODELS_DIR / "scaler_sentences.json")
    p.add_argument("--epochs", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    landmarks_dir = args.landmarks_dir
    labels_path = args.labels_path
    model_out = args.model_out
    scaler_out = args.scaler_out

    x_train_path = landmarks_dir / "sentences_X_train.npy"
    if not x_train_path.exists():
        print("[train_sentences] Sentence landmarks not found. Running src/preprocess_sentences.py...")
        from src.preprocess_sentences import main as run_preprocess
        run_preprocess()

    if not x_train_path.exists():
        print(f"[train_sentences] Error: {x_train_path} missing.")
        return 1

    X_train_raw = np.load(landmarks_dir / "sentences_X_train.npy").astype(np.float32)
    y_train = np.load(landmarks_dir / "sentences_y_train.npy").astype(np.int32)
    X_val_raw = np.load(landmarks_dir / "sentences_X_val.npy").astype(np.float32)
    y_val = np.load(landmarks_dir / "sentences_y_val.npy").astype(np.int32)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    idx_to_label = labels_data["idx_to_label"]
    num_classes = len(idx_to_label)

    print(f"[train_sentences] Training sentence model on {len(X_train_raw)} train samples across {num_classes} classes.")

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

    clf = MLPClassifier(
        hidden_layer_sizes=(384, 192, 96),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=min(32, len(X_train_scaled)),
        learning_rate_init=0.001,
        max_iter=args.epochs,
        random_state=config.SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        verbose=False,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred_train = clf.predict(X_train_scaled)
    y_pred_val = clf.predict(X_val_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)

    print(f"[train_sentences] Train Accuracy: {train_acc * 100:.2f}% | Validation Accuracy: {val_acc * 100:.2f}%")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"[train_sentences] Sentence model saved -> {model_out}")

    if _TF_OK:
        try:
            keras_path = config.MODELS_DIR / "sentence_model.keras"
            inp = Input(shape=(X_train_raw.shape[1], X_train_raw.shape[2]))
            x = Bidirectional(LSTM(128, return_sequences=True))(inp)
            x = Dropout(0.3)(x)
            x = Bidirectional(LSTM(64))(x)
            x = Dropout(0.3)(x)
            out = Dense(num_classes, activation="softmax")(x)
            k_model = Model(inp, out)
            k_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            k_model.fit(
                X_train_raw, to_categorical(y_train, num_classes),
                validation_data=(X_val_raw, to_categorical(y_val, num_classes)),
                epochs=min(args.epochs, 40), batch_size=32, verbose=0
            )
            k_model.save(keras_path)
            print(f"[train_sentences] Keras model saved -> {keras_path}")
        except Exception as e:
            print(f"[train_sentences] Keras save note: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
