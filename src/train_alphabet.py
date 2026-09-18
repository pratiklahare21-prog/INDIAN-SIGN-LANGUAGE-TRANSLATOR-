import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config

try:
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.callbacks import (
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
    )
    from tensorflow.keras.layers import (
        BatchNormalization,
        Dense,
        Dropout,
    )
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    _TF_OK = True
except Exception as _e:
    _TF_ERR = str(_e)
    _TF_OK = False

try:
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except Exception as _e:
    _SKLEARN_ERR = str(_e)
    _SKLEARN_OK = False
    confusion_matrix = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]


def parse_args():
    p = argparse.ArgumentParser(description="Train alphabet MLP classifier on precomputed hand landmarks.")
    p.add_argument("--landmarks-dir", type=Path, default=config.LANDMARKS_DIR,
                   help="Directory containing alphabet_X_train.npy etc.")
    p.add_argument("--labels-path", type=Path, default=config.ALPHABET_LABELS_PATH,
                   help="Path to labels_alphabet.json.")
    p.add_argument("--model-out", type=Path, default=config.ALPHABET_MODEL_PATH,
                   help="Output path for trained .keras model.")
    p.add_argument("--epochs", type=int, default=int(config.ALPHABET_TRAIN_KWARGS.get("epochs", 50)),
                   help="Max training epochs.")
    p.add_argument("--quick", action="store_true",
                   help="Quick run: 1 epoch, 200 train / 50 val samples, val_split 0.1.")
    return p.parse_args()


def _set_seeds():
    seed = int(config.SEED)
    random.seed(seed)
    np.random.seed(seed)
    if _TF_OK:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass


def _top_confused_pairs(cm, idx_to_label, topk=3):
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append((cm[i, j], idx_to_label.get(str(i), str(i)), idx_to_label.get(str(j), str(j))))
    pairs.sort(reverse=True)
    return pairs[:topk]


def main():
    args = parse_args()
    _set_seeds()

    if not _TF_OK:
        print(f"[train_alphabet] TensorFlow unavailable: {_TF_ERR}")
        print("Install tensorflow to train the alphabet model.")
        return 0
    if not _SKLEARN_OK:
        print(f"[train_alphabet] scikit-learn unavailable: {_SKLEARN_ERR}")
        print("Install scikit-learn (for StandardScaler + confusion-matrix analysis).")
        return 0

    landmarks_dir: Path = args.landmarks_dir
    labels_path: Path = args.labels_path
    model_out: Path = args.model_out

    model_out.parent.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        X_train = np.load(landmarks_dir / "alphabet_X_train.npy").astype(np.float32)
        y_train = np.load(landmarks_dir / "alphabet_y_train.npy").astype(np.int32)
        X_val = np.load(landmarks_dir / "alphabet_X_val.npy").astype(np.float32)
        y_val = np.load(landmarks_dir / "alphabet_y_val.npy").astype(np.int32)
    except Exception as e:
        print(f"[train_alphabet] Failed to load landmarks from {landmarks_dir}: {e}")
        print("Run preprocess_alphabet.py first.")
        return 0

    with open(labels_path, "r") as f:
        labels = json.load(f)
    idx_to_label = labels["idx_to_label"]
    label_to_idx = labels["label_to_idx"]
    num_classes = len(idx_to_label)
    if num_classes < 2 or sorted(label_to_idx.values()) != list(range(num_classes)):
        print("[train_alphabet] Invalid labels: expected contiguous indices for every detected class.")
        return 1
    if int(np.max(y_train)) >= num_classes or int(np.max(y_val)) >= num_classes:
        print("[train_alphabet] Label arrays contain an index absent from labels_alphabet.json.")
        return 1
    print(f"[train_alphabet] num_classes={num_classes}")

    if args.quick:
        X_train = X_train[:200]
        y_train = y_train[:200]
        X_val = X_val[:50]
        y_val = y_val[:50]
        epochs = 1
        batch_size = 16
        val_split = 0.1
        patience = 2
        lr = 1e-3
        print("[train_alphabet] QUICK mode: epochs=1, cap 200/50 train/val.")
    else:
        epochs = int(args.epochs)
        batch_size = int(config.ALPHABET_TRAIN_KWARGS.get("batch_size", 64))
        val_split = float(config.ALPHABET_TRAIN_KWARGS.get("validation_split", 0.2))
        patience = int(config.ALPHABET_TRAIN_KWARGS.get("early_stopping_patience", 8))
        lr = float(config.ALPHABET_TRAIN_KWARGS.get("learning_rate", 1e-3))

    print(f"[train_alphabet] X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[train_alphabet] X_val={X_val.shape},   y_val={y_val.shape}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    scaler_path = config.MODELS_DIR / "scaler_alphabet.json"
    with open(scaler_path, "w") as f:
        json.dump(
            {
                "mean": scaler.mean_.tolist(),
                "var": scaler.var_.tolist(),
                "scale": scaler.scale_.tolist(),
                "n_features_in": int(scaler.n_features_in_),
            },
            f,
            indent=2,
        )
    print(f"[train_alphabet] scaler saved -> {scaler_path}")

    y_train_oh = to_categorical(y_train, num_classes=num_classes).astype(np.float32)
    y_val_oh = to_categorical(y_val, num_classes=num_classes).astype(np.float32)

    m_kwargs = config.ALPHABET_MODEL_KWARGS
    hidden_units = tuple(m_kwargs.get("hidden_units", (256, 128, 64)))
    dropout = float(m_kwargs.get("dropout", 0.3))
    use_bn = bool(m_kwargs.get("batch_norm", True))

    inputs = Input(shape=(config.LANDMARK_DIM,), name="alphabet_input")
    x = inputs
    for units in hidden_units:
        x = Dense(units, activation="relu")(x)
        if use_bn:
            x = BatchNormalization()(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax", name="alphabet_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="alphabet_mlp")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda line: print(f"  {line}"))

    best_ckpt = model_out.parent / "alphabet_model_best.keras"
    csv_path = config.DATA_DIR / "training_log_alphabet.csv"
    curves_path = config.MODELS_DIR / "alphabet_training_curves.png"

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger(str(csv_path)),
        ModelCheckpoint(
            str(best_ckpt),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train_s, y_train_oh,
        validation_data=(X_val_s, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    val_pred_prob = model.predict(X_val_s, verbose=0)
    val_pred = np.argmax(val_pred_prob, axis=1)
    cm = confusion_matrix(y_val, val_pred, labels=list(range(num_classes)))
    top_pairs = _top_confused_pairs(cm, idx_to_label, topk=3)

    final_val_acc = float(np.mean(val_pred == y_val))
    print(f"[train_alphabet] final val_accuracy = {final_val_acc:.4f}")
    print(f"[train_alphabet] top-3 confused pairs (true->pred, count):")
    for count, t, p in top_pairs:
        print(f"  {t} -> {p} : {count}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        h = history.history
        epochs_axis = list(range(1, len(h["loss"]) + 1))
        axes[0].plot(epochs_axis, h["loss"], label="train")
        if "val_loss" in h:
            axes[0].plot(epochs_axis, h["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs_axis, h["accuracy"], label="train")
        if "val_accuracy" in h:
            axes[1].plot(epochs_axis, h["val_accuracy"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(curves_path), dpi=140)
        plt.close(fig)
        print(f"[train_alphabet] training curves saved -> {curves_path}")
    except Exception as e:
        print(f"[train_alphabet] could not plot training curves ({e}); skipping.")

    model.save(str(model_out), overwrite=True)
    print(f"[train_alphabet] model saved -> {model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
