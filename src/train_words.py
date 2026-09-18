import argparse
import json
import random
import sys
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
        Bidirectional,
        Dense,
        Dropout,
        LSTM,
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
    p = argparse.ArgumentParser(description="Train word LSTM classifier on precomputed hand-landmark sequences.")
    p.add_argument("--landmarks-dir", type=Path, default=config.LANDMARKS_DIR,
                   help="Directory containing words_X_train.npy etc.")
    p.add_argument("--labels-path", type=Path, default=config.WORD_LABELS_PATH,
                   help="Path to labels_word.json.")
    p.add_argument("--model-out", type=Path, default=config.WORD_MODEL_PATH,
                   help="Output path for trained .keras model.")
    p.add_argument("--epochs", type=int, default=int(config.WORD_TRAIN_KWARGS.get("epochs", 80)),
                   help="Max training epochs.")
    p.add_argument("--quick", action="store_true",
                   help="Quick run: 1 epoch, first 100 train / 25 val samples, sequence dim kept.")
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
        print(f"[train_words] TensorFlow unavailable: {_TF_ERR}")
        print("Install tensorflow to train the word model.")
        return 0
    if not _SKLEARN_OK:
        print(f"[train_words] scikit-learn unavailable: {_SKLEARN_ERR}")
        print("Install scikit-learn (for StandardScaler + confusion-matrix analysis).")
        return 0

    landmarks_dir: Path = args.landmarks_dir
    labels_path: Path = args.labels_path
    model_out: Path = args.model_out

    model_out.parent.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        X_train = np.load(landmarks_dir / "words_X_train.npy").astype(np.float32)
        y_train = np.load(landmarks_dir / "words_y_train.npy").astype(np.int32)
        X_val = np.load(landmarks_dir / "words_X_val.npy").astype(np.float32)
        y_val = np.load(landmarks_dir / "words_y_val.npy").astype(np.int32)
    except Exception as e:
        print(f"[train_words] Failed to load landmarks from {landmarks_dir}: {e}")
        print("Run preprocess_words.py first.")
        return 0

    seq_len = int(X_train.shape[1])
    feat_dim = int(X_train.shape[2])
    assert feat_dim == config.LANDMARK_DIM

    with open(labels_path, "r") as f:
        labels = json.load(f)
    idx_to_label = labels["idx_to_label"]
    label_to_idx = labels["label_to_idx"]
    num_classes = len(idx_to_label)
    if num_classes < 2 or sorted(label_to_idx.values()) != list(range(num_classes)):
        print("[train_words] Invalid labels: expected contiguous indices for every detected class.")
        return 1
    if int(np.max(y_train)) >= num_classes or int(np.max(y_val)) >= num_classes:
        print("[train_words] Label arrays contain an index absent from labels_word.json.")
        return 1
    print(f"[train_words] num_classes={num_classes}, seq_len={seq_len}")

    if args.quick:
        X_train = X_train[:100]
        y_train = y_train[:100]
        X_val = X_val[:25]
        y_val = y_val[:25]
        epochs = 1
        batch_size = 8
        val_split = 0.1
        patience = 2
        lr = 1e-3
        print("[train_words] QUICK mode: epochs=1, cap 100/25 train/val.")
    else:
        epochs = int(args.epochs)
        batch_size = int(config.WORD_TRAIN_KWARGS.get("batch_size", 32))
        val_split = float(config.WORD_TRAIN_KWARGS.get("validation_split", 0.2))
        patience = int(config.WORD_TRAIN_KWARGS.get("early_stopping_patience", 12))
        lr = float(config.WORD_TRAIN_KWARGS.get("learning_rate", 1e-3))

    print(f"[train_words] X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[train_words] X_val={X_val.shape},   y_val={y_val.shape}")

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    X_train_flat = X_train.reshape(-1, feat_dim)
    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    X_train_s = scaler.transform(X_train_flat).reshape(n_train, seq_len, feat_dim).astype(np.float32)
    X_val_s = scaler.transform(X_val.reshape(-1, feat_dim)).reshape(n_val, seq_len, feat_dim).astype(np.float32)

    scaler_path = config.MODELS_DIR / "scaler_words.json"
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
    print(f"[train_words] scaler saved -> {scaler_path}")

    y_train_oh = to_categorical(y_train, num_classes=num_classes).astype(np.float32)
    y_val_oh = to_categorical(y_val, num_classes=num_classes).astype(np.float32)

    w_kwargs = config.WORD_MODEL_KWARGS
    lstm_units = tuple(w_kwargs.get("lstm_units", (128, 64)))
    dense_units = tuple(w_kwargs.get("dense_units", (64,)))
    dropout = float(w_kwargs.get("dropout", 0.3))
    bidirectional = bool(w_kwargs.get("bidirectional", True))

    inputs = Input(shape=(seq_len, feat_dim), name="word_input")
    x = inputs
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        if bidirectional:
            x = Bidirectional(LSTM(units, return_sequences=return_sequences))(x)
        else:
            x = LSTM(units, return_sequences=return_sequences)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
    for units in dense_units:
        x = Dense(units, activation="relu")(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax", name="word_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="word_lstm")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda line: print(f"  {line}"))

    best_ckpt = model_out.parent / "word_model_best.keras"
    csv_path = config.DATA_DIR / "training_log_words.csv"
    curves_path = config.MODELS_DIR / "word_training_curves.png"

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
    print(f"[train_words] final val_accuracy = {final_val_acc:.4f}")
    print(f"[train_words] top-3 confused pairs (true->pred, count):")
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
        print(f"[train_words] training curves saved -> {curves_path}")
    except Exception as e:
        print(f"[train_words] could not plot training curves ({e}); skipping.")

    model.save(str(model_out), overwrite=True)
    print(f"[train_words] model saved -> {model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
