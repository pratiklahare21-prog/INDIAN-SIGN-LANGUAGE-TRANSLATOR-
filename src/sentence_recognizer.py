import json
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.feature_extraction import extract_sequence_features


class SentenceRecognizer:
    """
    Real-time ISL Continuous Sentence Recognizer.
    - Captures continuous gesture sequences over a sliding window.
    - Classifies full Indian Sign Language continuous sentences.
    - Returns recognized sentence gloss/phrase and confidence.
    """
    def __init__(
        self,
        model_path: Path | str | None = None,
        labels_path: Path | str | None = None,
        scaler_path: Path | str | None = None,
        seq_len: int | None = None,
    ):
        self._available = False
        self._status = "Initializing..."
        self._model = None
        self._backend = "none"
        self._idx_to_label: dict[str, str] = {}
        self._label_to_idx: dict[str, int] = {}
        self._scaler_mean = None
        self._scaler_scale = None
        self._num_classes = 0
        self._seq_len: int = int(seq_len) if seq_len is not None else 30
        self._buffer: deque = deque(maxlen=self._seq_len)
        self._recent_probs: deque = deque(maxlen=5)

        if labels_path is None:
            labels_path = config.SENTENCE_LABELS_PATH
        if scaler_path is None:
            scaler_path = config.MODELS_DIR / "scaler_sentences.json"

        labels_path = Path(labels_path)
        scaler_path = Path(scaler_path)

        if labels_path.exists():
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                self._idx_to_label = {str(k): v for k, v in labels["idx_to_label"].items()}
                self._label_to_idx = {k: int(v) for k, v in labels["label_to_idx"].items()}
                self._num_classes = len(self._idx_to_label)
            except Exception as e:
                self._status = f"Failed loading sentence labels: {e}"
                return
        else:
            self._status = f"Missing sentence labels: {labels_path}"
            return

        if scaler_path.exists():
            try:
                with open(scaler_path, "r", encoding="utf-8") as f:
                    scaler = json.load(f)
                self._scaler_mean = np.asarray(scaler["mean"], dtype=np.float32)
                self._scaler_scale = np.asarray(scaler["scale"], dtype=np.float32)
            except Exception:
                self._scaler_mean = None
                self._scaler_scale = None

        joblib_path = config.MODELS_DIR / "sentence_model.joblib" if model_path is None else Path(model_path)
        keras_path = config.MODELS_DIR / "sentence_model.keras"

        if joblib_path.exists() and str(joblib_path).endswith((".joblib", ".pkl")):
            try:
                import joblib
                self._model = joblib.load(joblib_path)
                self._backend = "joblib"
                self._available = True
                self._status = f"Sentence model ready ({self._num_classes} classes, scikit-learn)"
                return
            except Exception as e:
                self._status = f"Failed loading sentence joblib model: {e}"

        if keras_path.exists() or (model_path and str(model_path).endswith(".keras")):
            try:
                import tensorflow as tf
                target_k = keras_path if model_path is None else Path(model_path)
                self._model = tf.keras.models.load_model(str(target_k), compile=False)
                self._backend = "keras"
                self._available = True
                self._status = f"Sentence model ready ({self._num_classes} classes, Keras)"
                return
            except Exception as e:
                self._status = f"Keras sentence model error: {e}"

        if not self._available:
            self._status = f"Sentence model not trained yet ({self._num_classes} classes indexed)"

    @property
    def model_available(self) -> bool:
        return self._available

    @property
    def status_text(self) -> str:
        return self._status

    @property
    def class_names(self) -> list[str]:
        return list(self._idx_to_label.values())

    def update(self, vector) -> None:
        if vector is None:
            return
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape == (config.LANDMARK_DIM,):
            self._buffer.append(vec.copy())

    def reset(self) -> None:
        self._buffer.clear()
        self._recent_probs.clear()

    def predict(self, threshold: float | None = None) -> tuple[str, float]:
        if not self._available or len(self._buffer) < max(10, int(self._seq_len * 0.5)):
            return "", 0.0

        if threshold is None:
            threshold = config.SENTENCE_THRESHOLD

        seq_arr = np.asarray(self._buffer, dtype=np.float32)
        if len(seq_arr) < self._seq_len:
            pad = np.repeat(seq_arr[-1:], self._seq_len - len(seq_arr), axis=0)
            seq_arr = np.concatenate([seq_arr, pad], axis=0)

        try:
            if self._backend == "joblib":
                feats = extract_sequence_features(seq_arr).reshape(1, -1)
                if self._scaler_mean is not None and self._scaler_scale is not None:
                    feats = (feats - self._scaler_mean) / np.maximum(self._scaler_scale, 1e-7)

                if hasattr(self._model, "predict_proba"):
                    probs = self._model.predict_proba(feats)[0]
                else:
                    pred_idx = self._model.predict(feats)[0]
                    return self._idx_to_label.get(str(pred_idx), ""), 1.0

            elif self._backend == "keras":
                inp = seq_arr[np.newaxis, ...]
                probs = self._model.predict(inp, verbose=0)[0]
            else:
                return "", 0.0

            self._recent_probs.append(probs)
            smoothed_probs = np.mean(self._recent_probs, axis=0)

            best_idx = int(np.argmax(smoothed_probs))
            conf = float(smoothed_probs[best_idx])

            if conf >= threshold:
                label = self._idx_to_label.get(str(best_idx), "")
                return label, conf
            return "", conf

        except Exception as e:
            warnings.warn(f"SentenceRecognizer predict error: {e}")
            return "", 0.0
