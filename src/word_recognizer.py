import json
import sys
import warnings
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config


class WordRecognizer:
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
        self._idx_to_label: dict[str, str] = {}
        self._label_to_idx: dict[str, int] = {}
        self._scaler_mean = None
        self._scaler_scale = None
        self._num_classes = 0
        self._seq_len: int = int(seq_len) if seq_len is not None else int(config.SEQUENCE_LENGTH)
        self._buffer: deque = deque(maxlen=self._seq_len)

        if model_path is None:
            model_path = config.WORD_MODEL_PATH
        if labels_path is None:
            labels_path = config.WORD_LABELS_PATH
        if scaler_path is None:
            scaler_path = config.MODELS_DIR / "scaler_words.json"

        model_path = Path(model_path)
        labels_path = Path(labels_path)
        scaler_path = Path(scaler_path)

        try:
            import numpy as np
            self._np = np
        except Exception as e:
            self._status = f"NumPy unavailable: {e}"
            return

        try:
            import tensorflow as tf
            self._tf = tf
        except Exception as e:
            self._status = f"TensorFlow unavailable: {e}. Train the word model first."
            return

        if not model_path.exists():
            self._status = (
                "Not trained: run preprocess_words.py + train_words.py "
                f"(missing model: {model_path})"
            )
            return

        if not labels_path.exists():
            self._status = (
                "Not trained: run preprocess_words.py "
                f"(missing labels: {labels_path})"
            )
            return

        try:
            with open(labels_path, "r") as f:
                labels = json.load(f)
            self._idx_to_label = {str(k): v for k, v in labels["idx_to_label"].items()}
            self._label_to_idx = {k: int(v) for k, v in labels["label_to_idx"].items()}
            self._num_classes = len(self._idx_to_label)
        except Exception as e:
            self._status = f"Failed to load labels: {e}"
            return

        if scaler_path.exists():
            try:
                with open(scaler_path, "r") as f:
                    scaler = json.load(f)
                self._scaler_mean = np.asarray(scaler["mean"], dtype=np.float32)
                self._scaler_scale = np.asarray(scaler["scale"], dtype=np.float32)
            except Exception as e:
                warnings.warn(f"WordRecognizer: failed to load scaler ({e}); proceeding without scaling.")
                self._scaler_mean = None
                self._scaler_scale = None

        try:
            self._model = tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as e:
            self._status = f"Failed to load model: {e}"
            return

        try:
            output_classes = int(self._model.output_shape[-1])
            if output_classes != self._num_classes:
                self._status = f"Model/labels mismatch: model has {output_classes} outputs, labels contain {self._num_classes}. Retrain."
                return
        except Exception as e:
            self._status = f"Could not validate model output classes: {e}"
            return

        self._available = True
        self._status = "Word model ready"

    @property
    def model_available(self) -> bool:
        return self._available

    @property
    def status_text(self) -> str:
        return self._status

    def update(self, vec_63) -> None:
        try:
            np = self._np
            if vec_63 is None:
                return
            try:
                vec = np.asarray(vec_63, dtype=np.float32)
            except Exception:
                return
            if vec.shape != (config.LANDMARK_DIM,):
                return
            self._buffer.append(vec.copy())
        except Exception as e:
            warnings.warn(f"WordRecognizer.update failed: {e}")

    def _ready(self) -> bool:
        return len(self._buffer) == self._seq_len

    def reset(self) -> None:
        try:
            self._buffer.clear()
        except Exception:
            try:
                self._buffer = deque(maxlen=self._seq_len)
            except Exception:
                pass

    def predict(self, threshold: float | None = None):
        try:
            np = self._np
            if threshold is None:
                threshold = float(config.WORD_THRESHOLD)

            if not self._available:
                return ("", 0.0)

            if not self._ready():
                return ("", 0.0)

            try:
                seq = np.stack(list(self._buffer), axis=0).astype(np.float32)
            except Exception:
                return ("", 0.0)

            if seq.shape != (self._seq_len, config.LANDMARK_DIM):
                return ("", 0.0)

            if self._scaler_mean is not None and self._scaler_scale is not None:
                flat = seq.reshape(-1, config.LANDMARK_DIM)
                denom = np.where(self._scaler_scale == 0, 1.0, self._scaler_scale)
                flat_s = (flat - self._scaler_mean) / denom
                seq = flat_s.reshape(self._seq_len, config.LANDMARK_DIM).astype(np.float32)

            batch = seq.reshape(1, self._seq_len, config.LANDMARK_DIM)
            try:
                probs = self._model.predict(batch, verbose=0)
            except Exception:
                probs = self._model(batch, training=False).numpy()
            probs = np.asarray(probs, dtype=np.float32).reshape(-1)
            if probs.size == 0:
                return ("", 0.0)

            max_idx = int(np.argmax(probs))
            max_prob = float(probs[max_idx])
            if max_prob < threshold:
                return ("", max_prob)

            label = self._idx_to_label.get(str(max_idx), "")
            return (label, max_prob)
        except Exception as e:
            warnings.warn(f"WordRecognizer.predict failed: {e}")
            return ("", 0.0)


if __name__ == "__main__":
    import numpy as _np
    rec = WordRecognizer()
    z = _np.zeros(config.LANDMARK_DIM, dtype=_np.float32)
    for _ in range(65):
        rec.update(z)
    out = rec.predict()
    assert isinstance(out, tuple) and len(out) == 2, f"unexpected return: {out}"
    label, score = out
    assert isinstance(label, str) and isinstance(score, (int, float))
    rec.reset()
    after_reset = rec.predict()
    assert isinstance(after_reset, tuple) and len(after_reset) == 2
    assert after_reset[0] == "" and after_reset[1] == 0.0
    print("WordRecognizer smoke PASSED")
