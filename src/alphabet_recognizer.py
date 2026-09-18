import json
import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config


class AlphabetRecognizer:
    def __init__(
        self,
        model_path: Path | str | None = None,
        labels_path: Path | str | None = None,
        scaler_path: Path | str | None = None,
    ):
        self._available = False
        self._status = "Initializing..."
        self._model = None
        self._idx_to_label: dict[str, str] = {}
        self._label_to_idx: dict[str, int] = {}
        self._scaler_mean = None
        self._scaler_scale = None
        self._num_classes = 0

        if model_path is None:
            model_path = config.ALPHABET_MODEL_PATH
        if labels_path is None:
            labels_path = config.ALPHABET_LABELS_PATH
        if scaler_path is None:
            scaler_path = config.MODELS_DIR / "scaler_alphabet.json"

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
            self._status = f"TensorFlow unavailable: {e}. Train the alphabet model first."
            return

        if not model_path.exists():
            self._status = (
                "Not trained: run preprocess_alphabet.py + train_alphabet.py "
                f"(missing model: {model_path})"
            )
            return

        if not labels_path.exists():
            self._status = (
                "Not trained: run preprocess_alphabet.py "
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
                warnings.warn(f"AlphabetRecognizer: failed to load scaler ({e}); proceeding without scaling.")
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
        self._status = "Alphabet model ready"

    @property
    def model_available(self) -> bool:
        return self._available

    @property
    def status_text(self) -> str:
        return self._status

    def predict(self, vec_63, threshold: float | None = None):
        try:
            np = self._np
            if threshold is None:
                threshold = float(config.ALPHABET_THRESHOLD)

            if vec_63 is None:
                return ("", 0.0)

            try:
                vec = np.asarray(vec_63, dtype=np.float32)
            except Exception:
                return ("", 0.0)

            if vec.shape != (config.LANDMARK_DIM,):
                return ("", 0.0)

            if np.allclose(vec, 0.0):
                return ("", 0.0)

            if not self._available:
                return ("", 0.0)

            x = vec
            if self._scaler_mean is not None and self._scaler_scale is not None:
                denom = np.where(self._scaler_scale == 0, 1.0, self._scaler_scale)
                x = (x - self._scaler_mean) / denom
                x = x.astype(np.float32)

            batch = x.reshape(1, config.LANDMARK_DIM)
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
            warnings.warn(f"AlphabetRecognizer.predict failed: {e}")
            return ("", 0.0)


if __name__ == "__main__":
    import numpy as _np
    rec = AlphabetRecognizer()
    out = rec.predict(_np.zeros(config.LANDMARK_DIM, dtype=_np.float32))
    assert isinstance(out, tuple) and len(out) == 2, f"unexpected return: {out}"
    label, score = out
    assert isinstance(label, str) and isinstance(score, (int, float))
    print("AlphabetRecognizer smoke PASSED (model may be absent)")
