import json
import sys
import warnings
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config

try:
    from src.gesture_rules import recognize_isl_gesture
except ImportError:
    try:
        from gesture_rules import recognize_isl_gesture
    except ImportError:
        recognize_isl_gesture = None


class AlphabetRecognizer:
    """
    Static & Single-Frame ISL Alphabet / Number Recognizer.
    - Supports scikit-learn MLP/RandomForest and Keras models.
    - Features intelligent rule-based ISL fallback if no trained alphabet model is loaded.
    """
    def __init__(
        self,
        model_path: Path | str | None = None,
        labels_path: Path | str | None = None,
        scaler_path: Path | str | None = None,
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

        if model_path is None:
            model_path = config.ALPHABET_MODEL_PATH
        if labels_path is None:
            labels_path = config.ALPHABET_LABELS_PATH
        if scaler_path is None:
            scaler_path = config.MODELS_DIR / "scaler_alphabet.json"

        model_path = Path(model_path)
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
                pass

        if scaler_path.exists():
            try:
                with open(scaler_path, "r", encoding="utf-8") as f:
                    scaler = json.load(f)
                self._scaler_mean = np.asarray(scaler["mean"], dtype=np.float32)
                self._scaler_scale = np.asarray(scaler["scale"], dtype=np.float32)
            except Exception:
                pass

        joblib_path = config.MODELS_DIR / "alphabet_model.joblib"
        if joblib_path.exists():
            try:
                import joblib
                self._model = joblib.load(joblib_path)
                self._backend = "joblib"
                self._available = True
                self._status = f"Alphabet model ready ({self._num_classes} classes, scikit-learn)"
                return
            except Exception:
                pass

        if model_path.exists() and str(model_path).endswith(".keras"):
            try:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(str(model_path), compile=False)
                self._backend = "keras"
                self._available = True
                self._status = f"Alphabet model ready ({self._num_classes} classes, Keras)"
                return
            except Exception:
                pass

        if recognize_isl_gesture is not None:
            self._available = True
            self._backend = "rules"
            self._status = "Alphabet ready (ISL rule-based engine)"
        else:
            self._status = "Alphabet recognizer not loaded"

    @property
    def model_available(self) -> bool:
        return self._available

    @property
    def status_text(self) -> str:
        return self._status

    @property
    def class_names(self) -> list[str]:
        return list(self._idx_to_label.values())

    def predict(self, landmark_vector, threshold: float | None = None) -> tuple[str, float]:
        if not self._available or landmark_vector is None:
            return "", 0.0

        if threshold is None:
            threshold = config.ALPHABET_THRESHOLD

        vec = np.asarray(landmark_vector, dtype=np.float32).reshape(-1)
        if vec.shape != (config.LANDMARK_DIM,):
            return "", 0.0

        try:
            if self._backend == "joblib":
                feats = vec.reshape(1, -1)
                if self._scaler_mean is not None and self._scaler_scale is not None:
                    feats = (feats - self._scaler_mean) / np.maximum(self._scaler_scale, 1e-7)
                if hasattr(self._model, "predict_proba"):
                    probs = self._model.predict_proba(feats)[0]
                    best_idx = int(np.argmax(probs))
                    conf = float(probs[best_idx])
                    if conf >= threshold:
                        return self._idx_to_label.get(str(best_idx), ""), conf
                    return "", conf
                else:
                    pred = self._model.predict(feats)[0]
                    return self._idx_to_label.get(str(pred), ""), 1.0

            elif self._backend == "keras":
                inp = vec.reshape(1, -1)
                if self._scaler_mean is not None and self._scaler_scale is not None:
                    inp = (inp - self._scaler_mean) / np.maximum(self._scaler_scale, 1e-7)
                probs = self._model.predict(inp, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                conf = float(probs[best_idx])
                if conf >= threshold:
                    return self._idx_to_label.get(str(best_idx), ""), conf
                return "", conf

            elif self._backend == "rules" and recognize_isl_gesture is not None:
                gest, conf = recognize_isl_gesture(vec)
                if gest and conf >= threshold and len(gest) == 1:
                    return gest, conf
                return "", conf

            return "", 0.0

        except Exception as e:
            warnings.warn(f"Alphabet predict error: {e}")
            return "", 0.0
