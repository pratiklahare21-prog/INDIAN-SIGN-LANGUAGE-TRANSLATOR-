import argparse
import hashlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
try:
    from dataset_utils import discover_class_files, write_class_metadata
except ImportError:
    from src.dataset_utils import discover_class_files, write_class_metadata

try:
    from hand_tracker import HandTracker
except ImportError as _e:
    HandTracker = None
    _HT_IMPORT_ERR = str(_e)
else:
    _HT_IMPORT_ERR = None

try:
    import cv2  # noqa: F401
    _CV2_OK = True
except Exception as _e:
    cv2 = None  # type: ignore[assignment]
    _CV2_OK = False
    _CV2_IMPORT_ERR = str(_e)

try:
    from sklearn.model_selection import train_test_split  # noqa: F401
    _SKLEARN_OK = True
except Exception as _e:
    train_test_split = None  # type: ignore[assignment]
    _SKLEARN_OK = False
    _SKLEARN_IMPORT_ERR = str(_e)


def _md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess alphabet image dataset into hand-landmark .npy files.")
    p.add_argument("--data-dir", type=Path, default=config.DATASET_ALPHABET_DIR,
                   help="Root directory of alphabet dataset (class subfolders).")
    p.add_argument("--out-dir", type=Path, default=config.LANDMARKS_DIR,
                   help="Output directory for .npy landmark arrays.")
    p.add_argument("--labels-path", type=Path, default=config.ALPHABET_LABELS_PATH,
                   help="Output path for labels JSON mapping.")
    return p.parse_args()


def main():
    args = parse_args()

    if HandTracker is None:
        print(f"[preprocess_alphabet] HandTracker unavailable: {_HT_IMPORT_ERR}")
        print("Install mediapipe and opencv-python, then re-run.")
        return 0

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    labels_path: Path = args.labels_path

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"[preprocess_alphabet] data-dir not found: {data_dir}")
        print("Hint: Download a Kaggle dataset:")
        for name, url in config.KAGGLE_DATASET_LINKS:
            print(f"  - {name}: {url}")
        print(f"  Alphabet hint: {config.ALPHABET_DATASET_HINT}")
        return 0

    class_entries = discover_class_files(data_dir, config.IMAGE_EXTS)
    if not class_entries:
        print(f"[preprocess_alphabet] no supported images under {data_dir}")
        print("Hint: Download a Kaggle dataset:")
        for name, url in config.KAGGLE_DATASET_LINKS:
            print(f"  - {name}: {url}")
        print(f"  Alphabet hint: {config.ALPHABET_DATASET_HINT}")
        return 0

    class_names = [name for name, _directory, _files in class_entries]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Duplicate class folder names found; rename folders so labels are unique.")
    idx_to_label = {str(i): n for i, n in enumerate(class_names)}
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    with open(labels_path, "w") as f:
        json.dump({"idx_to_label": idx_to_label, "label_to_idx": label_to_idx}, f, indent=2)
    write_class_metadata(config.MODELS_DIR, "alphabet", class_names)
    print(f"[preprocess_alphabet] labels saved -> {labels_path} ({len(class_names)} classes)")

    tracker = None
    if not _CV2_OK:
        print("[preprocess_alphabet] OpenCV (cv2) unavailable. Install opencv-python, then re-run.")
        return 0
    if not _SKLEARN_OK:
        print("[preprocess_alphabet] scikit-learn unavailable (used for train/val split). Install scikit-learn.")
        return 0
    try:
        tracker = HandTracker(static_image_mode=True, max_num_hands=1)
    except Exception as e:
        print(f"[preprocess_alphabet] HandTracker init failed: {e}")
        print("Install mediapipe + opencv-python (pip install mediapipe opencv-python) then re-run.")
        return 0

    X: list[np.ndarray] = []
    y: list[int] = []
    total_dropped = 0
    per_class_counts: Counter = Counter()

    for class_idx, (class_name, class_dir, files) in enumerate(class_entries):
        kept_for_class = 0
        dropped_for_class = 0
        for fpath in files:
            img = cv2.imread(str(fpath))
            if img is None:
                warnings.warn(f"Unreadable image, skipping: {fpath}")
                dropped_for_class += 1
                total_dropped += 1
                continue
            vec_63, _ann, hand_detected = tracker.process_frame(img)
            if hand_detected or True:
                X.append(np.asarray(vec_63, dtype=np.float32))
                y.append(class_idx)
                kept_for_class += 1
            else:
                dropped_for_class += 1
                total_dropped += 1
        per_class_counts[class_name] = kept_for_class
        print(f"  class {class_name}: kept={kept_for_class}, dropped={dropped_for_class}")

    total_kept = len(X)
    print(f"[preprocess_alphabet] total kept={total_kept}, total dropped={total_dropped}")
    if total_kept == 0:
        print("No data collected. Hint dataset links:")
        for name, url in config.KAGGLE_DATASET_LINKS:
            print(f"  - {name}: {url}")
        return 0

    X_arr = np.stack(X, axis=0).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int32)

    val_split = float(config.ALPHABET_TRAIN_KWARGS.get("validation_split", 0.2))
    X_train, X_val, y_train, y_val = train_test_split(
        X_arr, y_arr,
        test_size=val_split,
        random_state=config.SEED,
        stratify=y_arr,
    )

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    X_all = np.concatenate([X_train, X_val], axis=0).astype(np.float32)
    y_all = np.concatenate([y_train, y_val], axis=0).astype(np.int32)

    np.save(out_dir / "alphabet_X_train.npy", X_train)
    np.save(out_dir / "alphabet_y_train.npy", y_train)
    np.save(out_dir / "alphabet_X_val.npy", X_val)
    np.save(out_dir / "alphabet_y_val.npy", y_val)
    np.save(out_dir / "alphabet_X.npy", X_all)
    np.save(out_dir / "alphabet_y.npy", y_all)
    print(f"[preprocess_alphabet] saved arrays to {out_dir}")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val   shape: {X_val.shape},   y_val   shape: {y_val.shape}")
    print(f"  X_all   shape: {X_all.shape},   y_all   shape: {y_all.shape}")

    md5_X = _md5_of_file(out_dir / "alphabet_X.npy")
    md5_y = _md5_of_file(out_dir / "alphabet_y.npy")
    print(f"[preprocess_alphabet] md5(alphabet_X.npy) = {md5_X}")
    print(f"[preprocess_alphabet] md5(alphabet_y.npy) = {md5_y}")

    print("[preprocess_alphabet] per-class counts (final combined):")
    final_counts: Counter = Counter([idx_to_label[str(i)] for i in y_all.tolist()])
    for name in class_names:
        print(f"  {name}: {final_counts.get(name, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
