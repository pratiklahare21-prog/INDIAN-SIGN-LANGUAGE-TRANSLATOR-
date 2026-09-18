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
    import cv2
    _CV2_OK = True
except Exception as _e:
    cv2 = None  # type: ignore[assignment]
    _CV2_OK = False
    _CV2_IMPORT_ERR = str(_e)

try:
    from sklearn.model_selection import train_test_split
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
    p = argparse.ArgumentParser(description="Preprocess word video dataset into hand-landmark sequence .npy files.")
    p.add_argument("--data-dir", type=Path, default=config.DATASET_WORDS_DIR,
                   help="Root directory of word video dataset (class subfolders).")
    p.add_argument("--out-dir", type=Path, default=config.LANDMARKS_DIR,
                   help="Output directory for .npy landmark arrays.")
    p.add_argument("--labels-path", type=Path, default=config.WORD_LABELS_PATH,
                   help="Output path for labels JSON mapping.")
    p.add_argument("--seq-len", type=int, default=config.SEQUENCE_LENGTH,
                   help="Target sequence length (frames) per video.")
    return p.parse_args()


def _process_video(tracker: HandTracker, video_path: Path, seq_len: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"Cannot open video: {video_path}")
        return None
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0
            while True:
                ok, _f = cap.read()
                if not ok:
                    break
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if total_frames <= 0:
            return None

        if total_frames >= seq_len:
            sampled_indices = np.linspace(0, total_frames - 1, num=seq_len, dtype=int)
        else:
            sampled_indices = np.arange(total_frames, dtype=int)

        vecs: list[np.ndarray] = []
        for idx in sampled_indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                vecs.append(np.zeros(config.LANDMARK_DIM, dtype=np.float32))
                continue
            vec_63, _ann, _det = tracker.process_frame(frame)
            vecs.append(np.asarray(vec_63, dtype=np.float32))

        if len(vecs) < seq_len:
            pad_vec = vecs[-1] if vecs else np.zeros(config.LANDMARK_DIM, dtype=np.float32)
            while len(vecs) < seq_len:
                vecs.append(pad_vec.copy())
        elif len(vecs) > seq_len:
            vecs = vecs[:seq_len]

        seq = np.stack(vecs, axis=0).astype(np.float32)
        assert seq.shape == (seq_len, config.LANDMARK_DIM)
        return seq
    finally:
        cap.release()


def _process_image(tracker: HandTracker, image_path: Path, seq_len: int) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        warnings.warn(f"Unreadable image, skipping: {image_path}")
        return None
    vec_63, _annotated, _detected = tracker.process_frame(image)
    vec = np.asarray(vec_63, dtype=np.float32)
    if vec.shape != (config.LANDMARK_DIM,):
        return None
    return np.repeat(vec[np.newaxis, :], seq_len, axis=0)


def main():
    args = parse_args()

    if HandTracker is None:
        print(f"[preprocess_words] HandTracker unavailable: {_HT_IMPORT_ERR}")
        print("Install mediapipe and opencv-python, then re-run.")
        return 0

    data_dir: Path = args.data_dir
    bundled_archive = (
        config.BASE_DIR / "archive" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "Frames_Word_Level"
    )
    out_dir: Path = args.out_dir
    labels_path: Path = args.labels_path
    seq_len: int = int(args.seq_len)

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"[preprocess_words] data-dir not found: {data_dir}")
        print("Hint: Download a Kaggle dataset:")
        for name, url in config.KAGGLE_DATASET_LINKS:
            print(f"  - {name}: {url}")
        return 0

    class_entries = discover_class_files(data_dir, (*config.VIDEO_EXTS, *config.IMAGE_EXTS))
    if not class_entries and data_dir.resolve() == config.DATASET_WORDS_DIR.resolve():
        archive_entries = discover_class_files(bundled_archive, config.IMAGE_EXTS)
        if archive_entries:
            data_dir = bundled_archive
            class_entries = archive_entries
            print(f"[preprocess_words] dataset/words is empty; using bundled frame corpus: {data_dir}")
    if not class_entries:
        print(f"[preprocess_words] no supported media under {data_dir}")
        print("Hint: Download a Kaggle dataset:")
        for name, url in config.KAGGLE_DATASET_LINKS:
            print(f"  - {name}: {url}")
        return 0

    class_names = [name for name, _directory, _files in class_entries]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Duplicate class folder names found; rename folders so labels are unique.")
    idx_to_label = {str(i): n for i, n in enumerate(class_names)}
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    with open(labels_path, "w") as f:
        json.dump({"idx_to_label": idx_to_label, "label_to_idx": label_to_idx}, f, indent=2)
    write_class_metadata(config.MODELS_DIR, "words", class_names)
    print(f"[preprocess_words] labels saved -> {labels_path} ({len(class_names)} classes)")

    tracker = None
    if not _CV2_OK:
        print("[preprocess_words] OpenCV (cv2) unavailable. Install opencv-python, then re-run.")
        return 0
    if not _SKLEARN_OK:
        print("[preprocess_words] scikit-learn unavailable (used for train/val split). Install scikit-learn.")
        return 0
    try:
        tracker = HandTracker(static_image_mode=False, max_num_hands=1)
    except Exception as e:
        print(f"[preprocess_words] HandTracker init failed: {e}")
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
            if fpath.suffix.lower() in config.VIDEO_EXTS:
                seq = _process_video(tracker, fpath, seq_len)
            else:
                seq = _process_image(tracker, fpath, seq_len)
            if seq is None:
                dropped_for_class += 1
                total_dropped += 1
                continue
            X.append(seq)
            y.append(class_idx)
            kept_for_class += 1
        per_class_counts[class_name] = kept_for_class
        print(f"  class {class_name}: kept={kept_for_class}, dropped={dropped_for_class}")

    total_kept = len(X)
    print(f"[preprocess_words] total kept={total_kept}, total dropped={total_dropped}")
    if total_kept == 0:
        print("No data collected. Hint dataset links:")
        for name, url in config.KAGGLE_DATASET_LINKS:
            print(f"  - {name}: {url}")
        return 0

    X_arr = np.stack(X, axis=0).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    assert X_arr.shape[1:] == (seq_len, config.LANDMARK_DIM), f"X shape: {X_arr.shape}"

    val_split = float(config.WORD_TRAIN_KWARGS.get("validation_split", 0.2))
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

    np.save(out_dir / "words_X_train.npy", X_train)
    np.save(out_dir / "words_y_train.npy", y_train)
    np.save(out_dir / "words_X_val.npy", X_val)
    np.save(out_dir / "words_y_val.npy", y_val)
    np.save(out_dir / "words_X.npy", X_all)
    np.save(out_dir / "words_y.npy", y_all)
    print(f"[preprocess_words] saved arrays to {out_dir}")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val   shape: {X_val.shape},   y_val   shape: {y_val.shape}")
    print(f"  X_all   shape: {X_all.shape},   y_all   shape: {y_all.shape}")

    md5_X = _md5_of_file(out_dir / "words_X.npy")
    print(f"[preprocess_words] md5(words_X.npy) = {md5_X}")

    print("[preprocess_words] per-class counts (final combined):")
    final_counts: Counter = Counter([idx_to_label[str(i)] for i in y_all.tolist()])
    for name in class_names:
        print(f"  {name}: {final_counts.get(name, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
