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
from src.dataset_utils import discover_class_files, write_class_metadata

try:
    from src.hand_tracker import HandTracker
    _HT_OK = True
except Exception as _e:
    HandTracker = None
    _HT_OK = False
    _HT_ERR = str(_e)

try:
    import cv2
    _CV2_OK = True
except Exception as _e:
    cv2 = None
    _CV2_OK = False

try:
    from sklearn.model_selection import train_test_split
    _SKLEARN_OK = True
except Exception as _e:
    train_test_split = None
    _SKLEARN_OK = False


def _process_video(tracker: HandTracker, video_path: Path, seq_len: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
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

        sampled_indices = np.linspace(0, total_frames - 1, num=seq_len, dtype=int)
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
        return seq
    finally:
        cap.release()


def _process_image(tracker: HandTracker, image_path: Path, seq_len: int) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    vec_63, _annotated, detected = tracker.process_frame(image)
    vec = np.asarray(vec_63, dtype=np.float32)
    if vec.shape != (config.LANDMARK_DIM,):
        return None
    # If no hand was detected, return None so empty background isn't learned
    if not detected and np.all(vec == 0):
        return None
    # Generate sequence repeating with minor temporal variation
    return np.repeat(vec[np.newaxis, :], seq_len, axis=0)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess ISL word dataset into hand-landmark sequences.")
    p.add_argument("--data-dir", type=Path, default=config.DATASET_WORDS_DIR)
    p.add_argument("--out-dir", type=Path, default=config.LANDMARKS_DIR)
    p.add_argument("--labels-path", type=Path, default=config.WORD_LABELS_PATH)
    p.add_argument("--seq-len", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    if not _HT_OK:
        print(f"[preprocess_words] HandTracker unavailable: {_HT_ERR}")
        return 1

    data_dir = args.data_dir
    out_dir = args.out_dir
    labels_path = args.labels_path
    seq_len = int(args.seq_len)

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    # Automatically ensure dataset is setup
    if not data_dir.exists() or len(list(data_dir.iterdir())) == 0:
        print("[preprocess_words] dataset/words empty. Auto-running setup_isl_dataset...")
        from src.setup_isl_dataset import setup_isl_dataset
        setup_isl_dataset(verbose=False)

    class_entries = discover_class_files(data_dir, (*config.VIDEO_EXTS, *config.IMAGE_EXTS))
    if not class_entries:
        print(f"[preprocess_words] No class files found under {data_dir}")
        return 1

    class_names = [name for name, _dir, _files in class_entries]
    idx_to_label = {str(i): n for i, n in enumerate(class_names)}
    label_to_idx = {n: i for i, n in enumerate(class_names)}

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"idx_to_label": idx_to_label, "label_to_idx": label_to_idx}, f, indent=2)
    write_class_metadata(config.MODELS_DIR, "words", class_names)
    print(f"[preprocess_words] Discovered {len(class_names)} ISL word classes.")

    tracker = HandTracker(static_image_mode=True, max_num_hands=2)

    X: list[np.ndarray] = []
    y: list[int] = []
    per_class_counts: Counter = Counter()

    for class_idx, (class_name, class_dir, files) in enumerate(class_entries):
        kept_class = 0
        for fpath in files:
            if fpath.suffix.lower() in config.VIDEO_EXTS:
                seq = _process_video(tracker, fpath, seq_len)
            else:
                seq = _process_image(tracker, fpath, seq_len)
            if seq is not None:
                X.append(seq)
                y.append(class_idx)
                kept_class += 1
                # If only 1 or 2 images per class, augment with slight landmark variations to enable robust learning & validation
                if len(files) <= 3:
                    for _ in range(4):
                        noise = np.random.normal(0, 0.005, seq.shape).astype(np.float32)
                        X.append(seq + noise)
                        y.append(class_idx)
                        kept_class += 1
        per_class_counts[class_name] = kept_class

    if not X:
        print("[preprocess_words] No landmark features extracted. Check image/video data.")
        return 1

    X_arr = np.stack(X, axis=0).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    print(f"[preprocess_words] Extracted {len(X_arr)} total samples across {len(class_names)} classes.")

    # Train / Val split
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=config.SEED, stratify=y_arr
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=config.SEED
        )

    np.save(out_dir / "words_X_train.npy", X_train)
    np.save(out_dir / "words_y_train.npy", y_train)
    np.save(out_dir / "words_X_val.npy", X_val)
    np.save(out_dir / "words_y_val.npy", y_val)
    np.save(out_dir / "words_X.npy", X_arr)
    np.save(out_dir / "words_y.npy", y_arr)

    print(f"[preprocess_words] Saved arrays to {out_dir}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
