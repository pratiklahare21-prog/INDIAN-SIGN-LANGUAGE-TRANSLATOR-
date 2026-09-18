import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from src.dataset_utils import write_class_metadata

try:
    import cv2
    from sklearn.model_selection import train_test_split
    from src.hand_tracker import HandTracker
    _HT_OK = True
except Exception as exc:
    cv2 = None
    train_test_split = None
    HandTracker = None
    _HT_OK = False
    _HT_ERR = str(exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess ISL sentence videos into temporal landmark sequences.")
    parser.add_argument("--data-dir", type=Path, default=config.DATASET_SENTENCES_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.LANDMARKS_DIR)
    parser.add_argument("--labels-path", type=Path, default=config.SENTENCE_LABELS_PATH)
    parser.add_argument("--seq-len", type=int, default=30)
    return parser.parse_args()


def discover_sentence_videos(root: Path):
    entries = []
    if not root.exists():
        return []
    for class_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
        videos = sorted(
            (p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS),
            key=lambda p: p.name.casefold(),
        )
        if videos:
            entries.append((class_dir.name, videos))
    return entries


def process_video(tracker, video_path: Path, seq_len: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 0
            while True:
                ok, _f = capture.read()
                if not ok:
                    break
                total += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if total <= 0:
            return None

        # Sample frames evenly across video length
        sample_indices = np.linspace(0, total - 1, num=seq_len, dtype=int)
        vectors = []
        for idx in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = capture.read()
            if not ok or frame is None:
                vectors.append(np.zeros(config.LANDMARK_DIM, dtype=np.float32))
                continue
            vec_63, _annotated, _detected = tracker.process_frame(frame)
            vectors.append(np.asarray(vec_63, dtype=np.float32))

        if len(vectors) < seq_len:
            pad_vec = vectors[-1] if vectors else np.zeros(config.LANDMARK_DIM, dtype=np.float32)
            while len(vectors) < seq_len:
                vectors.append(pad_vec.copy())
        elif len(vectors) > seq_len:
            vectors = vectors[:seq_len]

        sequence = np.stack(vectors, axis=0).astype(np.float32)
        return sequence
    finally:
        capture.release()


def main():
    args = parse_args()
    if not _HT_OK:
        print(f"[preprocess_sentences] dependencies unavailable: {_HT_ERR}")
        return 1

    data_dir = args.data_dir
    out_dir = args.out_dir
    labels_path = args.labels_path
    seq_len = int(args.seq_len)

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    entries = discover_sentence_videos(data_dir)
    if not entries:
        # Fallback to archive location
        archive_dir = config.BASE_DIR / "archive" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "Videos_Sentence_Level"
        entries = discover_sentence_videos(archive_dir)

    if not entries:
        print(f"[preprocess_sentences] No sentence videos found under {data_dir}")
        return 1

    class_names = [name for name, _videos in entries]
    labels = {
        "idx_to_label": {str(index): name for index, name in enumerate(class_names)},
        "label_to_idx": {name: index for index, name in enumerate(class_names)},
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    write_class_metadata(config.MODELS_DIR, "sentences", class_names)
    print(f"[preprocess_sentences] Discovered {len(class_names)} ISL sentence classes.")

    tracker = HandTracker(static_image_mode=False, max_num_hands=2)
    sequences, targets = [], []
    counts = Counter()

    for class_index, (class_name, videos) in enumerate(entries):
        for video_path in videos:
            sequence = process_video(tracker, video_path, seq_len)
            if sequence is not None:
                sequences.append(sequence)
                targets.append(class_index)
                counts[class_name] += 1

    if len(sequences) < 2:
        print("[preprocess_sentences] Insufficient sequence samples extracted.")
        return 1

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int32)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=config.SEED, stratify=y
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=config.SEED
        )

    for name, array in {
        "sentences_X": X,
        "sentences_y": y,
        "sentences_X_train": X_train,
        "sentences_y_train": y_train,
        "sentences_X_val": X_val,
        "sentences_y_val": y_val,
    }.items():
        np.save(out_dir / f"{name}.npy", array)

    print(f"[preprocess_sentences] Saved sentences dataset:")
    print(f"  Total: {X.shape}, Train: {X_train.shape}, Val: {X_val.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
