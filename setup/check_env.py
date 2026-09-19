from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

REQUIRED_DIRS = [
    config.DATASET_DIR,
    config.DATASET_ALPHABET_DIR,
    config.DATASET_WORDS_DIR,
    config.MODELS_DIR,
    config.DATA_DIR,
    config.LANDMARKS_DIR,
]

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("tensorflow", "tensorflow"),
    ("pyttsx3", "pyttsx3"),
    ("streamlit", "streamlit"),
]

SUGGESTED_PY = ((3, 10), (3, 11), (3, 12))


def check_python() -> tuple[bool, str]:
    v = sys.version_info[:2]
    ok = v in SUGGESTED_PY
    msg = f"Python {v[0]}.{v[1]} (recommended: 3.10-3.12)"
    if not ok:
        msg += " [UNSUPPORTED: TensorFlow 2.16 may fail on this version]"
    return ok, msg


def check_packages() -> list[tuple[str, bool, str]]:
    out = []
    for imp_name, pip_name in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(imp_name)
        ok = spec is not None
        out.append((pip_name, ok, "installed" if ok else "MISSING (run: pip install -r requirements.txt)"))
    return out


def check_dirs() -> list[tuple[str, bool]]:
    out = []
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out.append((str(d.relative_to(ROOT)), d.exists()))
    return out


def dataset_summary(root: Path) -> dict[str, int]:
    from src.dataset_utils import count_classes
    archive_root = config.BASE_DIR / "archive" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "Frames_Word_Level"
    sentence_root = config.DATASET_SENTENCES_DIR
    return {
        "alphabet": count_classes(config.DATASET_ALPHABET_DIR, config.IMAGE_EXTS),
        "words": count_classes(config.DATASET_WORDS_DIR, (*config.VIDEO_EXTS, *config.IMAGE_EXTS)),
        "archive_words": count_classes(archive_root, config.IMAGE_EXTS),
        "sentences": len([p for p in sentence_root.iterdir() if p.is_dir()]) if sentence_root.is_dir() else 0,
    }


def model_summary() -> dict[str, bool]:
    word_joblib = config.MODELS_DIR / "word_model.joblib"
    sent_joblib = config.MODELS_DIR / "sentence_model.joblib"
    return {
        "alphabet_model": config.ALPHABET_MODEL_PATH.is_file(),
        "word_model": config.WORD_MODEL_PATH.is_file() or word_joblib.is_file(),
        "labels_alphabet": config.ALPHABET_LABELS_PATH.is_file(),
        "labels_word": config.WORD_LABELS_PATH.is_file(),
        "sentence_model": config.SENTENCE_MODEL_PATH.is_file() or sent_joblib.is_file(),
        "labels_sentences": config.SENTENCE_LABELS_PATH.is_file(),
    }


def next_steps(stats: dict[str, int], models: dict[str, bool]) -> list[str]:
    steps = []
    if importlib.util.find_spec("tensorflow") is None:
        steps.append("1. Install dependencies: pip install -r requirements.txt")
    if stats["words"] == 0:
        if stats["archive_words"]:
            steps.append(
                f"2. Bundled frame corpus detected ({stats['archive_words']} classes): "
                "run python src\\preprocess_words.py to use it."
            )
        else:
            steps.append(
                "2. Download Kaggle word dataset: "
                "https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset"
                "  -> extract into dataset\\words\\<ClassName>\\<videos>..."
            )
    if stats["sentences"] and not models["sentence_model"]:
        steps.append("3. Preprocess+train sentence videos: python src\\preprocess_sentences.py && python src\\train_sentences.py")
    if stats["alphabet"] == 0:
        steps.append(
            "3. Put A-Z images into dataset\\alphabet\\A\\, ...\\Z\\ (>= 10 images each)."
        )
    if stats["alphabet"] > 0 and not models["alphabet_model"]:
        steps.append("4. Preprocess+train alphabet: python src\\preprocess_alphabet.py && python src\\train_alphabet.py")
    if stats["words"] > 0 and not models["word_model"]:
        steps.append("5. Preprocess+train words: python src\\preprocess_words.py && python src\\train_words.py")
    if models["alphabet_model"] and models["word_model"]:
        steps.append("*. Models found. Launch: run.bat  OR  streamlit run app.py")
    return steps


def main() -> int:
    print("=" * 64)
    print("  ISL Sign2Speech  Environment Check")
    print("=" * 64)

    py_ok, py_msg = check_python()
    print(f"\n[Python] ", py_msg)

    print("\n[Packages]")
    for pkg, ok, info in check_packages():
        print(f"  - {pkg:<18} {info}")

    print("\n[Folders] (auto-created if missing)")
    for name, ok in check_dirs():
        print(f"  - {name:<22} {'OK' if ok else 'MISSING'}")

    stats = dataset_summary(ROOT)
    print("\n[Dataset class counts]")
    print(f"  - alphabet classes: {stats['alphabet']}  (want 26)")
    print(f"  - word classes:     {stats['words']}     (from dataset/words)")
    print(f"  - archive classes:  {stats['archive_words']} (from bundled Frames_Word_Level)")
    print(f"  - sentence folders: {stats['sentences']} (from Videos_Sentence_Level)")

    models = model_summary()
    print("\n[Models]")
    for k, v in models.items():
        print(f"  - {k:<20} {'present' if v else 'ABSENT'}")

    print("\n[Next steps]")
    for s in next_steps(stats, models):
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
