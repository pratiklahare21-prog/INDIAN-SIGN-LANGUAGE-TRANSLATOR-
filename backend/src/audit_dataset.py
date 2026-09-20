from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from dataset_utils import discover_class_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit detected ISL alphabet and word classes.")
    parser.add_argument("--words-dir", type=Path, default=config.DATASET_WORDS_DIR)
    parser.add_argument("--alphabet-dir", type=Path, default=config.DATASET_ALPHABET_DIR)
    args = parser.parse_args()

    archive_dir = config.BASE_DIR / "archive" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "Frames_Word_Level"
    roots = [
        ("alphabet", args.alphabet_dir, config.IMAGE_EXTS),
        ("words", args.words_dir, (*config.VIDEO_EXTS, *config.IMAGE_EXTS)),
    ]
    for name, root, extensions in roots:
        entries = discover_class_files(root, extensions)
        print(f"{name}: root={root}")
        print(f"{name}: classes={len(entries)}, samples={sum(len(files) for _label, _path, files in entries)}")
        print("  " + ", ".join(label for label, _path, _files in entries))

    if not discover_class_files(args.words_dir, (*config.VIDEO_EXTS, *config.IMAGE_EXTS)):
        entries = discover_class_files(archive_dir, config.IMAGE_EXTS)
        print(f"archive_words: root={archive_dir}")
        print(f"archive_words: classes={len(entries)}, samples={sum(len(files) for _label, _path, files in entries)}")
        print("  " + ", ".join(label for label, _path, _files in entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
