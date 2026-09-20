import json
import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

def setup_isl_dataset(verbose: bool = True):
    """
    Sets up and indexes the authentic Indian Sign Language dataset (ISL-CSLRT).
    - Links/copies/syncs words and sentences to dataset/words and dataset/sentences.
    - Parses ISL sign gloss to spoken sentence mappings.
    - Generates class registries.
    """
    corpus_root = config.BASE_DIR / "archive" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus"
    if not corpus_root.exists():
        # Search fallback
        candidates = list((config.BASE_DIR / "archive").rglob("Frames_Word_Level"))
        if candidates:
            corpus_root = candidates[0].parent
        else:
            raise FileNotFoundError(f"Could not find ISL_CSLRT_Corpus under {config.BASE_DIR / 'archive'}")

    words_src = corpus_root / "Frames_Word_Level"
    sentences_src = corpus_root / "Videos_Sentence_Level"
    csv_dir = corpus_root / "corpus_csv_files"

    dataset_words_dir = config.DATASET_WORDS_DIR
    dataset_sentences_dir = config.DATASET_DIR / "sentences"
    dataset_words_dir.mkdir(parents=True, exist_ok=True)
    dataset_sentences_dir.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Setup Word Classes
    word_classes = []
    if words_src.exists():
        for item in sorted(words_src.iterdir()):
            if item.is_dir():
                target_word_dir = dataset_words_dir / item.name
                target_word_dir.mkdir(parents=True, exist_ok=True)
                imgs = [p for p in item.iterdir() if p.suffix.lower() in config.IMAGE_EXTS or p.suffix.lower() in config.VIDEO_EXTS]
                for img in imgs:
                    dest = target_word_dir / img.name
                    if not dest.exists():
                        shutil.copy2(img, dest)
                word_classes.append(item.name)

    # 2. Setup Sentence Classes
    sentence_classes = []
    if sentences_src.exists():
        for item in sorted(sentences_src.iterdir()):
            if item.is_dir():
                target_sent_dir = dataset_sentences_dir / item.name
                target_sent_dir.mkdir(parents=True, exist_ok=True)
                vids = [p for p in item.iterdir() if p.suffix.lower() in config.VIDEO_EXTS]
                for vid in vids:
                    dest = target_sent_dir / vid.name
                    if not dest.exists():
                        shutil.copy2(vid, dest)
                sentence_classes.append(item.name)

    # 3. Parse Gloss-to-Sentence Mappings
    gloss_map = {}
    csv_file = csv_dir / "ISL Corpus sign glosses.csv"
    if csv_file.exists():
        try:
            import csv
            with open(csv_file, "r", encoding="utf-8-sig", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        spoken_sent = row[0].strip()
                        gloss = row[1].strip()
                        if spoken_sent and gloss:
                            norm_gloss = " ".join(gloss.upper().split())
                            gloss_map[norm_gloss] = spoken_sent
                            clean_gloss = "".join(c for c in norm_gloss if c.isalnum() or c == " ")
                            gloss_map[clean_gloss] = spoken_sent
        except Exception as e:
            if verbose:
                print(f"[setup_isl_dataset] Warning reading gloss CSV: {e}")

    gloss_map_path = config.MODELS_DIR / "gloss_to_sentence.json"
    with open(gloss_map_path, "w", encoding="utf-8") as f:
        json.dump(gloss_map, f, indent=2, ensure_ascii=False)

    summary = {
        "dataset_name": "ISL-CSLRT (Indian Sign Language Continuous Sign Language Recognition & Translation Corpus)",
        "source": "SASTRA University & Navajeevan Residential School for the Deaf, Andhra Pradesh, India",
        "word_classes_count": len(word_classes),
        "word_classes": word_classes,
        "sentence_classes_count": len(sentence_classes),
        "sentence_classes": sentence_classes,
        "gloss_mappings_count": len(gloss_map),
        "dataset_words_dir": str(dataset_words_dir),
        "dataset_sentences_dir": str(dataset_sentences_dir),
        "gloss_map_path": str(gloss_map_path),
    }

    info_path = config.MODELS_DIR / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print("=" * 65)
        print("[ISL Sign2Speech] ISL Dataset Setup Complete!")
        print(f"Dataset: {summary['dataset_name']}")
        print(f"Detected Word Classes: {len(word_classes)}")
        print(f"Detected Sentence Classes: {len(sentence_classes)}")
        print(f"Gloss Mappings Indexed: {len(gloss_map)}")
        print(f"Dataset Info Saved: {info_path}")
        print("=" * 65)

    return summary


if __name__ == "__main__":
    setup_isl_dataset()
