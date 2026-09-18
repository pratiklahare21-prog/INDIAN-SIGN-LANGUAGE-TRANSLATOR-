"""
Utility script to download or link authentic ISL datasets (AI4Bharat INCLUDE / Kaggle ISL).
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

DATASET_REPOSITORIES = {
    "INCLUDE": {
        "title": "AI4Bharat INCLUDE (Indian Sign Language Dataset)",
        "hf_url": "https://huggingface.co/datasets/ai4bharat/INCLUDE",
        "github_url": "https://github.com/AI4Bharat/INCLUDE",
        "description": "263 ISL word classes across 4,287 videos recorded by deaf signers."
    },
    "INCLUDE-50": {
        "title": "AI4Bharat INCLUDE-50 Subset",
        "description": "50 high-frequency ISL word classes for quick training & evaluation."
    },
    "ISL-CSLRT": {
        "title": "ISL-CSLRT Continuous Sign Language Recognition & Translation Corpus",
        "description": "114 word classes, 101 sentence classes, 700 sentence videos recorded at Navajeevan Deaf School & SASTRA."
    },
    "KAGGLE-ISL": {
        "title": "Indian Sign Language Video Dataset (Kaggle / prasadshet)",
        "url": "https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset",
        "description": "60 dynamic ISL word classes (~3.48 GB)."
    }
}

def print_dataset_catalog():
    print("=" * 65)
    print("Authentic Indian Sign Language (ISL) Datasets Supported:")
    print("=" * 65)
    for key, info in DATASET_REPOSITORIES.items():
        print(f"[{key}] {info['title']}")
        print(f"  Description: {info['description']}")
        if "hf_url" in info:
            print(f"  HuggingFace: {info['hf_url']}")
        if "github_url" in info:
            print(f"  GitHub:     {info['github_url']}")
        if "url" in info:
            print(f"  URL:        {info['url']}")
        print()

def main():
    parser = argparse.ArgumentParser(description="ISL Dataset Manager")
    parser.add_argument("--list", action="store_true", help="List supported authentic ISL datasets")
    args = parser.parse_args()
    print_dataset_catalog()

if __name__ == "__main__":
    main()
