import os
import random
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
DATASET_ALPHABET_DIR = DATASET_DIR / "alphabet"
DATASET_WORDS_DIR = DATASET_DIR / "words"

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LANDMARKS_DIR = DATA_DIR / "landmarks"

ALPHABET_MODEL_PATH = MODELS_DIR / "alphabet_model.keras"
WORD_MODEL_PATH = MODELS_DIR / "word_model.keras"
ALPHABET_LABELS_PATH = MODELS_DIR / "labels_alphabet.json"
WORD_LABELS_PATH = MODELS_DIR / "labels_word.json"

LANDMARKS_PER_HAND = 21
COORD_PER_LANDMARK = 3
LANDMARK_DIM = LANDMARKS_PER_HAND * COORD_PER_LANDMARK

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

SEQUENCE_LENGTH = 60
WORD_SAMPLE_STRIDE = 2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

ALPHABET_THRESHOLD = 0.70
WORD_THRESHOLD = 0.75
DEBOUNCE_FRAMES = 15

ALPHABET_MODEL_KWARGS = dict(
    hidden_units=(256, 128, 64),
    dropout=0.3,
    batch_norm=True,
)
ALPHABET_TRAIN_KWARGS = dict(
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    learning_rate=1e-3,
    early_stopping_patience=8,
)

WORD_MODEL_KWARGS = dict(
    lstm_units=(128, 64),
    dense_units=(64,),
    dropout=0.3,
    bidirectional=True,
)
WORD_TRAIN_KWARGS = dict(
    batch_size=32,
    epochs=80,
    validation_split=0.2,
    learning_rate=1e-3,
    early_stopping_patience=12,
)

PYTTSX_RATE = 175
PYTTSX_VOLUME = 0.9

WEBCAM_INDEX = 0
WEBCAM_FPS_TARGET = 20

KAGGLE_DATASET_LINKS = [
    (
        "Indian Sign Language Video Dataset (prasadshet, 60 word classes, ~3.48 GB)",
        "https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset",
    ),
]
ALPHABET_DATASET_HINT = (
    "Place A-Z folders under dataset/alphabet/A .. dataset/alphabet/Z"
)

MAX_HISTORY_TOKENS = 20
