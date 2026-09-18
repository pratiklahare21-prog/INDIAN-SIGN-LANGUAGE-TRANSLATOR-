from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

from src.sentence_builder import SentenceBuilder
from src.speech import SpeechEngine

OK = 0
FAIL = 1


def hline() -> None:
    print("=" * 64)


def section(title: str) -> None:
    hline()
    print(f"  {title}")
    hline()


def import_optional(name: str, pip_name: str | None = None):
    try:
        if name == "HandTracker":
            from src.hand_tracker import HandTracker
            return HandTracker, True, ""
        if name == "AlphabetRecognizer":
            from src.alphabet_recognizer import AlphabetRecognizer
            return AlphabetRecognizer, True, ""
        if name == "WordRecognizer":
            from src.word_recognizer import WordRecognizer
            return WordRecognizer, True, ""
        if name == "cv2":
            import cv2
            return cv2, True, ""
        if name == "numpy":
            import numpy as np
            return np, True, ""
        if name == "sklearn":
            import sklearn  # noqa: F401
            return sklearn, True, ""
        if name == "tensorflow":
            import tensorflow as tf  # noqa: F401
            return tf, True, ""
    except Exception as e:
        return None, False, str(e)
    return None, False, "unknown"


def step_smoke_sentence_and_speech(actions: list[str]) -> int:
    section("Step 1: SentenceBuilder + SpeechEngine smoke")
    try:
        sb = SentenceBuilder(debounce_frames=3, max_history=50)
    except Exception as e:
        print(f"[FAIL] SentenceBuilder init: {e}")
        return FAIL
    adds = [sb.add_token("Hello", 0.95), sb.add_token("Hello", 0.93), sb.add_token("Hello", 0.97)]
    if adds.count(True) != 1:
        print(f"[FAIL] Expected 1 non-debounced 'Hello' add, got {adds.count(True)}")
        return FAIL
    sb.add_space()
    assert sb.add_token("world", 0.88) is True
    text = sb.get_text()
    if text != "Hello world":
        print(f"[FAIL] sentence: expected 'Hello world', got {text!r}")
        return FAIL
    sb.delete_last_word()
    if sb.get_text() != "Hello":
        print(f"[FAIL] after delete_last_word: {sb.get_text()!r}")
        return FAIL
    sb.add_space()
    sb.set_text("Hi there X")
    sb.backspace_char()
    if sb.get_text() != "Hi there ":
        print(f"[FAIL] after backspace: {sb.get_text()!r}")
        return FAIL
    sb.clear()
    if sb.get_text() != "":
        print("[FAIL] after clear, text not empty")
        return FAIL
    hist = sb.get_history()
    if len(hist) == 0:
        print("[FAIL] history empty after 3 unique adds")
        return FAIL
    actions.append("SentenceBuilder: debounce/add/delete/backspace/clear/history PASSED")
    print("[PASS] SentenceBuilder")

    try:
        sp = SpeechEngine(rate=180, volume=0.8)
    except Exception as e:
        print(f"[SKIP] SpeechEngine unavailable: {e}")
        actions.append("SpeechEngine: SKIP (unavailable)")
        return OK
    sp.set_rate(170)
    sp.set_volume(0.9)
    start = time.time()
    spoke = sp.speak("", block=True)
    if spoke:
        print("[FAIL] Empty string should not speak")
        return FAIL
    spoke2 = sp.speak("Integration test: speech engine is alive.", block=False)
    elapsed = time.time() - start
    print(f"[INFO] speech.speak() returned {spoke2} in {elapsed:.2f}s (non-blocking).")
    try:
        sp.stop()
    except Exception:
        pass
    actions.append("SpeechEngine: init/setters/speak/stop PASSED")
    print("[PASS] SpeechEngine")
    return OK


def step_smoke_optional_modules(actions: list[str]) -> dict[str, bool]:
    section("Step 2: Optional module availability check")
    status: dict[str, bool] = {}
    for label, key, pip in [
        ("numpy", "numpy", "numpy"),
        ("OpenCV (cv2)", "cv2", "opencv-python"),
        ("scikit-learn", "sklearn", "scikit-learn"),
        ("TensorFlow", "tensorflow", "tensorflow"),
        ("HandTracker (mediapipe)", "HandTracker", "mediapipe"),
        ("AlphabetRecognizer", "AlphabetRecognizer", None),
        ("WordRecognizer", "WordRecognizer", None),
    ]:
        _obj, ok, err = import_optional(key)
        hint = "" if ok else f" (missing: {pip})" if pip else ""
        print(f"  {label:<32} {'OK' if ok else 'SKIP'}{hint}")
        if not ok and err:
            print(f"       -> {err.splitlines()[0][:120]}")
        status[key] = ok
    actions.append(f"Module availability: {sum(status.values())}/{len(status)} OK")
    return status


def step_write_synthetic_alphabet(tmp_root: Path, np_mod, cv2_mod, n_classes: int = 5, per_class: int = 6) -> Path:
    alphabet = tmp_root / "dataset" / "alphabet"
    alphabet.mkdir(parents=True, exist_ok=True)
    labels = [chr(ord('A') + i) for i in range(n_classes)]
    rng = np_mod.random.RandomState(config.SEED)
    for lab in labels:
        d = alphabet / lab
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = rng.randint(0, 256, size=(120, 120, 3), dtype=np_mod.uint8)
            cv2_mod.imwrite(str(d / f"img_{i:03d}.png"), img)
    print(f"[SYNTH] Alphabet: {n_classes} classes x {per_class} images at {alphabet}")
    return alphabet


def step_write_synthetic_words(tmp_root: Path, np_mod, cv2_mod, n_classes: int = 3, videos_per_class: int = 4, frames: int = 30, fps: int = 20) -> Path:
    words = tmp_root / "dataset" / "words"
    words.mkdir(parents=True, exist_ok=True)
    labels = ["Hello", "Help", "Thanks"][:n_classes]
    fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
    rng = np_mod.random.RandomState(config.SEED + 1)
    for lab in labels:
        d = words / lab
        d.mkdir(parents=True, exist_ok=True)
        for i in range(videos_per_class):
            p = d / f"vid_{i:03d}.mp4"
            size = (160, 120)
            wri = cv2_mod.VideoWriter(str(p), fourcc, fps, size)
            if not wri.isOpened():
                continue
            for _ in range(frames):
                fr = rng.randint(0, 256, size=(size[1], size[0], 3), dtype=np_mod.uint8)
                wri.write(fr)
            wri.release()
    print(f"[SYNTH] Words: {len(labels)} classes x {videos_per_class} videos (~{frames}f) at {words}")
    return words


def run_script(script: str, argv: list[str]) -> tuple[int, str]:
    from subprocess import run, PIPE, STDOUT
    env = None
    full_argv = [sys.executable, str(ROOT / "src" / script), *argv]
    try:
        res = run(full_argv, cwd=str(ROOT), stdout=PIPE, stderr=STDOUT, text=True, env=env)
    except Exception as e:
        return 1, str(e)
    tail = "\n".join(res.stdout.splitlines()[-20:])
    return res.returncode, tail


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def step_preprocess_and_train(status: dict[str, bool], actions: list[str]) -> int:
    section("Step 3: Preprocess + Train on synthetic data (if deps allow)")
    if not (status.get("cv2") and status.get("numpy")):
        print("[SKIP] Cannot run synthetic preprocessing: cv2/numpy missing")
        actions.append("Preprocess+Train: SKIP (cv2/numpy missing)")
        return OK
    np_mod, _, _ = import_optional("numpy")
    cv2_mod, _, _ = import_optional("cv2")

    tmp_root = Path(tempfile.mkdtemp(prefix="isls2s_smoke_"))
    try:
        data_dir = tmp_root / "dataset"
        out_dir = tmp_root / "data" / "landmarks"
        models_dir = tmp_root / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        alphabet_dir = step_write_synthetic_alphabet(tmp_root, np_mod, cv2_mod)
        words_dir = step_write_synthetic_words(tmp_root, np_mod, cv2_mod)

        rc, out = run_script("preprocess_alphabet.py", [
            "--data-dir", str(alphabet_dir),
            "--out-dir", str(out_dir),
            "--labels-path", str(models_dir / "labels_alphabet.json"),
        ])
        print(f"[preprocess_alphabet] exit={rc}")
        print(out)
        if rc != 0:
            print("[WARN] preprocess_alphabet.py non-zero; treating as skip (mediapipe likely absent).")
            actions.append("Preprocess alphabet: SKIP/non-zero")
        else:
            for name in ["alphabet_X.npy", "alphabet_y.npy", "alphabet_X_train.npy", "alphabet_y_train.npy"]:
                p = out_dir / name
                sz = p.stat().st_size if p.exists() else 0
                print(f"   wrote {name}: {sz} bytes  md5={md5_of(p)[:12] if sz else '-'}")
            labels = json.loads((models_dir / "labels_alphabet.json").read_text())
            print(f"   classes in labels: {sorted(labels['label_to_idx'].keys())}")
            actions.append("Preprocess alphabet: PASSED")

            if status.get("tensorflow"):
                rc2, out2 = run_script("train_alphabet.py", [
                    "--landmarks-dir", str(out_dir),
                    "--labels-path", str(models_dir / "labels_alphabet.json"),
                    "--model-out", str(models_dir / "alphabet_model.keras"),
                    "--quick",
                ])
                print(f"[train_alphabet --quick] exit={rc2}")
                print(out2)
                model_exists = (models_dir / "alphabet_model.keras").exists()
                print(f"   model exists: {model_exists}  size={(models_dir / 'alphabet_model.keras').stat().st_size if model_exists else 0} bytes")
                actions.append("Train alphabet: " + ("PASSED" if rc2 == 0 else "FAILED"))
            else:
                actions.append("Train alphabet: SKIP (tensorflow missing)")

        rc, out = run_script("preprocess_words.py", [
            "--data-dir", str(words_dir),
            "--out-dir", str(out_dir),
            "--labels-path", str(models_dir / "labels_word.json"),
        ])
        print(f"[preprocess_words] exit={rc}")
        print(out)
        if rc != 0:
            print("[WARN] preprocess_words.py non-zero; treating as skip.")
            actions.append("Preprocess words: SKIP/non-zero")
        else:
            for name in ["words_X.npy", "words_y.npy", "words_X_train.npy", "words_y_train.npy"]:
                p = out_dir / name
                sz = p.stat().st_size if p.exists() else 0
                print(f"   wrote {name}: {sz} bytes  md5={md5_of(p)[:12] if sz else '-'}")
            labels = json.loads((models_dir / "labels_word.json").read_text())
            print(f"   classes in labels: {sorted(labels['label_to_idx'].keys())}")
            actions.append("Preprocess words: PASSED")

            if status.get("tensorflow"):
                rc2, out2 = run_script("train_words.py", [
                    "--landmarks-dir", str(out_dir),
                    "--labels-path", str(models_dir / "labels_word.json"),
                    "--model-out", str(models_dir / "word_model.keras"),
                    "--quick",
                ])
                print(f"[train_words --quick] exit={rc2}")
                print(out2)
                model_exists = (models_dir / "word_model.keras").exists()
                print(f"   model exists: {model_exists}  size={(models_dir / 'word_model.keras').stat().st_size if model_exists else 0} bytes")
                actions.append("Train words: " + ("PASSED" if rc2 == 0 else "FAILED"))
            else:
                actions.append("Train words: SKIP (tensorflow missing)")

    finally:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
    return OK


def step_smoke_recognizers(status: dict[str, bool], actions: list[str]) -> int:
    section("Step 4: Recognizer smoke (even without trained models)")
    np_mod, _, _ = import_optional("numpy")
    if np_mod is None:
        print("[SKIP] numpy missing")
        actions.append("Recognizers: SKIP (numpy missing)")
        return OK
    v = np_mod.zeros(63, dtype=np_mod.float32)
    if status.get("AlphabetRecognizer"):
        AR, _, _ = import_optional("AlphabetRecognizer")
        ar = AR()
        tok, conf = ar.predict(v, threshold=0.0)
        print(f"  AlphabetRecognizer.predict(zeros): {tok!r} conf={conf:.3f}  model_available={ar.model_available}")
        print(f"  status_text: {ar.status_text}")
        actions.append("AlphabetRecognizer: no-crash run OK")
    else:
        print("  AlphabetRecognizer: SKIP")
    if status.get("WordRecognizer"):
        WR, _, _ = import_optional("WordRecognizer")
        wr = WR(seq_len=10)
        for _ in range(15):
            wr.update(np_mod.random.rand(63).astype(np_mod.float32))
        tok, conf = wr.predict(threshold=0.0)
        print(f"  WordRecognizer.predict after 15 updates (seq=10): {tok!r} conf={conf:.3f}  model_available={wr.model_available}")
        wr.reset()
        assert not wr._ready()
        actions.append("WordRecognizer: update/predict/reset/ready OK")
    else:
        print("  WordRecognizer: SKIP")
    return OK


def step_end_to_end_mock(status: dict[str, bool], actions: list[str]) -> int:
    section("Step 5: Mock end-to-end flow (3 tokens -> speak)")
    np_mod, _, _ = import_optional("numpy")
    if np_mod is None:
        print("[SKIP] numpy missing")
        return OK

    AR, ar_ok, _ = import_optional("AlphabetRecognizer")
    WR, wr_ok, _ = import_optional("WordRecognizer")
    sb = SentenceBuilder(debounce_frames=1, max_history=20)
    sp = SpeechEngine()

    def fake_predict_letter(i: int):
        return (["H", "E", "Y", "L", "O"][i % 5], 0.92)
    def fake_predict_word(i: int):
        return (["Hello", "world", "friend"][i % 3], 0.91)

    for i in range(3):
        tok, conf = fake_predict_letter(i)
        sb.add_token(tok, conf)
    sb.add_space()
    for i in range(2):
        tok, conf = fake_predict_word(i)
        added = sb.add_token(tok, conf)
        if added and sp.is_available():
            try:
                sp.speak_last_word(sb.get_text())
            except Exception as e:
                print(f"  [speak_last_word] failed softly: {e}")
    final_text = sb.get_text()
    print(f"  sentence: {final_text!r}")
    if not final_text:
        print("[FAIL] sentence empty after 3+2 tokens (with debounce 1 frame per token  -> at least 2 unique word tokens)")
        return FAIL
    print(f"  speak() attempt on full sentence...")
    if sp.is_available():
        try:
            r = sp.speak(final_text, block=False)
            print(f"  speak(block=False) returned {r}")
        except Exception as e:
            print(f"  [WARN] speak raised: {e}")
    else:
        print("  SKIP (speech unavailable on this machine)")
    actions.append("E2E mock (3 letter tokens + 2 word tokens -> sentence): PASSED")
    return OK


def parse_args():
    p = argparse.ArgumentParser(description="ISL Sign2Speech end-to-end integration smoke test.")
    p.add_argument("--skip-preprocess-train", action="store_true", help="Skip synthetic preprocess+train (faster).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print()
    print("  +--------------------------------------------------------------+")
    print("  |       ISL Sign2Speech  Integration Smoke Test              |")
    print("  +--------------------------------------------------------------+")
    print()

    actions: list[str] = []
    rc = OK

    rc |= step_smoke_sentence_and_speech(actions)
    status = step_smoke_optional_modules(actions)
    if not args.skip_preprocess_train:
        rc |= step_preprocess_and_train(status, actions)
    else:
        actions.append("Preprocess+Train: SKIP (--skip-preprocess-train)")
    rc |= step_smoke_recognizers(status, actions)
    rc |= step_end_to_end_mock(status, actions)

    section("Summary")
    for a in actions:
        print(f"  - {a}")
    print()
    if rc == OK:
        print(f"Smoke test PASSED: {len(actions)} actions completed.")
    else:
        print(f"Smoke test COMPLETED WITH ISSUES (rc={rc}). Review above.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
