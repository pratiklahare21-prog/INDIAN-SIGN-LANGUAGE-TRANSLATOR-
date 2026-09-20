"""
ISL Sign2Speech — Standalone Real-Time Webcam Application
Indian Sign Language -> Word Recognition -> Sentence Construction -> Speech
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.alphabet_recognizer import AlphabetRecognizer
from src.hand_tracker import HandTracker
from src.sentence_builder import SentenceBuilder
from src.sentence_recognizer import SentenceRecognizer
from src.speech import SpeechEngine
from src.word_recognizer import WordRecognizer


def main():
    print("=" * 70)
    print("🤟 ISL Sign2Speech — Indian Sign Language Translator")
    print("=" * 70)

    print("[1/5] Initializing MediaPipe Hand Tracker...")
    tracker = HandTracker(static_image_mode=False, max_num_hands=2)

    print("[2/5] Initializing Alphabet Recognizer...")
    alphabet_rec = AlphabetRecognizer()
    print(f"      -> {alphabet_rec.status_text}")

    print("[3/5] Initializing Word Recognizer...")
    word_rec = WordRecognizer()
    print(f"      -> {word_rec.status_text}")

    print("[4/5] Initializing Sentence Recognizer...")
    sentence_rec = SentenceRecognizer()
    print(f"      -> {sentence_rec.status_text}")

    print("[5/5] Initializing Sentence Builder & Speech Engine...")
    sentence_builder = SentenceBuilder(debounce_frames=config.DEBOUNCE_FRAMES)
    speech_engine = SpeechEngine(rate=config.PYTTSX_RATE, volume=config.PYTTSX_VOLUME)
    print(f"      -> Speech Available: {speech_engine.is_available()}")

    cam_idx = config.WEBCAM_INDEX
    print(f"\nOpening Webcam (Index {cam_idx})...")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {cam_idx}. Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: No webcam found.")
            return

    # Hold-to-confirm parameters
    REQUIRED_HOLD_FRAMES = 12
    hold_frames = 0
    current_candidate = ""
    candidate_conf = 0.0
    locked_sign = ""
    confirmed_flash_frames = 0

    print("\nControls:")
    print("  [C] - Clear Sentence")
    print("  [D] - Delete Last Word")
    print("  [S] - Speak Full Sentence")
    print("  [Q] - Quit")
    print("=" * 70)

    fps_time = time.time()
    fps_counter = 0
    fps = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Hand tracking and landmark extraction
            vec_63, annotated, detected = tracker.process_frame(frame)
            num_hands = tracker.last_num_hands

            # Update sequence recognizers
            word_label, word_conf = "", 0.0
            sentence_label, sentence_conf = "", 0.0
            alpha_label, alpha_conf = "", 0.0

            if detected and vec_63 is not None:
                word_rec.update(vec_63)
                sentence_rec.update(vec_63)

                if word_rec.model_available:
                    word_label, word_conf = word_rec.predict(threshold=config.WORD_THRESHOLD)

                if sentence_rec.model_available:
                    sentence_label, sentence_conf = sentence_rec.predict(threshold=config.SENTENCE_THRESHOLD)

                if alphabet_rec.model_available and not word_label:
                    alpha_label, alpha_conf = alphabet_rec.predict(vec_63, threshold=config.ALPHABET_THRESHOLD)
            else:
                # Decay buffer when no hand detected
                if hold_frames > 0:
                    hold_frames -= 1

            # Candidate determination
            frame_best_sign = ""
            frame_best_conf = 0.0
            if sentence_label and sentence_conf >= config.SENTENCE_THRESHOLD:
                frame_best_sign = sentence_label
                frame_best_conf = sentence_conf
            elif word_label and word_conf >= config.WORD_THRESHOLD:
                frame_best_sign = word_label
                frame_best_conf = word_conf
            elif alpha_label and alpha_conf >= config.ALPHABET_THRESHOLD:
                frame_best_sign = alpha_label
                frame_best_conf = alpha_conf

            # Hold-to-confirm logic
            if frame_best_sign:
                if frame_best_sign == current_candidate:
                    hold_frames += 1
                else:
                    current_candidate = frame_best_sign
                    candidate_conf = frame_best_conf
                    hold_frames = 1
                    if current_candidate != locked_sign:
                        locked_sign = ""
            else:
                if hold_frames > 0:
                    hold_frames -= 1
                if hold_frames == 0:
                    current_candidate = ""
                    locked_sign = ""

            hold_progress = min(1.0, hold_frames / float(REQUIRED_HOLD_FRAMES))

            # Confirmation action
            if hold_progress >= 1.0 and current_candidate and current_candidate != locked_sign:
                locked_sign = current_candidate
                confirmed_flash_frames = 12
                sentence_builder.add_token(current_candidate, candidate_conf)
                # Speak confirmed token
                speech_engine.speak(current_candidate, block=False)
                print(f"\n[CONFIRMED] Token: '{current_candidate}' -> Sentence: \"{sentence_builder.get_text()}\"")

            # FPS calculation
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            # --- UI HUD DRAWING ---
            # Top Header Bar
            cv2.rectangle(annotated, (0, 0), (w, 50), (25, 25, 30), -1)
            cv2.putText(annotated, f"ISL Sign2Speech | Hands: {num_hands} | FPS: {fps}", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

            # Live Recognition Badge
            if current_candidate and hold_frames > 0:
                badge_text = f"Sign: {current_candidate.upper()} ({int(candidate_conf * 100)}%)"
                cv2.rectangle(annotated, (w - 320, 10), (w - 15, 42), (50, 60, 80), -1)
                cv2.putText(annotated, badge_text, (w - 310, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

            # Bottom Sentence Panel
            cv2.rectangle(annotated, (0, h - 90), (w, h), (18, 20, 26), -1)
            cv2.line(annotated, (0, h - 90), (w, h - 90), (60, 60, 70), 1)

            # Sentence Text
            current_sent = sentence_builder.get_text()
            display_sent = current_sent if current_sent else "Perform sign gestures to construct sentence..."
            sent_color = (255, 255, 255) if current_sent else (140, 140, 140)
            cv2.putText(annotated, display_sent, (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, sent_color, 2, cv2.LINE_AA)

            # Key Shortcuts Hint
            cv2.putText(annotated, "[C] Clear | [D] Del Word | [S] Speak | [Q] Quit", (20, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

            # Hold-to-Confirm Progress Bar
            if current_candidate and hold_frames > 0:
                bar_width = int((w - 40) * hold_progress)
                bar_y = h - 94
                if hold_progress >= 1.0 or confirmed_flash_frames > 0:
                    cv2.rectangle(annotated, (20, bar_y), (w - 20, bar_y + 4), (0, 255, 120), -1)
                else:
                    cv2.rectangle(annotated, (20, bar_y), (20 + bar_width, bar_y + 4), (0, 180, 255), -1)

            if confirmed_flash_frames > 0:
                confirmed_flash_frames -= 1

            cv2.imshow("ISL Sign2Speech", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('c'):
                sentence_builder.clear()
                print("\n[Action] Cleared sentence.")
            elif key == ord('d'):
                sentence_builder.delete_last_word()
                print(f"\n[Action] Deleted last word -> \"{sentence_builder.get_text()}\"")
            elif key == ord('s'):
                text_to_speak = sentence_builder.get_text()
                if text_to_speak:
                    speech_engine.speak(text_to_speak, block=False)
                    print(f"\n[Action] Speaking: \"{text_to_speak}\"")
                else:
                    print("\n[Action] Sentence is empty.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_engine.stop()
        print("\nISL Sign2Speech closed successfully.")


if __name__ == "__main__":
    main()
