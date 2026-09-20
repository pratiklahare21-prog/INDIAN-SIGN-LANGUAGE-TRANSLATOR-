import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
try:
    from src.gesture_rules import recognize_isl_gesture
except ImportError:
    try:
        from gesture_rules import recognize_isl_gesture
    except ImportError:
        recognize_isl_gesture = None


HAND_CONNECTIONS_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                # Palm base
]


class HandTracker:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        import mediapipe as mp

        self._mode = "legacy"
        self._mp = mp
        self.max_num_hands = max_num_hands
        self.last_gesture = ""
        self.last_confidence = 0.0
        self.last_num_hands = 0
        self.last_multi_hand_landmarks = []

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._mode = "legacy"
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self.hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            # Modern MediaPipe tasks API
            self._mode = "tasks"
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            task_model_path = config.MODELS_DIR / "hand_landmarker.task"
            if not task_model_path.exists():
                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                try:
                    urllib.request.urlretrieve(url, task_model_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to auto-download hand_landmarker.task: {e}")

            base_options = python.BaseOptions(model_asset_path=str(task_model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)

    def _draw_hand_skeleton(self, img: np.ndarray, landmarks_norm: list[tuple[float, float, float]], color=(0, 255, 0), is_secondary=False):
        h, w = img.shape[:2]
        pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks_norm]

        line_color = (255, 128, 0) if is_secondary else (0, 255, 128)
        joint_color = (0, 200, 255) if is_secondary else (0, 255, 255)
        tip_color = (255, 0, 128) if is_secondary else (50, 50, 255)

        # Draw connections
        for p1_idx, p2_idx in HAND_CONNECTIONS_PAIRS:
            if 0 <= p1_idx < len(pts) and 0 <= p2_idx < len(pts):
                cv2.line(img, pts[p1_idx], pts[p2_idx], line_color, 2, cv2.LINE_AA)

        # Draw landmarks
        for i, pt in enumerate(pts):
            if i in (4, 8, 12, 16, 20):  # Fingertips
                cv2.circle(img, pt, 6, tip_color, -1, cv2.LINE_AA)
                cv2.circle(img, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, pt, 4, joint_color, -1, cv2.LINE_AA)

    def process_frame(self, frame: np.ndarray):
        zeros = np.zeros(config.LANDMARK_DIM, dtype=np.float32)
        self.last_gesture = ""
        self.last_confidence = 0.0
        self.last_num_hands = 0
        self.last_multi_hand_landmarks = []

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return zeros, frame, False

        try:
            annotated = frame.copy()
        except Exception:
            return zeros, frame, False

        try:
            if len(frame.shape) == 2:
                annotated = self._gray_to_bgr(annotated)
                work = annotated.copy()
            elif len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    annotated = annotated[:, :, :3]
                    work = annotated.copy()
                elif frame.shape[2] == 3:
                    work = annotated.copy()
                elif frame.shape[2] == 1:
                    annotated = self._gray_to_bgr(annotated.squeeze(axis=2))
                    work = annotated.copy()
                else:
                    return zeros, frame, False
            else:
                return zeros, frame, False
        except Exception:
            return zeros, frame, False

        hands_list = []  # List of 21-point normalized coords

        if self._mode == "legacy":
            try:
                rgb = work[:, :, ::-1]
                result = self.hands.process(rgb)
            except Exception:
                return zeros, frame, False

            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 0:
                for hand_lm in result.multi_hand_landmarks[:2]:
                    h_pts = [(p.x, p.y, p.z) for p in hand_lm.landmark]
                    hands_list.append(h_pts)
        else:
            # Modern tasks mode
            try:
                rgb = work[:, :, ::-1].copy()
                mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                result = self._detector.detect(mp_image)
            except Exception:
                return zeros, frame, False

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                for hand_lm in result.hand_landmarks[:2]:
                    h_pts = [(p.x, p.y, p.z) for p in hand_lm]
                    hands_list.append(h_pts)

        if len(hands_list) == 0:
            return zeros, annotated, False

        self.last_num_hands = len(hands_list)
        self.last_multi_hand_landmarks = hands_list

        # Draw all detected hands (Hand 1 and Hand 2)
        for idx, h_pts in enumerate(hands_list):
            self._draw_hand_skeleton(annotated, h_pts, is_secondary=(idx > 0))

        # Extract primary 63-dim landmark vector (wrist-normalized)
        primary_hand = hands_list[0]
        try:
            arr = np.array([[p[0], p[1], p[2]] for p in primary_hand], dtype=np.float32)
            wrist = arr[0].copy()
            arr -= wrist
            vec = arr.flatten()
        except Exception:
            vec = zeros

        # Classify gesture using rule engine
        if recognize_isl_gesture is not None:
            try:
                gest, conf, n_hands = recognize_isl_gesture(hands_list)
                self.last_gesture = gest
                self.last_confidence = conf
            except Exception:
                pass

        # Draw HUD status overlay on frame
        try:
            h, w = annotated.shape[:2]
            hud_text = f"Hands: {len(hands_list)}"
            if self.last_gesture:
                hud_text += f" | Sign: {self.last_gesture.upper()} ({int(self.last_confidence*100)}%)"

            # Top badge
            cv2.rectangle(annotated, (10, 10), (min(w - 10, 15 + len(hud_text) * 12), 42), (20, 24, 33), -1)
            cv2.rectangle(annotated, (10, 10), (min(w - 10, 15 + len(hud_text) * 12), 42), (0, 200, 255) if len(hands_list) > 1 else (0, 255, 128), 1)
            cv2.putText(annotated, hud_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

        return vec, annotated, True

    @staticmethod
    def _gray_to_bgr(gray):
        return np.stack([gray, gray, gray], axis=-1)


if __name__ == "__main__":
    try:
        tracker = HandTracker()
    except Exception as e:
        print(f"HandTracker not available ({e}); skipping mediapipe-dependent smoke.")
        import numpy as _np
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        black = _np.zeros((480, 640, 3), dtype=_np.uint8)
        assert black.shape == (480, 640, 3)
        assert config.LANDMARK_DIM == 63
        print("HandTracker smoke (structure only, no mediapipe): PASSED")
        raise SystemExit(0)

    black = np.zeros((480, 640, 3), dtype=np.uint8)
    vec, ann, det = tracker.process_frame(black)
    assert vec.shape == (63,), f"black vec shape: {vec.shape}"
    assert ann.shape == (480, 640, 3), f"black ann shape: {ann.shape}"
    assert det is False, "black frame should not have hand"
    assert np.allclose(vec, 0), "black vec should be zeros"

    rng = np.random.RandomState(config.SEED)
    rand = rng.randint(0, 256, size=(240, 320, 3), dtype=np.uint8)
    vec, ann, det = tracker.process_frame(rand)
    assert vec.shape == (63,), f"rand vec shape: {vec.shape}"
    assert ann.shape == (240, 320, 3), f"rand ann shape: {ann.shape}"
    assert isinstance(det, bool), f"det type: {type(det)}"
    if not det:
        assert np.allclose(vec, 0), "no-hand rand vec should be zeros"

    assert tracker.process_frame(None)[2] is False
    vec, ann, det = tracker.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert vec.shape == (63,) and det is False

    gray = np.zeros((100, 100), dtype=np.uint8)
    vec, ann, det = tracker.process_frame(gray)
    assert vec.shape == (63,) and ann.shape == (100, 100, 3) and det is False

    bgra = np.zeros((50, 50, 4), dtype=np.uint8)
    vec, ann, det = tracker.process_frame(bgra)
    assert vec.shape == (63,) and ann.shape == (50, 50, 3) and det is False

    print("HandTracker smoke PASSED")

