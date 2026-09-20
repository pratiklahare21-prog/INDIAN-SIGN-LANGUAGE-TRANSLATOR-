from __future__ import annotations

import math
from typing import Tuple, List, Dict, Any


def dist_2d(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def dist_3d(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def get_angle(a, b, c) -> float:
    """Calculates angle ABC (in degrees) formed by points a, b, c."""
    ab = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    cb = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)
    if mag_ab * mag_cb == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cosine))


def analyze_hand(landmarks_norm: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    """
    Detailed 3D geometric breakdown of 21 hand landmarks.
    landmarks_norm: 21 (x, y, z) normalized coords.
    """
    if len(landmarks_norm) < 21:
        return {}

    lm = landmarks_norm
    wrist = lm[0]

    # Reference scale: distance from wrist to Middle MCP (joint 9)
    palm_len = max(0.01, dist_3d(wrist, lm[9]))

    # Finger Tip indices: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
    # PIP indices:        Thumb=3, Index=6, Middle=10, Ring=14, Pinky=18
    # MCP indices:        Thumb=2, Index=5, Middle=9,  Ring=13, Pinky=17

    # 1. Extension using distance-from-wrist ratio (rotation-invariant)
    # If Tip is significantly farther from wrist than PIP is, finger is extended.
    index_ext = (dist_3d(lm[8], wrist) > dist_3d(lm[6], wrist) * 1.12) and (dist_3d(lm[8], lm[5]) > dist_3d(lm[6], lm[5]) * 1.15)
    middle_ext = (dist_3d(lm[12], wrist) > dist_3d(lm[10], wrist) * 1.12) and (dist_3d(lm[12], lm[9]) > dist_3d(lm[10], lm[9]) * 1.15)
    ring_ext = (dist_3d(lm[16], wrist) > dist_3d(lm[14], wrist) * 1.12) and (dist_3d(lm[16], lm[13]) > dist_3d(lm[14], lm[13]) * 1.15)
    pinky_ext = (dist_3d(lm[20], wrist) > dist_3d(lm[18], wrist) * 1.12) and (dist_3d(lm[20], lm[17]) > dist_3d(lm[18], lm[17]) * 1.15)

    # Joint angles for curl detection (straight > 140 deg, curled < 100 deg)
    index_angle = get_angle(lm[5], lm[6], lm[8])
    middle_angle = get_angle(lm[9], lm[10], lm[12])
    ring_angle = get_angle(lm[13], lm[14], lm[16])
    pinky_angle = get_angle(lm[17], lm[18], lm[20])

    # Thumb extension: tip distance from pinky base (17) vs IP joint distance
    d_thumb_tip_pinky_base = dist_3d(lm[4], lm[17])
    d_thumb_ip_pinky_base = dist_3d(lm[3], lm[17])
    thumb_ext = (d_thumb_tip_pinky_base > d_thumb_ip_pinky_base * 1.15) or (dist_3d(lm[4], lm[5]) > palm_len * 0.70)

    # Thumb vertical direction
    thumb_pointing_up = (lm[4][1] < lm[2][1] - palm_len * 0.25)
    thumb_pointing_down = (lm[4][1] > lm[2][1] + palm_len * 0.25)

    # Relative separations
    d_index_middle_tips = dist_3d(lm[8], lm[12]) / palm_len
    d_thumb_index_tips = dist_3d(lm[4], lm[8]) / palm_len
    d_thumb_middle_tips = dist_3d(lm[4], lm[12]) / palm_len

    # Pinches
    pinch_thumb_index = d_thumb_index_tips < 0.28
    pinch_thumb_middle = d_thumb_middle_tips < 0.28
    all_tips_pinched = (
        d_thumb_index_tips < 0.38
        and d_thumb_middle_tips < 0.38
        and (dist_3d(lm[4], lm[16]) / palm_len) < 0.38
    )

    # Hand orientation: palm facing forward / up / side
    is_vertical_hand = (lm[9][1] < wrist[1])  # Middle MCP is higher than wrist

    return {
        "thumb": bool(thumb_ext),
        "index": bool(index_ext),
        "middle": bool(middle_ext),
        "ring": bool(ring_ext),
        "pinky": bool(pinky_ext),
        "fingers": [int(thumb_ext), int(index_ext), int(middle_ext), int(ring_ext), int(pinky_ext)],
        "thumb_up": bool(thumb_pointing_up),
        "thumb_down": bool(thumb_pointing_down),
        "pinch_index": bool(pinch_thumb_index),
        "pinch_middle": bool(pinch_thumb_middle),
        "all_pinched": bool(all_tips_pinched),
        "d_index_middle": d_index_middle_tips,
        "d_thumb_index": d_thumb_index_tips,
        "palm_len": palm_len,
        "wrist": wrist,
        "index_tip": lm[8],
        "thumb_tip": lm[4],
        "middle_tip": lm[12],
        "ring_tip": lm[16],
        "pinky_tip": lm[20],
        "is_vertical": is_vertical_hand,
    }


def classify_single_hand_isl(h: Dict[str, Any]) -> Tuple[str, float]:
    """
    Classifies single-hand ISL signs and alphabets with confidence.
    """
    if not h:
        return ("", 0.0)

    f = h["fingers"]
    thumb, index, middle, ring, pinky = f
    pinch_ti = h["pinch_index"]
    all_p = h["all_pinched"]
    thumb_up = h["thumb_up"]
    thumb_down = h["thumb_down"]
    d_im = h["d_index_middle"]
    d_ti = h["d_thumb_index"]

    # --- 1. Common Words / Universal Gestures ---

    # HELLO / STOP / 5: All 5 fingers extended upright
    if thumb and index and middle and ring and pinky:
        return ("Hello", 0.96)

    # I LOVE YOU (ILY): Thumb, Index, Pinky extended, Middle & Ring folded
    if thumb and index and not middle and not ring and pinky:
        return ("I Love You", 0.98)

    # GOOD / YES / SUPER: Thumbs Up
    if thumb_up and not index and not middle and not ring and not pinky:
        return ("Good", 0.97)

    # BAD / NO: Thumbs Down
    if thumb_down and not index and not middle and not ring and not pinky:
        return ("Bad", 0.97)

    # PEACE / VICTORY / V: Index & Middle UP separated, others down
    if index and middle and not ring and not pinky:
        if d_im > 0.28:
            return ("Peace", 0.95)
        else:
            return ("U", 0.93)

    # OK / OKAY / F: Thumb and Index touching circle, other 3 UP
    if pinch_ti and middle and ring and pinky:
        return ("OK", 0.96)

    # Y / PHONE: Thumb and Pinky extended
    if thumb and not index and not middle and not ring and pinky:
        return ("Y", 0.95)

    # L: Thumb and Index at right angle
    if thumb and index and not middle and not ring and not pinky:
        if d_ti > 0.40:
            return ("L", 0.96)

    # W / 3: Index, Middle, Ring UP
    if not thumb and index and middle and ring and not pinky:
        return ("W", 0.94)

    # B / 4: 4 fingers UP, Thumb across palm
    if not thumb and index and middle and ring and pinky:
        return ("B", 0.95)

    # I: Pinky only UP
    if not thumb and not index and not middle and not ring and pinky:
        return ("I", 0.96)

    # D / 1: Index only UP, thumb touching folded fingers
    if not thumb and index and not middle and not ring and not pinky:
        return ("D", 0.94)

    # O: All fingertips touching thumb forming circle
    if all_p and not (index and middle and ring and pinky):
        return ("O", 0.92)

    # C: C-shape curve
    if thumb and index and not pinch_ti and not all_p:
        if 0.30 <= d_ti <= 0.85 and not (middle and ring and pinky):
            return ("C", 0.88)

    # A / FIST: All fingers curled, thumb rested on side
    if not index and not middle and not ring and not pinky:
        if thumb or thumb_up:
            return ("A", 0.92)
        return ("A", 0.89)

    # E: All fingers tightly curled with thumb tucked across
    if not index and not middle and not ring and not pinky and not thumb:
        return ("E", 0.88)

    return ("", 0.0)


def classify_two_hands_isl(h1: Dict[str, Any], h2: Dict[str, Any]) -> Tuple[str, float]:
    """
    Classifies two-hand ISL signs (Namaste, Help, Thank You, Welcome, Friend, Super).
    """
    w1 = h1["wrist"]
    w2 = h2["wrist"]
    wrist_dist = dist_2d(w1, w2)
    avg_palm = (h1["palm_len"] + h2["palm_len"]) / 2.0

    f1 = h1["fingers"]
    f2 = h2["fingers"]

    # 1. NAMASTE / PRAY / HELLO: Both palms together, fingers pointing up
    # Wrists close together, index & middle extended on both hands
    if wrist_dist < avg_palm * 2.6 and (f1[1] and f1[2] and f2[1] and f2[2]):
        return ("Namaste", 0.98)

    # 2. HELP: One flat hand supporting the other fist / thumbs-up
    h1_flat = f1[1] and f1[2] and f1[3] and f1[4]
    h2_flat = f2[1] and f2[2] and f2[3] and f2[4]
    h1_fist = (not f1[1] and not f1[2] and not f1[3] and not f1[4])
    h2_fist = (not f2[1] and not f2[2] and not f2[3] and not f2[4])

    if (h1_flat and h2_fist) or (h2_flat and h1_fist):
        if wrist_dist < avg_palm * 3.2:
            return ("Help", 0.96)

    # 3. SUPER / GREAT: Both thumbs up
    if h1["thumb_up"] and h2["thumb_up"] and not f1[1] and not f2[1]:
        return ("Super", 0.97)

    # 4. WELCOME / OPEN: Both open hands with palms forward/up side by side
    if h1_flat and h2_flat:
        if wrist_dist > avg_palm * 1.4:
            return ("Welcome", 0.94)
        else:
            return ("Thank You", 0.94)

    return ("", 0.0)


def recognize_isl_gesture(hands_landmarks_norm: List[List[Tuple[float, float, float]]]) -> Tuple[str, float, int]:
    """
    Master recognition function for live MediaPipe hand coordinates.
    Returns: (sign_name, confidence, num_hands)
    """
    num_hands = len(hands_landmarks_norm)
    if num_hands == 0:
        return ("", 0.0, 0)

    analyzed_hands = [analyze_hand(h) for h in hands_landmarks_norm]

    # Two-hand signs take priority if 2 hands are visible
    if num_hands >= 2 and analyzed_hands[0] and analyzed_hands[1]:
        two_h_sign, two_h_conf = classify_two_hands_isl(analyzed_hands[0], analyzed_hands[1])
        if two_h_sign and two_h_conf >= 0.70:
            return (two_h_sign, two_h_conf, 2)

    # Single-hand sign classification
    sign1, conf1 = classify_single_hand_isl(analyzed_hands[0])

    if num_hands >= 2 and analyzed_hands[1] and (not sign1 or conf1 < 0.70):
        sign2, conf2 = classify_single_hand_isl(analyzed_hands[1])
        if conf2 > conf1:
            return (sign2, conf2, num_hands)

    return (sign1, conf1, num_hands)
