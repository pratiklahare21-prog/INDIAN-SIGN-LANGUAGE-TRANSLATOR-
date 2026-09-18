import numpy as np


def normalize_hand_landmarks(vec_63: np.ndarray) -> np.ndarray:
    """
    Normalizes a 63-dimensional landmark vector (21 landmarks x 3) for translation & scale invariance.
    """
    vec = np.asarray(vec_63, dtype=np.float32).reshape(-1, 3)
    if len(vec) < 21:
        return vec_63

    wrist = vec[0]
    # Center relative to wrist
    rel = vec - wrist

    # Calculate scale factor: distance between wrist (0) and middle MCP (9)
    scale = np.linalg.norm(rel[9])
    if scale < 1e-4:
        # Fallback to max bounding box size
        scale = np.max(np.abs(rel))
    if scale < 1e-4:
        scale = 1.0

    norm_landmarks = rel / scale

    # Additional geometric features: fingertip distances to thumb tip & wrist
    # Fingertips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    tips = [4, 8, 12, 16, 20]
    tip_to_wrist = [np.linalg.norm(norm_landmarks[t]) for t in tips]
    tip_to_thumb = [np.linalg.norm(norm_landmarks[t] - norm_landmarks[4]) for t in tips[1:]]

    # Inter-finger distances
    inter_tips = [
        np.linalg.norm(norm_landmarks[8] - norm_landmarks[12]),   # Index - Middle
        np.linalg.norm(norm_landmarks[12] - norm_landmarks[16]),  # Middle - Ring
        np.linalg.norm(norm_landmarks[16] - norm_landmarks[20]),  # Ring - Pinky
    ]

    extra_geom = np.array(tip_to_wrist + tip_to_thumb + inter_tips, dtype=np.float32)
    return np.concatenate([norm_landmarks.flatten(), extra_geom])


def extract_sequence_features(seq: np.ndarray) -> np.ndarray:
    """
    Extracts translation-invariant motion, geometry, and statistical features
    from a sequence of hand landmarks.
    Input shape: (seq_len, 63)
    Output shape: 1D feature vector.
    """
    seq = np.asarray(seq, dtype=np.float32)
    seq_len, dim = seq.shape

    # Normalize each frame's landmarks
    norm_frames = [normalize_hand_landmarks(seq[t]) for t in range(seq_len)]
    norm_seq = np.stack(norm_frames, axis=0)  # shape: (seq_len, norm_dim)
    norm_dim = norm_seq.shape[1]

    # 1. Statistical aggregates across time
    mean_feat = np.mean(norm_seq, axis=0)
    std_feat = np.std(norm_seq, axis=0)
    min_feat = np.min(norm_seq, axis=0)
    max_feat = np.max(norm_seq, axis=0)

    # 2. Key temporal snapshots (start, 25%, mid, 75%, end)
    idx_0 = 0
    idx_1 = max(0, int(seq_len * 0.25))
    idx_2 = max(0, int(seq_len * 0.50))
    idx_3 = max(0, min(seq_len - 1, int(seq_len * 0.75)))
    idx_4 = max(0, seq_len - 1)

    frame_0 = norm_seq[idx_0]
    frame_1 = norm_seq[idx_1]
    frame_2 = norm_seq[idx_2]
    frame_3 = norm_seq[idx_3]
    frame_4 = norm_seq[idx_4]

    # 3. Dynamic Velocity & Motion Trajectory
    if seq_len > 1:
        deltas = np.diff(norm_seq, axis=0)
        vel_mean = np.mean(deltas, axis=0)
        vel_std = np.std(deltas, axis=0)
        vel_max = np.max(np.abs(deltas), axis=0)
        total_displacement = norm_seq[-1] - norm_seq[0]
    else:
        vel_mean = np.zeros(norm_dim, dtype=np.float32)
        vel_std = np.zeros(norm_dim, dtype=np.float32)
        vel_max = np.zeros(norm_dim, dtype=np.float32)
        total_displacement = np.zeros(norm_dim, dtype=np.float32)

    features = np.concatenate([
        mean_feat,
        std_feat,
        min_feat,
        max_feat,
        frame_0,
        frame_1,
        frame_2,
        frame_3,
        frame_4,
        vel_mean,
        vel_std,
        vel_max,
        total_displacement,
    ])
    return features.astype(np.float32)


def batch_extract_features(sequences: np.ndarray) -> np.ndarray:
    sequences = np.asarray(sequences, dtype=np.float32)
    feats = [extract_sequence_features(seq) for seq in sequences]
    return np.stack(feats, axis=0).astype(np.float32)
