#!/usr/bin/env python3
"""
Dual OAK-D Pro: Fused REBA Scoring from Two Cameras
====================================================
Uses two OAK-D Pro cameras (profile/side + front) to compute a fused REBA
score that auto-detects posture conditions previously requiring manual CLI
flags: trunk twist, trunk side-bend, neck twist, neck side-bend, arm
abduction, shoulder raised, and unilateral stance.

The profile (side) camera measures flexion/extension angles, while the
front camera detects lateral movements and asymmetries.  The two views
are fused per body region, and the overall REBA score (1-15) is computed
via the standard Table A / B / C lookup.

On startup a **camera preview phase** shows a side-by-side live view so
you can verify which camera is PROFILE (side) and which is FRONT before
REBA analysis begins.  Preview controls:
    's'              – swap PROFILE / FRONT assignments
    Enter / Space / 'r' – confirm assignments and start REBA analysis
    'q'              – quit without starting REBA

Display layout (during REBA analysis):
    [PROFILE view | FUSED REBA panel | FRONT view]

Install:
    pip install depthai depthai-nodes opencv-python numpy

Run:
    python oak_dual_camera_reba.py --list-devices
    python oak_dual_camera_reba.py --profile-camera <MXID> --front-camera <MXID>

Press 'q' in the OpenCV window to quit.
"""

import os
import sys
import time
import math
import argparse
import contextlib
import collections
import numpy as np
import cv2

try:
    import depthai as dai  # type: ignore[import-untyped]
except ImportError:
    print(
        "ERROR: depthai is not installed.\n"
        "Run:  pip install depthai depthai-nodes opencv-python numpy",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from depthai_nodes.node import ParsingNeuralNetwork  # type: ignore[import-untyped]
except ImportError:
    print(
        "ERROR: depthai-nodes is not installed.\n"
        "Run:  pip install depthai-nodes",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_SLUG = "luxonis/yolov8-nano-pose-estimation:coco-512x288"
KP_THRESHOLD = 0.3
DET_THRESHOLD = 0.4
DISPLAY_SCALE = 1.5
SMOOTHING_WINDOW = 5
ALERT_COOLDOWN_SECONDS = 10.0
SYNC_TOLERANCE_SEC = 0.10


# ---------------------------------------------------------------------------
# COCO-17 keypoint indices
# ---------------------------------------------------------------------------
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


# ---------------------------------------------------------------------------
# REBA score color palette (BGR for OpenCV)
# ---------------------------------------------------------------------------
REBA_COLORS = {
    0: (130, 130, 130),
    1: (0, 180, 0),
    2: (0, 220, 220),
    3: (0, 150, 255),
    4: (0, 0, 230),
    5: (180, 0, 180),
}


# ---------------------------------------------------------------------------
# REBA Table A: Neck (rows 1-3) x Trunk (cols 1-5) x Legs (layers 1-4)
# Indexed as REBA_TABLE_A[neck-1][trunk-1][legs-1]
# ---------------------------------------------------------------------------
REBA_TABLE_A = [
    # Neck = 1
    [
        [1, 2, 3, 4],   # Trunk = 1, Legs 1-4
        [2, 3, 4, 5],   # Trunk = 2
        [2, 4, 5, 6],   # Trunk = 3
        [3, 5, 6, 7],   # Trunk = 4
        [4, 6, 7, 8],   # Trunk = 5
    ],
    # Neck = 2
    [
        [1, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ],
    # Neck = 3
    [
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 9],
    ],
]

# ---------------------------------------------------------------------------
# REBA Table B: Upper Arm (rows 1-6) x Lower Arm (cols 1-2) x Wrist (layers 1-3)
# Indexed as REBA_TABLE_B[upper_arm-1][lower_arm-1][wrist-1]
# ---------------------------------------------------------------------------
REBA_TABLE_B = [
    # Upper Arm = 1
    [
        [1, 2, 2],   # Lower Arm = 1, Wrist 1-3
        [1, 2, 3],   # Lower Arm = 2
    ],
    # Upper Arm = 2
    [
        [1, 2, 3],
        [2, 3, 4],
    ],
    # Upper Arm = 3
    [
        [3, 4, 5],
        [4, 5, 5],
    ],
    # Upper Arm = 4
    [
        [4, 5, 5],
        [5, 6, 7],
    ],
    # Upper Arm = 5
    [
        [6, 7, 8],
        [7, 8, 8],
    ],
    # Upper Arm = 6
    [
        [7, 8, 8],
        [8, 9, 9],
    ],
]

# ---------------------------------------------------------------------------
# REBA Table C: Score A (rows 1-12) x Score B (cols 1-12)
# Indexed as REBA_TABLE_C[score_a-1][score_b-1]
# ---------------------------------------------------------------------------
REBA_TABLE_C = [
    # Score A = 1
    [1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],
    # Score A = 2
    [1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],
    # Score A = 3
    [2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],
    # Score A = 4
    [3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],
    # Score A = 5
    [4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9, 10],
    # Score A = 6
    [6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 10, 10],
    # Score A = 7
    [7,  7,  7,  8,  9,  9,  9, 10, 10, 11, 11, 11],
    # Score A = 8
    [8,  8,  8,  9, 10, 10, 10, 10, 10, 11, 11, 11],
    # Score A = 9
    [9,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    # Score A = 10
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    # Score A = 11
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    # Score A = 12
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------
RISK_LEVELS = [
    (1,  "Negligible", (0, 180, 0)),      # green
    (3,  "Low",        (0, 220, 220)),     # yellow
    (7,  "Medium",     (0, 150, 255)),     # orange
    (10, "High",       (0, 0, 230)),       # red
    (15, "Very High",  (180, 0, 180)),     # purple
]


def get_risk_level(score):
    """Return (name, bgr_color) for a fused REBA score."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "N/A", (130, 130, 130)
    s = int(round(score))
    for threshold, name, color in RISK_LEVELS:
        if s <= threshold:
            return name, color
    return "Very High", (180, 0, 180)


# ---------------------------------------------------------------------------
# Geometry helpers  (copied from oak_pose_angles.py)
# ---------------------------------------------------------------------------
def get_midpoint(a, b):
    """Return midpoint of two points, or None if either is None."""
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle_at(p1, vertex, p3):
    """Unsigned angle in degrees at vertex formed by p1-vertex-p3."""
    if p1 is None or vertex is None or p3 is None:
        return float("nan")
    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]], dtype=np.float64)
    v2 = np.array([p3[0] - vertex[0], p3[1] - vertex[1]], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def signed_angle_from_vertical(p_bottom, p_top):
    """Signed angle (degrees) of vector p_bottom->p_top vs vertical-up.

    Positive = top is right of bottom (lean right in image).
    Negative = top is left of bottom (lean left in image).
    Zero = perfectly upright.
    """
    if p_bottom is None or p_top is None:
        return float("nan")
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    return math.degrees(math.atan2(dx, -dy))


def unsigned_angle_between(v1, v2):
    """Unsigned angle in degrees between two 2D vectors."""
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return float("nan")
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def _seg_length(p1, p2):
    """Euclidean distance between two 2D points, or NaN if either is None."""
    if p1 is None or p2 is None:
        return float("nan")
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# REBA scoring functions  (copied from oak_pose_angles.py)
# ---------------------------------------------------------------------------
def score_neck(angle, twist=False, side_bend=False):
    """Neck score: 0-20 deg = 1, >20 deg = 2. +1 twist, +1 side bend."""
    if math.isnan(angle):
        return float("nan")
    s = 1 if abs(angle) <= 20 else 2
    if twist:
        s += 1
    if side_bend:
        s += 1
    return s


def score_trunk(angle, twist=False, side_bend=False):
    """Trunk score: <5 = 1, 5-20 = 2, 20-60 = 3, >60 = 4.
    +1 twist, +1 side bend."""
    if math.isnan(angle):
        return float("nan")
    a = abs(angle)
    if a < 5:
        s = 1
    elif a <= 20:
        s = 2
    elif a <= 60:
        s = 3
    else:
        s = 4
    if twist:
        s += 1
    if side_bend:
        s += 1
    return s


def score_upper_arm(angle, shoulder_raised=False, abducted=False, supported=False):
    """Upper arm: 0-20 = 1, 20-45 = 2, 45-90 = 3, >90 = 4.
    +1 raised, +1 abducted, -1 supported."""
    if math.isnan(angle):
        return float("nan")
    a = abs(angle)
    if a <= 20:
        s = 1
    elif a <= 45:
        s = 2
    elif a <= 90:
        s = 3
    else:
        s = 4
    if shoulder_raised:
        s += 1
    if abducted:
        s += 1
    if supported:
        s -= 1
    return max(1, s)


def score_lower_arm(elbow_angle):
    """Lower arm: 80-120 deg included angle = 1, otherwise = 2."""
    if math.isnan(elbow_angle):
        return float("nan")
    if 80 <= elbow_angle <= 120:
        return 1
    return 2


def score_wrist(angle, deviated=False, twisted=False):
    """Wrist: 0-15 = 1, >15 = 2. +1 if deviated OR twisted (single adj)."""
    if math.isnan(angle):
        return float("nan")
    s = 1 if abs(angle) <= 15 else 2
    if deviated or twisted:
        s += 1
    return s


def score_legs(l_knee_angle, r_knee_angle, unilateral=False):
    """Legs base: 1 bilateral, 2 unilateral. Knee flexion: +1 30-60, +2 >60."""
    base = 2 if unilateral else 1
    max_flex = 0.0
    for ka in [l_knee_angle, r_knee_angle]:
        if not math.isnan(ka):
            flex = max(0.0, 180.0 - ka)
            if flex > max_flex:
                max_flex = flex
    if max_flex > 60:
        return base + 2
    if max_flex > 30:
        return base + 1
    return base


# ---------------------------------------------------------------------------
# Keypoint extraction  (copied from oak_pose_angles.py)
# ---------------------------------------------------------------------------
def keypoints_to_pixels(keypoints, w, h):
    """Convert normalized (0-1) Keypoint objects to pixel coords or None."""
    pts = []
    for kp in keypoints:
        if kp.confidence >= KP_THRESHOLD:
            pts.append((kp.imageCoordinates.x * w, kp.imageCoordinates.y * h))
        else:
            pts.append(None)
    return pts


def best_detection(detections):
    """Return the highest-confidence detection above threshold, or None."""
    best = None
    for det in detections:
        if det.confidence >= DET_THRESHOLD:
            if best is None or det.confidence > best.confidence:
                best = det
    return best


# ---------------------------------------------------------------------------
# Per-camera REBA computation  (copied from oak_pose_angles.py)
# ---------------------------------------------------------------------------
def compute_reba(pts, flags):
    """Compute all REBA angles and component scores from keypoints.

    Returns dict with angle and score values (NaN where not computable).
    """
    nan = float("nan")
    r = {
        "neck_angle": nan, "neck_score": nan,
        "trunk_angle": nan, "trunk_score": nan,
        "l_upper_arm_angle": nan, "l_upper_arm_score": nan,
        "r_upper_arm_angle": nan, "r_upper_arm_score": nan,
        "l_lower_arm_angle": nan, "l_lower_arm_score": nan,
        "r_lower_arm_angle": nan, "r_lower_arm_score": nan,
        "l_wrist_angle": nan, "l_wrist_score": nan,
        "r_wrist_angle": nan, "r_wrist_score": nan,
        "l_knee_angle": nan, "r_knee_angle": nan,
        "legs_score": nan,
    }

    if len(pts) < 17:
        return r

    mid_sh = get_midpoint(pts[L_SHOULDER], pts[R_SHOULDER])
    mid_hp = get_midpoint(pts[L_HIP], pts[R_HIP])

    # --- Trunk: angle of mid_hip->mid_shoulder from vertical ---
    trunk_ang = signed_angle_from_vertical(mid_hp, mid_sh)
    r["trunk_angle"] = trunk_ang
    r["trunk_score"] = score_trunk(
        trunk_ang, twist=flags["trunk_twist"], side_bend=flags["trunk_side_bend"]
    )

    # --- Neck: angle between neck vector and trunk vector ---
    if mid_sh is not None and mid_hp is not None and pts[NOSE] is not None:
        trunk_vec = (mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
        neck_vec = (pts[NOSE][0] - mid_sh[0], pts[NOSE][1] - mid_sh[1])
        neck_ang = unsigned_angle_between(trunk_vec, neck_vec)
        r["neck_angle"] = neck_ang
        r["neck_score"] = score_neck(
            neck_ang, twist=flags["neck_twist"], side_bend=flags["neck_side_bend"]
        )

    # --- Upper arms: angle of shoulder->elbow from trunk downward ---
    for side, sh_i, el_i in [("l", L_SHOULDER, L_ELBOW),
                              ("r", R_SHOULDER, R_ELBOW)]:
        ua_ang = nan
        if (pts[sh_i] is not None and pts[el_i] is not None
                and mid_hp is not None and mid_sh is not None):
            trunk_vec = (mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
            down_trunk = (-trunk_vec[0], -trunk_vec[1])
            arm_vec = (pts[el_i][0] - pts[sh_i][0], pts[el_i][1] - pts[sh_i][1])
            ua_ang = unsigned_angle_between(arm_vec, down_trunk)
        r[f"{side}_upper_arm_angle"] = ua_ang
        r[f"{side}_upper_arm_score"] = score_upper_arm(
            ua_ang,
            shoulder_raised=flags["shoulder_raised"],
            abducted=flags["arm_abducted"],
            supported=flags["arm_supported"],
        )

    # --- Lower arms: elbow angle ---
    r["l_lower_arm_angle"] = angle_at(pts[L_SHOULDER], pts[L_ELBOW], pts[L_WRIST])
    r["r_lower_arm_angle"] = angle_at(pts[R_SHOULDER], pts[R_ELBOW], pts[R_WRIST])
    r["l_lower_arm_score"] = score_lower_arm(r["l_lower_arm_angle"])
    r["r_lower_arm_score"] = score_lower_arm(r["r_lower_arm_angle"])

    # --- Wrists: not measurable with COCO-17 (no hand keypoints) ---
    # l_wrist_angle, r_wrist_angle, l_wrist_score, r_wrist_score stay NaN

    # --- Knees ---
    r["l_knee_angle"] = angle_at(pts[L_HIP], pts[L_KNEE], pts[L_ANKLE])
    r["r_knee_angle"] = angle_at(pts[R_HIP], pts[R_KNEE], pts[R_ANKLE])
    r["legs_score"] = score_legs(
        r["l_knee_angle"], r["r_knee_angle"],
        unilateral=flags["unilateral_legs"],
    )

    return r


# ---------------------------------------------------------------------------
# Overall REBA score from component scores (Table A / B / C)
# ---------------------------------------------------------------------------
def compute_reba_overall(neck_s, trunk_s, legs_s, upper_arm_s, lower_arm_s,
                         wrist_s, force_load=0, coupling=0, activity=0):
    """Compute the overall REBA score (1-15) from component scores.

    Takes the worst-case (max) left/right arm scores.
    Applies REBA adjustments: force/load to Score A, coupling to Score B,
    activity to final score.
    Returns (score_a, score_b, reba_score) or (NaN, NaN, NaN).
    """
    nan = float("nan")

    # Validate all inputs
    for v in [neck_s, trunk_s, legs_s, upper_arm_s, lower_arm_s, wrist_s]:
        if isinstance(v, float) and math.isnan(v):
            return nan, nan, nan

    neck_i = max(0, min(int(round(neck_s)) - 1, 2))
    trunk_i = max(0, min(int(round(trunk_s)) - 1, 4))
    legs_i = max(0, min(int(round(legs_s)) - 1, 3))
    score_a = REBA_TABLE_A[neck_i][trunk_i][legs_i]
    score_a = min(score_a + force_load, 12)

    ua_i = max(0, min(int(round(upper_arm_s)) - 1, 5))
    la_i = max(0, min(int(round(lower_arm_s)) - 1, 1))
    wr_i = max(0, min(int(round(wrist_s)) - 1, 2))
    score_b = REBA_TABLE_B[ua_i][la_i][wr_i]
    score_b = min(score_b + coupling, 12)

    sa_i = max(0, min(score_a - 1, 11))
    sb_i = max(0, min(score_b - 1, 11))
    reba_score = REBA_TABLE_C[sa_i][sb_i]
    reba_score = min(reba_score + activity, 15)

    return score_a, score_b, reba_score


# ---------------------------------------------------------------------------
# Fusion: combine profile + front camera REBA measurements
# ---------------------------------------------------------------------------
def fuse_reba_dicts(profile_rb, front_rb, profile_pts, front_pts, manual_flags):
    """Fuse REBA measurements from profile (side) and front cameras.

    Returns a dict with fused scores and auto-detected flags.
    """
    nan = float("nan")
    fused = {
        "neck_angle": nan, "neck_score": nan,
        "trunk_angle": nan, "trunk_score": nan,
        "l_upper_arm_angle": nan, "l_upper_arm_score": nan,
        "r_upper_arm_angle": nan, "r_upper_arm_score": nan,
        "l_lower_arm_angle": nan, "l_lower_arm_score": nan,
        "r_lower_arm_angle": nan, "r_lower_arm_score": nan,
        "l_wrist_angle": nan, "l_wrist_score": nan,
        "r_wrist_angle": nan, "r_wrist_score": nan,
        "l_knee_angle": nan, "r_knee_angle": nan,
        "legs_score": nan,
        # Detected flags
        "trunk_twist": False,
        "trunk_side_bend": False,
        "neck_twist": False,
        "neck_side_bend": False,
        "l_arm_abducted": False,
        "r_arm_abducted": False,
        "l_shoulder_raised": False,
        "r_shoulder_raised": False,
        "unilateral_legs": False,
        # Overall scores
        "score_a": nan,
        "score_b": nan,
        "fused_reba_score": nan,
    }

    has_profile = len(profile_pts) >= 17
    has_front = len(front_pts) >= 17

    if not has_profile and not has_front:
        return fused

    # --- Auto-detect flags from front camera ---
    if has_front:
        f_mid_sh = get_midpoint(front_pts[L_SHOULDER], front_pts[R_SHOULDER])
        f_mid_hp = get_midpoint(front_pts[L_HIP], front_pts[R_HIP])
        shoulder_width = _seg_length(front_pts[L_SHOULDER], front_pts[R_SHOULDER])

        # Trunk side-bend: lateral lean of mid-shoulder vs mid-hip > 10 deg
        if f_mid_sh is not None and f_mid_hp is not None:
            lateral_lean = signed_angle_from_vertical(f_mid_hp, f_mid_sh)
            if abs(lateral_lean) > 10:
                fused["trunk_side_bend"] = True

        # Trunk twist: L/R torso-side length asymmetry (ratio < 0.75)
        l_torso = _seg_length(front_pts[L_SHOULDER], front_pts[L_HIP])
        r_torso = _seg_length(front_pts[R_SHOULDER], front_pts[R_HIP])
        if not math.isnan(l_torso) and not math.isnan(r_torso):
            if l_torso > 1e-6 and r_torso > 1e-6:
                ratio = min(l_torso, r_torso) / max(l_torso, r_torso)
                if ratio < 0.85:
                    fused["trunk_twist"] = True

        # Neck side-bend: nose lateral offset from mid-shoulder > 15% shoulder width
        if (front_pts[NOSE] is not None and f_mid_sh is not None
                and not math.isnan(shoulder_width) and shoulder_width > 1e-6):
            nose_offset = abs(front_pts[NOSE][0] - f_mid_sh[0])
            if nose_offset > 0.15 * shoulder_width:
                fused["neck_side_bend"] = True

        # Neck twist: ear visibility asymmetry (one ear hidden)
        l_ear_vis = front_pts[L_EAR] is not None
        r_ear_vis = front_pts[R_EAR] is not None
        if l_ear_vis != r_ear_vis:
            fused["neck_twist"] = True

        # Arm abduction: front-view arm angle > 30 deg from trunk vertical
        for side, sh_i, el_i, key in [
            ("l", L_SHOULDER, L_ELBOW, "l_arm_abducted"),
            ("r", R_SHOULDER, R_ELBOW, "r_arm_abducted"),
        ]:
            if (front_pts[sh_i] is not None and front_pts[el_i] is not None
                    and f_mid_hp is not None and f_mid_sh is not None):
                trunk_vec = (f_mid_sh[0] - f_mid_hp[0], f_mid_sh[1] - f_mid_hp[1])
                down_trunk = (-trunk_vec[0], -trunk_vec[1])
                arm_vec = (front_pts[el_i][0] - front_pts[sh_i][0],
                           front_pts[el_i][1] - front_pts[sh_i][1])
                abd_angle = unsigned_angle_between(arm_vec, down_trunk)
                if not math.isnan(abd_angle) and abd_angle > 20:
                    fused[key] = True

        # Shoulder raised: shoulder height difference > 10% of torso length
        torso_len = _seg_length(f_mid_sh, f_mid_hp)
        if (front_pts[L_SHOULDER] is not None and front_pts[R_SHOULDER] is not None
                and not math.isnan(torso_len) and torso_len > 1e-6):
            l_sh_y = front_pts[L_SHOULDER][1]
            r_sh_y = front_pts[R_SHOULDER][1]
            height_diff = abs(l_sh_y - r_sh_y)
            if height_diff > 0.05 * torso_len:
                # The higher shoulder (lower y in image) is raised
                if l_sh_y < r_sh_y:
                    fused["l_shoulder_raised"] = True
                else:
                    fused["r_shoulder_raised"] = True

        # Unilateral stance: hip height asymmetry > 10% of shoulder width
        if (front_pts[L_HIP] is not None and front_pts[R_HIP] is not None
                and not math.isnan(shoulder_width) and shoulder_width > 1e-6):
            hip_diff = abs(front_pts[L_HIP][1] - front_pts[R_HIP][1])
            if hip_diff > 0.05 * shoulder_width:
                fused["unilateral_legs"] = True

    # --- Fuse trunk ---
    # Flexion from profile, side-bend/twist flags from front
    if has_profile and not math.isnan(profile_rb["trunk_angle"]):
        fused["trunk_angle"] = profile_rb["trunk_angle"]
    elif has_front and not math.isnan(front_rb["trunk_angle"]):
        fused["trunk_angle"] = front_rb["trunk_angle"]

    if not math.isnan(fused["trunk_angle"]):
        fused["trunk_score"] = score_trunk(
            fused["trunk_angle"],
            twist=fused["trunk_twist"],
            side_bend=fused["trunk_side_bend"],
        )

    # --- Fuse neck ---
    # Flexion from profile, side-bend/twist flags from front
    if has_profile and not math.isnan(profile_rb["neck_angle"]):
        fused["neck_angle"] = profile_rb["neck_angle"]
    elif has_front and not math.isnan(front_rb["neck_angle"]):
        fused["neck_angle"] = front_rb["neck_angle"]

    if not math.isnan(fused["neck_angle"]):
        fused["neck_score"] = score_neck(
            fused["neck_angle"],
            twist=fused["neck_twist"],
            side_bend=fused["neck_side_bend"],
        )

    # --- Fuse upper arms ---
    # Flexion from profile, abduction/raised flags from front
    for side in ["l", "r"]:
        ang_key = f"{side}_upper_arm_angle"
        scr_key = f"{side}_upper_arm_score"
        abd_key = f"{side}_arm_abducted"
        rsd_key = f"{side}_shoulder_raised"

        if has_profile and not math.isnan(profile_rb[ang_key]):
            fused[ang_key] = profile_rb[ang_key]
        elif has_front and not math.isnan(front_rb[ang_key]):
            fused[ang_key] = front_rb[ang_key]

        if not math.isnan(fused[ang_key]):
            fused[scr_key] = score_upper_arm(
                fused[ang_key],
                shoulder_raised=fused[rsd_key],
                abducted=fused[abd_key],
                supported=manual_flags["arm_supported"],
            )

    # --- Fuse lower arms ---
    # Profile preferred, front fallback
    for side in ["l", "r"]:
        ang_key = f"{side}_lower_arm_angle"
        scr_key = f"{side}_lower_arm_score"

        if has_profile and not math.isnan(profile_rb[ang_key]):
            fused[ang_key] = profile_rb[ang_key]
        elif has_front and not math.isnan(front_rb[ang_key]):
            fused[ang_key] = front_rb[ang_key]

        if not math.isnan(fused[ang_key]):
            fused[scr_key] = score_lower_arm(fused[ang_key])

    # --- Wrists: not measurable, apply manual flags ---
    # COCO-17 has no hand keypoints; use neutral angle (0) with manual flags
    for side in ["l", "r"]:
        fused[f"{side}_wrist_angle"] = 0.0
        fused[f"{side}_wrist_score"] = score_wrist(
            0.0,
            deviated=manual_flags["wrist_deviated"],
            twisted=manual_flags["wrist_twisted"],
        )

    # --- Fuse knees ---
    # Profile preferred, front fallback
    for side in ["l", "r"]:
        ka_key = f"{side}_knee_angle"
        if has_profile and not math.isnan(profile_rb[ka_key]):
            fused[ka_key] = profile_rb[ka_key]
        elif has_front and not math.isnan(front_rb[ka_key]):
            fused[ka_key] = front_rb[ka_key]

    fused["legs_score"] = score_legs(
        fused["l_knee_angle"], fused["r_knee_angle"],
        unilateral=fused["unilateral_legs"],
    )

    # --- Overall REBA score ---
    # Take worst-case (max) L/R for arm scores
    upper_arm_s = nan
    lower_arm_s = nan
    wrist_s = nan

    l_ua = fused["l_upper_arm_score"]
    r_ua = fused["r_upper_arm_score"]
    if not math.isnan(l_ua) and not math.isnan(r_ua):
        upper_arm_s = max(l_ua, r_ua)
    elif not math.isnan(l_ua):
        upper_arm_s = l_ua
    elif not math.isnan(r_ua):
        upper_arm_s = r_ua

    l_la = fused["l_lower_arm_score"]
    r_la = fused["r_lower_arm_score"]
    if not math.isnan(l_la) and not math.isnan(r_la):
        lower_arm_s = max(l_la, r_la)
    elif not math.isnan(l_la):
        lower_arm_s = l_la
    elif not math.isnan(r_la):
        lower_arm_s = r_la

    l_wr = fused["l_wrist_score"]
    r_wr = fused["r_wrist_score"]
    if not math.isnan(l_wr) and not math.isnan(r_wr):
        wrist_s = max(l_wr, r_wr)
    elif not math.isnan(l_wr):
        wrist_s = l_wr
    elif not math.isnan(r_wr):
        wrist_s = r_wr

    score_a, score_b, reba_score = compute_reba_overall(
        fused["neck_score"], fused["trunk_score"], fused["legs_score"],
        upper_arm_s, lower_arm_s, wrist_s,
        force_load=manual_flags.get("force_load", 0),
        coupling=manual_flags.get("coupling", 0),
        activity=manual_flags.get("activity", 0),
    )
    fused["score_a"] = score_a
    fused["score_b"] = score_b
    fused["fused_reba_score"] = reba_score

    return fused


# ---------------------------------------------------------------------------
# ScoreSmoother: rolling average to reduce noise
# ---------------------------------------------------------------------------
class ScoreSmoother:
    """Rolling average over SMOOTHING_WINDOW frames."""

    def __init__(self, window=SMOOTHING_WINDOW):
        self._window = window
        self._buf = collections.deque(maxlen=window)

    def update(self, score):
        """Add a score and return the smoothed value."""
        if isinstance(score, float) and math.isnan(score):
            return float("nan")
        self._buf.append(score)
        return sum(self._buf) / len(self._buf)

    def current(self):
        """Return the current smoothed value without adding a new sample."""
        if not self._buf:
            return float("nan")
        return sum(self._buf) / len(self._buf)


# ---------------------------------------------------------------------------
# AlertLogger: CSV alert logging with cooldown
# ---------------------------------------------------------------------------
class AlertLogger:
    """Logs REBA risk alerts to a CSV file with cooldown."""

    def __init__(self, path):
        self._path = os.path.abspath(path)
        self._file = open(self._path, "w", newline="")
        self._file.write("timestamp,reba_score,risk_level,highest_components\n")
        self._file.flush()
        self._last_risk = "N/A"
        self._last_alert_time = 0.0
        self._count = 0

    def check(self, score, fused_rb):
        """Log an alert if risk level changed or cooldown elapsed for Medium+."""
        if isinstance(score, float) and math.isnan(score):
            return
        risk_name, _ = get_risk_level(score)
        now = time.time()

        should_log = False
        if risk_name != self._last_risk:
            should_log = True
        elif score >= 4 and (now - self._last_alert_time) >= ALERT_COOLDOWN_SECONDS:
            should_log = True

        if should_log:
            # Find highest-scoring components
            components = []
            for key in ["neck_score", "trunk_score", "legs_score",
                        "l_upper_arm_score", "r_upper_arm_score",
                        "l_lower_arm_score", "r_lower_arm_score"]:
                val = fused_rb.get(key, float("nan"))
                if not (isinstance(val, float) and math.isnan(val)):
                    components.append((key.replace("_score", ""), int(val)))
            components.sort(key=lambda x: x[1], reverse=True)
            top = "; ".join(f"{n}={v}" for n, v in components[:3])

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self._file.write(f"{ts},{score:.1f},{risk_name},{top}\n")
            self._file.flush()
            self._last_risk = risk_name
            self._last_alert_time = now
            self._count += 1

    def close(self):
        self._file.close()

    @property
    def path(self):
        return self._path

    @property
    def count(self):
        return self._count


# ---------------------------------------------------------------------------
# FrameSynchronizer: sync frames from two cameras
# ---------------------------------------------------------------------------
class FrameSynchronizer:
    """Hold latest frame+detection from each camera and return synced pairs."""

    def __init__(self, tolerance=SYNC_TOLERANCE_SEC):
        self._tolerance = tolerance
        self._profile = None   # (host_timestamp, frame, reba_dict, pts)
        self._front = None
        self._last_pair_time = time.monotonic()

    def update_profile(self, frame, rb, pts):
        self._profile = (time.monotonic(), frame, rb, pts)

    def update_front(self, frame, rb, pts):
        self._front = (time.monotonic(), frame, rb, pts)

    def get_synced_pair(self):
        """Return (profile_data, front_data) if both within tolerance, else None.

        Each data tuple: (frame, reba_dict, pts)
        """
        if self._profile is None or self._front is None:
            return None
        t_p, frame_p, rb_p, pts_p = self._profile
        t_f, frame_f, rb_f, pts_f = self._front
        if abs(t_p - t_f) <= self._tolerance:
            self._last_pair_time = time.monotonic()
            # Consume the pair
            self._profile = None
            self._front = None
            return (frame_p, rb_p, pts_p), (frame_f, rb_f, pts_f)
        # Discard the older one
        if t_p < t_f:
            self._profile = None
        else:
            self._front = None
        return None

    def seconds_since_last_pair(self):
        return time.monotonic() - self._last_pair_time


# ---------------------------------------------------------------------------
# Drawing helpers  (copied from oak_pose_angles.py)
# ---------------------------------------------------------------------------
def reba_color(score):
    """Return BGR color for a REBA component score."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return REBA_COLORS[0]
    s = int(round(score))
    if s <= 0:
        return REBA_COLORS[0]
    return REBA_COLORS.get(min(s, 5), REBA_COLORS[5])


def _ipt(pt):
    return (int(round(pt[0])), int(round(pt[1])))


def _cline(frame, p1, p2, score, thickness=2):
    """Draw a line segment colored by REBA score."""
    if p1 is not None and p2 is not None:
        cv2.line(frame, _ipt(p1), _ipt(p2), reba_color(score), thickness, cv2.LINE_AA)


def draw_reba_skeleton(frame, pts, rb):
    """Draw skeleton with per-segment REBA coloring and angle labels."""
    if len(pts) < 17:
        return

    mid_sh = get_midpoint(pts[L_SHOULDER], pts[R_SHOULDER])
    mid_hp = get_midpoint(pts[L_HIP], pts[R_HIP])

    # Face (neutral gray, thin)
    for i, j in [(NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR)]:
        if pts[i] is not None and pts[j] is not None:
            cv2.line(frame, _ipt(pts[i]), _ipt(pts[j]),
                     REBA_COLORS[0], 1, cv2.LINE_AA)

    # Neck: mid-shoulder to nose
    _cline(frame, mid_sh, pts[NOSE], rb["neck_score"], 3)

    # Trunk: mid-hip to mid-shoulder
    _cline(frame, mid_hp, mid_sh, rb["trunk_score"], 3)

    # Shoulder bar, hip bar, torso sides (trunk color, thin)
    _cline(frame, pts[L_SHOULDER], pts[R_SHOULDER], rb["trunk_score"], 1)
    _cline(frame, pts[L_HIP], pts[R_HIP], rb["trunk_score"], 1)
    _cline(frame, pts[L_SHOULDER], pts[L_HIP], rb["trunk_score"], 1)
    _cline(frame, pts[R_SHOULDER], pts[R_HIP], rb["trunk_score"], 1)

    # Upper arms
    _cline(frame, pts[L_SHOULDER], pts[L_ELBOW], rb["l_upper_arm_score"], 2)
    _cline(frame, pts[R_SHOULDER], pts[R_ELBOW], rb["r_upper_arm_score"], 2)

    # Lower arms
    _cline(frame, pts[L_ELBOW], pts[L_WRIST], rb["l_lower_arm_score"], 2)
    _cline(frame, pts[R_ELBOW], pts[R_WRIST], rb["r_lower_arm_score"], 2)

    # Legs
    _cline(frame, pts[L_HIP], pts[L_KNEE], rb["legs_score"], 2)
    _cline(frame, pts[L_KNEE], pts[L_ANKLE], rb["legs_score"], 2)
    _cline(frame, pts[R_HIP], pts[R_KNEE], rb["legs_score"], 2)
    _cline(frame, pts[R_KNEE], pts[R_ANKLE], rb["legs_score"], 2)

    # Wrist markers (colored ring, gray if N/A)
    for wpt, ws in [(pts[L_WRIST], rb["l_wrist_score"]),
                    (pts[R_WRIST], rb["r_wrist_score"])]:
        if wpt is not None:
            cv2.circle(frame, _ipt(wpt), 6, reba_color(ws), -1, cv2.LINE_AA)
            cv2.circle(frame, _ipt(wpt), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # White keypoint dots
    for pt in pts:
        if pt is not None:
            cv2.circle(frame, _ipt(pt), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Angle labels at joints
    def _alabel(pt, angle, score, dx=8, dy=-6):
        if pt is not None and not math.isnan(angle):
            cv2.putText(frame, f"{angle:.0f}", (_ipt(pt)[0] + dx, _ipt(pt)[1] + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        reba_color(score), 1, cv2.LINE_AA)

    _alabel(mid_sh, rb["neck_angle"], rb["neck_score"], 10, -8)
    trunk_mid = get_midpoint(mid_hp, mid_sh)
    _alabel(trunk_mid, rb["trunk_angle"], rb["trunk_score"], 10, 0)
    _alabel(pts[L_SHOULDER], rb["l_upper_arm_angle"], rb["l_upper_arm_score"], -38, -6)
    _alabel(pts[R_SHOULDER], rb["r_upper_arm_angle"], rb["r_upper_arm_score"], 10, -6)
    _alabel(pts[L_ELBOW], rb["l_lower_arm_angle"], rb["l_lower_arm_score"], -38, -6)
    _alabel(pts[R_ELBOW], rb["r_lower_arm_angle"], rb["r_lower_arm_score"], 10, -6)
    _alabel(pts[L_KNEE], rb["l_knee_angle"], rb["legs_score"], -38, -6)
    _alabel(pts[R_KNEE], rb["r_knee_angle"], rb["legs_score"], 10, -6)


def draw_reba_table(frame, rb, fps, label=""):
    """Draw REBA scores table with dark background in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.37
    lh = 17
    pad = 5

    def _fa(a):
        return f"{a:>4.0f} deg" if not math.isnan(a) else "     ---"

    def _fs(s):
        if isinstance(s, float) and math.isnan(s):
            return "[-]"
        return f"[{int(s)}]"

    rows = [
        (f"FPS: {fps:.1f}  {label}",                           (0, 255, 0)),
        ("---- REBA Scores ----",                               (200, 200, 200)),
        (f"Neck    {_fa(rb['neck_angle'])}  {_fs(rb['neck_score'])}",
         reba_color(rb["neck_score"])),
        (f"Trunk   {_fa(rb['trunk_angle'])}  {_fs(rb['trunk_score'])}",
         reba_color(rb["trunk_score"])),
        (f"L UArm  {_fa(rb['l_upper_arm_angle'])}  {_fs(rb['l_upper_arm_score'])}",
         reba_color(rb["l_upper_arm_score"])),
        (f"R UArm  {_fa(rb['r_upper_arm_angle'])}  {_fs(rb['r_upper_arm_score'])}",
         reba_color(rb["r_upper_arm_score"])),
        (f"L LArm  {_fa(rb['l_lower_arm_angle'])}  {_fs(rb['l_lower_arm_score'])}",
         reba_color(rb["l_lower_arm_score"])),
        (f"R LArm  {_fa(rb['r_lower_arm_angle'])}  {_fs(rb['r_lower_arm_score'])}",
         reba_color(rb["r_lower_arm_score"])),
        (f"L Wrist      N/A  [-]",  REBA_COLORS[0]),
        (f"R Wrist      N/A  [-]",  REBA_COLORS[0]),
        (f"L Knee  {_fa(rb['l_knee_angle'])}",
         reba_color(rb["legs_score"])),
        (f"R Knee  {_fa(rb['r_knee_angle'])}",
         reba_color(rb["legs_score"])),
        (f"Legs              {_fs(rb['legs_score'])}",
         reba_color(rb["legs_score"])),
    ]

    # Semi-transparent dark background
    table_w = 235
    table_h = len(rows) * lh + pad * 2
    table_h = min(table_h, frame.shape[0])
    table_w = min(table_w, frame.shape[1])
    sub = frame[0:table_h, 0:table_w]
    dark = np.zeros_like(sub)
    cv2.addWeighted(sub, 0.35, dark, 0.65, 0, sub)

    # Draw text
    for i, (text, color) in enumerate(rows):
        y = pad + (i + 1) * lh - 3
        cv2.putText(frame, text, (pad, y), font, fs, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# New drawing functions for dual-camera display
# ---------------------------------------------------------------------------
def draw_alert_border(frame, score, thickness=6):
    """Draw a colored border around the frame based on risk level."""
    _, color = get_risk_level(score)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


def draw_fused_panel(panel, fused_rb, fps, smoothed_score, adjustments=None):
    """Draw the center info panel with fused REBA score and detected flags."""
    h, w = panel.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    nan = float("nan")
    if adjustments is None:
        adjustments = {}

    # Dark background
    panel[:] = (30, 30, 30)

    score = fused_rb.get("fused_reba_score", nan)
    risk_name, risk_color = get_risk_level(smoothed_score)

    y = 30
    # Title
    cv2.putText(panel, "FUSED REBA", (10, y), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    y += 40

    # Large score
    if not (isinstance(smoothed_score, float) and math.isnan(smoothed_score)):
        score_text = f"{smoothed_score:.0f}"
    else:
        score_text = "---"
    cv2.putText(panel, score_text, (10, y), font, 1.8, risk_color, 3, cv2.LINE_AA)
    y += 20

    # Risk level
    cv2.putText(panel, risk_name, (10, y), font, 0.6, risk_color, 1, cv2.LINE_AA)
    y += 30

    # Score A / B / C
    score_a = fused_rb.get("score_a", nan)
    score_b = fused_rb.get("score_b", nan)

    def _fv(v):
        if isinstance(v, float) and math.isnan(v):
            return "---"
        return f"{int(v)}"

    cv2.putText(panel, f"Score A: {_fv(score_a)}", (10, y), font, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(panel, f"Score B: {_fv(score_b)}", (10, y), font, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(panel, f"Score C: {_fv(score)}", (10, y), font, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)
    y += 22

    # REBA adjustments
    fl = adjustments.get("force_load", 0)
    cp = adjustments.get("coupling", 0)
    ac = adjustments.get("activity", 0)
    adj_color = (0, 180, 255) if (fl or cp or ac) else (100, 100, 100)
    cv2.putText(panel, f"Force/Load: +{fl}", (10, y), font, 0.37,
                adj_color, 1, cv2.LINE_AA)
    y += 16
    cv2.putText(panel, f"Coupling:   +{cp}", (10, y), font, 0.37,
                adj_color, 1, cv2.LINE_AA)
    y += 16
    cv2.putText(panel, f"Activity:   +{ac}", (10, y), font, 0.37,
                adj_color, 1, cv2.LINE_AA)
    y += 22

    # Detected flags
    cv2.putText(panel, "Detected:", (10, y), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    y += 18

    flag_items = [
        ("trunk_twist", "Trunk twist"),
        ("trunk_side_bend", "Trunk side-bend"),
        ("neck_twist", "Neck twist"),
        ("neck_side_bend", "Neck side-bend"),
        ("l_arm_abducted", "L arm abducted"),
        ("r_arm_abducted", "R arm abducted"),
        ("l_shoulder_raised", "L shoulder raised"),
        ("r_shoulder_raised", "R shoulder raised"),
        ("unilateral_legs", "Unilateral stance"),
    ]
    active_count = 0
    for key, label in flag_items:
        if fused_rb.get(key, False):
            color = (0, 180, 255)  # orange for active flags
            cv2.putText(panel, f"  {label}", (10, y), font, 0.37,
                        color, 1, cv2.LINE_AA)
            y += 16
            active_count += 1

    if active_count == 0:
        cv2.putText(panel, "  (none)", (10, y), font, 0.37,
                    (100, 100, 100), 1, cv2.LINE_AA)
        y += 16

    # Component scores
    y += 10
    cv2.putText(panel, "Components:", (10, y), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    y += 18

    comp_items = [
        ("Neck", fused_rb.get("neck_score", nan)),
        ("Trunk", fused_rb.get("trunk_score", nan)),
        ("Legs", fused_rb.get("legs_score", nan)),
        ("L UArm", fused_rb.get("l_upper_arm_score", nan)),
        ("R UArm", fused_rb.get("r_upper_arm_score", nan)),
        ("L LArm", fused_rb.get("l_lower_arm_score", nan)),
        ("R LArm", fused_rb.get("r_lower_arm_score", nan)),
    ]
    for name, val in comp_items:
        if y + 16 > h - 20:
            break
        vstr = _fv(val)
        cv2.putText(panel, f"  {name}: {vstr}", (10, y), font, 0.37,
                    reba_color(val), 1, cv2.LINE_AA)
        y += 16

    # FPS at bottom
    cv2.putText(panel, f"FPS: {fps:.1f}", (10, h - 15), font, 0.42,
                (0, 255, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# CSV helpers (extended format for dual camera)
# ---------------------------------------------------------------------------
DUAL_CSV_HEADER = (
    "timestamp,source,"
    "neck_angle,neck_score,"
    "trunk_angle,trunk_score,"
    "L_upper_arm_angle,L_upper_arm_score,"
    "R_upper_arm_angle,R_upper_arm_score,"
    "L_lower_arm_angle,L_lower_arm_score,"
    "R_lower_arm_angle,R_lower_arm_score,"
    "L_wrist_angle,L_wrist_score,"
    "R_wrist_angle,R_wrist_score,"
    "L_knee_angle,R_knee_angle,legs_score,"
    "fused_reba_score,risk_level,"
    "trunk_twist,trunk_side_bend,"
    "neck_twist,neck_side_bend,"
    "l_arm_abducted,r_arm_abducted,"
    "l_shoulder_raised,r_shoulder_raised,"
    "force_load,coupling,activity,"
    "FPS\n"
)


def reba_to_csv_line(rb, fps):
    """Format one CSV row from REBA results (single-camera compatible)."""
    ts = time.strftime("%H:%M:%S")

    def _v(val):
        if isinstance(val, float) and math.isnan(val):
            return ""
        if isinstance(val, float):
            return f"{val:.1f}"
        return str(val)

    fields = [
        ts,
        _v(rb["neck_angle"]), _v(rb["neck_score"]),
        _v(rb["trunk_angle"]), _v(rb["trunk_score"]),
        _v(rb["l_upper_arm_angle"]), _v(rb["l_upper_arm_score"]),
        _v(rb["r_upper_arm_angle"]), _v(rb["r_upper_arm_score"]),
        _v(rb["l_lower_arm_angle"]), _v(rb["l_lower_arm_score"]),
        _v(rb["r_lower_arm_angle"]), _v(rb["r_lower_arm_score"]),
        _v(rb["l_wrist_angle"]), _v(rb["l_wrist_score"]),
        _v(rb["r_wrist_angle"]), _v(rb["r_wrist_score"]),
        _v(rb["l_knee_angle"]), _v(rb["r_knee_angle"]), _v(rb["legs_score"]),
        f"{fps:.1f}",
    ]
    return ",".join(fields) + "\n"


def dual_reba_to_csv_line(rb, source, fps, fused_rb=None, adjustments=None):
    """Format one CSV row in extended dual-camera format."""
    ts = time.strftime("%H:%M:%S")
    if adjustments is None:
        adjustments = {}

    def _v(val):
        if isinstance(val, float) and math.isnan(val):
            return ""
        if isinstance(val, float):
            return f"{val:.1f}"
        if isinstance(val, bool):
            return "1" if val else "0"
        return str(val)

    # For fused rows, pull flags and overall score from fused_rb
    fr = fused_rb if fused_rb is not None else rb
    fused_score = fr.get("fused_reba_score", float("nan"))
    risk_name, _ = get_risk_level(fused_score)
    if isinstance(fused_score, float) and math.isnan(fused_score):
        risk_name = ""

    fields = [
        ts, source,
        _v(rb["neck_angle"]), _v(rb["neck_score"]),
        _v(rb["trunk_angle"]), _v(rb["trunk_score"]),
        _v(rb["l_upper_arm_angle"]), _v(rb["l_upper_arm_score"]),
        _v(rb["r_upper_arm_angle"]), _v(rb["r_upper_arm_score"]),
        _v(rb["l_lower_arm_angle"]), _v(rb["l_lower_arm_score"]),
        _v(rb["r_lower_arm_angle"]), _v(rb["r_lower_arm_score"]),
        _v(rb["l_wrist_angle"]), _v(rb["l_wrist_score"]),
        _v(rb["r_wrist_angle"]), _v(rb["r_wrist_score"]),
        _v(rb["l_knee_angle"]), _v(rb["r_knee_angle"]), _v(rb["legs_score"]),
        _v(fused_score), risk_name,
        _v(fr.get("trunk_twist", False)),
        _v(fr.get("trunk_side_bend", False)),
        _v(fr.get("neck_twist", False)),
        _v(fr.get("neck_side_bend", False)),
        _v(fr.get("l_arm_abducted", False)),
        _v(fr.get("r_arm_abducted", False)),
        _v(fr.get("l_shoulder_raised", False)),
        _v(fr.get("r_shoulder_raised", False)),
        str(adjustments.get("force_load", 0)),
        str(adjustments.get("coupling", 0)),
        str(adjustments.get("activity", 0)),
        f"{fps:.1f}",
    ]
    return ",".join(fields) + "\n"


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_pipeline():
    """Construct a DepthAI pipeline with ColorCamera + ParsingNeuralNetwork.

    Returns (pipeline, det_queue_name, pass_queue_name).
    """
    pipeline = dai.Pipeline()

    # Try v2-style Camera node first, fall back to ColorCamera
    try:
        cam = pipeline.create(dai.node.Camera).build()
    except (AttributeError, TypeError):
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(512, 288)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(15)

    # Neural network - try ParsingNeuralNetwork, fall back to NeuralNetwork
    model_desc = dai.NNModelDescription(MODEL_SLUG)
    try:
        nn = pipeline.create(ParsingNeuralNetwork).build(cam, model_desc)
    except (AttributeError, TypeError, RuntimeError) as exc:
        log(f"ParsingNeuralNetwork incompatible, falling back to NeuralNetwork: {exc}")
        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(model_desc)
        if hasattr(cam, "preview"):
            cam.preview.link(nn.input)
        else:
            cam.build().link(nn.input)

    return pipeline, nn


# ---------------------------------------------------------------------------
# Device discovery & initialization
# ---------------------------------------------------------------------------
def list_available_devices():
    """List all connected OAK devices and return the list."""
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("No OAK devices found.")
        print("  1. Make sure OAK-D Pro is plugged into a USB 3 port")
        print("  2. Try a different USB cable (must be data-capable)")
        print("  3. Check Device Manager for Movidius / Myriad X")
        return []
    print(f"Found {len(devices)} OAK device(s):")
    for i, info in enumerate(devices):
        state = info.state.name if hasattr(info.state, "name") else str(info.state)
        print(f"  [{i}] MxID: {info.deviceId}  State: {state}")
    return devices


def find_device_by_mxid(devices, mxid):
    """Find a device info by MxID string, or None."""
    for info in devices:
        if info.deviceId == mxid:
            return info
    return None


def initialize_devices(profile_mxid, front_mxid):
    """Initialize two OAK devices. Returns (profile_device, front_device) context.

    Uses contextlib.ExitStack so the caller manages cleanup.
    """
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("ERROR: No OAK devices found.", file=sys.stderr)
        list_available_devices()
        sys.exit(1)

    if profile_mxid == front_mxid:
        print("ERROR: --profile-camera and --front-camera cannot be the same MxID.",
              file=sys.stderr)
        sys.exit(1)

    profile_info = find_device_by_mxid(devices, profile_mxid)
    if profile_info is None:
        print(f"ERROR: Profile camera MxID '{profile_mxid}' not found.", file=sys.stderr)
        list_available_devices()
        sys.exit(1)

    front_info = find_device_by_mxid(devices, front_mxid)
    if front_info is None:
        print(f"ERROR: Front camera MxID '{front_mxid}' not found.", file=sys.stderr)
        list_available_devices()
        sys.exit(1)

    return profile_info, front_info


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Empty REBA dict
# ---------------------------------------------------------------------------
def _empty_reba():
    nan = float("nan")
    return {
        "neck_angle": nan, "neck_score": nan,
        "trunk_angle": nan, "trunk_score": nan,
        "l_upper_arm_angle": nan, "l_upper_arm_score": nan,
        "r_upper_arm_angle": nan, "r_upper_arm_score": nan,
        "l_lower_arm_angle": nan, "l_lower_arm_score": nan,
        "r_lower_arm_angle": nan, "r_lower_arm_score": nan,
        "l_wrist_angle": nan, "l_wrist_score": nan,
        "r_wrist_angle": nan, "r_wrist_score": nan,
        "l_knee_angle": nan, "r_knee_angle": nan,
        "legs_score": nan,
    }


# ---------------------------------------------------------------------------
# Camera preview: wait for devices to re-enumerate after preview closes
# ---------------------------------------------------------------------------
def _wait_for_devices_available(mxid_a, mxid_b, timeout=20.0, poll=0.5):
    """Poll until both MXIDs reappear in ``dai.Device.getAllAvailableDevices()``.

    After closing preview device connections, USB re-enumeration can take a few
    seconds.  This helper avoids stale-DeviceInfo issues by waiting for fresh
    enumeration results.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = {d.deviceId for d in dai.Device.getAllAvailableDevices()}
        if mxid_a in found and mxid_b in found:
            log(f"Both devices available ({mxid_a}, {mxid_b}).")
            return True
        missing = []
        if mxid_a not in found:
            missing.append(mxid_a)
        if mxid_b not in found:
            missing.append(mxid_b)
        log(f"Waiting for device(s) {', '.join(missing)} ...")
        time.sleep(poll)
    log("WARNING: Timed out waiting for devices to re-enumerate.")
    return False


# ---------------------------------------------------------------------------
# Camera preview: interactive side-by-side view for camera assignment
# ---------------------------------------------------------------------------
def run_camera_preview(profile_mxid, front_mxid):
    """Show a live side-by-side preview so the user can verify/swap cameras.

    Opens lightweight camera-only pipelines (no NN).
    Controls:
        's'              – swap PROFILE / FRONT
        Enter/Space/'r'  – confirm and start REBA
        'q'              – quit

    Returns ``(final_profile_mxid, final_front_mxid)`` on confirmation,
    or ``None`` if the user quits.
    """
    # Enumerate devices fresh
    devices = dai.Device.getAllAvailableDevices()
    profile_info = None
    front_info = None
    for d in devices:
        if d.deviceId == profile_mxid:
            profile_info = d
        if d.deviceId == front_mxid:
            front_info = d

    if profile_info is None:
        log(f"Preview: profile camera '{profile_mxid}' not found – skipping preview.")
        return (profile_mxid, front_mxid)
    if front_info is None:
        log(f"Preview: front camera '{front_mxid}' not found – skipping preview.")
        return (profile_mxid, front_mxid)

    result = None  # will be set on confirm

    try:
        with contextlib.ExitStack() as stack:
            # Connect by MXID string (not DeviceInfo) for reliable USB
            # negotiation – matches how the REBA analysis phase connects.
            log("Opening preview – connecting to profile camera...")
            profile_dev = stack.enter_context(
                dai.Device(profile_mxid))
            log(f"  Connected: {profile_dev.getDeviceName()}")
            time.sleep(2.0)

            log("Opening preview – connecting to front camera...")
            front_dev = stack.enter_context(
                dai.Device(front_mxid))
            log(f"  Connected: {front_dev.getDeviceName()}")
            time.sleep(2.0)

            # Camera-only pipelines (no NN)
            log("Building profile preview pipeline...")
            profile_pipeline = stack.enter_context(
                dai.Pipeline(profile_dev))
            log("  Creating profile camera node...")
            p_cam = profile_pipeline.create(dai.node.Camera).build()
            log("  Creating profile output queue...")
            p_q = p_cam.requestOutput((512, 288)).createOutputQueue()
            log("  Starting profile pipeline...")
            profile_pipeline.start()
            log("  Profile pipeline started.")
            time.sleep(2.0)

            log("Building front preview pipeline...")
            front_pipeline = stack.enter_context(
                dai.Pipeline(front_dev))
            log("  Creating front camera node...")
            f_cam = front_pipeline.create(dai.node.Camera).build()
            log("  Creating front output queue...")
            f_q = f_cam.requestOutput((512, 288)).createOutputQueue()
            log("  Starting front pipeline...")
            front_pipeline.start()
            log("  Front pipeline started.")
            log("  's' swap  |  Enter/Space/'r' confirm  |  'q' quit\n")

            swapped = False
            profile_frame = None
            front_frame = None
            fps_count = 0
            fps_timer = time.monotonic()
            fps_value = 0.0

            while (profile_pipeline.isRunning()
                   and front_pipeline.isRunning()):
                p_msg = p_q.tryGet()
                f_msg = f_q.tryGet()

                if p_msg is not None:
                    raw = p_msg.getCvFrame()
                    if DISPLAY_SCALE != 1.0:
                        nw = int(raw.shape[1] * DISPLAY_SCALE)
                        nh = int(raw.shape[0] * DISPLAY_SCALE)
                        profile_frame = cv2.resize(raw, (nw, nh))
                    else:
                        profile_frame = raw.copy()

                if f_msg is not None:
                    raw = f_msg.getCvFrame()
                    if DISPLAY_SCALE != 1.0:
                        nw = int(raw.shape[1] * DISPLAY_SCALE)
                        nh = int(raw.shape[0] * DISPLAY_SCALE)
                        front_frame = cv2.resize(raw, (nw, nh))
                    else:
                        front_frame = raw.copy()

                if profile_frame is not None and front_frame is not None:
                    fps_count += 1
                    elapsed = time.monotonic() - fps_timer
                    if elapsed >= 1.0:
                        fps_value = fps_count / elapsed
                        fps_count = 0
                        fps_timer = time.monotonic()

                    cam_h = max(profile_frame.shape[0], front_frame.shape[0])
                    pf = profile_frame
                    ff = front_frame
                    if pf.shape[0] != cam_h:
                        s = cam_h / pf.shape[0]
                        pf = cv2.resize(pf, (int(pf.shape[1] * s), cam_h))
                    if ff.shape[0] != cam_h:
                        s = cam_h / ff.shape[0]
                        ff = cv2.resize(ff, (int(ff.shape[1] * s), cam_h))

                    # Labels depend on swap state
                    if swapped:
                        left_label, right_label = "FRONT", "PROFILE"
                        left_id = front_mxid[-6:]
                        right_id = profile_mxid[-6:]
                    else:
                        left_label, right_label = "PROFILE", "FRONT"
                        left_id = profile_mxid[-6:]
                        right_id = front_mxid[-6:]

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    for frame, label, dev_id in [
                        (pf, left_label, left_id),
                        (ff, right_label, right_id),
                    ]:
                        color = ((0, 255, 0) if label == "PROFILE"
                                 else (255, 200, 0))
                        cv2.putText(frame, label, (10, 30), font, 0.8,
                                    (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(frame, label, (10, 30), font, 0.8,
                                    color, 2, cv2.LINE_AA)
                        cv2.putText(frame, f"ID: ...{dev_id}",
                                    (10, 55), font, 0.45,
                                    (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"ID: ...{dev_id}",
                                    (10, 55), font, 0.45,
                                    (200, 200, 200), 1, cv2.LINE_AA)

                    sep = np.full((cam_h, 3, 3), (100, 100, 100),
                                  dtype=np.uint8)
                    combined = np.hstack([pf, sep, ff])

                    h, w = combined.shape[:2]
                    bar_h = 30
                    combined[-bar_h:, :] = (40, 40, 40)
                    cv2.putText(
                        combined,
                        f"FPS: {fps_value:.1f}  |  's' swap  |  "
                        f"Enter/Space/'r' start REBA  |  'q' quit",
                        (10, h - 10), font, 0.45,
                        (180, 180, 180), 1, cv2.LINE_AA)

                    cv2.imshow("Camera Preview – Verify Assignments",
                               combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    result = None
                    break
                if key == ord("s"):
                    swapped = not swapped
                    state = "SWAPPED" if swapped else "ORIGINAL"
                    log(f"Camera assignments {state}")
                if key in (13, 32, ord("r")):  # Enter, Space, 'r'
                    if swapped:
                        result = (front_mxid, profile_mxid)
                    else:
                        result = (profile_mxid, front_mxid)
                    break

            cv2.destroyAllWindows()

    except RuntimeError as exc:
        if "X_LINK" in str(exc) or "device" in str(exc).lower():
            log("\nPreview ERROR: Could not connect to a camera.")
            log("Falling back to configured assignments.\n")
            return (profile_mxid, front_mxid)
        raise

    if result is None:
        return None

    # Wait for USB re-enumeration before returning
    log("Preview closed – waiting for devices to re-enumerate...")
    _wait_for_devices_available(result[0], result[1])
    # Extra settle time for USB stack to be fully ready
    time.sleep(3.0)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Dual OAK-D Pro fused REBA scoring (two-camera setup)")
    parser.add_argument(
        "--front-camera", metavar="MXID",
        help="Front-facing camera MxID")
    parser.add_argument(
        "--profile-camera", metavar="MXID",
        help="Side-view (profile) camera MxID")
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List connected OAK devices and exit")
    parser.add_argument(
        "--csv", default="dual_reba_angles.csv",
        help="Output CSV file (default: dual_reba_angles.csv)")
    parser.add_argument(
        "--video", default="dual_reba_overlay.mp4",
        help="Output video file (default: dual_reba_overlay.mp4)")
    parser.add_argument(
        "--alert-log", default="reba_alerts.csv",
        help="Alert log CSV (default: reba_alerts.csv)")

    # Manual flags (conditions that can't be detected by camera)
    parser.add_argument("--arm-supported", action="store_true",
                        help="Arm supported or leaning (-1 upper arm score)")
    parser.add_argument("--wrist-deviated", action="store_true",
                        help="Wrist bent from midline (+1 wrist score)")
    parser.add_argument("--wrist-twisted", action="store_true",
                        help="Wrist is twisted (+1 wrist score)")

    # REBA adjustment scores (Force/Load, Coupling, Activity)
    parser.add_argument(
        "--load-score", type=int, default=0, choices=[0, 1, 2, 3],
        help="Force/Load adjustment for Score A (0: <5kg, 1: 5-10kg, "
             "2: >10kg, 3: >10kg + shock)")
    parser.add_argument(
        "--coupling-score", type=int, default=0, choices=[0, 1, 2, 3],
        help="Coupling/grip adjustment for Score B (0: good, 1: fair, "
             "2: poor, 3: unacceptable)")
    parser.add_argument(
        "--activity-score", type=int, default=0, choices=[0, 1, 2, 3],
        help="Activity adjustment for final score (+1 each: static "
             "posture >1min, repetitive action, rapid change)")

    args = parser.parse_args()

    # Generous USB timeouts – must be set before any device operations
    os.environ.setdefault("DEPTHAI_WATCHDOG_INITIAL_DELAY", "60000")
    os.environ.setdefault("DEPTHAI_BOOTUP_TIMEOUT", "60000")

    # --- List devices mode ---
    if args.list_devices:
        list_available_devices()
        sys.exit(0)

    # --- Validate camera args ---
    if not args.profile_camera or not args.front_camera:
        # Auto-detect if exactly two devices
        devices = dai.Device.getAllAvailableDevices()
        if len(devices) == 2 and not args.profile_camera and not args.front_camera:
            args.profile_camera = devices[0].deviceId
            args.front_camera = devices[1].deviceId
            log(f"Auto-detected 2 cameras:")
            log(f"  Profile (side): {args.profile_camera}")
            log(f"  Front:          {args.front_camera}")
            log("  (Use --profile-camera / --front-camera to override)")
        else:
            print("ERROR: Both --profile-camera and --front-camera MxIDs are required.",
                  file=sys.stderr)
            print("Use --list-devices to find connected camera MxIDs.", file=sys.stderr)
            if devices:
                print(f"\nFound {len(devices)} device(s):")
                for info in devices:
                    print(f"  MxID: {info.deviceId}")
            sys.exit(1)

    # --- Camera preview phase ---
    preview_result = run_camera_preview(
        args.profile_camera, args.front_camera)
    if preview_result is None:
        log("User quit during camera preview.")
        sys.exit(0)
    args.profile_camera, args.front_camera = preview_result

    csv_path = os.path.abspath(args.csv)
    video_path = os.path.abspath(args.video)

    manual_flags = {
        "arm_supported": args.arm_supported,
        "wrist_deviated": args.wrist_deviated,
        "wrist_twisted": args.wrist_twisted,
        "force_load": args.load_score,
        "coupling": args.coupling_score,
        "activity": args.activity_score,
    }

    # Per-camera flags (no manual twist/side-bend/abduction/raised --
    # those are auto-detected from the front camera)
    profile_flags = {
        "neck_twist": False,
        "neck_side_bend": False,
        "trunk_twist": False,
        "trunk_side_bend": False,
        "shoulder_raised": False,
        "arm_abducted": False,
        "arm_supported": args.arm_supported,
        "wrist_deviated": args.wrist_deviated,
        "wrist_twisted": args.wrist_twisted,
        "unilateral_legs": False,
    }
    front_flags = dict(profile_flags)

    log("=" * 56)
    log("Dual OAK-D Pro Fused REBA Scoring")
    log("=" * 56)
    log(f"Model  : {MODEL_SLUG}")
    log(f"Profile: {args.profile_camera}")
    log(f"Front  : {args.front_camera}")
    log(f"CSV out: {csv_path}")
    log(f"Video  : {video_path}")
    log(f"Alerts : {os.path.abspath(args.alert_log)}")
    active_flags = [k for k, v in manual_flags.items()
                    if v and k not in ("force_load", "coupling", "activity")]
    if active_flags:
        log(f"Manual : {', '.join(active_flags)}")
    if manual_flags["force_load"] or manual_flags["coupling"] or manual_flags["activity"]:
        log(f"Adjust : Force/Load +{manual_flags['force_load']}, "
            f"Coupling +{manual_flags['coupling']}, "
            f"Activity +{manual_flags['activity']}")
    log("")

    # --- Validate devices are available (fresh enumeration) ---
    initialize_devices(args.profile_camera, args.front_camera)

    smoother = ScoreSmoother()
    alert_logger = AlertLogger(args.alert_log)
    synchronizer = FrameSynchronizer()

    csv_file = open(csv_path, "w", newline="")
    csv_file.write(DUAL_CSV_HEADER)
    csv_file.flush()
    row_count = 0

    video_writer = None
    fps_value = 0.0
    frame_count = 0
    fps_timer = time.monotonic()

    # Panel dimensions (center info panel)
    PANEL_WIDTH = 200

    try:
        with contextlib.ExitStack() as stack:
            # Connect by MXID string (not DeviceInfo) for fresh USB lookup
            log("Connecting to profile camera...")
            profile_dev = stack.enter_context(
                dai.Device(args.profile_camera))
            log(f"Profile camera connected: {profile_dev.getDeviceName()}")

            log("Connecting to front camera...")
            front_dev = stack.enter_context(
                dai.Device(args.front_camera))
            log(f"Front camera connected: {front_dev.getDeviceName()}")

            # Build and start profile camera pipeline
            log("Starting profile camera pipeline...")
            profile_pipeline = stack.enter_context(
                dai.Pipeline(profile_dev))
            p_cam = profile_pipeline.create(dai.node.Camera).build()
            p_model_desc = dai.NNModelDescription(MODEL_SLUG)
            p_nn = profile_pipeline.create(ParsingNeuralNetwork).build(
                p_cam, p_model_desc)
            p_det_q = p_nn.out.createOutputQueue()
            p_pass_q = p_nn.passthrough.createOutputQueue()
            profile_pipeline.start()
            log("Profile camera started.")

            time.sleep(1.0)  # XLink stability stagger

            # Build and start front camera pipeline
            log("Starting front camera pipeline...")
            front_pipeline = stack.enter_context(
                dai.Pipeline(front_dev))
            f_cam = front_pipeline.create(dai.node.Camera).build()
            f_model_desc = dai.NNModelDescription(MODEL_SLUG)
            f_nn = front_pipeline.create(ParsingNeuralNetwork).build(
                f_cam, f_model_desc)
            f_det_q = f_nn.out.createOutputQueue()
            f_pass_q = f_nn.passthrough.createOutputQueue()
            front_pipeline.start()
            log("Front camera started.")
            log("Press 'q' in the OpenCV window to quit.\n")

            # Tracking variables
            profile_display = None
            front_display = None
            combined_frame = None

            try:
                while (profile_pipeline.isRunning()
                       and front_pipeline.isRunning()):

                    # --- Poll profile camera ---
                    p_frame_msg = p_pass_q.tryGet()
                    p_det_msg = p_det_q.tryGet()

                    if p_frame_msg is not None:
                        raw = p_frame_msg.getCvFrame()
                        if DISPLAY_SCALE != 1.0:
                            nw = int(raw.shape[1] * DISPLAY_SCALE)
                            nh = int(raw.shape[0] * DISPLAY_SCALE)
                            profile_display = cv2.resize(raw, (nw, nh))
                        else:
                            profile_display = raw.copy()

                    if (p_det_msg is not None
                            and profile_display is not None):
                        ph, pw = profile_display.shape[:2]
                        p_rb = _empty_reba()
                        p_pts = []
                        det = best_detection(p_det_msg.detections)
                        if det is not None and len(det.getKeypoints()) >= 17:
                            p_pts = keypoints_to_pixels(
                                det.getKeypoints(), pw, ph)
                            p_rb = compute_reba(p_pts, profile_flags)
                        draw_reba_skeleton(profile_display, p_pts, p_rb)
                        draw_reba_table(
                            profile_display, p_rb, fps_value,
                            label="PROFILE")
                        synchronizer.update_profile(
                            profile_display.copy(), p_rb, p_pts)

                    # --- Poll front camera ---
                    f_frame_msg = f_pass_q.tryGet()
                    f_det_msg = f_det_q.tryGet()

                    if f_frame_msg is not None:
                        raw = f_frame_msg.getCvFrame()
                        if DISPLAY_SCALE != 1.0:
                            nw = int(raw.shape[1] * DISPLAY_SCALE)
                            nh = int(raw.shape[0] * DISPLAY_SCALE)
                            front_display = cv2.resize(raw, (nw, nh))
                        else:
                            front_display = raw.copy()

                    if (f_det_msg is not None
                            and front_display is not None):
                        fh, fw = front_display.shape[:2]
                        f_rb = _empty_reba()
                        f_pts = []
                        det = best_detection(f_det_msg.detections)
                        if det is not None and len(det.getKeypoints()) >= 17:
                            f_pts = keypoints_to_pixels(
                                det.getKeypoints(), fw, fh)
                            f_rb = compute_reba(f_pts, front_flags)
                        draw_reba_skeleton(front_display, f_pts, f_rb)
                        draw_reba_table(
                            front_display, f_rb, fps_value,
                            label="FRONT")
                        synchronizer.update_front(
                            front_display.copy(), f_rb, f_pts)

                    # --- Check for synced pair ---
                    pair = synchronizer.get_synced_pair()
                    if pair is not None:
                        (p_frame, p_rb, p_pts), \
                            (f_frame, f_rb, f_pts) = pair

                        # Fuse
                        fused_rb = fuse_reba_dicts(
                            p_rb, f_rb, p_pts, f_pts, manual_flags)

                        fused_score = fused_rb["fused_reba_score"]
                        smoothed = smoother.update(fused_score)

                        # Alert check
                        alert_logger.check(smoothed, fused_rb)

                        # FPS
                        frame_count += 1
                        elapsed = time.monotonic() - fps_timer
                        if elapsed >= 1.0:
                            fps_value = frame_count / elapsed
                            frame_count = 0
                            fps_timer = time.monotonic()

                        # Build combined display
                        cam_h = max(p_frame.shape[0], f_frame.shape[0])
                        cam_w_p = p_frame.shape[1]
                        cam_w_f = f_frame.shape[1]

                        # Resize frames to same height
                        if p_frame.shape[0] != cam_h:
                            scale = cam_h / p_frame.shape[0]
                            p_frame = cv2.resize(
                                p_frame,
                                (int(p_frame.shape[1] * scale), cam_h))
                            cam_w_p = p_frame.shape[1]
                        if f_frame.shape[0] != cam_h:
                            scale = cam_h / f_frame.shape[0]
                            f_frame = cv2.resize(
                                f_frame,
                                (int(f_frame.shape[1] * scale), cam_h))
                            cam_w_f = f_frame.shape[1]

                        # Center panel
                        panel = np.zeros(
                            (cam_h, PANEL_WIDTH, 3), dtype=np.uint8)
                        draw_fused_panel(
                            panel, fused_rb, fps_value, smoothed,
                            adjustments=manual_flags)

                        # Combine: [PROFILE | PANEL | FRONT]
                        combined_frame = np.hstack(
                            [p_frame, panel, f_frame])

                        # Alert border
                        draw_alert_border(combined_frame, smoothed)

                        # Initialize video writer on first frame
                        if video_writer is None:
                            ch, cw = combined_frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(
                                video_path, fourcc, 15.0, (cw, ch))
                            if not video_writer.isOpened():
                                log(f"WARNING: Could not open video "
                                    f"writer at {video_path}")

                        # Record video
                        if (video_writer is not None
                                and video_writer.isOpened()):
                            video_writer.write(combined_frame)

                        # CSV: 3 rows per synced frame
                        csv_file.write(dual_reba_to_csv_line(
                            p_rb, "profile", fps_value, fused_rb,
                            adjustments=manual_flags))
                        csv_file.write(dual_reba_to_csv_line(
                            f_rb, "front", fps_value, fused_rb,
                            adjustments=manual_flags))
                        csv_file.write(dual_reba_to_csv_line(
                            fused_rb, "fused", fps_value, fused_rb,
                            adjustments=manual_flags))
                        csv_file.flush()
                        row_count += 3

                    # --- Watchdog: no pairs for too long ---
                    no_pair_sec = synchronizer.seconds_since_last_pair()
                    if no_pair_sec > 5.0 and no_pair_sec <= 5.5:
                        log("WARNING: No synced camera pair for >5s. "
                            "Check camera connections.")
                    if no_pair_sec > 15.0:
                        log("ERROR: No synced pair for >15s. "
                            "Saving and exiting.")
                        break

                    # --- Display ---
                    if combined_frame is not None:
                        cv2.imshow(
                            "Dual OAK-D Pro REBA", combined_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            finally:
                if video_writer is not None:
                    video_writer.release()
                csv_file.close()
                alert_logger.close()
                cv2.destroyAllWindows()
                log(f"\nSaved {row_count} CSV rows to {csv_path}")
                log(f"Saved video to {video_path}")
                log(f"Logged {alert_logger.count} alerts to "
                    f"{alert_logger.path}")
                # Stop pipelines explicitly before ExitStack closes devices
                log("Stopping pipelines...")
                try:
                    profile_pipeline.stop()
                except Exception:
                    pass
                try:
                    front_pipeline.stop()
                except Exception:
                    pass
                log("Closing devices...")
                try:
                    profile_dev.close()
                except Exception:
                    pass
                try:
                    front_dev.close()
                except Exception:
                    pass

    except RuntimeError as exc:
        if "X_LINK" in str(exc) or "device" in str(exc).lower():
            log("")
            log("ERROR: Could not connect to an OAK device.")
            log("Troubleshooting steps:")
            log("  1. Make sure both OAK-D Pro cameras are plugged into "
                "USB 3 ports")
            log("  2. Try different USB cables (must be data-capable)")
            log("  3. Use --list-devices to check device availability")
            log("  4. Ensure no other application is using the cameras")
            log("  5. Try unplugging and re-plugging both cameras")
            sys.exit(1)
        raise

    log("Done.")


if __name__ == "__main__":
    main()
