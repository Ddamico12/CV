#!/usr/bin/env python3
"""
OAK-D Pro: On-Device Pose Estimation with REBA Scoring
=======================================================
Uses YOLOv8 Nano Pose from the Luxonis Model Zoo running entirely
on the OAK's VPU (Myriad X).  The model is downloaded automatically
on the first run.

Computes REBA (Rapid Entire Body Assessment) component scores for each
body region and displays color-coded skeleton overlay in real time.

All REBA angles and component scores are saved to CSV each frame.
The overlayed video is also saved to an MP4 file.

Wrist flexion/extension cannot be measured with COCO-17 keypoints
(no hand/finger points beyond the wrist), so wrist is reported as N/A.

Install:
    pip install depthai depthai-nodes opencv-python numpy

Run:
    python oak_pose_angles.py
    python oak_pose_angles.py --csv reba.csv --video reba.mp4

REBA adjustment flags (applied globally to all frames):
    python oak_pose_angles.py --neck-twist --trunk-side-bend
    python oak_pose_angles.py --shoulder-raised --arm-abducted

Press 'q' in the OpenCV window to quit.
"""

import os
import sys
import time
import math
import argparse
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


def reba_color(score):
    """Return BGR color for a REBA component score."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return REBA_COLORS[0]
    s = int(round(score))
    if s <= 0:
        return REBA_COLORS[0]
    return REBA_COLORS.get(min(s, 5), REBA_COLORS[5])


# ---------------------------------------------------------------------------
# Geometry helpers
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


# ---------------------------------------------------------------------------
# REBA scoring functions
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
    """Lower arm: 60-100 deg = 1, otherwise = 2."""
    if math.isnan(elbow_angle):
        return float("nan")
    if 60 <= elbow_angle <= 100:
        return 1
    return 2


def score_wrist(angle, deviated=False, twisted=False):
    """Wrist: 0-15 = 1, >15 = 2. +1 deviated, +1 twisted."""
    if math.isnan(angle):
        return float("nan")
    s = 1 if abs(angle) <= 15 else 2
    if deviated:
        s += 1
    if twisted:
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
# Keypoint extraction (OAK DepthAI normalized keypoints -> pixel coords)
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
# REBA computation
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
# Drawing
# ---------------------------------------------------------------------------
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


def draw_reba_table(frame, rb, fps):
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
        (f"FPS: {fps:.1f}",                                  (0, 255, 0)),
        ("---- REBA Scores ----",                             (200, 200, 200)),
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

    # OAK label at bottom
    cv2.putText(
        frame, "[OAK-D Pro VPU]", (8, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
CSV_HEADER = (
    "timestamp,"
    "neck_angle,neck_score,"
    "trunk_angle,trunk_score,"
    "L_upper_arm_angle,L_upper_arm_score,"
    "R_upper_arm_angle,R_upper_arm_score,"
    "L_lower_arm_angle,L_lower_arm_score,"
    "R_lower_arm_angle,R_lower_arm_score,"
    "L_wrist_angle,L_wrist_score,"
    "R_wrist_angle,R_wrist_score,"
    "L_knee_angle,R_knee_angle,legs_score,"
    "FPS\n"
)


def reba_to_csv_line(rb, fps):
    """Format one CSV row from REBA results."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="OAK-D Pro REBA pose scoring (on-device VPU inference)")
    parser.add_argument(
        "--csv", default="joint_angles.csv",
        help="Output CSV file path (default: joint_angles.csv)")
    parser.add_argument(
        "--video", default="pose_overlay.mp4",
        help="Output video file path (default: pose_overlay.mp4)")

    # REBA adjustment flags
    parser.add_argument("--neck-twist", action="store_true",
                        help="Neck is twisted (+1 neck score)")
    parser.add_argument("--neck-side-bend", action="store_true",
                        help="Neck has side bend (+1 neck score)")
    parser.add_argument("--trunk-twist", action="store_true",
                        help="Trunk is twisted (+1 trunk score)")
    parser.add_argument("--trunk-side-bend", action="store_true",
                        help="Trunk has side bend (+1 trunk score)")
    parser.add_argument("--shoulder-raised", action="store_true",
                        help="Shoulder is raised (+1 upper arm score)")
    parser.add_argument("--arm-abducted", action="store_true",
                        help="Upper arm is abducted (+1 upper arm score)")
    parser.add_argument("--arm-supported", action="store_true",
                        help="Arm supported or leaning (-1 upper arm score)")
    parser.add_argument("--wrist-deviated", action="store_true",
                        help="Wrist bent from midline (+1 wrist score)")
    parser.add_argument("--wrist-twisted", action="store_true",
                        help="Wrist is twisted (+1 wrist score)")
    parser.add_argument("--unilateral-legs", action="store_true",
                        help="Unilateral weight bearing (legs base = 2)")

    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    video_path = os.path.abspath(args.video)

    flags = {
        "neck_twist": args.neck_twist,
        "neck_side_bend": args.neck_side_bend,
        "trunk_twist": args.trunk_twist,
        "trunk_side_bend": args.trunk_side_bend,
        "shoulder_raised": args.shoulder_raised,
        "arm_abducted": args.arm_abducted,
        "arm_supported": args.arm_supported,
        "wrist_deviated": args.wrist_deviated,
        "wrist_twisted": args.wrist_twisted,
        "unilateral_legs": args.unilateral_legs,
    }

    def log(msg):
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    log("=" * 56)
    log("OAK-D Pro REBA Pose Scoring")
    log("=" * 56)
    log(f"Model  : {MODEL_SLUG}")
    log(f"CSV out: {csv_path}")
    log(f"Video  : {video_path}")

    active_flags = [k for k, v in flags.items() if v]
    if active_flags:
        log(f"Flags  : {', '.join(active_flags)}")
    else:
        log("Flags  : none (use --help to see REBA adjustment flags)")

    log("The model will be downloaded on the first run.")
    log("This may take a minute depending on your connection.")
    log("")

    # Default empty REBA dict (all NaN) for frames with no detection
    nan = float("nan")
    empty_reba = {
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

    # ------------------------------------------------------------------
    # Build the DepthAI v3 pipeline
    # ------------------------------------------------------------------
    try:
        with dai.Pipeline() as pipeline:
            # RGB camera (unified Camera node in v3)
            cam = pipeline.create(dai.node.Camera).build()

            # Neural network with built-in output parsing.
            model_desc = dai.NNModelDescription(MODEL_SLUG)
            nn = pipeline.create(ParsingNeuralNetwork).build(cam, model_desc)

            # Output queues
            det_q = nn.out.createOutputQueue()
            pass_q = nn.passthrough.createOutputQueue()

            # Start the pipeline on the device
            pipeline.start()
            log("Pipeline started. Press 'q' in the OpenCV window to quit.")
            log("")

            # CSV file
            csv_file = open(csv_path, "w", newline="")
            csv_file.write(CSV_HEADER)
            csv_file.flush()
            row_count = 0

            # Video writer (initialized on first frame)
            video_writer = None

            # Tracking variables
            display_frame = None
            fps_value = 0.0
            frame_count = 0
            fps_timer = time.monotonic()

            try:
                # ---- Main loop ----
                while pipeline.isRunning():
                    # Non-blocking reads so we can service the OpenCV window
                    frame_msg = pass_q.tryGet()
                    det_msg = det_q.tryGet()

                    # Keep the latest camera frame
                    if frame_msg is not None:
                        raw_frame = frame_msg.getCvFrame()
                        # Upscale for easier viewing
                        if DISPLAY_SCALE != 1.0:
                            new_w = int(raw_frame.shape[1] * DISPLAY_SCALE)
                            new_h = int(raw_frame.shape[0] * DISPLAY_SCALE)
                            display_frame = cv2.resize(raw_frame, (new_w, new_h))
                        else:
                            display_frame = raw_frame.copy()

                    # Process detections when available
                    if det_msg is not None and display_frame is not None:
                        h, w = display_frame.shape[:2]

                        # Initialize video writer on first detection frame
                        if video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(
                                video_path, fourcc, 30.0, (w, h))
                            if not video_writer.isOpened():
                                log(f"WARNING: Could not open video writer at {video_path}")

                        # FPS counter
                        frame_count += 1
                        elapsed = time.monotonic() - fps_timer
                        if elapsed >= 1.0:
                            fps_value = frame_count / elapsed
                            frame_count = 0
                            fps_timer = time.monotonic()

                        rb = dict(empty_reba)
                        pts = []

                        # Pick the most confident person
                        det = best_detection(det_msg.detections)
                        if det is not None and len(det.getKeypoints()) >= 17:
                            pts = keypoints_to_pixels(det.getKeypoints(), w, h)
                            rb = compute_reba(pts, flags)

                        # Draw
                        draw_reba_skeleton(display_frame, pts, rb)
                        draw_reba_table(display_frame, rb, fps_value)

                        # Record video
                        if video_writer is not None and video_writer.isOpened():
                            video_writer.write(display_frame)

                        # CSV
                        csv_file.write(reba_to_csv_line(rb, fps_value))
                        csv_file.flush()
                        row_count += 1

                    # Show frame
                    if display_frame is not None:
                        cv2.imshow("OAK-D Pro REBA Pose Scoring", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            finally:
                if video_writer is not None:
                    video_writer.release()
                csv_file.close()
                cv2.destroyAllWindows()
                log(f"Saved {row_count} rows to {csv_path}")
                log(f"Saved video to {video_path}")

    except RuntimeError as exc:
        if "X_LINK" in str(exc) or "device" in str(exc).lower():
            log("")
            log("ERROR: Could not connect to the OAK device.")
            log("Troubleshooting steps:")
            log("  1. Make sure the OAK-D Pro is plugged into a USB 3 port")
            log("  2. Try a different USB cable (must be data-capable)")
            log("  3. Open Device Manager and look for Movidius / Myriad")
            log("  4. Run: python -c \"import depthai as dai; "
                "print(dai.Device.getAllAvailableDevices())\"")
            sys.exit(1)
        raise

    log("Done.")


if __name__ == "__main__":
    main()
