#!/usr/bin/env python3
"""
OAK-D Pro: On-Device Pose Estimation with Real-Time Joint Angles
================================================================
Uses YOLOv8 Nano Pose from the Luxonis Model Zoo running entirely
on the OAK's VPU (Myriad X).  The model is downloaded automatically
on the first run.

Outputs:
  - OpenCV window with skeleton overlay and angle readouts
  - CSV lines to stdout (redirect to a file if desired)

Press 'q' in the OpenCV window to quit.

Install (PowerShell, Python 3.10 or 3.11):
    pip install depthai depthai-nodes opencv-python numpy

Why YOLOv8 Pose instead of human-pose-estimation-0001?
    The Intel Open Model Zoo model outputs heatmaps + Part Affinity
    Fields that require 200+ lines of non-trivial association code.
    YOLOv8 Pose outputs keypoints directly per detection, and the
    depthai-nodes ParsingNeuralNetwork decodes them automatically.
"""

import sys
import time
import math
import numpy as np
import cv2

try:
    import depthai as dai
except ImportError:
    print(
        "ERROR: depthai is not installed.\n"
        "Run:  pip install depthai depthai-nodes opencv-python numpy",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from depthai_nodes.node import ParsingNeuralNetwork
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
KP_THRESHOLD = 0.3      # minimum keypoint confidence
DET_THRESHOLD = 0.4      # minimum detection (person) confidence
DISPLAY_SCALE = 1.5      # upscale the display window for readability


# ---------------------------------------------------------------------------
# COCO-17 keypoint indices (standard YOLOv8-pose order)
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


# Skeleton edges for drawing (pairs of keypoint indices)
SKELETON_PAIRS = [
    (NOSE, L_EYE), (NOSE, R_EYE),
    (L_EYE, L_EAR), (R_EYE, R_EAR),
    (L_SHOULDER, R_SHOULDER),                     # across shoulders
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),   # left arm
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),   # right arm
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),    # torso sides
    (L_HIP, R_HIP),                               # across hips
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),          # left leg
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),          # right leg
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def angle_at(p1, vertex, p3):
    """Return the angle in degrees at *vertex* formed by rays to p1 and p3.

    Each point is (x, y) or None.  Returns NaN when any point is missing.
    """
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


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------
def keypoints_to_pixels(keypoints, w, h):
    """Convert normalized (0-1) Keypoint objects to pixel coords or None."""
    pts = []
    for kp in keypoints:
        if kp.confidence >= KP_THRESHOLD:
            pts.append((kp.x * w, kp.y * h))
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
# Drawing
# ---------------------------------------------------------------------------
def _ipt(pt):
    """Convert float (x, y) to integer tuple for OpenCV."""
    return (int(round(pt[0])), int(round(pt[1])))


def draw_skeleton(frame, pts):
    """Draw skeleton lines and keypoint dots."""
    for (i, j) in SKELETON_PAIRS:
        if i < len(pts) and j < len(pts) and pts[i] is not None and pts[j] is not None:
            cv2.line(frame, _ipt(pts[i]), _ipt(pts[j]), (0, 200, 0), 2, cv2.LINE_AA)
    for pt in pts:
        if pt is not None:
            cv2.circle(frame, _ipt(pt), 4, (0, 255, 255), -1, cv2.LINE_AA)


def draw_overlay(frame, angles, fps):
    """Draw FPS and angle values in the top-left corner."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    labels = [
        "L Elbow", "R Elbow", "L Knee", "R Knee",
        "L Shoulder", "R Shoulder", "L Hip", "R Hip",
    ]
    for k, (lbl, val) in enumerate(zip(labels, angles)):
        txt = f"{lbl}: {val:.0f} deg" if not math.isnan(val) else f"{lbl}: ---"
        cv2.putText(
            frame, txt, (8, 44 + k * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 0), 1, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    def log(msg):
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    log("=" * 56)
    log("OAK-D Pose Estimation with Joint Angles")
    log("=" * 56)
    log(f"Model : {MODEL_SLUG}")
    log("The model will be downloaded on the first run.")
    log("This may take a minute depending on your connection.")
    log("")

    # ------------------------------------------------------------------
    # Build the DepthAI v3 pipeline
    # ------------------------------------------------------------------
    try:
        with dai.Pipeline() as pipeline:
            # RGB camera (unified Camera node in v3)
            cam = pipeline.create(dai.node.Camera).build()

            # Neural network with built-in output parsing.
            # ParsingNeuralNetwork downloads the blob, sets up image
            # resize, runs inference on device, and decodes keypoints.
            model_desc = dai.NNModelDescription(MODEL_SLUG)
            nn = pipeline.create(ParsingNeuralNetwork).build(cam, model_desc)

            # Output queues
            det_q = nn.out.createOutputQueue()
            pass_q = nn.passthrough.createOutputQueue()

            # Print CSV header to stdout
            print(
                "timestamp,L_elbow,R_elbow,L_knee,R_knee,"
                "L_shoulder,R_shoulder,L_hip,R_hip,FPS"
            )
            sys.stdout.flush()

            # Start the pipeline on the device
            pipeline.start()
            log("Pipeline started. Press 'q' in the OpenCV window to quit.")
            log("")

            # Tracking variables
            display_frame = None
            fps_value = 0.0
            frame_count = 0
            fps_timer = time.monotonic()

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

                    # FPS counter
                    frame_count += 1
                    elapsed = time.monotonic() - fps_timer
                    if elapsed >= 1.0:
                        fps_value = frame_count / elapsed
                        frame_count = 0
                        fps_timer = time.monotonic()

                    # Start with unknown angles
                    angles = [float("nan")] * 8
                    pts = []

                    # Pick the most confident person
                    det = best_detection(det_msg.detections)
                    if det is not None and len(det.keypoints) >= 17:
                        pts = keypoints_to_pixels(det.keypoints, w, h)

                        # 8 joint angles: elbow, knee, shoulder, hip (left + right)
                        angles = [
                            angle_at(pts[L_SHOULDER], pts[L_ELBOW], pts[L_WRIST]),
                            angle_at(pts[R_SHOULDER], pts[R_ELBOW], pts[R_WRIST]),
                            angle_at(pts[L_HIP],      pts[L_KNEE],  pts[L_ANKLE]),
                            angle_at(pts[R_HIP],      pts[R_KNEE],  pts[R_ANKLE]),
                            angle_at(pts[L_ELBOW],    pts[L_SHOULDER], pts[L_HIP]),
                            angle_at(pts[R_ELBOW],    pts[R_SHOULDER], pts[R_HIP]),
                            angle_at(pts[L_SHOULDER], pts[L_HIP],    pts[L_KNEE]),
                            angle_at(pts[R_SHOULDER], pts[R_HIP],    pts[R_KNEE]),
                        ]

                    # Draw on the display frame
                    draw_skeleton(display_frame, pts)
                    draw_overlay(display_frame, angles, fps_value)

                    # CSV line to stdout
                    ts = time.strftime("%H:%M:%S")
                    vals = ",".join(f"{a:.1f}" for a in angles)
                    print(f"{ts},{vals},{fps_value:.1f}")
                    sys.stdout.flush()

                # Show frame
                if display_frame is not None:
                    cv2.imshow("OAK-D Pose Estimation", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except RuntimeError as exc:
        if "X_LINK" in str(exc) or "device" in str(exc).lower():
            log("")
            log("ERROR: Could not connect to the OAK device.")
            log("Troubleshooting steps:")
            log("  1. Make sure the OAK-D Pro is plugged into a USB 3 port")
            log("  2. Try a different USB cable (must be data-capable)")
            log("  3. Open Device Manager and look for Movidius / Myriad")
            log("  4. Run: python -c \"import depthai as dai; print(dai.Device.getAllAvailableDevices())\"")
            sys.exit(1)
        raise

    cv2.destroyAllWindows()
    log("Done.")


if __name__ == "__main__":
    main()
