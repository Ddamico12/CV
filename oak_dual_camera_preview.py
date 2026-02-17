#!/usr/bin/env python3
"""
Dual OAK-D Pro: Camera Preview for Placement
==============================================
Displays live feeds from two OAK-D Pro cameras side by side so you can
position them before running the REBA scoring script.

Each feed is labeled with its assigned role (PROFILE / FRONT) and device ID.

Install:
    pip install depthai opencv-python numpy

Run:
    python oak_dual_camera_preview.py --list-devices
    python oak_dual_camera_preview.py --profile-camera <MXID> --front-camera <MXID>
    python oak_dual_camera_preview.py   # auto-detect if exactly 2 cameras

Press 'q' in the OpenCV window to quit.
Press 's' to swap the PROFILE and FRONT camera assignments.
"""

import os
import sys
import time
import argparse
import contextlib
import numpy as np
import cv2

try:
    import depthai as dai
except ImportError:
    print(
        "ERROR: depthai is not installed.\n"
        "Run:  pip install depthai opencv-python numpy",
        file=sys.stderr,
    )
    sys.exit(1)

DISPLAY_SCALE = 1.5


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def list_available_devices():
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("No OAK devices found.")
        return []
    print(f"Found {len(devices)} OAK device(s):")
    for i, info in enumerate(devices):
        print(f"  [{i}] DeviceID: {info.deviceId}  Name: {info.name}  "
              f"State: {info.state}")
    return devices


def main():
    parser = argparse.ArgumentParser(
        description="Dual OAK-D Pro camera preview for placement")
    parser.add_argument(
        "--front-camera", metavar="MXID",
        help="Front-facing camera DeviceID")
    parser.add_argument(
        "--profile-camera", metavar="MXID",
        help="Side-view (profile) camera DeviceID")
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List connected OAK devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_available_devices()
        sys.exit(0)

    # Auto-detect if both cameras not specified
    if not args.profile_camera or not args.front_camera:
        devices = dai.Device.getAllAvailableDevices()
        if len(devices) == 2 and not args.profile_camera and not args.front_camera:
            args.profile_camera = devices[0].deviceId
            args.front_camera = devices[1].deviceId
            log(f"Auto-detected 2 cameras:")
            log(f"  Profile (side): {args.profile_camera}")
            log(f"  Front:          {args.front_camera}")
            log("  Press 's' to swap assignments")
        else:
            print("ERROR: Both --profile-camera and --front-camera are required.",
                  file=sys.stderr)
            print("Use --list-devices to find connected camera DeviceIDs.",
                  file=sys.stderr)
            sys.exit(1)

    profile_mxid = args.profile_camera
    front_mxid = args.front_camera

    if profile_mxid == front_mxid:
        print("ERROR: --profile-camera and --front-camera cannot be the same.",
              file=sys.stderr)
        sys.exit(1)

    # Find device infos
    devices = dai.Device.getAllAvailableDevices()
    profile_info = None
    front_info = None
    for d in devices:
        if d.deviceId == profile_mxid:
            profile_info = d
        if d.deviceId == front_mxid:
            front_info = d

    if profile_info is None:
        print(f"ERROR: Profile camera '{profile_mxid}' not found.", file=sys.stderr)
        list_available_devices()
        sys.exit(1)
    if front_info is None:
        print(f"ERROR: Front camera '{front_mxid}' not found.", file=sys.stderr)
        list_available_devices()
        sys.exit(1)

    os.environ.setdefault("DEPTHAI_WATCHDOG_INITIAL_DELAY", "60000")
    os.environ.setdefault("DEPTHAI_BOOTUP_TIMEOUT", "60000")

    log("=" * 48)
    log("Dual OAK-D Pro Camera Preview")
    log("=" * 48)
    log(f"Profile: {profile_mxid}")
    log(f"Front  : {front_mxid}")
    log("")

    try:
        with contextlib.ExitStack() as stack:
            log("Connecting to profile camera...")
            profile_dev = stack.enter_context(
                dai.Device(profile_info, dai.UsbSpeed.HIGH))
            log(f"  Connected: {profile_dev.getDeviceName()}")
            time.sleep(2.0)

            log("Connecting to front camera...")
            front_dev = stack.enter_context(
                dai.Device(front_info, dai.UsbSpeed.HIGH))
            log(f"  Connected: {front_dev.getDeviceName()}")
            time.sleep(2.0)

            # Build pipelines (camera only, no NN)
            log("Building pipelines...")
            profile_pipeline = stack.enter_context(
                dai.Pipeline(profile_dev))
            p_cam = profile_pipeline.create(dai.node.Camera).build()
            p_q = p_cam.requestOutput((512, 288)).createOutputQueue()

            front_pipeline = stack.enter_context(
                dai.Pipeline(front_dev))
            f_cam = front_pipeline.create(dai.node.Camera).build()
            f_q = f_cam.requestOutput((512, 288)).createOutputQueue()

            log("Starting cameras...")
            profile_pipeline.start()
            time.sleep(2.0)
            front_pipeline.start()
            log("Both cameras running. Press 'q' to quit, 's' to swap.\n")

            swapped = False
            profile_frame = None
            front_frame = None
            fps_count = 0
            fps_timer = time.monotonic()
            fps_value = 0.0

            while profile_pipeline.isRunning() and front_pipeline.isRunning():
                # Poll both cameras
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

                    # Resize to same height
                    cam_h = max(profile_frame.shape[0], front_frame.shape[0])
                    pf = profile_frame
                    ff = front_frame
                    if pf.shape[0] != cam_h:
                        s = cam_h / pf.shape[0]
                        pf = cv2.resize(pf, (int(pf.shape[1] * s), cam_h))
                    if ff.shape[0] != cam_h:
                        s = cam_h / ff.shape[0]
                        ff = cv2.resize(ff, (int(ff.shape[1] * s), cam_h))

                    # Determine labels based on swap state
                    if swapped:
                        left_label = "FRONT"
                        right_label = "PROFILE"
                        left_id = front_mxid[-6:]
                        right_id = profile_mxid[-6:]
                    else:
                        left_label = "PROFILE"
                        right_label = "FRONT"
                        left_id = profile_mxid[-6:]
                        right_id = front_mxid[-6:]

                    # Draw labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    for frame, label, dev_id in [
                        (pf, left_label, left_id),
                        (ff, right_label, right_id),
                    ]:
                        color = (0, 255, 0) if label == "PROFILE" else (255, 200, 0)
                        cv2.putText(frame, label, (10, 30), font, 0.8,
                                    (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(frame, label, (10, 30), font, 0.8,
                                    color, 2, cv2.LINE_AA)
                        cv2.putText(frame, f"ID: ...{dev_id}", (10, 55), font, 0.45,
                                    (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"ID: ...{dev_id}", (10, 55), font, 0.45,
                                    (200, 200, 200), 1, cv2.LINE_AA)

                    # Separator line
                    sep = np.full((cam_h, 3, 3), (100, 100, 100), dtype=np.uint8)
                    combined = np.hstack([pf, sep, ff])

                    # FPS and instructions at bottom
                    h, w = combined.shape[:2]
                    bar_h = 30
                    combined[-bar_h:, :] = (40, 40, 40)
                    cv2.putText(combined,
                                f"FPS: {fps_value:.1f}  |  'q' quit  |  's' swap cameras",
                                (10, h - 10), font, 0.45,
                                (180, 180, 180), 1, cv2.LINE_AA)

                    cv2.imshow("Dual OAK-D Pro Preview", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    swapped = not swapped
                    state = "SWAPPED" if swapped else "ORIGINAL"
                    log(f"Camera assignments {state}")

            cv2.destroyAllWindows()

    except RuntimeError as exc:
        if "X_LINK" in str(exc) or "device" in str(exc).lower():
            log("\nERROR: Could not connect to an OAK device.")
            log("  1. Make sure both cameras are plugged in")
            log("  2. Close OAK Viewer or other apps using the cameras")
            log("  3. Use --list-devices to check availability")
            sys.exit(1)
        raise

    log("Done.")


if __name__ == "__main__":
    main()
