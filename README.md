# REBA Pose Estimation

Real-time human pose estimation with **REBA (Rapid Entire Body Assessment)** ergonomic scoring. Uses YOLOv8 Nano Pose to detect 17 body keypoints, compute joint angles, and assign color-coded REBA risk scores per body region.

Three operating modes:

| Script | Inference | Hardware needed |
|---|---|---|
| `test_pose_angles_webcam.py` | Host CPU (webcam) | Any webcam |
| `oak_pose_angles.py` | OAK-D Pro VPU (on-device) | 1x Luxonis OAK-D Pro + USB 3 cable |
| `oak_dual_camera_reba.py` | OAK-D Pro VPU (on-device, dual) | 2x Luxonis OAK-D Pro + USB cables |

## Requirements

- **OS**: Windows 10 or 11 (x64)
- **Python**: 3.10 or 3.11 recommended (3.14 may work but some packages lack wheels)
- **Webcam**: Built-in or USB webcam for the test script
- **OAK-D Pro**: 1x required for `oak_pose_angles.py`, 2x required for `oak_dual_camera_reba.py`

## Installation

### 1. Install Python 3.11

Download from [python.org](https://www.python.org/downloads/) or use winget:

```powershell
winget install Python.Python.3.11
```

Verify:

```powershell
python --version
# Should show Python 3.11.x
```

### 2. Create a virtual environment

```powershell
cd C:\Users\<your-username>\SDP\CV
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

**For the webcam test script** (no OAK device needed):

```powershell
pip install ultralytics opencv-python numpy
```

Note: `ultralytics` pulls in PyTorch as a dependency. The first install is approximately 2 GB.

**For the OAK-D Pro on-device script** (additional packages):

```powershell
pip install depthai depthai-nodes opencv-python numpy
```

### 4. First run

The YOLOv8n-pose model weights (~6 MB) are downloaded automatically on the first run. Make sure you have internet access.

## Usage

### Webcam test script (REBA scoring)

```powershell
# Basic run with default webcam
python test_pose_angles_webcam.py

# Use a specific webcam index
python test_pose_angles_webcam.py --source 1

# Use a video file as input
python test_pose_angles_webcam.py --source C:\path\to\video.mp4

# Custom output file paths
python test_pose_angles_webcam.py --csv session1.csv --video session1.mp4
```

Press **q** in the OpenCV window to stop recording and quit.

### OAK-D Pro single-camera script

```powershell
# Plug in OAK-D Pro via USB 3 port, then:
python oak_pose_angles.py

# Custom output paths
python oak_pose_angles.py --csv session1.csv --video session1.mp4
```

### OAK-D Pro dual-camera script

Uses two OAK-D Pro cameras (one profile/side view, one front view) to compute a fused REBA score. The front camera auto-detects posture conditions that normally require manual CLI flags: trunk twist, trunk side-bend, neck twist, neck side-bend, arm abduction, shoulder raised, and unilateral stance.

```powershell
# List connected devices to find their MxIDs
python oak_dual_camera_reba.py --list-devices

# Run with explicit camera assignment
python oak_dual_camera_reba.py --profile-camera <MXID> --front-camera <MXID>

# Auto-detect (when exactly 2 cameras are connected)
python oak_dual_camera_reba.py

# With manual flags for conditions that can't be auto-detected
python oak_dual_camera_reba.py --arm-supported --wrist-deviated
```

The display shows three panels side by side: `[PROFILE view | FUSED REBA panel | FRONT view]`. The center panel shows the fused REBA score, risk level, detected flags, and component scores. A colored border indicates the overall risk level.

#### Auto-detected flags (from front camera)

| Condition | Detection method |
|---|---|
| Trunk twist | L/R torso-side length asymmetry (ratio < 0.75) |
| Trunk side-bend | Lateral lean of mid-shoulder vs mid-hip > 10 degrees |
| Neck twist | Ear visibility asymmetry (one ear hidden) |
| Neck side-bend | Nose lateral offset from mid-shoulder > 15% shoulder width |
| Arm abduction | Front-view arm angle > 30 degrees from trunk vertical |
| Shoulder raised | Shoulder height difference > 10% of torso length |
| Unilateral stance | Hip height asymmetry > 10% of shoulder width |

#### Dual-camera output files

| File | Description |
|---|---|
| `dual_reba_angles.csv` | 3 rows per synced frame (profile, front, fused) with angles, scores, and detected flags |
| `dual_reba_overlay.mp4` | Video of the three-panel display |
| `reba_alerts.csv` | Timestamped alerts when risk level changes or stays Medium+ for 10 seconds |

### REBA adjustment flags

The following flags apply REBA manual adjustments globally to all frames. Enable them when you visually observe the condition during recording:

| Flag | Effect | REBA region |
|---|---|---|
| `--neck-twist` | +1 neck score | Neck |
| `--neck-side-bend` | +1 neck score | Neck |
| `--trunk-twist` | +1 trunk score | Trunk |
| `--trunk-side-bend` | +1 trunk score | Trunk |
| `--shoulder-raised` | +1 upper arm score | Upper arm (both) |
| `--arm-abducted` | +1 upper arm score | Upper arm (both) |
| `--arm-supported` | -1 upper arm score | Upper arm (both) |
| `--wrist-deviated` | +1 wrist score | Wrist (both) |
| `--wrist-twisted` | +1 wrist score | Wrist (both) |
| `--unilateral-legs` | Legs base score = 2 | Legs |

Example with multiple flags:

```powershell
python test_pose_angles_webcam.py --neck-twist --trunk-side-bend --shoulder-raised
```

## Output files

Each run produces two output files:

### CSV (`joint_angles.csv` by default)

One row per frame with 21 data columns:

| Column | Description |
|---|---|
| `timestamp` | Wall clock time (HH:MM:SS) |
| `neck_angle` | Neck deviation from trunk line (degrees) |
| `neck_score` | REBA neck component score |
| `trunk_angle` | Trunk deviation from vertical (signed degrees) |
| `trunk_score` | REBA trunk component score |
| `L_upper_arm_angle` | Left upper arm angle from trunk (degrees) |
| `L_upper_arm_score` | REBA left upper arm score |
| `R_upper_arm_angle` | Right upper arm angle from trunk (degrees) |
| `R_upper_arm_score` | REBA right upper arm score |
| `L_lower_arm_angle` | Left elbow angle (degrees) |
| `L_lower_arm_score` | REBA left lower arm score |
| `R_lower_arm_angle` | Right elbow angle (degrees) |
| `R_lower_arm_score` | REBA right lower arm score |
| `L_wrist_angle` | Left wrist angle (always blank, see limitations) |
| `L_wrist_score` | Left wrist score (always blank) |
| `R_wrist_angle` | Right wrist angle (always blank, see limitations) |
| `R_wrist_score` | Right wrist score (always blank) |
| `L_knee_angle` | Left knee angle (degrees) |
| `R_knee_angle` | Right knee angle (degrees) |
| `legs_score` | REBA legs component score |
| `FPS` | Frames per second |

Blank cells mean the keypoint was not detected with sufficient confidence in that frame.

### Video (`pose_overlay.mp4` by default)

An MP4 recording of the webcam feed with all overlays baked in:
- Color-coded skeleton segments per REBA score
- Angle values drawn at each joint
- REBA scores table in the top-left corner
- FPS counter

## REBA scoring reference

### Score bins

**Neck** (angle = deviation of neck from trunk line):
- Score 1: 0 to 20 degrees
- Score 2: greater than 20 degrees (flexion or extension)

**Trunk** (angle = deviation from vertical):
- Score 1: less than 5 degrees (approximately erect)
- Score 2: 5 to 20 degrees
- Score 3: 20 to 60 degrees
- Score 4: greater than 60 degrees

**Upper arm** (angle = arm deviation from trunk downward):
- Score 1: 0 to 20 degrees
- Score 2: 20 to 45 degrees
- Score 3: 45 to 90 degrees
- Score 4: greater than 90 degrees

**Lower arm** (angle = elbow angle):
- Score 1: 60 to 100 degrees (comfortable mid-range)
- Score 2: less than 60 or greater than 100 degrees

**Wrist** (not measurable with current keypoints):
- Score 1: 0 to 15 degrees
- Score 2: greater than 15 degrees

**Legs** (base + knee flexion adjustment):
- Base 1 (bilateral weight bearing) or Base 2 (unilateral, via `--unilateral-legs`)
- +1 if knee flexion is 30 to 60 degrees
- +2 if knee flexion is greater than 60 degrees

### Color coding

The skeleton overlay and table text are colored by REBA component score:

| Score | Color | Meaning |
|---|---|---|
| 1 | Green | Low risk |
| 2 | Yellow | Moderate |
| 3 | Orange | High |
| 4 | Red | Very high |
| 5+ | Purple | Extreme |

## Known limitations

- **2D only**: Pose estimation is from a single camera with no depth information. Flexion vs extension is approximate, especially from a front-facing camera. For best trunk and neck readings, position the camera to the side of the subject. The dual-camera script mitigates this by fusing side and front views.
- **Wrist angles**: COCO-17 keypoints include the wrist but not the hand or fingers. Without a point beyond the wrist, wrist flexion/extension cannot be measured. Wrist columns in the CSV are always blank. Use `--wrist-deviated` and `--wrist-twisted` flags when applicable.
- **Twist/side-bend detection (single camera)**: Neck and trunk twist cannot be detected from a single 2D image. Use the `--neck-twist` and `--trunk-twist` CLI flags when you observe twisting. The dual-camera script auto-detects these from the front camera.
- **REBA flags are global**: Adjustment flags apply the same to every frame in the recording. They cannot be toggled mid-session.
- **Single person**: Only the highest-confidence detected person is scored per frame.
- **Host CPU speed**: The webcam test script runs at approximately 8 to 10 FPS on a typical laptop CPU. The OAK-D Pro on-device script targets approximately 30 FPS.
- **Dual-camera USB bandwidth**: Both OAK-D Pro cameras sharing the same USB host controller can cause intermittent XLink disconnects. For best stability, plug cameras into separate USB controllers (e.g., one front panel, one rear panel, or use a PCIe USB card).

## Troubleshooting

### Webcam not opening

- Try a different source index: `--source 1` or `--source 2`
- Check that no other application is using the webcam
- Verify OpenCV can see your camera:
  ```powershell
  python -c "import cv2; cap = cv2.VideoCapture(0); print('Open:', cap.isOpened()); cap.release()"
  ```

### OAK-D Pro not found

- Use a USB 3.0 data cable (not charge-only). Look for a blue USB port interior.
- Avoid USB hubs. Plug directly into the computer.
- Check Device Manager for "Movidius MyriadX" or "USB Boot Device"
- Install WinUSB drivers with [Zadig](https://zadig.akeo.ie/) if the device is not recognized
- Test detection:
  ```powershell
  python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
  ```

### Model download fails

- Ensure you have internet access on the first run
- If behind a corporate proxy, set the environment variable before running:
  ```powershell
  $env:HTTPS_PROXY = "http://your-proxy:port"
  ```

### Black or frozen OpenCV window

- The model takes a few seconds to load on the first frame. Wait 3 to 5 seconds.
- Make sure the camera lens is not blocked or covered
- Click on the OpenCV window to give it focus

### Import errors

- Make sure the virtual environment is activated: `.\.venv\Scripts\Activate.ps1`
- Verify packages: `pip show ultralytics opencv-python numpy`
- If using the OAK script, also verify: `pip show depthai depthai-nodes`

## Project structure

```
CV/
  test_pose_angles_webcam.py   # REBA scoring via webcam (host CPU)
  oak_pose_angles.py           # Single-camera REBA on OAK-D Pro (on-device)
  oak_dual_camera_reba.py      # Dual-camera fused REBA on 2x OAK-D Pro
  README.md                    # This file
  joint_angles.csv             # Generated: single-camera angles/scores per frame
  pose_overlay.mp4             # Generated: single-camera video with overlay
  dual_reba_angles.csv         # Generated: dual-camera fused angles/scores
  dual_reba_overlay.mp4        # Generated: dual-camera three-panel video
  reba_alerts.csv              # Generated: dual-camera risk-level alerts
  yolov8n-pose.pt              # Auto-downloaded model weights (not committed)
  .depthai_cached_models/      # Auto-downloaded OAK models (not committed)
```
