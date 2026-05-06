# 🛒 Smart Retail Theft Detection System

A real-time AI-powered retail surveillance system using **YOLOv8** and **OpenCV** that monitors CCTV feeds for suspicious activity, zone intrusions, and potential theft — with a live web dashboard.

---

## Features

| Feature | Description |
|---|---|
| 👤 Person Tracking | Tracks individuals across frames with unique IDs and motion trails |
| 🎒 Bag Detection | Detects backpacks, handbags, and suitcases |
| 📱 Mobile Detection | Flags phones near checkout areas |
| 🔪 Suspicious Items | Detects knives, scissors, and other flagged objects |
| 🚧 Zone Intrusion | Alerts when a person lingers in a restricted area |
| 👻 Sudden Disappearance | Flags persons who vanish abruptly (possible concealment) |
| 📸 Screenshot Capture | Auto-saves a snapshot to `alerts/` on every danger event |
| 📊 Live Dashboard | Real-time web UI with video feed, stats, and alert feed |

---

## Demo

```
┌──────────────────────────────────┬────────────────┐
│                                  │  Alert Feed    │
│       Live CCTV Feed             │────────────────│
│   [bounding boxes + trails]      │ ⚠ Zone Intrusion│
│                                  │ ⚠ Mobile Phone  │
│                                  │ ✓ Bag Detected  │
├────────┬────────┬────────┬───────┴────────────────┤
│Persons │  Bags  │Mobiles │ Zone Alerts │ Total     │
└────────┴────────┴────────┴─────────────┴──────────┘
```

---

## Requirements

- Python 3.8+
- Webcam or video file
- Internet connection on first run (downloads YOLOv8 weights ~6 MB)

---

## Installation

**1. Clone or download the project**

```bash
git clone https://github.com/srikarkarthik-678/smart-retail-theft-detection
cd smart-retail-theft-detection
```

**2. (Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install ultralytics opencv-python flask flask-socketio
```

---

## Usage

**Run the system**

```bash
python retail_theft_detection.py
```

**Open the dashboard**

```
http://localhost:5000
```

The system will automatically download `yolov8n.pt` (~6 MB) on the first run.

---

## Configuration

All settings are at the top of `retail_theft_detection.py`:

```python
VIDEO_SOURCE      = 0          # 0 = webcam, or "path/to/video.mp4"
ALERT_COOLDOWN    = 5          # seconds between repeated alerts (same type)
SCREENSHOT_DIR    = "alerts"   # folder where alert snapshots are saved
CONFIDENCE_THRESH = 0.40       # YOLO detection confidence (0.0 – 1.0)
LINGER_THRESHOLD  = 8          # seconds in restricted zone before alert fires
```

### Defining Restricted Zones

Zones are defined as fractions of the frame (0.0 – 1.0):

```python
RESTRICTED_ZONES = [
    (x1_frac, y1_frac, x2_frac, y2_frac),
]
```

**Example — mark top-right corner as restricted:**

```python
RESTRICTED_ZONES = [
    (0.65, 0.0, 1.0, 0.6),   # right 35% × top 60%
]
```

You can add multiple zones:

```python
RESTRICTED_ZONES = [
    (0.65, 0.0, 1.0, 0.6),   # staff area
    (0.0,  0.8, 0.3, 1.0),   # back storeroom
]
```

---

## Project Structure

```
smart-retail-theft-detection/
│
├── retail_theft_detection.py   # Main script (detection + dashboard)
├── alerts/                     # Auto-created; stores alert screenshots
│   └── 20260506_143201_Zone_Intrusion.jpg
└── README.md
```

---

## How It Works

```
Video Source (webcam / file)
        │
        ▼
  YOLOv8 .track()  ←── persists track IDs across frames
        │
        ▼
  Per-detection logic
  ├── Person?   → update trail, check restricted zones, check disappearance
  ├── Bag?      → fire warning alert
  ├── Mobile?   → fire warning alert + screenshot
  └── Weapon?   → fire danger alert + screenshot
        │
        ▼
  Flask + SocketIO  →  Browser dashboard (MJPEG feed + live alerts)
```

---

## Alert Severity Levels

| Level | Colour | Examples |
|---|---|---|
| 🔴 Danger | Red | Zone intrusion, weapon detected |
| 🟡 Warning | Amber | Bag detected, mobile phone in use, person vanished |
| 🔵 Info | Blue | General activity |

---

## Tech Stack

- **[YOLOv8](https://github.com/ultralytics/ultralytics)** — object detection and tracking
- **[OpenCV](https://opencv.org/)** — video capture and frame processing
- **[Flask](https://flask.palletsprojects.com/)** — web server
- **[Flask-SocketIO](https://flask-socketio.readthedocs.io/)** — real-time alert streaming

---

## Known Limitations

- Detection accuracy depends on camera angle, lighting, and resolution.
- YOLOv8-nano (`yolov8n`) is used for speed; swap to `yolov8s.pt` or `yolov8m.pt` for better accuracy at the cost of performance.
- Zone intrusion requires the person's bounding-box centre to enter the zone — partial overlap is not flagged.

---

## Possible Improvements

- Add face blurring for privacy compliance
- Integrate with IP cameras via RTSP stream (`VIDEO_SOURCE = "rtsp://..."`)
- Add email / SMS notifications on danger alerts
- Store alerts in SQLite for historical review
- Add multi-camera support

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with YOLOv8 + OpenCV · Part of a retail AI surveillance research project*
