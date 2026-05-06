"""
Smart Retail Theft Detection System
====================================
Uses YOLOv8 + OpenCV for real-time retail surveillance.
Detects: person tracking, bag/mobile detection, zone intrusion,
suspicious behaviour, item removal alerts, and screenshot capture.

Requirements:
    pip install ultralytics opencv-python flask flask-socketio

Run:
    python retail_theft_detection.py

Then open: http://localhost:5000
"""

import cv2
import threading
import time
import base64
import os
import json
import random
from datetime import datetime
from collections import defaultdict, deque
from flask import Flask, render_template_string, Response, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VIDEO_SOURCE      = 0          # 0 = webcam; or path to video file e.g. "shop.mp4"
ALERT_COOLDOWN    = 5          # seconds between repeated alerts for same type
SCREENSHOT_DIR    = "alerts"   # folder to save alert screenshots
CONFIDENCE_THRESH = 0.40       # YOLO detection confidence threshold
LINGER_THRESHOLD  = 8          # seconds a person must stay in zone to trigger alert
MAX_ALERTS        = 50         # max alerts to keep in memory

# Restricted zones: list of (x1, y1, x2, y2) as fractions of frame size [0.0–1.0]
RESTRICTED_ZONES = [
    (0.65, 0.0, 1.0, 0.6),    # top-right zone (e.g. storage/staff area)
]

# Classes of interest from COCO dataset (used by YOLOv8)
PERSON_CLASS      = 0
BAG_CLASSES       = {24, 26, 28}          # backpack, handbag, suitcase
MOBILE_CLASS      = 67                    # cell phone
BOTTLE_CLASS      = 39
SUSPICIOUS_ITEMS  = {67, 76, 77, 78, 79}  # phone, scissors, knife, etc.

# ─────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Shared state (written by detector thread, read by Flask)
state = {
    "frame_jpeg":    None,
    "alerts":        deque(maxlen=MAX_ALERTS),
    "stats": {
        "persons":          0,
        "bags":             0,
        "mobiles":          0,
        "zone_intrusions":  0,
        "total_alerts":     0,
        "fps":              0.0,
        "status":           "Initialising…",
    },
    "lock": threading.Lock(),
}

alert_timers   = {}   # alert_type -> last trigger time
person_linger  = defaultdict(float)   # track_id -> time entered zone
track_history  = defaultdict(lambda: deque(maxlen=30))  # track_id -> [(cx, cy), …]

# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────
def now_str():
    return datetime.now().strftime("%H:%M:%S")

def can_alert(key):
    """Return True if enough time has passed since last alert of this type."""
    t = time.time()
    if t - alert_timers.get(key, 0) >= ALERT_COOLDOWN:
        alert_timers[key] = t
        return True
    return False

def push_alert(level, title, detail, frame=None):
    """Append an alert to the shared list and emit via SocketIO."""
    ts = now_str()
    screenshot_path = None

    if frame is not None:
        fname = f"{SCREENSHOT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{title.replace(' ', '_')}.jpg"
        cv2.imwrite(fname, frame)
        screenshot_path = fname

    alert = {
        "time":       ts,
        "level":      level,      # "danger" | "warning" | "info"
        "title":      title,
        "detail":     detail,
        "screenshot": screenshot_path,
    }

    with state["lock"]:
        state["alerts"].appendleft(alert)
        state["stats"]["total_alerts"] += 1

    socketio.emit("alert", alert)
    print(f"[{ts}] [{level.upper()}] {title}: {detail}")

def in_zone(cx, cy, zone, w, h):
    """Check if pixel point (cx, cy) is inside a fractional zone."""
    x1, y1, x2, y2 = zone
    return x1 * w <= cx <= x2 * w and y1 * h <= cy <= y2 * h

def draw_zones(frame, zones):
    h, w = frame.shape[:2]
    for zone in zones:
        x1 = int(zone[0] * w); y1 = int(zone[1] * h)
        x2 = int(zone[2] * w); y2 = int(zone[3] * h)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(frame, "RESTRICTED", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

def draw_hud(frame, stats):
    """Overlay a minimal HUD on the frame."""
    h, w = frame.shape[:2]
    # Semi-transparent bar at top
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 36), (10, 10, 10), -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    ts  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    fps = stats["fps"]
    cv2.putText(frame, f"RETAIL WATCH   {ts}   {fps:.1f} FPS",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Stats row at bottom
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - 30), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(bar2, 0.6, frame, 0.4, 0, frame)

    row = (f"  Persons: {stats['persons']}    "
           f"Bags: {stats['bags']}    "
           f"Mobiles: {stats['mobiles']}    "
           f"Zone Alerts: {stats['zone_intrusions']}    "
           f"Total Alerts: {stats['total_alerts']}")
    cv2.putText(frame, row, (4, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 160), 1)

# ─────────────────────────────────────────────
# CORE DETECTION LOOP
# ─────────────────────────────────────────────
def detection_loop():
    model = YOLO("yolov8n.pt")   # nano — fast, auto-downloads ~6 MB
    cap   = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        with state["lock"]:
            state["stats"]["status"] = "ERROR: Cannot open video source."
        print("ERROR: Cannot open video source.")
        return

    with state["lock"]:
        state["stats"]["status"] = "Running"

    fps_counter = 0
    fps_clock   = time.time()
    prev_person_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video files; re-open webcam on drop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

        h, w = frame.shape[:2]

        # ── Run YOLO tracking ──────────────────
        results = model.track(
            frame,
            persist=True,
            conf=CONFIDENCE_THRESH,
            verbose=False,
            classes=list({PERSON_CLASS, MOBILE_CLASS, BOTTLE_CLASS} | BAG_CLASSES | SUSPICIOUS_ITEMS),
        )

        persons_this_frame  = 0
        bags_this_frame     = 0
        mobiles_this_frame  = 0
        current_person_ids  = set()

        boxes = results[0].boxes if results[0].boxes is not None else []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_id = int(box.id[0]) if box.id is not None else -1

            label  = model.names[cls_id]
            colour = (60, 220, 60)

            # ── Person tracking ───────────────
            if cls_id == PERSON_CLASS:
                persons_this_frame += 1
                colour = (220, 180, 60)
                current_person_ids.add(track_id)

                if track_id >= 0:
                    track_history[track_id].append((cx, cy))
                    # Draw trail
                    pts = list(track_history[track_id])
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i-1], pts[i], (100, 200, 255), 1)

                # ── Zone intrusion ────────────
                for zone in RESTRICTED_ZONES:
                    if in_zone(cx, cy, zone, w, h):
                        elapsed = time.time() - person_linger.get(track_id, time.time())
                        person_linger.setdefault(track_id, time.time())

                        if elapsed >= LINGER_THRESHOLD and can_alert(f"zone_{track_id}"):
                            with state["lock"]:
                                state["stats"]["zone_intrusions"] += 1
                            push_alert("danger", "Zone Intrusion",
                                       f"Person #{track_id} in restricted zone for {elapsed:.0f}s",
                                       frame.copy())
                        colour = (0, 0, 255)
                        cv2.putText(frame, "⚠ RESTRICTED", (x1, y1 - 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        person_linger.pop(track_id, None)

            # ── Bag detection ─────────────────
            elif cls_id in BAG_CLASSES:
                bags_this_frame += 1
                colour = (255, 140, 0)
                if can_alert(f"bag_{track_id}"):
                    push_alert("warning", "Bag Detected",
                               f"{label.title()} spotted near ID #{track_id}")

            # ── Mobile phone ──────────────────
            elif cls_id == MOBILE_CLASS:
                mobiles_this_frame += 1
                colour = (180, 0, 255)
                if can_alert(f"mobile_{track_id}"):
                    push_alert("warning", "Mobile Phone Detected",
                               f"Phone in use near checkout — ID #{track_id}",
                               frame.copy())

            # ── Suspicious items ──────────────
            elif cls_id in SUSPICIOUS_ITEMS:
                colour = (0, 0, 200)
                if can_alert(f"susp_{cls_id}_{track_id}"):
                    push_alert("danger", "Suspicious Item",
                               f"{label.title()} detected! ID #{track_id}",
                               frame.copy())

            # ── Draw bounding box + label ─────
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            tag = f"{label} #{track_id} {conf:.0%}" if track_id >= 0 else f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ── Sudden person disappearance ────────
        vanished = prev_person_ids - current_person_ids
        for pid in vanished:
            if can_alert(f"vanish_{pid}"):
                push_alert("warning", "Person Left Suddenly",
                           f"Person #{pid} disappeared abruptly — possible item concealment")
        prev_person_ids = current_person_ids

        # ── Draw restricted zones ──────────────
        draw_zones(frame, RESTRICTED_ZONES)

        # ── HUD overlay ───────────────────────
        with state["lock"]:
            state["stats"].update({
                "persons": persons_this_frame,
                "bags":    bags_this_frame,
                "mobiles": mobiles_this_frame,
            })
            stats_snap = dict(state["stats"])

        draw_hud(frame, stats_snap)

        # ── FPS calculation ───────────────────
        fps_counter += 1
        if time.time() - fps_clock >= 1.0:
            with state["lock"]:
                state["stats"]["fps"] = fps_counter
            fps_counter = 0
            fps_clock   = time.time()

        # ── Encode frame as JPEG ──────────────
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with state["lock"]:
            state["frame_jpeg"] = buf.tobytes()

        # ── Push live stats to dashboard ──────
        socketio.emit("stats", stats_snap)

    cap.release()


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
def gen_frames():
    """MJPEG generator for the live video feed."""
    while True:
        with state["lock"]:
            frame = state["frame_jpeg"]
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/alerts")
def api_alerts():
    with state["lock"]:
        return jsonify(list(state["alerts"]))


@app.route("/api/stats")
def api_stats():
    with state["lock"]:
        return jsonify(state["stats"])


# ─────────────────────────────────────────────
# DASHBOARD HTML (single-file, minimal UI)
# ─────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retail Watch — Theft Detection</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0d0f14;
    --surface:  #161a22;
    --border:   #252b38;
    --text:     #c8cdd8;
    --dim:      #5a6070;
    --accent:   #3b82f6;
    --danger:   #ef4444;
    --warning:  #f59e0b;
    --ok:       #22c55e;
    --font:     'Courier New', monospace;
  }

  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; }

  /* ── Layout ── */
  .shell   { display: grid; grid-template-rows: 42px 1fr; height: 100vh; }
  .main    { display: grid; grid-template-columns: 1fr 300px; overflow: hidden; }
  .left    { display: flex; flex-direction: column; overflow: hidden; }
  .right   { border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }

  /* ── Header ── */
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 16px;
    letter-spacing: .06em;
  }
  .logo  { color: #fff; font-weight: 700; font-size: 13px; }
  .logo span { color: var(--danger); }
  #clock { color: var(--dim); font-size: 11px; }

  /* ── Video pane ── */
  .video-wrap {
    flex: 1; background: #000; display: flex; align-items: center; justify-content: center; overflow: hidden;
  }
  .video-wrap img { max-width: 100%; max-height: 100%; display: block; }
  .no-feed { color: var(--dim); font-size: 12px; text-align: center; }

  /* ── Stats bar ── */
  .stats-bar {
    background: var(--surface); border-top: 1px solid var(--border);
    display: flex; gap: 0; flex-shrink: 0;
  }
  .stat {
    flex: 1; padding: 8px 12px; border-right: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 2px;
  }
  .stat:last-child { border-right: none; }
  .stat-label { font-size: 9px; color: var(--dim); letter-spacing: .1em; text-transform: uppercase; }
  .stat-value { font-size: 22px; font-weight: 700; color: #fff; line-height: 1; }
  .stat-value.danger  { color: var(--danger); }
  .stat-value.warning { color: var(--warning); }
  .stat-value.ok      { color: var(--ok); }

  /* ── Alert panel ── */
  .panel-head {
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    font-size: 10px; letter-spacing: .1em; color: var(--dim); text-transform: uppercase;
    display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;
  }
  .panel-head .badge {
    background: var(--danger); color: #fff; border-radius: 9px;
    padding: 1px 7px; font-size: 10px;
  }
  #alert-list { flex: 1; overflow-y: auto; }
  #alert-list::-webkit-scrollbar { width: 4px; }
  #alert-list::-webkit-scrollbar-thumb { background: var(--border); }

  .alert-item {
    padding: 9px 14px; border-bottom: 1px solid var(--border);
    display: grid; grid-template-columns: 3px 1fr; gap: 10px;
    animation: slideIn .25s ease;
  }
  @keyframes slideIn { from { opacity:0; transform: translateX(8px); } to { opacity:1; transform: none; } }

  .alert-bar { border-radius: 2px; }
  .alert-bar.danger  { background: var(--danger); }
  .alert-bar.warning { background: var(--warning); }
  .alert-bar.info    { background: var(--accent); }

  .alert-body { min-width: 0; }
  .alert-title { color: #fff; font-weight: 700; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .alert-detail { color: var(--dim); font-size: 10px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .alert-time { color: var(--dim); font-size: 9px; margin-top: 3px; }

  /* ── Status dot ── */
  #status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); flex-shrink: 0; }
  #status-dot.running { background: var(--ok); box-shadow: 0 0 6px var(--ok); animation: pulse 2s infinite; }
  #status-dot.error   { background: var(--danger); }
  @keyframes pulse { 0%,100%{ opacity:1 } 50%{ opacity:.4 } }

  #fps-display { color: var(--dim); font-size: 11px; }

  .empty-msg { padding: 24px 14px; color: var(--dim); font-size: 11px; text-align: center; }
</style>
</head>
<body>
<div class="shell">
  <header>
    <div style="display:flex;align-items:center;gap:10px">
      <div id="status-dot"></div>
      <div class="logo">RETAIL<span>WATCH</span></div>
      <div id="status-text" style="color:var(--dim);font-size:11px">Connecting…</div>
    </div>
    <div style="display:flex;gap:16px;align-items:center">
      <div id="fps-display">— FPS</div>
      <div id="clock">--:--:--</div>
    </div>
  </header>

  <div class="main">
    <!-- LEFT: video + stats -->
    <div class="left">
      <div class="video-wrap">
        <img id="feed" src="/video_feed" alt="Live Feed"
             onerror="this.style.display='none';document.getElementById('no-feed').style.display='block'">
        <div id="no-feed" class="no-feed" style="display:none">
          No video feed.<br>Check VIDEO_SOURCE in the script.
        </div>
      </div>

      <div class="stats-bar">
        <div class="stat">
          <div class="stat-label">Persons</div>
          <div class="stat-value ok" id="s-persons">0</div>
        </div>
        <div class="stat">
          <div class="stat-label">Bags</div>
          <div class="stat-value warning" id="s-bags">0</div>
        </div>
        <div class="stat">
          <div class="stat-label">Mobiles</div>
          <div class="stat-value warning" id="s-mobiles">0</div>
        </div>
        <div class="stat">
          <div class="stat-label">Zone Alerts</div>
          <div class="stat-value danger" id="s-zones">0</div>
        </div>
        <div class="stat">
          <div class="stat-label">Total Alerts</div>
          <div class="stat-value" id="s-total">0</div>
        </div>
      </div>
    </div>

    <!-- RIGHT: alert feed -->
    <div class="right">
      <div class="panel-head">
        Alert Feed
        <span class="badge" id="alert-count">0</span>
      </div>
      <div id="alert-list">
        <div class="empty-msg" id="empty-msg">No alerts yet.</div>
      </div>
    </div>
  </div>
</div>

<script>
  // Clock
  setInterval(() => {
    document.getElementById('clock').textContent =
      new Date().toLocaleTimeString('en-GB');
  }, 1000);

  // SocketIO
  const socket = io();
  let alertCount = 0;

  socket.on('connect', () => {
    document.getElementById('status-dot').className = 'running';
    document.getElementById('status-text').textContent = 'Connected';
  });
  socket.on('disconnect', () => {
    document.getElementById('status-dot').className = 'error';
    document.getElementById('status-text').textContent = 'Disconnected';
  });

  socket.on('stats', s => {
    document.getElementById('s-persons').textContent = s.persons ?? 0;
    document.getElementById('s-bags').textContent    = s.bags    ?? 0;
    document.getElementById('s-mobiles').textContent = s.mobiles ?? 0;
    document.getElementById('s-zones').textContent   = s.zone_intrusions ?? 0;
    document.getElementById('s-total').textContent   = s.total_alerts ?? 0;
    document.getElementById('fps-display').textContent = (s.fps ?? 0).toFixed(1) + ' FPS';
    if (s.status) document.getElementById('status-text').textContent = s.status;
  });

  socket.on('alert', a => {
    const list = document.getElementById('alert-list');
    const empty = document.getElementById('empty-msg');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'alert-item';
    item.innerHTML = `
      <div class="alert-bar ${a.level}"></div>
      <div class="alert-body">
        <div class="alert-title">${escHtml(a.title)}</div>
        <div class="alert-detail">${escHtml(a.detail)}</div>
        <div class="alert-time">${escHtml(a.time)}</div>
      </div>`;
    list.prepend(item);

    alertCount++;
    document.getElementById('alert-count').textContent = alertCount;

    // Trim old items
    while (list.children.length > 50) list.removeChild(list.lastChild);
  });

  // Load existing alerts on page load
  fetch('/api/alerts').then(r => r.json()).then(alerts => {
    alerts.forEach(a => {
      const e = new Event('alert');
      socket.emit('alert', a);  // re-use handler
    });
  });

  function escHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Retail Theft Detection System")
    print("  Dashboard → http://localhost:5000")
    print("=" * 55)
    print(f"  Video source : {VIDEO_SOURCE}")
    print(f"  Screenshots  : ./{SCREENSHOT_DIR}/")
    print(f"  Confidence   : {CONFIDENCE_THRESH}")
    print("=" * 55)

    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    socketio.run(app, host="0.0.0.0", port=5000, debug=False)