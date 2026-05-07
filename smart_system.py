"""
YAARIOK SMART SYSTEM v2.0
- Real-time person detection (YOLOv8)
- Auto visitor greetings + visitor counter
- Security mode / Karaoke mode
- Live camera window with overlays
- Smart voice commands with camera integration
- Visitor log file
- Live terminal dashboard
"""

import warnings
warnings.filterwarnings("ignore")
import os, sys

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"       # suppress HEVC/codec warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"   # suppress FFmpeg warnings

import cv2, threading, time, datetime

import sqlite3
def init_db():
    conn = sqlite3.connect("snapshots.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS snapshots
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  timestamp TEXT,
                  visitor_type TEXT,
                  notes TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT,
                  timestamp TEXT,
                  date TEXT,
                  alert_type TEXT,
                  status TEXT,
                  person_count INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  log_type TEXT,
                  message TEXT,
                  person_count INTEGER,
                  timestamp TEXT,
                  date TEXT)''')
    try:
        c.execute("ALTER TABLE logs ADD COLUMN person_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE alerts ADD COLUMN date TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE alerts ADD COLUMN person_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def save_alert(filename, alert_type="Person Detected", status="Active", person_count=0):
    conn = sqlite3.connect("snapshots.db")
    c = conn.cursor()
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    date = now.strftime("%Y-%m-%d")
    c.execute("INSERT INTO alerts (image_path, timestamp, date, alert_type, status, person_count) VALUES (?, ?, ?, ?, ?, ?)",
              (filename, ts, date, alert_type, status, person_count))
    conn.commit()
    conn.close()

def play_alert_sound():
    try:
        if _neural_ok:
            # We already have pygame initialized
            # Make a simple beep sound
            arr = np.array([4096 * np.sin(2.0 * np.pi * 1000 * x / 44100) for x in range(0, 44100//4)]).astype(np.int16)
            arr = np.column_stack((arr, arr))
            snd = _pygame.sndarray.make_sound(arr)
            snd.play()
    except:
        print("\a", end="", flush=True)

def trigger_alert(frame, num_people, should_beep=True):
    fname = datetime.datetime.now().strftime("alert_%H%M%S.jpg")
    os.makedirs("./alerts/", exist_ok=True)
    snap = "./alerts/" + fname
    cv2.imwrite(snap, frame)
    save_alert(fname, alert_type=f"{num_people} Person(s) Detected", person_count=num_people)
    save_log_db("alert", f"Security alert: {num_people} person(s) detected — {fname}", person_count=num_people)
    if should_beep:
        play_alert_sound()
    S(last_live_alert=fname, last_live_alert_time=datetime.datetime.now().strftime("%H:%M:%S"))

def delete_today_snapshots():
    """Delete all snapshot files and DB rows from today. Returns count deleted."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("snapshots.db")
    c = conn.cursor()
    c.execute("SELECT filename FROM snapshots WHERE timestamp LIKE ?", (f"{today}%",))
    rows = c.fetchall()
    deleted = len(rows)
    for row in rows:
        fpath = os.path.join(SNAPSHOT_DIR, row[0])
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
    c.execute("DELETE FROM snapshots WHERE timestamp LIKE ?", (f"{today}%",))
    conn.commit()
    conn.close()
    with _lock:
        state["snapshots_taken"] = 0
    return deleted

def save_snapshot_db(filename, vtype="", notes=""):
    conn = sqlite3.connect("snapshots.db")
    c = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO snapshots (filename, timestamp, visitor_type, notes) VALUES (?, ?, ?, ?)",
              (filename, ts, vtype, notes))
    conn.commit()
    conn.close()

def save_log_db(log_type, message, person_count=0):
    try:
        conn = sqlite3.connect("snapshots.db")
        c = conn.cursor()
        now = datetime.datetime.now()
        ts   = now.strftime("%Y-%m-%d %H:%M:%S")
        date = now.strftime("%Y-%m-%d")
        c.execute("INSERT INTO logs (log_type, message, person_count, timestamp, date) VALUES (?, ?, ?, ?, ?)",
                  (log_type, message, person_count, ts, date))
        conn.commit()
        conn.close()
    except Exception:
        pass

import speech_recognition as sr
import webbrowser, urllib.parse, re, requests, tempfile, base64
import numpy as np

RTSP = "rtsp://Test:Nanta@123@192.168.29.118:554/1/1?transmode=unicast&profile=vam"
SNAPSHOT_DIR = "./snapshots/"
LOG_FILE = "./visitor_log.txt"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Entry zone for door-based greeting: (x1, y1, x2, y2) as fractions of detection frame (640x360)
# Tune these values to match where your reception door is in the camera view.
ENTRY_ZONE = (0.1, 0.2, 0.9, 1.0)

# YOLO loads lazily inside camera_thread — keeps startup instant
_model      = None
YOLO_OK     = False
_cam_frame  = [None]   # latest raw frame — shared with dashboard
_ann_frame  = [None]   # annotated frame with overlays — shared with dashboard

# ── ANSI colors ────────────────────────────────────────────────────────────────
C = {
    "reset": "\033[0m",   "bold": "\033[1m",
    "red":   "\033[91m",  "green":  "\033[92m",
    "yellow":"\033[93m",  "blue":   "\033[94m",
    "cyan":  "\033[96m",  "gray":   "\033[90m",
    "purple":"\033[95m",
}

# ── Shared state ───────────────────────────────────────────────────────────────
_lock = threading.Lock()
state = {
    "system_running":  False,
    "voice_running":   False,
    "people_now":      0,
    "visitors_today":  0,
    "alerts_today":    0,
    "snapshots_taken": 0,
    "security_mode":   False,
    "karaoke_mode":    False,
    "show_camera":     False,
    "take_snapshot":   False,
    "last_command":    "—",
    "last_alert_time": "—",
    "robot_status":    "Starting...",
    "robot_speech":    "System starting up...",
    "camera_status":   "Connecting...",
    "running":         True,
    "dash_command":    "",
    "last_visitor_b64": "",   # base64 JPEG of last visitor snap for dashboard
    "ai_alert":        "",    # Claude's security analysis text
}

_visitor_frame_buf = [None]   # raw numpy frame of last visitor, for Claude vision

def S(**kw):
    with _lock:
        state.update(kw)

# ── TTS ────────────────────────────────────────────────────────────────────────
# Neural voice via edge-tts (Microsoft Edge neural TTS, free, no API key needed)
# Falls back to pyttsx3 if edge-tts/pygame are not installed.
_EDGE_VOICE = "en-US-AriaNeural"   # change to "en-IN-NeerjaNeural" for Indian accent
_neural_ok  = False
_tts_lock   = threading.Lock()

# pyttsx3 fallback engine
try:
    import pyttsx3 as _pyttsx3
    _engine = _pyttsx3.init()
    _engine.setProperty("rate", 165)
    _engine.setProperty("volume", 1.0)
    for _v in _engine.getProperty("voices"):
        if "zira" in _v.name.lower():
            _engine.setProperty("voice", _v.id)
            break
except Exception:
    _engine = None

# Try to initialise neural TTS
try:
    import edge_tts as _edge_tts   # pip install edge-tts
    import pygame as _pygame        # pip install pygame
    _pygame.mixer.pre_init(44100, -16, 2, 512)
    _pygame.mixer.init()
    _pygame.mixer.music.set_volume(1.0)
    _neural_ok = True
except Exception:
    pass


def _speak_neural(text: str) -> bool:
    """Generate speech with Edge neural TTS and play via pygame. Returns True on success."""
    if not _neural_ok:
        return False
    try:
        tmp = tempfile.mktemp(suffix=".mp3")

        def _generate():
            import asyncio
            async def _run():
                tts = _edge_tts.Communicate(text, voice=_EDGE_VOICE, rate="+8%", volume="+0%")
                await tts.save(tmp)
            asyncio.run(_run())

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()
        gen_thread.join(timeout=12)

        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            return False

        _pygame.mixer.music.load(tmp)
        _pygame.mixer.music.play()
        while _pygame.mixer.music.get_busy():
            time.sleep(0.05)

        try:
            os.unlink(tmp)
        except Exception:
            pass
        return True
    except Exception as ex:
        print(f"  [TTS-neural] {ex}")
        return False


_recent_responses = {}       # Track response IDs to block duplicates

def speak(text: str):
    global _last_speak_time
    if not text or not text.strip():
        return
        
    # Unique Response ID: use the text itself to track uniqueness
    response_id = hash(text)
    
    # Block duplicate same-ID response within 10 seconds
    now = time.monotonic()
    if response_id in _recent_responses and (now - _recent_responses[response_id] < 10.0):
        print(f"{C['gray']}  [TTS] Blocked duplicate response ID: {response_id}{C['reset']}")
        return
        
    _recent_responses[response_id] = now
    
    # State: lock (do not listen, no echo)
    S(voice_state="lock")
    S(robot_status=f"Speaking: {text[:45]}", robot_speech=text)
    print(f"\n{C['cyan']}  [Robot] {text}{C['reset']}")
    
    with _tts_lock:
        if not _speak_neural(text):
            if _engine:
                _engine.say(text)
                _engine.runAndWait()
                
    _last_speak_time = time.monotonic()
    # State: back to listen
    S(voice_state="listen")
    S(robot_status="Waiting for wake word...")

# ── Sounds (routed through real speakers via cross-platform print/play) ───────
def beep(times=3):
    def _b():
        for _ in range(times):
            print("\a", end="", flush=True)  # Terminal bell fallback
            time.sleep(0.6)
    threading.Thread(target=_b, daemon=True).start()

def success_sound():
    print("\a", end="", flush=True)

# ── Visitor log ────────────────────────────────────────────────────────────────
def log(event: str, log_type: str = "system", person_count: int = 0):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {event}\n")
    save_log_db(log_type, event, person_count)

def notify(title: str, msg: str):
    def _n():
        try:
            from plyer import notification
            notification.notify(title=title, message=msg,
                                app_name="Yaariok Smart System", timeout=4)
        except Exception:
            pass
    threading.Thread(target=_n, daemon=True).start()

import json

CONFIG_FILE = "vision_config.json"
def load_vision_config():
    default_config = {
        "confidence_threshold": 0.60,
        "min_person_area": 1800,
        "min_person_height": 45,
        "min_person_aspect": 1.1,
        "target_fps": 15,
        "box_smoothing_alpha": 0.3,
        "door_polygon": []
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                loaded = json.load(f)
                default_config.update(loaded)
    except Exception as e:
        pass
    return default_config

def save_vision_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=4)
    except Exception as e:
        pass

vision_config = load_vision_config()
is_calibrating = False

def run_calibration():
    global is_calibrating, vision_config
    if is_calibrating: return
    is_calibrating = True
    
    def _cal_thread():
        speak("Calibration started. Please place a person at the farthest distance you care about, and walk around for 60 seconds to calibrate real people.")
        time.sleep(5)
        
        real_people_scores = []
        real_areas = []
        t0 = time.time()
        while time.time() - t0 < 60:
            frame = _cam_frame[0]
            if frame is not None and YOLO_OK and _model is not None:
                small = cv2.resize(frame, (640, 360))
                res = _model(small, classes=[0], verbose=False, conf=0.1, half=True)[0]
                for b in res.boxes:
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    real_people_scores.append(conf)
                    real_areas.append(area)
            time.sleep(0.1)
            
        speak("Now please leave the room empty. Calibrating false positives for 60 seconds.")
        time.sleep(5)
        
        fake_scores = []
        t0 = time.time()
        while time.time() - t0 < 60:
            frame = _cam_frame[0]
            if frame is not None and YOLO_OK and _model is not None:
                small = cv2.resize(frame, (640, 360))
                res = _model(small, classes=[0], verbose=False, conf=0.1, half=True)[0]
                for b in res.boxes:
                    conf = float(b.conf[0])
                    fake_scores.append(conf)
            time.sleep(0.1)
            
        p95 = np.percentile(real_people_scores, 5) if real_people_scores else 0.5
        f10 = np.percentile(fake_scores, 90) if fake_scores else 0.2
        
        thresh = (p95 + f10) / 2.0
        if p95 <= f10:
            thresh = p95
            
        min_area = np.min(real_areas) * 0.7 if real_areas else 5000
        
        vision_config["confidence_threshold"] = float(max(0.2, min(thresh, 0.95)))
        vision_config["min_person_area"] = float(min_area)
        save_vision_config(vision_config)
        
        speak(f"Calibration complete. Threshold set to {vision_config['confidence_threshold']:.2f}.")
        global is_calibrating
        is_calibrating = False

    threading.Thread(target=_cal_thread, daemon=True).start()

def _person_in_entry_zone(boxes, fw=640, fh=360):
    """Return True if any YOLO box center falls inside ENTRY_ZONE or door_polygon."""
    if not boxes:
        return False
        
    poly = vision_config.get("door_polygon", [])
    has_poly = len(poly) >= 3
    if has_poly:
        scaled_poly = np.array(poly, np.int32)
        
    x1z = ENTRY_ZONE[0] * fw
    y1z = ENTRY_ZONE[1] * fh
    x2z = ENTRY_ZONE[2] * fw
    y2z = ENTRY_ZONE[3] * fh
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        if has_poly:
            if cv2.pointPolygonTest(scaled_poly, (cx, cy), False) >= 0:
                return True
        else:
            if x1z <= cx <= x2z and y1z <= cy <= y2z:
                return True
    return False

_recent_snapshots = [] # list of (time, hist)

def _is_new_person(frame, box, threshold=0.85, memory=120, update_memory=True):
    global _recent_snapshots
    now = time.time()
    # clean old memories
    _recent_snapshots = [(t, h) for t, h in _recent_snapshots if now - t <= memory]
    
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return True
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    for _, prev_hist in _recent_snapshots:
        sim = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
        if sim >= threshold:
            return False # Same person
            
    # New person
    if update_memory:
        _recent_snapshots.append((now, hist))
    return True

# ── Camera thread ──────────────────────────────────────────────────────────────
def camera_thread():
    # Load YOLO here so startup is instant (torch takes ~15s to import)
    global _model, YOLO_OK
    S(camera_status="LOADING AI")
    _device = "cpu"
    try:
        import torch
        from ultralytics import YOLO as _YOLO
        _device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _model  = _YOLO("yolov8s.pt")   # small model — better accuracy than nano, still fast
        _model.to(_device)
        YOLO_OK = True
        print(f"\n{C['green']}  [YOLO] Person detection ready on {_device.upper()}.{C['reset']}")
    except Exception as e:
        YOLO_OK = False
        print(f"\n{C['yellow']}  [YOLO] Not available — using motion detection. ({e}){C['reset']}")

    # ── Frame grabber thread: drains RTSP buffer so processing gets latest frame ─
    _live = [None]
    _flk  = threading.Lock()

    def _open_cap():
        c = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c

    def _grabber():
        cap = _open_cap()
        if cap.isOpened():
            S(camera_status="LIVE")
            log("Camera connected")
            notify("Camera Live", "Reception camera connected and streaming")
        else:
            S(camera_status="FAILED")
            notify("Camera Failed", "Could not connect to RTSP stream")
            return
        while state["system_running"] and state["running"]:
            ret, frm = cap.read()
            if ret:
                with _flk:
                    _live[0] = frm
            else:
                S(camera_status="RECONNECTING")
                notify("Camera Disconnected", "Reconnecting to RTSP stream...")
                cap.release()
                time.sleep(4)
                cap = _open_cap()
                if cap.isOpened():
                    S(camera_status="LIVE")
                    notify("Camera Reconnected", "RTSP stream restored")

    threading.Thread(target=_grabber, daemon=True).start()

    t0 = time.time()
    while _live[0] is None and state["running"]:
        time.sleep(0.05)
        if time.time() - t0 > 15:
            S(camera_status="FAILED")
            return

    bg_sub       = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
    frame_n      = 0
    latest_frame = _cam_frame   # module-level ref for dashboard
    was_showing  = False        # track show_camera transitions

    # ── Decoupled Detection State ──────────────────────────────────────────────
    _det_lock = threading.Lock()
    _det_state = {
        "boxes": [], # list of dicts: {'id', 'xyxy', 'conf', 'is_new'}
        "people_count": 0
    }
    
    _smoothed_boxes = {} # For drawing smooth bounding boxes
    target_fps = vision_config.get("target_fps", 5)
    box_alpha = vision_config.get("box_smoothing_alpha", 0.3)

    def _detect_loop():
        person_present       = False
        detection_count      = 0
        REQUIRED_FRAMES      = 3
        absence_start_time       = 0.0
        last_greeted             = 0
        GREET_COOLDOWN           = 120
        last_alert_beep_time     = 0.0
        beep_cooldown            = 15
        
        _cfg_reload_ctr = 0  # counter for hot-reloading config every 30 frames
        import collections
        detection_history = collections.deque(maxlen=10)
        
        while state["system_running"] and state["running"]:
            t0 = time.time()
            # Hot-reload config every 30 frames so changes to vision_config.json take effect immediately
            _cfg_reload_ctr += 1
            if _cfg_reload_ctr % 30 == 0:
                vision_config.update(load_vision_config())
            with _flk:
                frame = _live[0]
            if frame is None:
                time.sleep(0.01)
                continue
                
            small = cv2.resize(frame, (640, 360))
            people = 0
            boxes_to_share = []
            
            if YOLO_OK:
                conf_thresh = vision_config["confidence_threshold"]
                min_area = vision_config["min_person_area"]
                last_results = _model.track(small, classes=[0], verbose=False,
                                            conf=conf_thresh, half=True, device=_device,
                                            persist=True, tracker="bytetrack.yaml")
                min_ar  = vision_config.get("min_person_aspect", 1.1)
                min_h   = vision_config.get("min_person_height", 45)
                _valid_boxes = []
                if last_results[0].boxes.id is not None:
                    for _b in last_results[0].boxes:
                        _x1, _y1, _x2, _y2 = _b.xyxy[0].tolist()
                        bw, bh = _x2 - _x1, _y2 - _y1
                        area = bw * bh
                        ar   = bh / bw if bw > 0 else 0
                        # Accept far/small people: need min height + correct aspect ratio
                        # Reject robots/furniture: too low (ar < min_ar) or too tiny (area + height)
                        if area >= min_area and bh >= min_h and ar >= min_ar:
                            _valid_boxes.append(_b)
                people = len(_valid_boxes)
                
                for _b in _valid_boxes:
                    is_new = _is_new_person(frame, _b, threshold=0.85, memory=120, update_memory=True)
                    boxes_to_share.append({
                        "id": int(_b.id[0]) if _b.id is not None else hash(str(_b.xyxy[0])),
                        "xyxy": _b.xyxy[0].tolist(),
                        "conf": float(_b.conf[0]),
                        "is_new": is_new
                    })
            else:
                mask = bg_sub.apply(small)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                people = 0
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    area = w * h
                    if area > vision_config["min_person_area"] * 0.5:
                        if 0.2 <= (w / float(h)) <= 0.5:
                            people += 1

            # ── Flicker Reduction (Hysteresis) ──
            detection_history.append(people > 0)
            if not person_present:
                is_stable = sum(detection_history) >= 2
            else:
                is_stable = sum(detection_history) > 0
                
            stable_people = max(1, people) if is_stable else 0
            S(people_now=stable_people)

            with _det_lock:
                _det_state["boxes"] = boxes_to_share
                _det_state["people_count"] = stable_people

            # ── Entry / exit detection ────────────────────────────────────────
            now = time.time()
            if stable_people > 0:
                absence_start_time = 0.0
                detection_count += 1
                
                if detection_count >= REQUIRED_FRAMES and not person_present:
                    # ── Confirmed new entry ───────────────────────────────────
                    person_present = True
                    print("  [Detect] Person entered")

                    with _lock:
                        state["visitors_today"] += 1
                        vnum = state["visitors_today"]
                    S(last_alert_time=datetime.datetime.now().strftime("%H:%M:%S"))
                    log(f"Visitor #{vnum} arrived")
                    save_log_db("visitor", f"Visitor #{vnum} arrived", person_count=stable_people)

                    # ── ONE snapshot per new entry ────────────────────────────
                    is_alert      = state["security_mode"]
                    fname_prefix  = "ALERT_visitor" if is_alert else "visitor"
                    fname         = datetime.datetime.now().strftime(f"{fname_prefix}_%H%M%S.jpg")
                    cv2.imwrite(SNAPSHOT_DIR + fname, frame)
                    save_snapshot_db(fname, "Alert" if is_alert else "Visitor")
                    with _lock:
                        if is_alert: state["alerts_today"] += 1
                        state["snapshots_taken"] += 1
                    notify("Visitor Detected", f"Visitor #{vnum} arrived — snapshot saved")
                    beep(4 if is_alert else 2)
                    _visitor_frame_buf[0] = frame.copy()
                    try:
                        _, buf = cv2.imencode(".jpg", cv2.resize(frame, (320, 180)),
                                             [cv2.IMWRITE_JPEG_QUALITY, 70])
                        S(last_visitor_b64=base64.b64encode(buf.tobytes()).decode())
                    except Exception:
                        pass
                    print(f"  [Detect] Snapshot saved: {fname}")

                if person_present:
                    # ── Greeting — fires once per visit, zone-gated ──────────
                    class MockBox:
                        def __init__(self, xyxy):
                            self.xyxy = [np.array(xyxy)]
                    mock_boxes = [MockBox(b["xyxy"]) for b in boxes_to_share]
                    in_zone = (not YOLO_OK) or (_person_in_entry_zone(mock_boxes) if mock_boxes else True)

                    if in_zone and (now - last_greeted > GREET_COOLDOWN):
                        last_greeted = now
                        print("  [Detect] Greeting triggered")
                        if state["security_mode"]:
                            should_beep = False
                            if now - last_alert_beep_time >= beep_cooldown:
                                last_alert_beep_time = now
                                should_beep = True
                            trigger_alert(frame, stable_people, should_beep=should_beep)
                            def _sec_greet(f=frame.copy()):
                                speak(get_claude_response("Security alert! Unauthorized person detected."))
                                analysis = llm_security_analysis(f)
                                S(ai_alert=analysis[:80])
                                if analysis: speak(analysis)
                            threading.Thread(target=_sec_greet, daemon=True).start()
                        elif state["karaoke_mode"]:
                            def _karaoke_greet(f=frame.copy(), v=state["visitors_today"]):
                                speak(llm_greet_vision(v, f))
                                play_youtube("party welcome music upbeat")
                            threading.Thread(target=_karaoke_greet, daemon=True).start()
                        else:
                            threading.Thread(
                                target=lambda v=state["visitors_today"], f=frame.copy(): speak(llm_greet_vision(v, f)),
                                daemon=True).start()

            else:
                detection_count = 0
                if person_present:
                    if absence_start_time == 0.0:
                        absence_start_time = now
                    if (now - absence_start_time) > 2.0:
                        print("  [Detect] Person left — resetting state")
                        person_present       = False
                        absence_start_time   = 0.0
                        last_alert_beep_time      = 0.0
                        _smoothed_boxes.clear()

            elapsed = time.time() - t0
            time.sleep(max(0.005, (1.0 / target_fps) - elapsed))

    # Start detection on separate thread
    threading.Thread(target=_detect_loop, daemon=True).start()

    # ── Display rendering loop (High FPS) ──────────────────────────────────────
    while state["system_running"] and state["running"]:
        with _flk:
            frame = _live[0]
        if frame is None:
            time.sleep(0.01)
            continue

        latest_frame[0] = frame.copy()
        
        # Manual snapshot
        if state["take_snapshot"]:
            fname = datetime.datetime.now().strftime("manual_%H%M%S.jpg")
            snap = SNAPSHOT_DIR + fname
            cv2.imwrite(snap, latest_frame[0])
            save_snapshot_db(fname, "Manual")
            with _lock:
                state["snapshots_taken"] += 1
                state["take_snapshot"] = False
            log("Manual snapshot taken from dashboard")
            notify("Snapshot Saved", "Manual snapshot saved")
            threading.Thread(target=speak, args=("Snapshot saved.",), daemon=True).start()

        # Build display frame
        display = cv2.resize(frame, (1280, 720))
        sec  = state["security_mode"]
        kar  = state["karaoke_mode"]
        t    = time.time()

        if sec: box_color = (0, 0, 255)
        elif kar:
            hue = int((t * 40) % 180)
            box_color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 220]]]), cv2.COLOR_HSV2BGR)[0][0])
        else: box_color = (34, 197, 94)

        with _det_lock:
            cur_boxes = list(_det_state["boxes"])
            cur_people = _det_state["people_count"]

        if YOLO_OK:
            h_orig, w_orig = frame.shape[:2]
            scale_x = w_orig / 640.0
            scale_y = h_orig / 360.0
            scale_disp_x = 1280 / w_orig
            scale_disp_y = 720 / h_orig
            
            active_ids = set()
            for b in cur_boxes:
                tid = b["id"]
                active_ids.add(tid)
                x1, y1, x2, y2 = b["xyxy"]
                
                # Scale from 640x360 to native frame size
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                
                if tid not in _smoothed_boxes:
                    _smoothed_boxes[tid] = [x1, y1, x2, y2]
                else:
                    ox1, oy1, ox2, oy2 = _smoothed_boxes[tid]
                    # EMA smoothing at display rate (creates gliding effect)
                    x1 = box_alpha * x1 + (1 - box_alpha) * ox1
                    y1 = box_alpha * y1 + (1 - box_alpha) * oy1
                    x2 = box_alpha * x2 + (1 - box_alpha) * ox2
                    y2 = box_alpha * y2 + (1 - box_alpha) * oy2
                    _smoothed_boxes[tid] = [x1, y1, x2, y2]
                
                # Scale to 960x540 display size
                dx1, dy1, dx2, dy2 = int(x1 * scale_disp_x), int(y1 * scale_disp_y), int(x2 * scale_disp_x), int(y2 * scale_disp_y)
                
                label = "ALERT" if sec else ("★" if kar else f"{b['conf']:.0%}")
                cv2.rectangle(display, (dx1, dy1), (dx2, dy2), box_color, 2)
                cv2.putText(display, label, (dx1 + 4, max(dy1 + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)
            
            # Prune lost IDs
            for k in list(_smoothed_boxes.keys()):
                if k not in active_ids:
                    del _smoothed_boxes[k]

        h, w = display.shape[:2]
        if sec:
            pulse = int(180 + 75 * abs(np.sin(t * 2.5)))
            cv2.rectangle(display, (0, 0), (w-1, h-1), (0, 0, pulse), 10)
            cv2.rectangle(display, (5, 5), (w-6, h-6), (0, 0, 120), 3)
            cv2.rectangle(display, (0, 0), (w, 48), (0, 0, 180), -1)
            cv2.putText(display, "SECURITY MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 60, 60), 2, cv2.LINE_AA)
            ai_txt = state.get("ai_alert", "")[:60]
            if ai_txt:
                cv2.putText(display, ai_txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 100, 100), 1, cv2.LINE_AA)
            if cur_people > 0 and int(t * 2) % 2 == 0:
                cv2.putText(display, f"  INTRUDER DETECTED  x{cur_people}", (w//2 - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        elif kar:
            hue2 = int((t * 50) % 180)
            kc = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
            cv2.rectangle(display, (0, 0), (w-1, h-1), kc, 8)
            cv2.rectangle(display, (0, 0), (w, 48), (100, 0, 140), -1)
            cv2.putText(display, "KARAOKE MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 50), 2, cv2.LINE_AA)
            if cur_people > 0:
                cv2.putText(display, "PARTY TIME!", (w//2 - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, kc, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(display, (0, 0), (w, 44), (8, 12, 20), -1)
            cv2.putText(display, "YAARIOK  RECEPTION", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 200, 220), 2, cv2.LINE_AA)

        cv2.rectangle(display, (0, h - 32), (w, h), (0, 0, 0), -1)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(display, f"People: {cur_people}  |  Visitors: {state['visitors_today']}  |  {ts}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 150, 180), 1, cv2.LINE_AA)
        
        # Debug Mode Overlay
        debug_on = vision_config.get("debug_mode", False)
        if debug_on:
            cv2.putText(display, f"Target FPS: {target_fps} | Alpha: {box_alpha}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        _ann_frame[0] = display
        
        time.sleep(0.010)

        # ── OpenCV popup window (show_camera only) ────────────────────────────────
        now_showing = state["show_camera"]
        if now_showing:
            cv2.imshow("Yaariok Smart Camera", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                S(show_camera=False)
                was_showing = False
                cv2.destroyAllWindows()
                continue
        elif was_showing:
            cv2.destroyAllWindows()
        was_showing = now_showing

    cv2.destroyAllWindows()  # cap is owned and released by _grabber thread

# ── STT ────────────────────────────────────────────────────────────────────────
_rec = sr.Recognizer()
_rec.pause_threshold          = 0.5    # faster end-of-speech detection (was 0.8)
_rec.energy_threshold         = 150
_rec.dynamic_energy_threshold = False

def listen(label="Listening", timeout=5) -> str:
    S(robot_status=f"{label}...")
    try:
        with sr.Microphone() as src:
            _rec.adjust_for_ambient_noise(src, duration=1.0)
            try:
                print(f"{C['gray']}  [{label}] listening (threshold={_rec.energy_threshold:.0f})...{C['reset']}", end="\r", flush=True)
                audio = _rec.listen(src, timeout=timeout, phrase_time_limit=8)
                text  = _rec.recognize_google(audio).lower().strip()
                print(f"{C['yellow']}  [You] {text}                          {C['reset']}")
                return text
            except sr.WaitTimeoutError:
                return ""
            except sr.UnknownValueError:
                print(f"{C['gray']}  [MIC] Could not understand audio        {C['reset']}", end="\r", flush=True)
                return ""
            except sr.RequestError as e:
                print(f"{C['red']}  [STT] Google STT unavailable: {e}{C['reset']}")
                return ""
    except OSError as e:
        print(f"{C['red']}  [MIC] Microphone error: {e}{C['reset']}")
        time.sleep(2)
        return ""

# ── YouTube ────────────────────────────────────────────────────────────────────
def play_youtube(query: str):
    notify("YouTube", f"Playing: {query}")
    speak(get_claude_response(f"Playing {query} for you.", user_input=f"play {query}"))
    try:
        h = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(
            f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}",
            headers=h, timeout=6
        ).text
        ids = re.findall(r'watch\?v=([a-zA-Z0-9_-]{11})', html)
        if ids:
            webbrowser.open(f"https://www.youtube.com/watch?v={ids[0]}&autoplay=1")
            log(f"YouTube: {query}")
            return
    except Exception:
        pass
    webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}")

# ── Conversational LLM (OpenRouter / Claude Sonnet) with tool calling ───────────
try:
    from keys import OPENROUTER_API_KEY as _OR_KEY
except Exception:
    _OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")

_OR_URL   = "https://openrouter.ai/api/v1/chat/completions"
_OR_MODEL = "anthropic/claude-sonnet-4-5"   # Claude Sonnet via OpenRouter
_OR_READY = False
_conv_history = []

LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "take_snapshot",
            "description": "Take a photo snapshot from the reception camera and save it to disk.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_nantatech",
            "description": "Open the Nanta Tech Limited company website in the browser.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_youtube",
            "description": "Open YouTube and play a song, video, or music by search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search and play on YouTube"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "camera_status",
            "description": "Get live camera and detection status: how many people, visitors today, camera state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_mode",
            "description": "Change operating mode: normal, security, or karaoke.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["normal", "security", "karaoke"],
                        "description": "The mode to activate",
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_camera",
            "description": "Show or hide the live camera window on the robot's screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visible": {
                        "type": "boolean",
                        "description": "true to show, false to hide",
                    }
                },
                "required": ["visible"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visitor_log",
            "description": "Get a summary of today's visitor log: entries, visitors, alerts.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _exec_tool(name: str, args: dict) -> str:
    if name == "take_snapshot":
        S(take_snapshot=True)
        notify("Snapshot", "Photo saved by voice command")
        return "Snapshot taken and saved."

    if name == "open_nantatech":
        webbrowser.open("https://www.nantatech.com")
        log("Opened nantatech.com via LLM tool")
        return "Opened Nanta Tech Limited website."

    if name == "play_youtube":
        query = args.get("query", "music")
        threading.Thread(target=play_youtube, args=(query,), daemon=True).start()
        return f"Opening YouTube for: {query}"

    if name == "camera_status":
        n    = state["people_now"]
        v    = state["visitors_today"]
        a    = state["alerts_today"]
        sn   = state["snapshots_taken"]
        cam  = state["camera_status"]
        mode = "security" if state["security_mode"] else "karaoke" if state["karaoke_mode"] else "normal"
        det  = "YOLOv8" if YOLO_OK else "motion sensor"
        return (f"Camera: {cam}. People in reception: {n}. "
                f"Visitors today: {v}. Alerts: {a}. Snapshots: {sn}. "
                f"Mode: {mode}. Detection: {det}.")

    if name == "set_mode":
        mode = args.get("mode", "normal")
        if mode == "security":
            S(security_mode=True, karaoke_mode=False)
            notify("Security Mode ON", "Every visitor will trigger an alert")
            beep(2)
            log("Security mode ON (LLM)", log_type="activity")
            return "Security mode activated."
        elif mode == "karaoke":
            S(karaoke_mode=True, security_mode=False)
            notify("Karaoke Mode ON", "Visitors welcomed with music!")
            beep(1)
            log("Karaoke mode ON (LLM)", log_type="activity")
            return "Karaoke mode activated."
        else:
            S(security_mode=False, karaoke_mode=False)
            notify("Normal Mode", "Back to normal")
            log("Normal mode (LLM)", log_type="activity")
            return "Normal mode activated."

    if name == "show_camera":
        visible = args.get("visible", True)
        S(show_camera=visible)
        return "Camera window opened." if visible else "Camera window closed."

    if name == "visitor_log":
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            lines = [l.strip() for l in open(LOG_FILE) if today in l]
            return (f"Today's log: {len(lines)} entries. "
                    f"{state['visitors_today']} visitors, {state['alerts_today']} alerts.")
        except Exception:
            return "No log entries yet today."

    return f"Unknown tool: {name}"


def _sys_prompt() -> str:
    n    = state["people_now"]
    mode = "security" if state["security_mode"] else "karaoke" if state["karaoke_mode"] else "normal"
    return (
        "You are Yaariok, a smart and friendly AI reception robot at the Yaariok office. "
        "Speak naturally like a real human assistant — warm, a little witty, conversational. "
        "Keep every spoken reply under 2 short sentences. "
        "No bullet points, no markdown, no special characters — output is read aloud via text-to-speech. "
        "When the user asks you to DO something (snapshot, music, lights, mode, camera), "
        "call the appropriate tool. For chitchat and questions, reply directly without tools. "
        f"Live context: {n} {'person' if n==1 else 'people'} in reception right now, "
        f"mode is {mode}, {state['visitors_today']} visitors today."
    )


def _or_call(messages: list, use_tools: bool = True, max_tokens: int = 256) -> dict:
    """Single OpenRouter API call. Returns the raw response dict."""
    import json as _json
    headers = {
        "Authorization": f"Bearer {_OR_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:5000",
        "X-Title":       "Yaariok Smart System",
    }
    body = {
        "model":       _OR_MODEL,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.7,
    }
    if use_tools:
        body["tools"]       = LLM_TOOLS
        body["tool_choice"] = "auto"
    resp = requests.post(_OR_URL, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def ask_claude(user_text: str) -> str:
    """Direct OpenRouter call for voice questions — simple, no tools, always responds."""
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_OR_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "http://localhost:5000",
                "X-Title":       "Yaariok Smart System",
            },
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are YaariOK, an AI receptionist robot at the Yaariok office. "
                            "Be warm, helpful, and brief. Keep replies under 2 sentences. "
                            "No markdown, no bullet points — output is spoken aloud via TTS."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": 120,
                "temperature": 0.7,
            },
            timeout=15,
        )
        data = resp.json()
        print(f"  [Claude] raw response: {data}")
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [Claude Error] {e}")
        return "I had trouble connecting to my brain. Please try again."


def chat(user_msg: str) -> str:
    global _conv_history
    if not _OR_READY:
        return "My AI brain is not connected yet. Please check the OpenRouter key in keys dot py."

    import json as _json

    _conv_history.append({"role": "user", "content": user_msg})
    if len(_conv_history) > 14:
        _conv_history = _conv_history[-14:]

    messages = [{"role": "system", "content": _sys_prompt()}] + _conv_history

    try:
        # Pass 1 — Claude decides whether to use a tool or reply directly
        data = _or_call(messages, use_tools=True, max_tokens=512)
        choice  = data["choices"][0]
        msg     = choice["message"]
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []

        if tool_calls:
            # Execute every tool the model requested
            tool_results = []
            assistant_msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
            for tc in tool_calls:
                fn   = tc["function"]
                args = _json.loads(fn.get("arguments") or "{}")
                result = _exec_tool(fn["name"], args)
                print(f"{C['cyan']}  [TOOL] {fn['name']}({args}) → {result}{C['reset']}")
                tool_results.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result,
                })

            # Pass 2 — get the spoken confirmation after tool execution
            data2  = _or_call(messages + [assistant_msg] + tool_results,
                              use_tools=False, max_tokens=150)
            reply  = (data2["choices"][0]["message"].get("content") or "").strip()
        else:
            reply = content

        reply = reply or "I'm here! Go ahead."
        _conv_history.append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        print(f"\n{C['red']}  [LLM] {e}{C['reset']}")
        return "I had trouble thinking just now — ask me again."


def llm_greet(visitor_num: int) -> str:
    """AI-generated unique welcome for each new visitor."""
    if not _OR_READY:
        return f"Welcome to Yaariok! You are visitor number {visitor_num} today."
    mode = "security" if state["security_mode"] else "karaoke" if state["karaoke_mode"] else "normal"
    try:
        data = _or_call(
            messages=[
                {"role": "system", "content": (
                    "You are Yaariok, a friendly AI reception robot. "
                    "Generate one short, warm, unique welcome greeting for a new visitor. "
                    "No markdown, no special characters — spoken aloud via TTS. Max 1 sentence."
                )},
                {"role": "user", "content": (
                    f"A new person just walked into reception. "
                    f"Visitor number {visitor_num} today. Mode: {mode}. Greet them!"
                )},
            ],
            use_tools=False,
            max_tokens=80,
        )
        return (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        return f"Welcome to Yaariok! You are visitor number {visitor_num} today."


def _frame_to_b64(frame, width=480) -> str:
    """Encode a camera frame as a base64 JPEG for Claude Vision."""
    h, w = frame.shape[:2]
    scale = width / w
    small = cv2.resize(frame, (width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _vision_messages(frame, system_text: str, user_text: str) -> list:
    """Build a messages list that includes the camera frame as a vision image."""
    b64 = _frame_to_b64(frame)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ]},
    ]


def llm_greet_vision(visitor_num: int, frame) -> str:
    """Claude sees the visitor via camera and generates a personalized greeting."""
    if not _OR_READY or frame is None:
        return llm_greet(visitor_num)
    mode = "security" if state["security_mode"] else "karaoke" if state["karaoke_mode"] else "normal"
    try:
        data = _or_call(
            messages=_vision_messages(
                frame,
                system_text=(
                    "You are Yaariok, a friendly AI reception robot with a live camera. "
                    "Look at the visitor and generate ONE warm, personalized welcome. "
                    "Mention something specific you can see — like their clothing color, "
                    "if they are smiling, carrying something, etc. Be natural and warm. "
                    "No markdown. Max 2 short sentences. Spoken aloud via TTS."
                ),
                user_text=(
                    f"Visitor number {visitor_num} just walked in. Mode: {mode}. "
                    "Welcome them warmly based on what you see in the camera!"
                ),
            ),
            use_tools=False,
            max_tokens=100,
        )
        greeting = (data["choices"][0]["message"].get("content") or "").strip()
        print(f"{C['cyan']}  [VISION] Greeting: {greeting}{C['reset']}")
        return greeting or llm_greet(visitor_num)
    except Exception as e:
        print(f"{C['gray']}  [VISION] {e}{C['reset']}")
        return llm_greet(visitor_num)


def llm_security_analysis(frame) -> str:
    """Claude analyzes a security camera frame and returns a threat description."""
    if not _OR_READY or frame is None:
        return ""
    try:
        data = _or_call(
            messages=_vision_messages(
                frame,
                system_text=(
                    "You are a security AI monitoring a reception camera. "
                    "Describe who you see: their appearance, behavior, and any concern level. "
                    "Be direct and brief. 1-2 sentences. No markdown. Spoken aloud via TTS."
                ),
                user_text="Security alert triggered. Analyze this camera frame and report what you see.",
            ),
            use_tools=False,
            max_tokens=80,
        )
        result = (data["choices"][0]["message"].get("content") or "").strip()
        print(f"{C['red']}  [SECURITY-AI] {result}{C['reset']}")
        return result
    except Exception as e:
        print(f"{C['gray']}  [SECURITY-AI] {e}{C['reset']}")
        return ""


def get_claude_response(context: str, user_input: str = "") -> str:
    """Generate a dynamic spoken response using Claude."""
    if not _OR_READY:
        return user_input if user_input else context
    try:
        data = _or_call(
            messages=[
                {"role": "system", "content": (
                    "You are YaariOK, an intelligent receptionist and security assistant robot. "
                    "Respond naturally, professionally, and briefly. "
                    "Be friendly for greetings. Be alert in security mode. Be energetic in karaoke mode. "
                    "Only return the spoken response. Keep it under 2 sentences. No markdown."
                )},
                {"role": "user", "content": f"Context: {context}\nUser Input: {user_input}\nGenerate a spoken response."}
            ],
            use_tools=False,
            max_tokens=80,
        )
        return (data["choices"][0]["message"].get("content") or "").strip() or (user_input if user_input else context)
    except Exception:
        return user_input if user_input else context

def _init_llm():
    global _OR_READY
    if not _OR_KEY:
        print(f"\n{C['yellow']}  [LLM] No OPENROUTER_API_KEY — add it to keys.py{C['reset']}\n")
        return
    try:
        # Quick connectivity test
        data = _or_call(
            messages=[{"role": "user", "content": "Say OK"}],
            use_tools=False, max_tokens=5,
        )
        _ = data["choices"][0]["message"]["content"]
        _OR_READY = True
        print(f"\n{C['green']}  [LLM] Claude Sonnet ready via OpenRouter — 7 tools active{C['reset']}")
    except Exception as e:
        print(f"\n{C['red']}  [LLM] OpenRouter init failed: {e}{C['reset']}")

# ── Command handler ────────────────────────────────────────────────────────────
FILLERS = ["please", "can you", "could you", "would you", "just",
           "some music", "a song", "for me", "music", "song"]

def clean(cmd: str) -> str:
    # Use word-boundary replacement so "me" inside "camera" is not destroyed
    for f in FILLERS:
        cmd = re.sub(r"\b" + re.escape(f) + r"\b", " ", cmd)
    return " ".join(cmd.split())

def handle(raw: str) -> bool:
    cmd = clean(raw)
    S(last_command=raw[:48])

    # ── Camera queries ──
    if any(p in cmd for p in ["how many people", "who is in", "anyone in",
                                "count people", "how many visitors", "anyone there"]):
        n = state["people_now"]
        v = state["visitors_today"]
        word = "person" if n == 1 else "people"
        speak(f"I can see {n} {word} in reception right now. "
              f"Total visitors today: {v}. Snapshots saved: {state['snapshots_taken']}.")

    elif any(p in cmd for p in ["take snapshot", "take photo", "take picture", "capture", "snapshot"]):
        take_snapshot()

    elif any(p in cmd for p in ["show camera", "open camera", "show feed", "camera on"]):
        S(show_camera=True)
        speak("Opening live camera feed. Press Q on the window to close.")

    elif any(p in cmd for p in ["hide camera", "close camera", "camera off"]):
        S(show_camera=False)   # camera thread sees this and closes the window itself
        speak("Camera closed.")

    # ── Modes ──
    elif any(p in cmd for p in ["normal mode", "reset mode", "disable all modes"]):
        set_mode("normal")

    elif any(p in cmd for p in ["disable security", "security off", "deactivate security"]):
        set_mode("normal")

    elif any(p in cmd for p in ["security mode", "enable security", "activate security",
                                  "lock down", "lockdown", "security on"]):
        set_mode("security")

    elif any(p in cmd for p in ["karaoke", "start karaoke", "enable karaoke"]):
        set_mode("karaoke")

    elif any(p in cmd for p in ["stop karaoke", "disable karaoke", "karaoke off"]):
        set_mode("normal")

    # ── Status report ──
    elif any(p in cmd for p in ["status", "report", "what is happening", "give update",
                                  "system status"]):
        n   = state["people_now"]
        v   = state["visitors_today"]
        a   = state["alerts_today"]
        sn  = state["snapshots_taken"]
        mode = "security" if state["security_mode"] else ("karaoke" if state["karaoke_mode"] else "normal")
        det  = "YOLOv8" if YOLO_OK else "motion sensor"
        speak(
            f"System status. Camera is {state['camera_status']}. "
            f"Currently {n} {'person' if n==1 else 'people'} in reception. "
            f"{v} visitors today, {a} alerts, {sn} snapshots. "
            f"Mode is {mode}. Detection using {det}."
        )

    # ── Visitor log ──
    elif any(p in cmd for p in ["show log", "visitor log", "read log", "today visitors"]):
        try:
            lines = open(LOG_FILE).readlines()
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            today_lines = [l.strip() for l in lines if today in l]
            speak(f"Today's log has {len(today_lines)} entries. "
                  f"{state['visitors_today']} visitors, {state['alerts_today']} alerts.")
        except:
            speak("No log file found yet.")

    # ── Nanta Tech website ──
    elif any(p in cmd for p in ["nantatech", "nanta tech", "nantacheck", "nanta check", "open nanta", "nanta website"]):
        speak("Opening Nanta Tech Limited website.")
        webbrowser.open("https://www.nantatech.com")
        log("Opened nantatech.com")

    # ── YouTube / Music ──
    elif "youtube" in cmd or "play" in cmd:
        song = ""
        for kw in ["open youtube play", "youtube play", "play"]:
            if kw in cmd:
                song = cmd.split(kw, 1)[-1].strip()
                break
        if song:
            play_youtube(song)
        else:
            speak("Opening YouTube.")
            webbrowser.open("https://www.youtube.com")

    # ── Time ──
    elif "time" in cmd:
        now = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"It is {now}.")

    # ── Help ──
    elif any(p in cmd for p in ["help", "what can you do", "commands"]):
        speak(
            "I can: play music on YouTube, control lights, show camera feed, "
            "count people in reception, take snapshots, enable security or karaoke mode, "
            "give a status report, and read the visitor log."
        )

    # ── Dashboard ──
    elif any(p in cmd for p in ["open dashboard", "show dashboard", "dashboard"]):
        webbrowser.open("http://localhost:5000")
        speak(get_claude_response("Opening the YaariOK dashboard now"))
        log("Dashboard opened via voice")

    # ── Delete today snapshots ──
    elif any(p in cmd for p in ["delete today snapshots", "delete snapshots",
                                  "clear snapshots", "clear today snapshots"]):
        n = delete_today_snapshots()
        speak(get_claude_response(f"Deleted {n} snapshots from today"))
        log(f"Deleted {n} today snapshots via voice")

    # ── Stop ──
    elif any(p in cmd for p in ["stop", "exit", "quit", "bye", "goodbye", "shut down", "shutdown"]):
        speak("Goodbye! Shutting down Yaariok Smart System.")
        log("=== System stopped ===")
        S(running=False, show_camera=False)
        cv2.destroyAllWindows()
        return False

    else:
        print(f"{C['blue']}  [LLM] → '{cmd}'{C['reset']}")
        reply = chat(cmd)
        speak(reply)

    return True

# ── Terminal dashboard ─────────────────────────────────────────────────────────
def dashboard_thread():
    while state["system_running"] and state["running"]:
        time.sleep(3)
        n    = state["people_now"]
        v    = state["visitors_today"]
        a    = state["alerts_today"]
        cam  = state["camera_status"]
        mode = f"{C['red']}SECURITY{C['gray']}" if state["security_mode"] else \
               (f"{C['yellow']}KARAOKE{C['gray']}"  if state["karaoke_mode"]  else
                f"{C['green']}NORMAL{C['gray']}")
        det  = "YOLO" if YOLO_OK else "MOTION"
        print(
            f"\r{C['gray']}[{det}] CAM:{cam}  People:{n}  Visitors:{v}  "
            f"Alerts:{a}  Mode:{mode}  | {state['robot_status'][:38]}{C['reset']}",
            end="", flush=True
        )

# ── Wake words ─────────────────────────────────────────────────────────────────
WAKE = [
    "hey robot", "hi robot", "okay robot", "ok robot",
    "a robot", "the robot", "hey robots", "robot",
    "hey robert", "hey rob",
    # common STT mishearings of "hey robot"
    "hey ravan", "hey rabot", "hey rubot", "hey robo",
    "hey ribbon", "hey robin", "air robot", "a robot",
    "ravan", "rubot", "rabot",
]

# ── Main ───────────────────────────────────────────────────────────────────────


_camera_t = None
_voice_t = None

def start_system():
    global _camera_t
    if not state["system_running"]:
        print("  [System Debug] System started")
        S(system_running=True)
        _camera_t = threading.Thread(target=camera_thread, daemon=True)
        _camera_t.start()
        log("System started")

def stop_system():
    if state["system_running"]:
        print("  [System Debug] System stopped")
        S(system_running=False, show_camera=False)
        log("System stopped")

def set_mode(mode):
    if mode == "karaoke":
        S(karaoke_mode=True, security_mode=False)
        notify("Karaoke Mode ON", "Visitors welcomed with music!")
        beep(1)
        speak(get_claude_response("Karaoke mode activated"))
        log("Karaoke mode ON (Voice)", log_type="activity")
    elif mode == "security":
        S(security_mode=True, karaoke_mode=False)
        notify("Security Mode ON", "Every visitor will trigger an alert")
        beep(2)
        speak(get_claude_response("Security mode activated"))
        log("Security mode ON (Voice)", log_type="activity")
    elif mode == "normal":
        S(security_mode=False, karaoke_mode=False)
        notify("Normal Mode", "Back to normal")
        speak(get_claude_response("Normal mode activated"))
        log("Normal mode (Voice)", log_type="activity")

def take_snapshot():
    S(take_snapshot=True)
    notify("Snapshot", "Photo saved by voice command")
    speak(get_claude_response("Snapshot taken"))

def process_voice_command(cmd):
    cmd = cmd.lower().strip()
    # Safety: strip any leftover wake-word fragment the STT loop may have missed
    if cmd.startswith("hey robot"):
        cmd = cmd.replace("hey robot", "", 1).strip()
    print(f"  [Voice] Clean command: '{cmd}'")
    save_log_db("voice", f"Command: {cmd}")

    # ── System commands — executed locally ────────────────────────────────────
    if "start system" in cmd:
        start_system()
        speak(get_claude_response("System started and camera is now active"))

    elif "stop system" in cmd:
        stop_system()
        speak(get_claude_response("System stopped"))

    elif any(p in cmd for p in ["security mode", "start security", "enable security",
                                 "activate security", "security on"]):
        set_mode("security")

    elif any(p in cmd for p in ["karaoke mode", "start karaoke", "enable karaoke",
                                 "karaoke on", "party mode", "start party"]):
        set_mode("karaoke")

    elif any(p in cmd for p in ["normal mode", "start normal", "reset mode",
                                 "back to normal", "normal on", "disable security",
                                 "disable karaoke", "default mode"]):
        set_mode("normal")

    elif any(p in cmd for p in ["take snapshot", "snapshot", "take photo",
                                 "take picture", "capture", "snap shot", "click photo"]):
        take_snapshot()

    elif "run calibration" in cmd or "calibrate" in cmd:
        run_calibration()
        speak("Initializing vision calibration sequence.")

    elif any(p in cmd for p in ["delete today snapshots", "delete snapshots",
                                  "clear snapshots", "clear today snapshots"]):
        n = delete_today_snapshots()
        speak(get_claude_response(f"Deleted {n} snapshots from today"))
        log(f"Deleted {n} today snapshots via voice")

    elif any(p in cmd for p in ["open dashboard", "show dashboard", "dashboard"]):
        webbrowser.open("http://localhost:5000")
        speak(get_claude_response("Opening the YaariOK dashboard"))
        log("Dashboard opened via voice")

    elif "play " in cmd:
        song = cmd.split("play ", 1)[-1].strip()
        if song:
            play_youtube(song)
        else:
            speak("What song should I play?")

    elif cmd.startswith("search ") or cmd == "search":
        query = cmd.replace("search", "", 1).strip()
        if query:
            import urllib.parse
            speak(get_claude_response(f"Searching Google for {query}"))
            encoded = urllib.parse.quote(query)
            url = f"https://www.google.com/search?q={encoded}"
            webbrowser.open(url)
            log(f"Google search: {query}", log_type="activity")
        else:
            global _pending_action
            _pending_action = "search"
            speak("What would you like me to search for?")

    elif "voice off" in cmd:
        speak(get_claude_response("Turning off voice listener"))
        stop_voice_listener()

    # ── Anything else → Claude AI answers ─────────────────────────────────────
    else:
        print(f"  [Voice→Claude] Claude fallback triggered for: '{cmd}'")
        reply = ask_claude(cmd)
        print(f"  [Voice→Claude] Response received: '{reply[:80]}'")
        speak(reply)

    return True

def voice_loop():
    print("Voice thread started")
    recognizer = sr.Recognizer()
    
    # VAD / Endpoint detection optimization (low latency)
    recognizer.pause_threshold = 0.4
    recognizer.non_speaking_duration = 0.3
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 300
    
    clarification_asked = False
    
    while state["voice_running"] and state["running"]:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                while state["voice_running"] and state["running"]:
                    
                    # State: lock (do not listen during TTS to prevent echo/re-trigger)
                    if state.get("voice_state") == "lock":
                        time.sleep(0.1)
                        continue
                        
                    dcmd = state.get("dash_command", "")
                    if dcmd:
                        S(dash_command="", last_command=f"[Dashboard] {dcmd}")
                        handle(dcmd)
                        continue
                    
                    # State: listen
                    S(voice_state="listen")
                    print("Listening...")
                    S(robot_status="Listening...")
                    
                    try:
                        audio = recognizer.listen(source, timeout=2, phrase_time_limit=8)
                        
                        # Cancel processing if system started speaking
                        if state.get("voice_state") == "lock":
                            continue
                            
                        # State: process
                        S(voice_state="process")
                        print("Voice detected")
                        
                        try:
                            text = recognizer.recognize_google(audio).lower().strip()
                            print(f"Speech recognized: {text}")
                            clarification_asked = False # Reset on successful transcription
                        except sr.UnknownValueError:
                            # High accuracy fallback: if unclear, ask ONCE
                            if not clarification_asked:
                                speak(get_claude_response("I didn't quite catch that, could you repeat?"))
                                clarification_asked = True
                            continue
                            
                        global _pending_action
                        
                        cmd = None
                        if globals().get("_pending_action") == "search":
                            # Treat the next phrase entirely as the search query
                            cmd = "search " + text
                            _pending_action = None
                        else:
                            has_wake = any(w in text for w in WAKE)
                            if has_wake:
                                cmd = text
                                for w in sorted(WAKE, key=len, reverse=True):
                                    cmd = cmd.replace(w, "")
                                cmd = " ".join(cmd.split()).strip()
                                
                                if not cmd:
                                    speak("Yes?")
                                    continue
                        
                        if cmd is not None:
                            # State: respond
                            S(voice_state="respond")
                            process_voice_command(cmd)
                            print("Command executed")
                            
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"  [Voice Error] {e}")
                        continue
        except Exception as e:
            print(f"  [Voice Loop Error] Restarting mic: {e}")
            time.sleep(2)
            continue


def start_voice_listener():
    global _voice_t
    if not state["voice_running"]:
        S(voice_running=True)
        _voice_t = threading.Thread(target=voice_loop, daemon=True)
        _voice_t.start()
        log("Voice started")

def stop_voice_listener():
    if state["voice_running"]:
        S(voice_running=False)
        log("Voice stopped")

def main():
    init_db()
    print(f"\\n{C['purple']}{C['bold']}")
    print("  ╔══════════════════════════════════════════╗")
    print("  ║     YAARIOK SMART SYSTEM  v2.0           ║")
    print("  ║  Voice  •  Camera  •  AI Detection       ║")
    print(f"  ╚══════════════════════════════════════════╝{C['reset']}\\n")
    

    _init_llm()
    # Start voice by default
    start_voice_listener()
    
    # Start web dashboard

    try:
        import dashboard as dash
        dash.app.smart_funcs = {
            "start_system": start_system,
            "stop_system": stop_system,
            "start_voice": start_voice_listener,
            "stop_voice": stop_voice_listener,
            "set_mode": set_mode,
            "take_snapshot": take_snapshot
        }
        port = dash.start_dashboard(state, _ann_frame, _lock, LOG_FILE)
        print(f"  Dashboard : {C['green']}http://localhost:{port}{C['reset']}  (open in browser)\\n")
        import webbrowser as _wb
        threading.Timer(1, lambda: _wb.open(f"http://localhost:{port}")).start()
    except Exception as e:
        print(f"  Dashboard : {C['yellow']}unavailable ({e}){C['reset']}\\n")

    while state["running"]:
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        S(running=False)
        cv2.destroyAllWindows()
        print(f"\n{C['gray']}[Stopped]{C['reset']}")
        sys.exit(0)
