"""VisionService — camera capture, YOLO person detection, and display rendering.

Publishes:
  visitor_detected  → when a confirmed new person enters (every GREET_COOLDOWN s)
  security_alert    → when a visitor is detected in security mode
  camera_status_changed → on connect / reconnect / failure

Never calls speak() or any AI function directly. All cross-service
effects are routed through the EventBus.
"""

import base64
import collections
import datetime
import json
import logging
import os
import threading
import time

import cv2
import numpy as np

from backend.config.settings import (
    ALERTS_DIR,
    ENTRY_ZONE,
    RTSP_URL,
    SNAPSHOT_DIR,
    VISION_CONFIG_FILE,
)
from backend.core.constants import (
    EV_CAMERA_STATUS,
    EV_SECURITY_ALERT,
    EV_SYSTEM_STARTED,
    EV_SYSTEM_STOPPED,
    EV_VISITOR_DETECTED,
    MODE_KARAOKE,
    MODE_SECURITY,
)
from backend.utils.tts import play_alert_sound

logger = logging.getLogger(__name__)

_DEFAULT_VISION_CONFIG: dict = {
    "confidence_threshold": 0.60,
    "min_person_area": 1800,
    "min_person_height": 45,
    "min_person_aspect": 1.1,
    "target_fps": 15,
    "box_smoothing_alpha": 0.3,
    "debug_mode": False,
    "door_polygon": [],
    "exclusion_zones": [],
}


class VisionService:

    def __init__(self, state, bus, db) -> None:
        self._state = state
        self._bus = bus
        self._db = db

        # Frame buffers shared with the dashboard (read-only outside this service)
        self._ann_frame: list = [None]   # annotated display frame
        self._cam_frame: list = [None]   # latest raw frame

        # YOLO model (loaded lazily in camera thread)
        self._model = None
        self._yolo_ok = False
        self._device = "cpu"

        # Vision config (hot-reloadable)
        self._vcfg: dict = self._load_config()

        # Histogram memory for person deduplication (120-second window)
        self._recent_snapshots: list = []

        # Calibration guard
        self._is_calibrating = False

        bus.subscribe(EV_SYSTEM_STARTED, self._on_system_started)
        bus.subscribe(EV_SYSTEM_STOPPED, self._on_system_stopped)

    # ── Public API ────────────────────────────────────────────────────────────────

    def get_annotated_frame(self):
        """Return the latest annotated display frame (or None)."""
        return self._ann_frame[0]

    def run_calibration(self) -> None:
        """Auto-calibrate YOLO confidence thresholds using live camera input."""
        if self._is_calibrating:
            return
        self._is_calibrating = True
        threading.Thread(target=self._calibration_thread, daemon=True).start()

    # ── Event handlers ────────────────────────────────────────────────────────────

    def _on_system_started(self) -> None:
        threading.Thread(target=self._camera_thread, daemon=True).start()

    def _on_system_stopped(self) -> None:
        pass  # camera thread exits when system_running becomes False

    # ── Camera thread (entry point) ────────────────────────────────────────────────

    def _camera_thread(self) -> None:
        self._state.update(camera_status="LOADING AI")
        self._load_yolo()

        live = [None]
        flk = threading.Lock()

        def _open_cap():
            c = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c

        def _grabber() -> None:
            cap = _open_cap()
            if cap.isOpened():
                self._state.update(camera_status="LIVE")
                self._db.log_event("Camera connected", "system")
                self._bus.publish(EV_CAMERA_STATUS, status="LIVE")
            else:
                self._state.update(camera_status="FAILED")
                self._bus.publish(EV_CAMERA_STATUS, status="FAILED")
                return
            while self._state.get("system_running") and self._state.get("running"):
                ret, frm = cap.read()
                if ret:
                    with flk:
                        live[0] = frm
                else:
                    self._state.update(camera_status="RECONNECTING")
                    cap.release()
                    time.sleep(4)
                    cap = _open_cap()
                    if cap.isOpened():
                        self._state.update(camera_status="LIVE")
                        self._bus.publish(EV_CAMERA_STATUS, status="LIVE")

        threading.Thread(target=_grabber, daemon=True).start()

        # Wait for first frame
        t0 = time.time()
        while live[0] is None and self._state.get("running"):
            time.sleep(0.05)
            if time.time() - t0 > 15:
                self._state.update(camera_status="FAILED")
                return

        bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
        smoothed_boxes: dict = {}

        det_lock = threading.Lock()
        det_state = {"boxes": [], "people_count": 0}

        threading.Thread(
            target=self._detect_loop,
            args=(live, flk, bg_sub, det_lock, det_state),
            daemon=True,
        ).start()

        self._render_loop(live, flk, det_lock, det_state, smoothed_boxes)
        cv2.destroyAllWindows()

    # ── Detection loop ─────────────────────────────────────────────────────────────

    def _detect_loop(self, live, flk, bg_sub, det_lock, det_state) -> None:
        person_present = False
        detection_count = 0
        REQUIRED_FRAMES = 3
        absence_start = 0.0
        last_greeted = 0.0
        GREET_COOLDOWN = 120.0
        last_alert_beep = 0.0
        beep_cooldown = 15.0
        cfg_ctr = 0
        detection_history = collections.deque(maxlen=10)

        while self._state.get("system_running") and self._state.get("running"):
            t0 = time.time()

            # Hot-reload vision config every 30 frames
            cfg_ctr += 1
            if cfg_ctr % 30 == 0:
                self._vcfg.update(self._load_config())

            with flk:
                frame = live[0]
            if frame is None:
                time.sleep(0.01)
                continue

            small = cv2.resize(frame, (640, 360))
            people, boxes_data = self._run_detection(small, bg_sub)

            # Hysteresis: require ≥2 of last 10 frames to flip "on"
            detection_history.append(people > 0)
            if not person_present:
                is_stable = sum(detection_history) >= 2
            else:
                is_stable = sum(detection_history) > 0

            stable_people = max(1, people) if is_stable else 0
            self._state.update(people_now=stable_people)

            with det_lock:
                det_state["boxes"] = boxes_data
                det_state["people_count"] = stable_people

            now = time.time()

            if stable_people > 0:
                absence_start = 0.0
                detection_count += 1

                if detection_count >= REQUIRED_FRAMES and not person_present:
                    # ── Confirmed new entry ──────────────────────────────────────
                    person_present = True
                    logger.info("[Detect] Person entered")

                    vnum = self._state.increment("visitors_today")
                    self._state.update(last_alert_time=datetime.datetime.now().strftime("%H:%M:%S"))
                    self._db.log_event(f"Visitor #{vnum} arrived", "visitor", person_count=stable_people)

                    # Save snapshot
                    is_alert = self._state.get("security_mode")
                    prefix = "ALERT_visitor" if is_alert else "visitor"
                    fname = datetime.datetime.now().strftime(f"{prefix}_%H%M%S.jpg")
                    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(SNAPSHOT_DIR, fname), frame)
                    self._db.save_snapshot(fname, "Alert" if is_alert else "Visitor")

                    if is_alert:
                        self._state.increment("alerts_today")
                    self._state.increment("snapshots_taken")

                    # Encode thumbnail for dashboard
                    try:
                        _, buf = cv2.imencode(".jpg", cv2.resize(frame, (320, 180)),
                                             [cv2.IMWRITE_JPEG_QUALITY, 70])
                        self._state.update(last_visitor_b64=base64.b64encode(buf.tobytes()).decode())
                    except Exception:
                        pass

                if person_present:
                    # ── Greeting — fires once per visit, zone-gated ──────────────
                    in_zone = (not self._yolo_ok) or self._person_in_entry_zone(boxes_data)
                    if in_zone and (now - last_greeted > GREET_COOLDOWN):
                        last_greeted = now
                        mode = self._current_mode()

                        if mode == MODE_SECURITY:
                            should_beep = now - last_alert_beep >= beep_cooldown
                            if should_beep:
                                last_alert_beep = now
                            self._trigger_security_alert(frame, stable_people, should_beep)
                        else:
                            self._bus.publish(
                                EV_VISITOR_DETECTED,
                                frame=frame.copy(),
                                visitor_num=vnum,
                                mode=mode,
                            )
            else:
                detection_count = 0
                if person_present:
                    if absence_start == 0.0:
                        absence_start = now
                    if now - absence_start > 2.0:
                        logger.info("[Detect] Person left — resetting")
                        person_present = False
                        absence_start = 0.0
                        last_alert_beep = 0.0

            elapsed = time.time() - t0
            target_fps = self._vcfg.get("target_fps", 15)
            time.sleep(max(0.005, (1.0 / target_fps) - elapsed))

    # ── Rendering loop ─────────────────────────────────────────────────────────────

    def _render_loop(self, live, flk, det_lock, det_state, smoothed_boxes) -> None:
        was_showing = False
        box_alpha = self._vcfg.get("box_smoothing_alpha", 0.3)

        while self._state.get("system_running") and self._state.get("running"):
            with flk:
                frame = live[0]
            if frame is None:
                time.sleep(0.01)
                continue

            self._cam_frame[0] = frame.copy()

            # Manual snapshot (set by dashboard or voice)
            if self._state.get("take_snapshot"):
                fname = datetime.datetime.now().strftime("manual_%H%M%S.jpg")
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, fname), frame)
                self._db.save_snapshot(fname, "Manual")
                self._state.increment("snapshots_taken")
                self._state.update(take_snapshot=False)
                self._db.log_event("Manual snapshot taken", "activity")
                self._bus.publish("speak_text", text="Snapshot saved.")

            display = self._draw_overlays(frame, det_lock, det_state, smoothed_boxes, box_alpha)
            self._ann_frame[0] = display

            time.sleep(0.010)

            now_showing = self._state.get("show_camera")
            if now_showing:
                cv2.imshow("Yaariok Smart Camera", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._state.update(show_camera=False)
                    cv2.destroyAllWindows()
                    was_showing = False
                    continue
            elif was_showing:
                cv2.destroyAllWindows()
            was_showing = now_showing

    # ── Drawing helpers ────────────────────────────────────────────────────────────

    def _draw_overlays(self, frame, det_lock, det_state, smoothed_boxes, box_alpha) -> np.ndarray:
        display = cv2.resize(frame, (1280, 720))
        sec = self._state.get("security_mode")
        kar = self._state.get("karaoke_mode")
        t = time.time()

        if sec:
            box_color = (0, 0, 255)
        elif kar:
            hue = int((t * 40) % 180)
            box_color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 220]]]), cv2.COLOR_HSV2BGR)[0][0])
        else:
            box_color = (34, 197, 94)

        with det_lock:
            cur_boxes = list(det_state["boxes"])
            cur_people = det_state["people_count"]

        h_orig, w_orig = frame.shape[:2]
        h, w = display.shape[:2]
        scale_x = w_orig / 640.0
        scale_y = h_orig / 360.0
        sx = w / w_orig
        sy = h / h_orig

        active_ids: set = set()
        for b in cur_boxes:
            tid = b["id"]
            active_ids.add(tid)
            x1, y1, x2, y2 = b["xyxy"]
            x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
            if tid not in smoothed_boxes:
                smoothed_boxes[tid] = [x1, y1, x2, y2]
            else:
                ox1, oy1, ox2, oy2 = smoothed_boxes[tid]
                x1 = box_alpha * x1 + (1 - box_alpha) * ox1
                y1 = box_alpha * y1 + (1 - box_alpha) * oy1
                x2 = box_alpha * x2 + (1 - box_alpha) * ox2
                y2 = box_alpha * y2 + (1 - box_alpha) * oy2
                smoothed_boxes[tid] = [x1, y1, x2, y2]
            dx1, dy1 = int(x1 * sx), int(y1 * sy)
            dx2, dy2 = int(x2 * sx), int(y2 * sy)
            label = "ALERT" if sec else ("★" if kar else f"{b['conf']:.0%}")
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), box_color, 2)
            cv2.putText(display, label, (dx1 + 4, max(dy1 + 16, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        for k in list(smoothed_boxes.keys()):
            if k not in active_ids:
                del smoothed_boxes[k]

        # Header bar
        if sec:
            pulse = int(180 + 75 * abs(np.sin(t * 2.5)))
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, pulse), 10)
            cv2.rectangle(display, (0, 0), (w, 48), (0, 0, 180), -1)
            cv2.putText(display, "SECURITY MODE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 60, 60), 2, cv2.LINE_AA)
            ai_txt = self._state.get("ai_alert", "")[:60]
            if ai_txt:
                cv2.putText(display, ai_txt, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 100, 100), 1, cv2.LINE_AA)
            if cur_people > 0 and int(t * 2) % 2 == 0:
                cv2.putText(display, f"  INTRUDER DETECTED  x{cur_people}", (w // 2 - 160, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        elif kar:
            hue2 = int((t * 50) % 180)
            kc = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), kc, 8)
            cv2.rectangle(display, (0, 0), (w, 48), (100, 0, 140), -1)
            cv2.putText(display, "KARAOKE MODE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 50), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(display, (0, 0), (w, 44), (8, 12, 20), -1)
            cv2.putText(display, "YAARIOK  RECEPTION", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 200, 220), 2, cv2.LINE_AA)

        # Footer bar
        cv2.rectangle(display, (0, h - 32), (w, h), (0, 0, 0), -1)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            display,
            f"People: {cur_people}  |  Visitors: {self._state.get('visitors_today')}  |  {ts}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 150, 180), 1, cv2.LINE_AA,
        )

        if self._vcfg.get("debug_mode"):
            fps = self._vcfg.get("target_fps", 15)
            alpha = self._vcfg.get("box_smoothing_alpha", 0.3)
            cv2.putText(display, f"Target FPS: {fps} | Alpha: {alpha}",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return display

    # ── Detection helpers ──────────────────────────────────────────────────────────

    def _run_detection(self, small: np.ndarray, bg_sub) -> tuple:
        """Returns (person_count, boxes_data_list)."""
        boxes_data = []

        if self._yolo_ok:
            conf_thresh = self._vcfg["confidence_threshold"]
            min_area = self._vcfg["min_person_area"]
            min_ar = self._vcfg.get("min_person_aspect", 1.1)
            min_h = self._vcfg.get("min_person_height", 45)

            results = self._model.track(
                small, classes=[0], verbose=False,
                conf=conf_thresh, half=True, device=self._device,
                persist=True, tracker="bytetrack.yaml",
            )
            valid = []
            if results[0].boxes.id is not None:
                for b in results[0].boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    bw, bh = x2 - x1, y2 - y1
                    area = bw * bh
                    ar = bh / bw if bw > 0 else 0
                    if area >= min_area and bh >= min_h and ar >= min_ar:
                        valid.append(b)

            for b in valid:
                is_new = self._is_new_person(small, b)
                boxes_data.append({
                    "id": int(b.id[0]) if b.id is not None else hash(str(b.xyxy[0])),
                    "xyxy": b.xyxy[0].tolist(),
                    "conf": float(b.conf[0]),
                    "is_new": is_new,
                })
            return len(valid), boxes_data

        # MOG2 fallback
        mask = bg_sub.apply(small)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh > self._vcfg["min_person_area"] * 0.5 and 0.2 <= (bw / float(bh)) <= 0.5:
                count += 1
        return count, []

    def _is_new_person(self, frame: np.ndarray, box, threshold: float = 0.85, memory: float = 120.0) -> bool:
        now = time.time()
        self._recent_snapshots = [(t, h) for t, h in self._recent_snapshots if now - t <= memory]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return True

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        for _, prev_hist in self._recent_snapshots:
            if cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL) >= threshold:
                return False

        self._recent_snapshots.append((now, hist))
        return True

    def _person_in_entry_zone(self, boxes_data: list, fw: int = 640, fh: int = 360) -> bool:
        if not boxes_data:
            return False
        poly = self._vcfg.get("door_polygon", [])
        has_poly = len(poly) >= 3
        if has_poly:
            scaled_poly = np.array(poly, np.int32)
        x1z, y1z = ENTRY_ZONE[0] * fw, ENTRY_ZONE[1] * fh
        x2z, y2z = ENTRY_ZONE[2] * fw, ENTRY_ZONE[3] * fh

        for b in boxes_data:
            x1, y1, x2, y2 = b["xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if has_poly:
                if cv2.pointPolygonTest(scaled_poly, (cx, cy), False) >= 0:
                    return True
            else:
                if x1z <= cx <= x2z and y1z <= cy <= y2z:
                    return True
        return False

    def _trigger_security_alert(self, frame, num_people: int, should_beep: bool) -> None:
        fname = datetime.datetime.now().strftime("alert_%H%M%S.jpg")
        os.makedirs(ALERTS_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(ALERTS_DIR, fname), frame)
        self._db.save_alert(
            fname,
            alert_type=f"{num_people} Person(s) Detected",
            person_count=num_people,
        )
        self._db.save_log("alert", f"Security alert: {num_people} person(s) detected — {fname}",
                          person_count=num_people)
        if should_beep:
            play_alert_sound()
        self._state.update(
            last_live_alert=fname,
            last_live_alert_time=datetime.datetime.now().strftime("%H:%M:%S"),
        )
        self._bus.publish(EV_SECURITY_ALERT, frame=frame.copy(), num_people=num_people)

    # ── Config helpers ─────────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        cfg = dict(_DEFAULT_VISION_CONFIG)
        try:
            if os.path.exists(VISION_CONFIG_FILE):
                with open(VISION_CONFIG_FILE) as f:
                    cfg.update(json.load(f))
        except Exception:
            pass
        return cfg

    def _save_config(self) -> None:
        try:
            with open(VISION_CONFIG_FILE, "w") as f:
                json.dump(self._vcfg, f, indent=4)
        except Exception:
            pass

    # ── YOLO loader ────────────────────────────────────────────────────────────────

    def _load_yolo(self) -> None:
        try:
            import torch
            from ultralytics import YOLO
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._model = YOLO("yolov8s.pt")
            self._model.to(self._device)
            self._yolo_ok = True
            logger.info("[YOLO] Person detection ready on %s", self._device.upper())
        except Exception as e:
            self._yolo_ok = False
            logger.warning("[YOLO] Not available — using motion detection (%s)", e)

    # ── Calibration ────────────────────────────────────────────────────────────────

    def _calibration_thread(self) -> None:
        from backend.core.constants import EV_SPEAK_TEXT
        self._bus.publish(EV_SPEAK_TEXT, text=(
            "Calibration started. Place a person at the farthest distance "
            "and walk around for 60 seconds to calibrate real people."
        ))
        time.sleep(5)

        real_scores, real_areas = [], []
        t0 = time.time()
        while time.time() - t0 < 60:
            frame = self._cam_frame[0]
            if frame is not None and self._yolo_ok:
                small = cv2.resize(frame, (640, 360))
                res = self._model(small, classes=[0], verbose=False, conf=0.1, half=True)[0]
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    real_scores.append(float(b.conf[0]))
                    real_areas.append((x2 - x1) * (y2 - y1))
            time.sleep(0.1)

        self._bus.publish(EV_SPEAK_TEXT, text="Now please leave the room empty. Calibrating for 60 seconds.")
        time.sleep(5)

        fake_scores = []
        t0 = time.time()
        while time.time() - t0 < 60:
            frame = self._cam_frame[0]
            if frame is not None and self._yolo_ok:
                small = cv2.resize(frame, (640, 360))
                res = self._model(small, classes=[0], verbose=False, conf=0.1, half=True)[0]
                for b in res.boxes:
                    fake_scores.append(float(b.conf[0]))
            time.sleep(0.1)

        p95 = float(np.percentile(real_scores, 5)) if real_scores else 0.5
        f10 = float(np.percentile(fake_scores, 90)) if fake_scores else 0.2
        thresh = (p95 + f10) / 2.0 if p95 > f10 else p95
        min_area = float(np.min(real_areas) * 0.7) if real_areas else 5000.0

        self._vcfg["confidence_threshold"] = max(0.2, min(thresh, 0.95))
        self._vcfg["min_person_area"] = min_area
        self._save_config()

        self._bus.publish(EV_SPEAK_TEXT,
                          text=f"Calibration complete. Threshold set to {self._vcfg['confidence_threshold']:.2f}.")
        self._is_calibrating = False

    def _current_mode(self) -> str:
        if self._state.get("security_mode"):
            return MODE_SECURITY
        if self._state.get("karaoke_mode"):
            return MODE_KARAOKE
        return "normal"
