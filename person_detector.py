"""
person_detector.py
──────────────────
Standalone, reusable person-detection module extracted from YaariOK smart_system.py.

Features
--------
- YOLOv8 person detection (nano model, GPU/CPU auto-select)
- MOG2 motion-based fallback when YOLO is unavailable
- RTSP / webcam frame-grabber on its own thread (low-latency, buffer=1)
- Auto-reconnect on camera failure
- Occupancy state machine with configurable cooldowns
- Clean class interface: get_state(), get_latest_frame(), stop()
- Callback interface: on_person_entered(frame, count)

No snapshots, no voice, no TTS, no Flask, no LLM, no lights.
"""

import cv2
import threading
import time
import numpy as np


# ── Default configuration ──────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Frame resizing before inference
    "resize_width":  640,
    "resize_height": 360,

    # Run detection every N frames (1 = every frame, 2 = every other, …)
    "detection_interval_frames": 2,

    # YOLO confidence threshold (0.0–1.0)
    "yolo_conf_threshold": 0.25,

    # "auto" → use cuda:0 if available, else cpu.  Or force "cpu" / "cuda:0"
    "yolo_device": "auto",

    # YOLO model file (must be on PATH or absolute path)
    "yolo_model": "yolov8n.pt",

    # Fallback MOG2 contour area threshold (pixels on resize_width × resize_height)
    "fallback_contour_area": 1500,

    # Minimum bounding-box area in pixels (on resized frame) to count as a person
    "min_person_area": 800,

    # Seconds of continuous absence before declaring the space empty
    "leave_seconds": 20,

    # Minimum seconds between consecutive "person entered" callbacks
    "greet_cooldown_seconds": 120,

    # VideoCapture buffer size
    "cap_buffer_size": 1,

    # Seconds to wait before retrying camera on failure
    "reconnect_wait_seconds": 4,
}


class PersonDetector:
    """
    Detects people in a camera stream and provides a simple polling + callback API.

    Parameters
    ----------
    source : str | int
        RTSP URL string or integer webcam index (e.g. 0).
    config : dict, optional
        Override any key from DEFAULT_CONFIG.
    on_person_entered : callable(frame, count) | None
        Called (in the detection thread) the first time a person is seen after the
        cooldown period.  ``frame`` is the raw numpy BGR image; ``count`` is the
        number of people detected.
    on_person_left : callable() | None
        Called when the space goes from occupied to empty.
    verbose : bool
        Print status messages to stdout.
    """

    def __init__(
        self,
        source,
        config: dict | None = None,
        on_person_entered=None,
        on_person_left=None,
        verbose: bool = True,
    ):
        self._source           = source
        self._cfg              = {**DEFAULT_CONFIG, **(config or {})}
        self._on_entered       = on_person_entered
        self._on_left          = on_person_left
        self._verbose          = verbose

        # ── Shared state ──────────────────────────────────────────────────────
        self._running          = False
        self._lock             = threading.Lock()
        self._live_frame       = None          # latest raw frame from grabber
        self._frame_lock       = threading.Lock()

        # Detection outputs (written by detect thread, read by caller)
        self._people_count     = 0
        self._just_entered     = False         # True for exactly one get_state() call
        self._occupied         = False

        # ── Occupancy state machine ───────────────────────────────────────────
        self._empty_since      = 0.0           # monotonic time when last person left
        self._last_greeted     = 0.0           # monotonic time of last on_person_entered

        # ── Model / fallback ──────────────────────────────────────────────────
        self._model            = None
        self._yolo_ok          = False
        self._bg_sub           = None          # MOG2 fallback
        self._device           = "cpu"

        # ── Threads ───────────────────────────────────────────────────────────
        self._grabber_thread   = None
        self._detect_thread    = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Load model, open camera, start background threads."""
        self._running = True
        self._load_model()
        self._grabber_thread = threading.Thread(
            target=self._grabber_loop, daemon=True, name="pd-grabber"
        )
        self._grabber_thread.start()

        # Wait up to 10 s for the first frame
        t0 = time.time()
        while self._live_frame is None and time.time() - t0 < 10:
            time.sleep(0.05)
        if self._live_frame is None:
            self._log("Camera did not produce a frame within 10 s — continuing anyway")

        self._detect_thread = threading.Thread(
            target=self._detect_loop, daemon=True, name="pd-detect"
        )
        self._detect_thread.start()
        self._log("PersonDetector started.")
        return self   # allow: detector = PersonDetector(...).start()

    def stop(self):
        """Signal threads to stop and wait for them."""
        self._running = False
        if self._grabber_thread:
            self._grabber_thread.join(timeout=3)
        if self._detect_thread:
            self._detect_thread.join(timeout=3)
        self._log("PersonDetector stopped.")

    def get_state(self):
        """
        Returns
        -------
        (people_count: int, just_entered: bool)
            ``just_entered`` is True only once per new arrival (after cooldown).
        """
        with self._lock:
            count  = self._people_count
            just_e = self._just_entered
            if just_e:
                self._just_entered = False   # consume the flag
        return count, just_e

    def get_latest_frame(self):
        """Return the latest raw BGR frame (numpy array) or None."""
        with self._frame_lock:
            return None if self._live_frame is None else self._live_frame.copy()

    def is_occupied(self) -> bool:
        """True while at least one person has been detected recently."""
        with self._lock:
            return self._occupied

    # ── Internal: model loading ────────────────────────────────────────────────

    def _load_model(self):
        try:
            import torch
            from ultralytics import YOLO

            cfg_dev = self._cfg["yolo_device"]
            if cfg_dev == "auto":
                self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                self._device = cfg_dev

            use_half = self._device.startswith("cuda")
            self._model  = YOLO(self._cfg["yolo_model"])
            self._model.to(self._device)
            # Warm-up inference so first real frame isn't slow
            dummy = np.zeros(
                (self._cfg["resize_height"], self._cfg["resize_width"], 3), dtype=np.uint8
            )
            self._model(
                dummy, classes=[0], verbose=False,
                conf=self._cfg["yolo_conf_threshold"],
                half=use_half,
            )
            self._yolo_ok = True
            self._log(f"[YOLO] Model '{self._cfg['yolo_model']}' ready on {self._device.upper()}")

        except Exception as exc:
            self._yolo_ok = False
            self._log(f"[YOLO] Not available ({exc}) — using MOG2 fallback")
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=50, detectShadows=False
            )

    # ── Internal: frame grabber thread ────────────────────────────────────────

    def _open_cap(self):
        cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cfg["cap_buffer_size"])
        return cap

    def _grabber_loop(self):
        """Continuously reads from camera into self._live_frame (latest frame only)."""
        cap = self._open_cap()
        if cap.isOpened():
            self._log(f"[Camera] Connected to {self._source}")
        else:
            self._log(f"[Camera] FAILED to open {self._source}")

        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._frame_lock:
                    self._live_frame = frame
            else:
                self._log(f"[Camera] Lost connection — retrying in {self._cfg['reconnect_wait_seconds']} s …")
                cap.release()
                time.sleep(self._cfg["reconnect_wait_seconds"])
                cap = self._open_cap()
                if cap.isOpened():
                    self._log("[Camera] Reconnected.")

        cap.release()

    # ── Internal: detection loop ───────────────────────────────────────────────

    def _detect_loop(self):
        frame_n  = 0
        interval = self._cfg["detection_interval_frames"]
        rw, rh   = self._cfg["resize_width"], self._cfg["resize_height"]
        use_half = self._yolo_ok and self._device.startswith("cuda")

        while self._running:
            with self._frame_lock:
                frame = self._live_frame

            if frame is None:
                time.sleep(0.02)
                continue

            frame_n += 1
            if frame_n % interval != 0:
                time.sleep(0.005)
                continue

            small   = cv2.resize(frame, (rw, rh))
            people  = self._run_detection(small, use_half)
            self._update_state(people, frame)

    def _run_detection(self, small: np.ndarray, use_half: bool) -> int:
        """Run YOLO or MOG2 fallback and return person count."""
        if self._yolo_ok:
            results     = self._model(
                small,
                classes=[0],
                verbose=False,
                conf=self._cfg["yolo_conf_threshold"],
                half=use_half,
                device=self._device,
            )
            min_area = self._cfg["min_person_area"]
            count = 0
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if (x2 - x1) * (y2 - y1) >= min_area:
                    count += 1
            return count
        else:
            # MOG2 fallback ────────────────────────────────────────────────────
            mask    = self._bg_sub.apply(small)
            mask    = cv2.GaussianBlur(mask, (5, 5), 0)
            _, thr  = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_a   = self._cfg["fallback_contour_area"]
            count   = 0
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if area > min_a:
                    aspect = w / float(h) if h > 0 else 0
                    if 0.15 <= aspect <= 0.7:   # rough human silhouette ratio
                        count += 1
            return count

    # ── Internal: occupancy state machine ────────────────────────────────────

    def _update_state(self, people: int, frame: np.ndarray):
        now             = time.monotonic()
        leave_secs      = self._cfg["leave_seconds"]
        cooldown_secs   = self._cfg["greet_cooldown_seconds"]

        with self._lock:
            self._people_count = people

            if people > 0:
                self._empty_since = 0.0   # reset absence timer

                if not self._occupied:
                    # Transition: empty → occupied
                    self._occupied = True
                    if now - self._last_greeted >= cooldown_secs:
                        self._last_greeted = now
                        self._just_entered = True
                        if self._on_entered:
                            # Fire callback in its own thread so detection never blocks
                            threading.Thread(
                                target=self._on_entered,
                                args=(frame.copy(), people),
                                daemon=True,
                            ).start()
            else:
                if self._occupied:
                    if self._empty_since == 0.0:
                        self._empty_since = now   # start absence timer

                    if now - self._empty_since >= leave_secs:
                        # Transition: occupied → empty
                        self._occupied    = False
                        self._empty_since = 0.0
                        if self._on_left:
                            threading.Thread(
                                target=self._on_left, daemon=True
                            ).start()

    # ── Utility ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self._verbose:
            print(f"[PersonDetector] {msg}")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PersonDetector live demo")
    parser.add_argument(
        "--source",
        default=0,
        help="Camera source: integer webcam index or RTSP URL (default: 0)",
    )
    parser.add_argument("--conf",    type=float, default=0.25,  help="YOLO confidence threshold")
    parser.add_argument("--min-area",type=int,   default=800,   help="Min bounding-box area (px²)")
    parser.add_argument("--fps",     type=int,   default=5,     help="Target detection FPS")
    parser.add_argument("--cooldown",type=int,   default=30,    help="Seconds between entry events")
    parser.add_argument("--leave",   type=int,   default=5,     help="Seconds of absence = empty")
    args = parser.parse_args()

    # Convert source to int if it looks like a webcam index
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def on_entered(frame, count):
        print(f"\n🟢  PERSON ENTERED — {count} person(s) detected")

    def on_left():
        print("\n🔴  Area is now EMPTY")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = {
        "yolo_conf_threshold":      args.conf,
        "min_person_area":          args.min_area,
        "detection_interval_frames": max(1, 5 // args.fps),  # rough N from target fps
        "greet_cooldown_seconds":   args.cooldown,
        "leave_seconds":            args.leave,
    }

    detector = PersonDetector(
        source           = source,
        config           = cfg,
        on_person_entered= on_entered,
        on_person_left   = on_left,
        verbose          = True,
    )

    # ── Main polling loop ─────────────────────────────────────────────────────
    print(f"Starting PersonDetector on source={source!r}. Press Ctrl+C to quit.\n")
    with detector:
        frame_count  = 0
        fps_timer    = time.time()

        while True:
            count, just_entered = detector.get_state()
            frame_count += 1

            # Print FPS + count every second
            if time.time() - fps_timer >= 1.0:
                elapsed      = time.time() - fps_timer
                fps          = frame_count / elapsed
                occupied_str = "OCCUPIED" if detector.is_occupied() else "EMPTY"
                print(
                    f"\r  Poll FPS: {fps:5.1f}  |  People: {count}  |  State: {occupied_str}     ",
                    end="", flush=True,
                )
                frame_count = 0
                fps_timer   = time.time()

            time.sleep(0.02)   # 50 Hz polling — adjust as needed
