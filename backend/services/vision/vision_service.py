"""VisionService — RTSP capture, YOLOv11+BoT-SORT tracking, and MQTT events.

Architecture:

    ┌──────────────┐  raw frame   ┌──────────────────┐  unknown_track
    │ RTSP grabber │ ───────────▶ │  Detection loop  │ ──────────────┐
    └──────────────┘              │  (YOLO+BoT-SORT) │               │
            │                     └──────────────────┘               ▼
            │  640x360 jpg                │                 ┌────────────────┐
            ▼                             │ annotated       │  Face worker   │
    ┌──────────────┐                      ▼ overlay         │ (InsightFace)  │
    │  MJPEG queue │              ┌──────────────┐          └────────────────┘
    └──────────────┘              │ Stream queue │                  │
                                  └──────────────┘                  │
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │   MQTT publish   │
                                                          │  vision/events   │
                                                          └──────────────────┘

Strict rules enforced in this file:
    - No SQLite, no disk writes for images, no audio side-effects.
    - The camera reading loop never calls InsightFace synchronously —
      heavy work is offloaded onto a bounded worker queue.
    - Frame queue is bounded (maxsize=5) and uses drop-oldest semantics
      so a slow consumer (dashboard) cannot back up producer.
    - RTSP loop is auto-recovering: any failure releases the cap and
      retries on a backoff.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from backend.config.settings import SNAPSHOT_COOLDOWN_SECONDS, ZONE_CONFIG_FILE
from backend.core.business_day import (
    now_ist,
    seconds_until_next_business_day_start,
)
from backend.services.vision.body_engine import BodyEngine
from backend.services.vision.face_engine import DetectedFace, FaceEngine
from backend.services.vision.snapshot_buffer import SnapshotBuffer
from backend.services.vision.vector_store import DailyFaceCache

logger = logging.getLogger(__name__)

# ── Configuration constants ─────────────────────────────────────────────────

MQTT_TOPIC_EVENTS: str = "vision/events"

DEFAULT_BLUR_THRESHOLD: float = 100.0
DEFAULT_MATCH_THRESHOLD: float = 0.30  # CCTV: tuned empirically; 0.385 same-person misses observed at 0.40
DEFAULT_DETECT_CONF: float = 0.45

# Plausibility filters applied to every YOLO person detection. Without
# these, YOLO will happily call posters, table objects, reflections, and
# distant figures "persons", each producing its own face embedding and
# its own NEW visitor registration. Diagnostic of live snapshots showed:
#   - small posters at the top of frame (57×93, 86×167)
#   - non-vertical table junk (163×88, h/w = 0.54)
# Tuned for a 1280×720 native stream.
MIN_PERSON_HEIGHT_PX: int = 150     # minimum vertical extent (head-to-feet)
MIN_PERSON_ASPECT: float = 1.3      # height / width — humans are taller than wide
MIN_PERSON_AREA_PX: int = 15000     # rough floor for "actually a person sized"
RECONNECT_BACKOFF_SECONDS: float = 4.0

# Stream output: bumped from 640x360@q75 to 960x540@q85 — noticeably sharper
# without blowing up the per-frame JPEG cost.
STREAM_FRAME_WIDTH: int = 960
STREAM_FRAME_HEIGHT: int = 540
STREAM_JPEG_QUALITY: int = 85

WORKER_QUEUE_MAX: int = 16
STREAM_QUEUE_MAX: int = 5

# How often we publish a presence_update event (active track ids + count)
# so the dashboard's people_now stays honest even when no NEW face arrives.
PRESENCE_PUBLISH_INTERVAL_SECONDS: float = 1.0

# ── Sprint 1 — Snapshot quality + lurker spam fix ───────────────────────────
#
# SNAPSHOT_BUFFER_WINDOW_SECONDS (Issue #4): we hold the snapshot decision
# open this long, scoring candidate frames by face_quality, before
# committing the sharpest one. 2.5 s ≈ 15 frames at 6 FPS — enough to
# catch a head turn or a clearer pass without delaying the photo
# noticeably for the user.
#
# ABSENT_DURATION_REQUIRED_SECONDS (Issue #5): a known visitor whose
# 10-minute snapshot cooldown has elapsed only qualifies for a refresh
# snapshot if they were ALSO absent from the camera for at least this
# many seconds. A stationary receptionist whose BoT-SORT track resets
# every 10 minutes would otherwise generate one fresh snapshot per
# cooldown window even though they never actually left.
SNAPSHOT_BUFFER_WINDOW_SECONDS: float = 2.5
ABSENT_DURATION_REQUIRED_SECONDS: float = 60.0

# Spatial-IoU sibling threshold — BoT-SORT track-split false-split fix.
#
# Symptom: a single physical person occasionally has their BoT-SORT
# track id flip from N to N+1 over consecutive frames. If the old
# track is still in the SAME frame as the new one when YOLO emits
# results, the co-occurrence exclusivity rule treats the new track
# as "someone different from whoever the old track is bound to" —
# and the new track is forbidden from re-matching the bound
# visitor. Result: a false-split (one person becomes two visitor_ids).
#
# Detection: if the new unknown track's bbox overlaps a BOUND
# track's bbox at IoU ≥ this threshold, the two are treated as the
# same physical person (siblings). The bound track's visitor_id is
# removed from the new track's co-visible exclusion set for that
# one dispatch, so the new track is free to re-match the same
# visitor via the normal body/face rules.
#
# 0.5 is conservative: two distinct people standing shoulder-to-
# shoulder rarely produce bbox IoU > 0.3, and BoT-SORT track
# splits typically produce IoU > 0.7. The middle is safe.
SIBLING_IOU_THRESHOLD: float = 0.5


# ── Internal work items ─────────────────────────────────────────────────────

@dataclass
class _FaceJob:
    """One unit of work for the face worker thread.

    ``is_in_zone`` records whether the person's feet (bottom-center of
    bbox) fell inside the user-configured entry polygon at detection
    time. It is plumbed all the way to the MQTT payload so the AI
    service can gate the VOICE greeting on zone membership while
    snapshots and counters fire for every person regardless.

    ``co_visible_track_ids`` is the snapshot of OTHER track_ids
    visible in the same frame at dispatch time. The worker turns
    those into the set of visitor_ids currently occupying the camera
    view and forbids the new face/body from matching any of them —
    the "co-occurrence exclusivity" rule that makes simultaneous
    arrivals impossible to false-merge.

    ``buffer_feed_for`` (Sprint 1 / Issue #4): when set, this job is
    a LIGHTWEIGHT candidate for the SnapshotBuffer keyed by the
    given visitor_id. The worker runs face detection + quality
    scoring only — no identity search, no MQTT publish, no binding
    work — and submits the result to the buffer. Used to keep
    feeding frames into an open buffer for visitors whose track is
    already bound (which would normally bypass the worker entirely
    via Gate 1).
    """
    frame: np.ndarray
    track_id: int
    bbox: Tuple[float, float, float, float]
    is_in_zone: bool = True
    co_visible_track_ids: tuple = ()   # tuple[int, ...] — other tracks in same frame
    buffer_feed_for: Optional[int] = None   # visitor_id, or None for normal identity work


# ── Service ─────────────────────────────────────────────────────────────────

class VisionService:
    """Stand-alone vision microservice.

    Construction is cheap; heavy resources (YOLO weights, InsightFace
    models, MQTT connection) are lazy-loaded in :meth:`start`. Call
    :meth:`stop` for graceful shutdown — it waits for threads to drain.
    """

    def __init__(
        self,
        rtsp_url: str,
        mqtt_client=None,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_client_id: str = "yaariok-vision",
        yolo_model_path: str = "yolo11n.pt",
        tracker_config: str = "botsort.yaml",
        on_midnight_reset=None,
        detect_conf: float = DEFAULT_DETECT_CONF,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self._rtsp_url: str = rtsp_url
        self._mqtt_host: str = mqtt_host
        self._mqtt_port: int = mqtt_port
        self._mqtt_client_id: str = mqtt_client_id
        self._yolo_model_path: str = yolo_model_path
        self._tracker_config: str = tracker_config
        self._detect_conf: float = detect_conf
        # Optional hook called after the daily FAISS / body-library reset.
        # main.py wires this to ``state.reset_daily_counters`` so the
        # dashboard's persistent counters also roll over at midnight.
        self._on_midnight_reset = on_midnight_reset
        self._blur_threshold: float = blur_threshold

        # When an external client is injected, this service neither
        # connects nor disconnects it — lifecycle is owned by main.py.
        self._mqtt_external: bool = mqtt_client is not None
        self._mqtt_client = mqtt_client

        # ── Lifecycle state ──────────────────────────────────────────────
        self._stop_event: threading.Event = threading.Event()
        self._threads: list = []

        # ── MJPEG stream queue (producer: inference loop, consumer: HTTP) ───
        self._stream_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=STREAM_QUEUE_MAX)

        # ── Face worker queue (decouples InsightFace from camera FPS) ────
        self._face_queue: "queue.Queue[_FaceJob]" = queue.Queue(maxsize=WORKER_QUEUE_MAX)

        # ── In-flight guard ──────────────────────────────────────────────
        # The camera/inference thread runs at ~15 FPS but the face worker
        # takes ~150 ms+ on CPU. Without a guard, the inference thread
        # would enqueue the SAME unknown track 5-10 times per second while
        # the worker is still evaluating the first job — instantly filling
        # the bounded queue with Person 1 and starving Person 2/3 who
        # entered the same frame.
        #
        # We mark a track-id "in flight" the moment we hand it to the
        # worker. The worker's finally-block always removes it on exit.
        # If Gate 4 succeeded the track-id will already be in
        # ``known_bot_sort_ids`` so it won't re-enqueue; if Gate 2 (blur)
        # rejected it, removal lets the next frame try a sharper crop.
        self._in_flight_track_ids: set = set()
        self._in_flight_lock: threading.Lock = threading.Lock()

        # ── Track stability tracker (recent-frames co-occurrence rule) ──
        # Set of track ids emitted by BoT-SORT in the PREVIOUS frame.
        # The co-occurrence exclusion is restricted to tracks that are
        # both (a) emitted in the current frame AND (b) were already
        # emitted in the previous frame — i.e., "stable" for ≥2 consecutive
        # frames. Brief BoT-SORT ghost re-emissions (Kalman-prediction
        # tracks that appear for one frame and vanish) are filtered out.
        # The result: when the same person is briefly occluded and
        # BoT-SORT spawns a new track id alongside the lingering old
        # one, the old one (count=1 this frame) doesn't exclude the
        # new one from matching the same visitor.
        self._prev_frame_track_ids: set = set()

        # ── Entry-zone polygon (greeting gate, NOT a detection filter) ──
        # Stored normalized (0..1) so it survives camera-resolution changes.
        # Empty list = default-allow: every detection is tagged in_zone=True.
        # The polygon is loaded from disk on boot and hot-reloaded whenever
        # /api/zone POSTs a new shape. Lock protects concurrent reads from
        # the inference thread vs writes from Flask's HTTP thread.
        self._zone_norm: list = []   # list[tuple[float, float]] of normalized (x, y)
        self._zone_lock: threading.Lock = threading.Lock()
        self.reload_zone()

        # ── Latest-frame slot (drop-oldest hand-off from grabber → inference) ──
        # Grabber thread writes the most recent decoded frame; inference
        # thread reads it. We never queue raw frames — keeping only the
        # newest one means YOLO/BoT-SORT always work on fresh data and
        # never fall behind the RTSP socket buffer.
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock: threading.Lock = threading.Lock()
        self._frame_ready: threading.Event = threading.Event()

        # Presence broadcast (so the dashboard's people_now stays accurate
        # even when no NEW visitor event fires).
        self._last_presence_publish: float = 0.0

        # ── Engines ──────────────────────────────────────────────────────
        self._face_engine: FaceEngine = FaceEngine(blur_threshold=blur_threshold)
        # OSNet-x0_25 body re-ID — replaces the prior HSV-histogram path.
        # Cheap to construct; weights load in :meth:`start` via initialize().
        self._body_engine: BodyEngine = BodyEngine()
        self._face_cache: DailyFaceCache = DailyFaceCache(match_threshold=match_threshold)

        # ── Snapshot buffer (Issue #4 — best-frame selection) ────────────
        # When the pipeline decides a snapshot should be committed, the
        # decision opens a buffer here instead of publishing immediately.
        # Candidate frames feed the buffer over the configured window;
        # the sharpest one is committed via the on_commit callback,
        # which publishes the real ``visitor_snapshot`` MQTT event.
        self._snapshot_buffer: SnapshotBuffer = SnapshotBuffer(
            window_seconds=SNAPSHOT_BUFFER_WINDOW_SECONDS,
            on_commit=self._commit_buffered_snapshot,
        )

        # YOLO loaded in start(). MQTT client may already be injected;
        # if not, _connect_mqtt() will create one. Do NOT reset it here.
        self._yolo = None
        self._yolo_device: str = "cpu"
        self._mqtt_connected: bool = False

    # ── Public lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Load models, connect MQTT, and spin up worker threads."""
        logger.info("VisionService starting …")
        self._stop_event.clear()

        self._load_yolo()
        if not self._face_engine.initialize():
            logger.warning(
                "FaceEngine init failed — service will still run but no "
                "embeddings will be published."
            )
        # BodyEngine loads OSNet weights. On failure the worker still
        # runs face-only: every body_emb will be None, the search() will
        # skip C2/C3, and identity decisions ride C1 alone. Degraded but
        # not crashing — same posture as FaceEngine failure.
        if not self._body_engine.initialize():
            logger.warning(
                "BodyEngine init failed — body re-ID disabled, falling "
                "back to face-only identity."
            )
        # Only manage the MQTT connection when no external client was injected.
        if not self._mqtt_external:
            self._connect_mqtt()
        elif self._mqtt_client is not None:
            logger.info("MQTT client injected — using externally-managed connection.")

        # Grabber thread: RTSP → latest frame slot (no inference, no JPEG).
        # Inference thread: latest frame → YOLO + BoT-SORT + face dispatch
        #                   + annotated JPEG → stream queue.
        # Splitting these means RTSP keeps draining at full speed even
        # when YOLO is slow — the source of the "lag when person detected"
        # symptom.
        self._spawn(self._run_grabber_loop,   name="vision-grabber")
        self._spawn(self._run_inference_loop, name="vision-inference")
        self._spawn(self._run_face_worker,    name="vision-face-worker")
        self._spawn(self._run_business_day_reset, name="vision-business-day")
        logger.info("VisionService started (%d threads).", len(self._threads))

    def stop(self, join_timeout: float = 5.0) -> None:
        """Signal threads to exit and wait for them to drain."""
        logger.info("VisionService stopping …")
        self._stop_event.set()

        # Unblock the face worker if it's parked on queue.get().
        try:
            self._face_queue.put_nowait(None)  # type: ignore[arg-type]
        except queue.Full:
            pass

        for t in self._threads:
            t.join(timeout=join_timeout)

        # Only tear down the client we created ourselves; an injected
        # client is owned by the caller (main.py).
        if self._mqtt_client is not None and not self._mqtt_external:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:
                pass
        logger.info("VisionService stopped.")

    # ── Public MJPEG generator ───────────────────────────────────────────

    def get_video_feed(self) -> Generator[bytes, None, None]:
        """Yield multipart MJPEG chunks for a FastAPI / Flask streaming endpoint.

        Usage (Flask):
            @app.route("/video")
            def video():
                return Response(
                    vision.get_video_feed(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                )

        Producer-side semantics: the camera loop drops the oldest frame
        when this consumer falls behind, so the served stream remains
        close to real-time at the cost of occasional skipped frames.
        """
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while not self._stop_event.is_set():
            try:
                jpeg = self._stream_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            yield boundary + jpeg + b"\r\n"

    # Alias matching the architectural rules' naming.
    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Alias for :meth:`get_video_feed`."""
        return self.get_video_feed()

    # ── Testing / first-time wipe ────────────────────────────────────────

    def factory_reset(self) -> None:
        """Wipe the in-memory daily identity cache.

        Called by the dashboard's "Start System First Time" button so
        previously-detected visitors become unknown again when
        re-testing the pipeline. Drops:
            - FAISS face index
            - per-visitor body histogram library
            - known_bot_sort_ids set
            - in-flight track id set (just in case)

        On-disk artefacts (snapshots/, DB rows, daily_state.json) are
        the route handler's responsibility — this method only touches
        the in-memory state owned by VisionService.
        """
        # Drain any pending snapshot buffers BEFORE wiping the
        # identity cache. The on_commit callback only publishes MQTT
        # — disk + DB persistence happens in the bridge, which is
        # fine to receive the in-flight snapshot. If we wiped first
        # the buffer would commit with stale visitor_ids that no
        # longer map to anything in the cache.
        self._snapshot_buffer.drain()
        self._face_cache.reset()
        with self._in_flight_lock:
            self._in_flight_track_ids.clear()
        logger.info("VisionService.factory_reset — daily identity cache wiped")

    # ── Entry-zone polygon (voice-greeting gate, NOT a detection filter) ─

    def reload_zone(self) -> None:
        """Hot-reload the normalized entry polygon from disk.

        Called once at construction and every time ``POST /api/zone``
        saves a new polygon. Failure modes (missing file, malformed
        JSON, invalid coords) all fall back to an empty polygon —
        which means "no zone configured" → every detection is tagged
        ``in_zone=True`` and voice greets everyone, as before.

        Coords on disk are normalized [0.0, 1.0]; we keep them
        normalized in memory and scale to pixel coordinates per-frame
        inside :meth:`_point_in_zone`.
        """
        polygon: list = []
        try:
            if os.path.exists(ZONE_CONFIG_FILE):
                with open(ZONE_CONFIG_FILE) as f:
                    data = json.load(f)
                raw = data.get("polygon", [])
                for pt in raw:
                    if (isinstance(pt, (list, tuple)) and len(pt) == 2
                            and all(isinstance(c, (int, float)) for c in pt)
                            and 0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0):
                        polygon.append((float(pt[0]), float(pt[1])))
                if 0 < len(polygon) < 3:
                    # A 1- or 2-point polygon can't enclose anything;
                    # treat as "no zone" rather than a degenerate one.
                    logger.warning(
                        "Zone polygon has only %d point(s) — ignoring "
                        "(need ≥3). Default-allow remains in effect.",
                        len(polygon),
                    )
                    polygon = []
        except Exception:
            logger.exception("reload_zone: failed to read %s", ZONE_CONFIG_FILE)
            polygon = []

        with self._zone_lock:
            self._zone_norm = polygon
        logger.info(
            "Entry zone reloaded: %d points (%s)",
            len(polygon),
            "active" if polygon else "default-allow",
        )

    def _point_in_zone(self, x: float, y: float, fw: int, fh: int) -> bool:
        """Return True if pixel (x, y) is inside the configured polygon.

        Default-allow: no polygon configured → True for everything. Uses
        ``cv2.pointPolygonTest`` (sub-microsecond), called once per
        person per frame.
        """
        with self._zone_lock:
            poly_norm = self._zone_norm
        if not poly_norm:
            return True
        # Scale normalized polygon to current frame size on the fly.
        # The polygon is small (a handful of points) so this is cheap;
        # done per-call to handle frame-size changes without bookkeeping.
        poly_px = np.array(
            [[int(px * fw), int(py * fh)] for px, py in poly_norm],
            dtype=np.int32,
        )
        return cv2.pointPolygonTest(poly_px, (float(x), float(y)), False) >= 0

    # ── Legacy compatibility shim ────────────────────────────────────────

    def run_calibration(self) -> None:
        """Deprecated no-op kept for backwards compatibility.

        The Phase A ``camera_service.VisionService`` exposed a YOLO
        confidence-calibration routine that the AutomationService and
        voice commands could invoke. The Phase B pipeline uses fixed
        YOLOv11 + BoT-SORT thresholds plus InsightFace's own detector
        scores, so runtime calibration is no longer meaningful.

        This stub exists solely so that legacy callers
        (``automation.run_calibration()``, voice "calibrate" command)
        do not raise ``AttributeError`` and crash the system.
        """
        logger.warning(
            "Calibration is deprecated in Phase B Auto-Tracking — no-op."
        )

    # ── Grabber loop ─────────────────────────────────────────────────────

    def _run_grabber_loop(self) -> None:
        """Drain RTSP into the latest-frame slot. NO inference here.

        This thread exists purely to keep the OS socket buffer empty.
        If we instead ran YOLO inline (the old design), CPU inference
        time would back-pressure into the RTSP buffer and produce
        visibly stuttery / laggy video — exactly the "video lags when
        person detected" symptom.

        Auto-recovery: any read failure releases the capture and
        reconnects after :data:`RECONNECT_BACKOFF_SECONDS`.
        """
        cap: Optional[cv2.VideoCapture] = None
        while not self._stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = self._open_capture()
                    if cap is None or not cap.isOpened():
                        logger.warning(
                            "RTSP open failed — retrying in %.1fs",
                            RECONNECT_BACKOFF_SECONDS,
                        )
                        self._sleep(RECONNECT_BACKOFF_SECONDS)
                        continue
                    logger.info("RTSP stream connected: %s", self._rtsp_url)

                ok, frame = cap.read()
                if not ok or frame is None:
                    logger.warning("RTSP read failed — reconnecting.")
                    cap.release()
                    cap = None
                    self._sleep(RECONNECT_BACKOFF_SECONDS)
                    continue

                # Publish the freshest frame, replacing whatever was there.
                # NOTE: we deliberately do NOT push to the stream queue
                # here. The inference loop pushes the *annotated* frame
                # (with bounding boxes) so the user can see which
                # detections the system is acting on. That caps the
                # MJPEG rate at YOLO's inference rate (~6 FPS on CPU);
                # the user accepted that trade-off for the detection
                # overlay.
                with self._frame_lock:
                    self._latest_frame = frame
                self._frame_ready.set()

            except Exception as exc:
                logger.exception("Grabber loop error: %s", exc)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                self._sleep(RECONNECT_BACKOFF_SECONDS)

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    # ── Inference loop ───────────────────────────────────────────────────

    def _run_inference_loop(self) -> None:
        """Run YOLO + BoT-SORT on the latest grabbed frame, emit annotated stream + presence."""
        while not self._stop_event.is_set():
            # Wait until the grabber has produced at least one frame, then
            # read the *latest* one (we deliberately skip stale frames).
            if not self._frame_ready.wait(timeout=1.0):
                continue
            with self._frame_lock:
                frame = self._latest_frame
                # Clear the ready flag so we wait again if the grabber stalls.
                self._frame_ready.clear()
            if frame is None:
                continue

            try:
                annotated, active_track_ids = self._process_frame(frame)
            except Exception as exc:
                logger.exception("Inference loop error: %s", exc)
                continue

            # Push the annotated frame (with green bounding boxes around
            # detected persons) to the MJPEG stream. This caps the
            # browser-visible rate at YOLO's inference rate (~6 FPS on
            # CPU), but in return the user can SEE which people the
            # system is currently tracking — a UX requirement that
            # outweighs the smoothness gain from streaming raw frames.
            self._enqueue_stream_frame(annotated)
            self._maybe_publish_presence(active_track_ids)

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open the RTSP stream with a small read buffer for low latency."""
        try:
            cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception as exc:
            logger.error("VideoCapture init failed: %s", exc)
            return None

    # ── Per-frame processing ─────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """Run YOLO + BoT-SORT, gate by track id, dispatch unknown faces.

        Returns:
            (annotated_stream_frame, active_track_ids)

        ``active_track_ids`` is the list of int track ids currently
        visible; the inference loop uses it to publish ``presence_update``
        events so the dashboard's people_now stays honest even when
        every visible person is already "known" and silent.
        """
        if self._yolo is None:
            return self._resize_for_stream(frame), []

        try:
            results = self._yolo.track(
                frame,
                classes=[0],
                conf=self._detect_conf,
                persist=True,
                tracker=self._tracker_config,
                verbose=False,
                device=self._yolo_device,
            )
        except Exception as exc:
            logger.debug("YOLO.track failed: %s", exc)
            return self._resize_for_stream(frame), []

        display = frame.copy()
        active_ids: list = []

        if not results:
            return self._resize_for_stream(display), active_ids

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.id is None:
            return self._resize_for_stream(display), active_ids

        try:
            xyxy_all = boxes.xyxy.cpu().numpy()
            ids_all = boxes.id.cpu().numpy().astype(int)
        except Exception:
            return self._resize_for_stream(display), active_ids

        fh, fw = display.shape[:2]

        # ── Pass 1: collect every plausible track in this frame ─────────
        # We need ALL valid track_ids up front so each face-job can be
        # told who its CO-VISIBLE siblings are. The dispatch loop below
        # is the second pass; this two-pass split is the only reason
        # we don't dispatch immediately inside the YOLO iteration.
        plausible: list = []   # list of (track_id, bbox, is_in_zone)
        for (x1, y1, x2, y2), tid in zip(xyxy_all, ids_all):
            track_id = int(tid)
            bbox = (float(x1), float(y1), float(x2), float(y2))

            bw = x2 - x1
            bh = y2 - y1
            if (bh < MIN_PERSON_HEIGHT_PX
                or bw <= 0
                or (bh / bw) < MIN_PERSON_ASPECT
                or (bw * bh) < MIN_PERSON_AREA_PX):
                # YOLO flagged a poster / table / reflection — skip
                # for ALL purposes including the presence broadcast.
                continue

            feet_x = (x1 + x2) * 0.5
            feet_y = y2
            is_in_zone = self._point_in_zone(feet_x, feet_y, fw, fh)
            plausible.append((track_id, bbox, is_in_zone))

        # Set of ALL track_ids currently visible in this frame.
        current_tids = {t for (t, _b, _z) in plausible}

        # ── Identity bookkeeping for this frame ───────────────────────────
        # One snapshot of the binding map drives THREE downstream things:
        #   (1) the co-occurrence pool below,
        #   (2) the per-frame ``touch_last_seen`` update (Issue #5),
        #   (3) the buffer-feed dispatch decision inside Gate 1 (Issue #4).
        # Taking the snapshot once under the cache's lock keeps all three
        # consistent with each other even if the worker thread binds a
        # new track mid-frame.
        track_to_visitor = self._face_cache.track_to_visitor_snapshot()
        visible_visitor_ids: set = {
            track_to_visitor[t] for t in current_tids if t in track_to_visitor
        }
        # Touch last-seen for every visitor with a bound track in view.
        # Drives the absent-duration guard so a stationary visitor's
        # last_seen_wall never goes stale, preventing the cooldown from
        # firing a "refresh" snapshot on a person who never left.
        self._face_cache.touch_last_seen(visible_visitor_ids)

        # ── Recent-frames co-occurrence pool ──────────────────────────────
        # A track contributes to another track's co_visible exclusion
        # ONLY IF it qualifies as "trustworthy" by either:
        #
        #   • stability — emitted in BOTH the current frame AND the
        #     previous frame (BoT-SORT ghost tracks that appear for
        #     just one frame fail this test), OR
        #   • already-bound — already resolved to a visitor_id by an
        #     earlier worker call. Once we know who a track belongs to,
        #     it's authoritative even if its stability count is fresh.
        #     This safeguards the "two strangers arrive same frame"
        #     guarantee: the moment track A is bound to visitor 0,
        #     track B's search excludes visitor 0 even on frame 1.
        stable_tids = current_tids & self._prev_frame_track_ids
        bound_tids  = set(track_to_visitor.keys()) & current_tids
        co_visible_pool = stable_tids | bound_tids

        # ── Sibling-IoU map (BoT-SORT track-split fix) ────────────────────
        # Per-track bbox of every BOUND track in this frame, used by
        # the Pass 2 dispatch to detect "the new unknown track is
        # spatially the same blob as a bound track." When that
        # happens, the bound track's visitor_id is removed from the
        # new track's co-occurrence exclusion set — defeating the
        # false-split where BoT-SORT spawns a fresh id for a person
        # whose previous id hasn't dropped yet, and the rule treats
        # them as different people.
        bound_bbox_by_tid: dict = {
            t: b for (t, b, _z) in plausible if t in track_to_visitor
        }

        # ── Pass 2: draw boxes, gate, dispatch ───────────────────────────
        for track_id, bbox, is_in_zone in plausible:
            active_ids.append(track_id)
            self._draw_box(display, bbox, track_id)

            # GATE 1 — already resolved today, skip identity work.
            # Buffer-feed exception (Issue #4): if this track's
            # visitor has an OPEN snapshot buffer, dispatch a
            # lightweight feed job so the buffer can compare this
            # frame's face_quality. Otherwise this skip would starve
            # the buffer of candidates — Gate 1 normally short-
            # circuits known tracks before they ever reach the worker.
            if self._face_cache.is_track_known(track_id):
                vid = track_to_visitor.get(track_id)
                if vid is not None and self._snapshot_buffer.is_open(vid):
                    try:
                        self._face_queue.put_nowait(
                            _FaceJob(
                                frame=frame.copy(),
                                track_id=track_id,
                                bbox=bbox,
                                buffer_feed_for=vid,
                            )
                        )
                    except queue.Full:
                        # Buffer already has at least the opener frame;
                        # missing one feed candidate is harmless.
                        pass
                continue

            # GATE 1b — already enqueued and being evaluated.
            with self._in_flight_lock:
                if track_id in self._in_flight_track_ids:
                    continue
                self._in_flight_track_ids.add(track_id)

            # ── Sibling-IoU exception (BoT-SORT track-split fix) ────────
            # If this new unknown track sits at high spatial overlap
            # with any BOUND track in the same frame, treat them as
            # the SAME physical person (BoT-SORT split). Remove those
            # bound tracks from THIS track's co-visible set so the
            # worker is allowed to re-match the same visitor through
            # the normal body/face rules.
            sibling_tids: set = set()
            for bt, bb in bound_bbox_by_tid.items():
                if bt == track_id:
                    continue
                if self._bbox_iou(bbox, bb) >= SIBLING_IOU_THRESHOLD:
                    sibling_tids.add(bt)
            if sibling_tids:
                sibling_vids = {
                    track_to_visitor[t] for t in sibling_tids
                    if t in track_to_visitor
                }
                logger.info(
                    "Sibling-IoU exception — new track_id=%d overlaps bound "
                    "tracks %s (visitors %s) — exclusion suspended for this dispatch",
                    track_id, sorted(sibling_tids), sorted(sibling_vids),
                )

            # Co-occurrence exclusivity: every OTHER trustworthy track
            # in the pool is forbidden from sharing this job's visitor_id,
            # EXCEPT siblings (BoT-SORT splits of the same person).
            # Cast to tuple so the _FaceJob is hashable / picklable.
            co_visible = tuple(
                t for t in co_visible_pool
                if t != track_id and t not in sibling_tids
            )

            try:
                self._face_queue.put_nowait(
                    _FaceJob(
                        frame=frame.copy(),
                        track_id=track_id,
                        bbox=bbox,
                        is_in_zone=is_in_zone,
                        co_visible_track_ids=co_visible,
                    )
                )
            except queue.Full:
                # Roll the in-flight marker back so the NEXT frame can
                # try again — otherwise this track would never recover.
                with self._in_flight_lock:
                    self._in_flight_track_ids.discard(track_id)
                logger.debug(
                    "Face queue full — dropping track_id=%d (will retry next frame)",
                    track_id,
                )

        # ── SnapshotBuffer tick (Issue #4) ─────────────────────────────
        # Two commit paths fire from the inference thread at ~6 Hz:
        #
        #   • commit_for_orphaned_visitors — anyone with an OPEN buffer
        #     whose visitor has no alive bound track in this frame.
        #     "Track lost" semantics from the spec: commit early
        #     rather than wait the full window on someone who left.
        #     Conservative: a BoT-SORT track reset that spawns a new
        #     id for the SAME visitor keeps the buffer open across
        #     the transition (the new track binds → visitor stays in
        #     ``visible_visitor_ids`` → not orphaned).
        #
        #   • commit_expired — anyone whose 2.5 s window elapsed,
        #     regardless of visibility. The hard deadline guarantees
        #     a snapshot eventually lands even for very brief visits.
        self._snapshot_buffer.commit_for_orphaned_visitors(visible_visitor_ids)
        self._snapshot_buffer.commit_expired()

        # Roll the previous-frame set forward so the next call's
        # stability check has accurate data. The set tracks ALL
        # plausibility-passing track ids from this frame, not just
        # the ones we dispatched.
        self._prev_frame_track_ids = current_tids
        return self._resize_for_stream(display), active_ids

    def _maybe_publish_presence(self, active_track_ids: list) -> None:
        """Throttled MQTT broadcast of currently-visible track ids.

        Bridges this to the dashboard so people_now reflects reality
        instead of decaying to 0 while the same person is still in frame
        (which is what the legacy decay-only design did wrong).
        """
        now = time.monotonic()
        if (now - self._last_presence_publish) < PRESENCE_PUBLISH_INTERVAL_SECONDS:
            return
        self._last_presence_publish = now
        self._publish_event({
            "type": "presence_update",
            "active_tracks": [int(t) for t in active_track_ids],
            "people_now": len(active_track_ids),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    # ── Face worker (heavy thread) ───────────────────────────────────────

    def _run_face_worker(self) -> None:
        """Consume _FaceJob items and execute Gates 2-4.

        Failures in InsightFace / FAISS are logged but never propagate —
        the worker self-heals to the next job.

        Concurrency invariant: every job pulled from the queue here was
        added to ``self._in_flight_track_ids`` by the inference thread
        before being put on the queue. We MUST discard it from that set
        before returning, no matter how this job finished — success,
        Gate 2/3/4 fall-through, or an exception. The ``finally`` block
        below is the only place that responsibility lives.
        """
        while not self._stop_event.is_set():
            try:
                job = self._face_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:  # shutdown sentinel — no in-flight membership
                break

            try:
                self._handle_face_job(job)
            except Exception as exc:
                logger.exception("Face worker error: %s", exc)
            finally:
                # ALWAYS release the in-flight slot. If Gate 4 matched
                # or registered, the track-id is now in
                # ``known_bot_sort_ids`` so Gate 1 in the camera thread
                # blocks future re-enqueues. If Gate 2 (blur) rejected
                # it, releasing here lets the next frame retry with a
                # sharper crop. If something raised, we don't want to
                # leak slots.
                with self._in_flight_lock:
                    self._in_flight_track_ids.discard(int(job.track_id))

    def _handle_face_job(self, job: _FaceJob) -> None:
        """Run the hybrid identity pipeline for one (track_id, bbox) pair.

        Pipeline (Phase C Sprint 1 — buffered snapshot, split events):

            Buffer-feed shortcut (Issue #4): if ``buffer_feed_for`` is
                set, this is a lightweight candidate for an already-open
                SnapshotBuffer. Score the face_quality and submit; no
                identity work, no MQTT publish, no binding mutation.

            Gate 1  : Track-id already in known set?  → bail (camera thread)
            Gate 1b : Re-check under the worker       → bail
            Gate 2  : Compute HSV body histogram      → always available
            Gate 3  : Try to extract a sharp face     → optional
            Gate 4  : Search identity store with BOTH signals
                       - body match wins immediately (clothing color)
                       - else face match wins
                       - else: brand-new visitor

            Decision (refresh path): the 10-min cooldown is necessary
                but no longer sufficient. The visitor must ALSO have
                been absent (no bound track in view) for
                ABSENT_DURATION_REQUIRED_SECONDS — the Issue #5
                "lurker spam fix".

            Emit (first sighting): publish ``visitor_announce`` INSTANTLY
                (fires voice + visitors_today counter via the bridge)
                and OPEN a SnapshotBuffer. The real ``visitor_snapshot``
                event fires later when the buffer commits the sharpest
                frame.

            Emit (refresh): no announce (no greet, no counter bump);
                open the SnapshotBuffer directly.
        """
        # ── Buffer-feed shortcut (Issue #4) ──────────────────────────
        # Lightweight job for known tracks whose visitor has an open
        # snapshot buffer. Compute face_quality only — no identity
        # search, no binding, no MQTT.
        if job.buffer_feed_for is not None:
            vid = int(job.buffer_feed_for)
            if not self._snapshot_buffer.is_open(vid):
                return
            face_quality = 0.0
            detected = self._face_engine.get_face(job.frame, job.bbox)
            if detected is not None:
                face_quality = self._face_engine.check_quality(detected.crop)
            self._snapshot_buffer.submit(vid, job.frame, face_quality)
            return

        # ── Gate 1b: re-check under the worker ───────────────────────
        if self._face_cache.is_track_known(job.track_id):
            return

        # ── Gate 2: body embedding (OSNet, always available when ready) ──
        # If BodyEngine failed to initialize, compute_body_embedding
        # returns None and we proceed face-only (search() naturally
        # skips C2/C3 when body_emb is None). If BodyEngine IS ready
        # but the crop is degenerate (zero-area, sub-resolution torso),
        # we still bail — same contract as the old HSV path.
        body_emb = self._body_engine.compute_body_embedding(job.frame, job.bbox)
        if body_emb is None and self._body_engine.is_ready:
            return  # zero-area crop, can't proceed

        # ── Gate 3: face (optional, may be None) ─────────────────────
        face_emb: Optional[np.ndarray] = None
        face_quality: float = 0.0
        detected: Optional[DetectedFace] = self._face_engine.get_face(
            job.frame, job.bbox
        )
        if detected is not None:
            face_quality = self._face_engine.check_quality(detected.crop)
            if face_quality >= self._blur_threshold:
                face_emb = self._face_engine.get_embedding(detected)
                if face_emb is None or face_emb.size == 0:
                    face_emb = None

        # ── Gate 4: identity search (hybrid, with co-occurrence guard) ──
        # Any visitor_id currently bound to ANOTHER active track in the
        # same frame must NOT be a candidate for this face — that's
        # the rule that mathematically prevents simultaneously-visible
        # strangers from being merged into the same visitor.
        excluded_vids = self._face_cache.visitor_ids_for_tracks(
            job.co_visible_track_ids,
        )
        match = self._face_cache.search(
            face_embedding=face_emb,
            body_emb=body_emb,
            excluded_visitor_ids=excluded_vids,
        )

        # Resolve the two policy flags up front so the rest of this
        # method is just "given the decisions, do the work".
        #
        #   is_first_sighting_today
        #       True  → this visitor is not yet in today's cache. Will
        #               be added below if we have a face. Greet fires
        #               (subject to zone). Snapshot is unconditional.
        #       False → known visitor on a new BoT-SORT track id. No
        #               greet ever. Snapshot only if cooldown elapsed
        #               AND the absent-duration guard (Issue #5) is
        #               satisfied.
        is_first_sighting = not match.matched

        if not is_first_sighting:
            visitor_id = int(match.index)
            # Read both timers BEFORE binding this new track. The
            # binding causes the next inference frame to call
            # ``touch_last_seen`` for this visitor — washing out the
            # absence window we need to measure here.
            time_since_snap = self._face_cache.time_since_snapshot(visitor_id)
            time_since_seen = self._face_cache.time_since_last_seen(visitor_id)
            self._face_cache.mark_track_known(job.track_id, visitor_id)

            cooldown_elapsed = time_since_snap >= SNAPSHOT_COOLDOWN_SECONDS
            absent_long_enough = time_since_seen >= ABSENT_DURATION_REQUIRED_SECONDS
            should_snapshot = cooldown_elapsed and absent_long_enough
            should_greet = False

            if not should_snapshot:
                reason = (
                    "within-cooldown" if not cooldown_elapsed
                    else "lurker-still-present"
                )
                logger.info(
                    "Re-identified via %s on new track_id=%d (sim=%.3f → "
                    "visitor #%d) — silent (%s: cooldown=%.0fs absent=%.0fs)",
                    match.kind, job.track_id, match.score, visitor_id,
                    reason, time_since_snap, time_since_seen,
                )
                return
            # Both gates passed → eligible for a refresh snapshot. Stamp
            # the cooldown NOW so concurrent worker jobs for the same
            # visitor don't all open buffers; only the first one through
            # will see the cooldown reset.
            self._face_cache.mark_snapshot(visitor_id)
        else:
            # No match. We still REQUIRE a face to confirm "brand new
            # visitor" — body color alone is too ambiguous (see the
            # prior duplicate-greet bug). Defer the track if face is
            # missing; the camera will retry on the next frame.
            if face_emb is None:
                logger.debug(
                    "Faceless candidate, deferring registration — "
                    "track_id=%d body_score=%.3f",
                    job.track_id, match.score,
                )
                return
            visitor_id = self._face_cache.add_visitor(
                face_embedding=face_emb, body_emb=body_emb,
            )
            self._face_cache.mark_track_known(job.track_id, visitor_id)
            should_snapshot = True
            should_greet = bool(job.is_in_zone)
            time_since_snap = 0.0  # just registered → freshly stamped

        # If we got here, ``should_snapshot`` is True. Two emissions:
        #
        # 1. visitor_announce (INSTANT, first-sighting only) — drives
        #    the voice greet and the visitors_today counter. Includes
        #    a JPEG of the current frame so the AI service has scene
        #    context for its captioning; encoding cost is paid here
        #    on the worker thread, not on the camera thread.
        #
        # 2. visitor_snapshot (DELAYED via SnapshotBuffer) — drives
        #    snapshots_taken, disk write, DB row, and the dashboard
        #    thumbnail. Fires from the on_commit callback when the
        #    buffer picks the sharpest frame. NEVER fires voice
        #    (announce already did).
        ist_ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")

        if is_first_sighting:
            self._publish_event({
                "type": "visitor_announce",
                "visitor_id": int(visitor_id),
                "track_id": int(job.track_id),
                "is_first_sighting": True,
                "should_greet": should_greet,
                "in_zone": bool(job.is_in_zone),
                "has_face": face_emb is not None,
                "face_quality": round(face_quality, 2),
                "image_base64": self._encode_snapshot_base64(job.frame),
                "timestamp": ist_ts,
            })

        payload_template = {
            "type": "visitor_snapshot",
            "visitor_id": int(visitor_id),
            "track_id": int(job.track_id),
            "is_first_sighting": is_first_sighting,
            # Voice has already fired via visitor_announce (or this is a
            # refresh, which never greets). The bridge ignores this
            # field on visitor_snapshot, but keep it explicit for any
            # downstream consumer that reads it.
            "should_greet": False,
            "in_zone": bool(job.is_in_zone),
            "timestamp": ist_ts,
        }

        self._snapshot_buffer.open(
            visitor_id=visitor_id,
            track_id=job.track_id,
            frame=job.frame,
            face_quality=face_quality,
            payload_template=payload_template,
        )

        if is_first_sighting:
            # Diagnostic: top per-visitor (body, face) tuples for the
            # near-miss candidates. Lets us see whether the match
            # missed because no SINGLE visitor cleared both rule bars
            # at the same time (versus best_body / best_face above
            # which are score-maxes across DIFFERENT visitors and
            # therefore can be misleading when tuning thresholds).
            cand_str = ", ".join(
                f"#{vid}(b={b:.3f},f={f:.3f})"
                for vid, b, f in match.top_candidates
            ) or "<none>"
            logger.info(
                "NEW visitor #%d — track_id=%d has_face=%s face_quality=%.1f "
                "in_zone=%s should_greet=%s (best_body=%.3f best_face=%.3f via=%s) "
                "→ announce fired, snapshot buffered for %.1fs | "
                "top candidates: %s",
                visitor_id, job.track_id, face_emb is not None, face_quality,
                "YES" if job.is_in_zone else "no",
                "YES" if should_greet else "no",
                match.body_score, match.face_score, match.kind,
                SNAPSHOT_BUFFER_WINDOW_SECONDS,
                cand_str,
            )
        else:
            logger.info(
                "Refresh snapshot buffered for visitor #%d — track_id=%d "
                "(via=%s sim=%.3f, %.0fs since last snapshot)",
                visitor_id, job.track_id, match.kind, match.score, time_since_snap,
            )

    # ── Snapshot-buffer commit callback (Issue #4) ───────────────────────

    def _commit_buffered_snapshot(
        self,
        payload_template: dict,
        best_frame: np.ndarray,
        best_quality: float,
    ) -> None:
        """SnapshotBuffer.on_commit — JPEG-encode the best frame and publish.

        Called from the inference thread (via ``commit_expired`` or
        ``commit_for_orphaned_visitors``) and from any thread that
        triggers ``drain`` (factory_reset, business-day rollover).
        JPEG encoding happens here, not at decision time, so the
        encoding cost is paid exactly once for the chosen frame
        rather than on every candidate that the buffer rejected.

        Failures are logged and swallowed — the visitor_announce
        event has already fired, so there's nothing to roll back; we
        just lose the gallery photo for that one event, which the
        bridge handles gracefully (no file written, no DB row).
        """
        try:
            image_b64 = self._encode_snapshot_base64(best_frame)
            payload = dict(payload_template)
            payload["image_base64"] = image_b64
            payload["has_face"] = best_quality > 0.0
            payload["face_quality"] = round(float(best_quality), 2)
            self._publish_event(payload)
            logger.info(
                "Snapshot committed — visitor #%s track=%s first=%s "
                "best_face_quality=%.1f",
                payload.get("visitor_id"), payload.get("track_id"),
                payload.get("is_first_sighting"), best_quality,
            )
        except Exception:
            logger.exception(
                "_commit_buffered_snapshot failed for visitor=%s",
                payload_template.get("visitor_id"),
            )

    # ── Streaming helpers ────────────────────────────────────────────────

    def _resize_for_stream(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (STREAM_FRAME_WIDTH, STREAM_FRAME_HEIGHT))

    def _enqueue_stream_frame(self, frame: np.ndarray) -> None:
        """JPEG-encode and push to the stream queue, dropping oldest on full.

        Drop-oldest semantics matter: a slow MJPEG consumer must never
        cause back-pressure on the camera loop. ``queue.Full`` triggers
        a non-blocking pop of the stale frame and a retry.
        """
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
            )
        except Exception:
            return
        if not ok:
            return
        jpeg_bytes = buf.tobytes()

        try:
            self._stream_queue.put_nowait(jpeg_bytes)
        except queue.Full:
            try:
                self._stream_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._stream_queue.put_nowait(jpeg_bytes)
            except queue.Full:
                pass

    @staticmethod
    def _bbox_iou(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> float:
        """Intersection-over-Union of two axis-aligned bboxes (xyxy).

        Returns 0.0 for non-overlapping pairs or degenerate (zero-area)
        inputs. Used by Pass 2 to detect BoT-SORT track splits: two
        boxes at IoU ≥ SIBLING_IOU_THRESHOLD are treated as the same
        physical person and the co-occurrence exclusion is suspended
        for that pair.
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = a_area + b_area - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _draw_box(
        canvas: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
    ) -> None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (34, 197, 94), 2)
        label = f"ID {track_id}"
        cv2.putText(
            canvas, label, (x1 + 4, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 2, cv2.LINE_AA,
        )

    @staticmethod
    def _encode_face_base64(face_crop: np.ndarray, quality: int = 85) -> str:
        """JPEG-encode a face crop and return a base64 ASCII string."""
        ok, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _encode_snapshot_base64(
        frame: np.ndarray,
        max_width: int = 1280,
        quality: int = 80,
    ) -> str:
        """JPEG-encode a FULL frame (the whole scene) for the snapshot file.

        The bridge writes this base64 directly to ``snapshots/visitor_*.jpg``
        — the gallery page expects scene-level context, not a face crop.

        We cap the long edge at ``max_width`` so 1080p+ camera streams
        produce a reasonable MQTT payload (typically 80–180 KB at q=80)
        without sacrificing recognizable detail.
        """
        if frame is None or frame.size == 0:
            return ""
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_size = (max_width, int(h * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ── MQTT ─────────────────────────────────────────────────────────────

    def _connect_mqtt(self) -> None:
        """Connect to the broker in non-blocking mode.

        Uses ``loop_start`` so reconnection is handled by paho's internal
        thread — we never block the camera loop on MQTT I/O.
        """
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.error("paho-mqtt not installed — events will be dropped.")
            self._mqtt_client = None
            return

        def _on_connect(client, _ud, _flags, rc, *_args):
            self._mqtt_connected = (rc == 0)
            if rc == 0:
                logger.info(
                    "MQTT connected to %s:%d", self._mqtt_host, self._mqtt_port
                )
            else:
                logger.warning("MQTT connect failed rc=%s", rc)

        def _on_disconnect(client, _ud, *_args):
            self._mqtt_connected = False
            logger.warning("MQTT disconnected")

        try:
            client = mqtt.Client(client_id=self._mqtt_client_id)
        except TypeError:
            # paho-mqtt 2.x requires CallbackAPIVersion
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION1,
                client_id=self._mqtt_client_id,
            )
        client.on_connect = _on_connect
        client.on_disconnect = _on_disconnect

        try:
            client.connect_async(self._mqtt_host, self._mqtt_port, keepalive=60)
            client.loop_start()
            self._mqtt_client = client
        except Exception as exc:
            logger.error("MQTT connect_async failed: %s", exc)
            self._mqtt_client = None

    def _publish_event(self, payload: dict) -> None:
        """Publish ``payload`` to :data:`MQTT_TOPIC_EVENTS`.

        If MQTT is unavailable, the event is logged at WARNING and
        dropped — no retry queue. Vision events are time-sensitive;
        replaying a stale greeting an hour later would be wrong.
        """
        if self._mqtt_client is None:
            logger.warning("MQTT not initialized — event dropped (type=%s)",
                           payload.get("type"))
            return
        try:
            body = json.dumps(payload, separators=(",", ":"))
            self._mqtt_client.publish(MQTT_TOPIC_EVENTS, body, qos=0, retain=False)
        except Exception as exc:
            logger.warning("MQTT publish failed: %s", exc)

    # ── YOLO loader ──────────────────────────────────────────────────────

    def _load_yolo(self) -> None:
        """Load the YOLOv11 detector. Falls back to CPU if no CUDA."""
        try:
            import torch
            from ultralytics import YOLO

            self._yolo_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._yolo = YOLO(self._yolo_model_path)
            self._yolo.to(self._yolo_device)
            logger.info(
                "YOLO loaded — model=%s device=%s",
                self._yolo_model_path, self._yolo_device,
            )
        except Exception as exc:
            logger.exception("YOLO load failed: %s", exc)
            self._yolo = None

    # ── Business-day rollover (09:00 IST) ────────────────────────────────

    def _run_business_day_reset(self) -> None:
        """Reset the daily identity cache at 09:00:00 IST.

        Anyone seen at 08:59 IST is part of yesterday's business day;
        anyone seen at 09:01 IST is part of today's. Sleeps in
        1-second slices so ``stop()`` can interrupt without waiting
        hours, then resets the cache and fires the on-midnight
        callback (which despite its legacy name now triggers at 9 AM).
        """
        while not self._stop_event.is_set():
            seconds_to_rollover = seconds_until_next_business_day_start()
            slept = 0.0
            while slept < seconds_to_rollover and not self._stop_event.is_set():
                step = min(1.0, seconds_to_rollover - slept)
                if self._stop_event.wait(timeout=step):
                    return
                slept += step
            if self._stop_event.is_set():
                return
            # Drain pending snapshot buffers first so we don't commit
            # photos for visitor_ids that the reset is about to delete.
            self._snapshot_buffer.drain()
            self._face_cache.reset()
            logger.info(
                "Business-day rollover (09:00 IST) — daily identity cache cleared.",
            )
            # Notify the rest of the system (StateManager + DB prune)
            # that the day has rolled over. The constructor arg name
            # is still ``on_midnight_reset`` for backward compat — the
            # callback contract is unchanged, only the trigger time.
            if self._on_midnight_reset is not None:
                try:
                    self._on_midnight_reset()
                except Exception:
                    logger.exception("on_midnight_reset callback raised")

    # ── Thread / sleep helpers ───────────────────────────────────────────

    def _spawn(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep that wakes immediately on ``stop()``."""
        self._stop_event.wait(timeout=seconds)


# ── Stand-alone runner ──────────────────────────────────────────────────────

def _main() -> None:
    """Run the service in isolation. Useful for smoke-testing without main.py.

    Reads RTSP_URL / MQTT_BROKER_HOST / MQTT_BROKER_PORT from env.
    """
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
    )

    rtsp_url = os.environ.get("RTSP_URL", "")
    if not rtsp_url:
        raise SystemExit("RTSP_URL env var is required.")

    service = VisionService(
        rtsp_url=rtsp_url,
        mqtt_host=os.environ.get("MQTT_BROKER_HOST", "localhost"),
        mqtt_port=int(os.environ.get("MQTT_BROKER_PORT", "1883")),
        yolo_model_path=os.environ.get("YOLO_MODEL", "yolo11n.pt"),
    )

    def _shutdown(signum, _frame):
        logger.info("Signal %s received — shutting down.", signum)
        service.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    service.start()
    # Block main thread until shutdown.
    while not service._stop_event.is_set():
        time.sleep(1.0)


if __name__ == "__main__":
    _main()
