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
from datetime import datetime, timedelta
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from backend.config.settings import ENTRY_ZONE
from backend.services.vision.face_engine import DetectedFace, FaceEngine
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


# ── Internal work items ─────────────────────────────────────────────────────

@dataclass
class _FaceJob:
    """One unit of work for the face worker thread."""
    frame: np.ndarray
    track_id: int
    bbox: Tuple[float, float, float, float]


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
        self._face_cache: DailyFaceCache = DailyFaceCache(match_threshold=match_threshold)

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
        self._spawn(self._run_midnight_reset, name="vision-midnight")
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
        self._face_cache.reset()
        with self._in_flight_lock:
            self._in_flight_track_ids.clear()
        logger.info("VisionService.factory_reset — daily identity cache wiped")

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
        for (x1, y1, x2, y2), tid in zip(xyxy_all, ids_all):
            track_id = int(tid)
            bbox = (float(x1), float(y1), float(x2), float(y2))

            # ── Plausibility filter — reject non-person YOLO bboxes ─────
            # Without this, YOLO will flag wall posters, table objects,
            # and reflections as "persons" and each gets registered as a
            # separate visitor with its own face embedding.
            bw = x2 - x1
            bh = y2 - y1
            if (bh < MIN_PERSON_HEIGHT_PX
                or bw <= 0
                or (bh / bw) < MIN_PERSON_ASPECT
                or (bw * bh) < MIN_PERSON_AREA_PX):
                # Don't include in active_ids either: the presence
                # broadcast should reflect real humans only.
                continue

            # ── Entry-zone filter ───────────────────────────────────────
            # Only consider bboxes whose centre falls inside the
            # configured ENTRY_ZONE (normalised x1, y1, x2, y2). This
            # silences wall posters / scenery / reflections that sit in
            # frame regions the receptionist would never see a real
            # visitor walk through.
            cx = (x1 + x2) * 0.5 / fw
            cy = (y1 + y2) * 0.5 / fh
            if not (ENTRY_ZONE[0] <= cx <= ENTRY_ZONE[2]
                    and ENTRY_ZONE[1] <= cy <= ENTRY_ZONE[3]):
                continue

            active_ids.append(track_id)
            self._draw_box(display, bbox, track_id)

            # GATE 1 — already greeted today, skip face work entirely.
            if self._face_cache.is_track_known(track_id):
                continue

            # GATE 1b — already enqueued and being evaluated by the
            # worker. Without this check we'd send 5–10 duplicate jobs
            # per second for the same brand-new person while the worker
            # is still chewing on the first one, filling the bounded
            # queue and starving anyone else who walked into the same
            # frame. Add-then-enqueue (not enqueue-then-add) closes the
            # race window where the worker could finish before the
            # camera thread marks the id in-flight.
            with self._in_flight_lock:
                if track_id in self._in_flight_track_ids:
                    continue
                self._in_flight_track_ids.add(track_id)

            try:
                self._face_queue.put_nowait(
                    _FaceJob(frame=frame.copy(), track_id=track_id, bbox=bbox)
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

        Pipeline (Phase B v2 — hybrid body + face):

            Gate 1  : Track-id already in known set?  → bail (camera thread)
            Gate 1b : Re-check under the worker       → bail
            Gate 2  : Compute HSV body histogram      → always available
            Gate 3  : Try to extract a sharp face     → optional
            Gate 4  : Search identity store with BOTH signals
                       - body match wins immediately (clothing color)
                       - else face match wins
                       - else: brand-new visitor
            Snapshot: ONE per visitor per day — full frame, not face crop

        Why both signals: CCTV faces are tiny / off-axis / occluded ~80%
        of the time. Face-only re-ID misses re-acquisitions across track
        resets, causing duplicate snapshots and duplicate greetings.
        Clothing color works on every frame and catches those cases.
        """
        # ── Gate 1b: re-check under the worker ───────────────────────
        if self._face_cache.is_track_known(job.track_id):
            return

        # ── Gate 2: body histogram (always available) ────────────────
        body_hist = self._face_engine.compute_body_histogram(job.frame, job.bbox)
        if body_hist is None:
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

        # ── Gate 4: identity search (hybrid) ─────────────────────────
        match = self._face_cache.search(face_embedding=face_emb, body_hist=body_hist)
        if match.matched:
            # Same person — BoT-SORT just gave them a new track id.
            # Mark the new id known and stay silent (no snapshot, no greet).
            # The cache itself appends the new body view internally on
            # every successful match, so we don't double-call here.
            self._face_cache.mark_track_known(job.track_id)
            logger.info(
                "Re-identified via %s on new track_id=%d (sim=%.3f → visitor #%d)",
                match.kind, job.track_id, match.score, match.index,
            )
            return

        # ── Defer if we have no face ─────────────────────────────────
        # A face is REQUIRED to register a new visitor. Without one we
        # cannot tell apart "new person walked in back-first" from "the
        # same person who just turned around". HSV body color is too
        # ambiguous on its own — that's the lesson from the prior
        # duplicate-greet bug.
        #
        # We do NOT mark the track id known: a subsequent frame may
        # catch a clear face for this same track, at which point we
        # can resolve identity definitively.
        if face_emb is None:
            logger.debug(
                "Faceless candidate, deferring registration — track_id=%d body_score=%.3f",
                job.track_id, match.score,
            )
            return

        # ── Decision: BRAND NEW VISITOR (face confirmed) ─────────────
        visitor_id = self._face_cache.add_visitor(
            face_embedding=face_emb, body_hist=body_hist
        )
        self._face_cache.mark_track_known(job.track_id)

        # Snapshot = the FULL frame (CCTV scene), not the face crop.
        # This matches Phase A's UX expectation in the gallery page.
        image_b64 = self._encode_snapshot_base64(job.frame)

        payload = {
            "type": "new_visitor_detected",
            "visitor_id": int(visitor_id),
            "track_id": int(job.track_id),
            "image_base64": image_b64,           # ← full frame JPEG, base64
            "has_face": face_emb is not None,
            "face_quality": round(face_quality, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self._publish_event(payload)
        logger.info(
            "NEW visitor #%d — track_id=%d has_face=%s face_quality=%.1f "
            "(best_body=%.3f best_face=%.3f via=%s)",
            visitor_id, job.track_id, face_emb is not None, face_quality,
            match.body_score, match.face_score, match.kind,
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

    # ── Midnight reset ───────────────────────────────────────────────────

    def _run_midnight_reset(self) -> None:
        """Reset the daily face cache once per local midnight.

        Computes seconds-until-midnight, sleeps in 1-second slices so
        ``stop()`` can interrupt without waiting hours, then resets.
        """
        while not self._stop_event.is_set():
            seconds_to_midnight = self._seconds_until_next_midnight()
            # Sleep in small slices so shutdown is responsive.
            slept = 0.0
            while slept < seconds_to_midnight and not self._stop_event.is_set():
                step = min(1.0, seconds_to_midnight - slept)
                if self._stop_event.wait(timeout=step):
                    return
                slept += step
            if self._stop_event.is_set():
                return
            self._face_cache.reset()
            logger.info("Midnight reset fired — daily face cache cleared.")
            # Notify the rest of the system (e.g. StateManager) that the
            # day has rolled over so persistent daily counters also reset.
            if self._on_midnight_reset is not None:
                try:
                    self._on_midnight_reset()
                except Exception:
                    logger.exception("on_midnight_reset callback raised")

    @staticmethod
    def _seconds_until_next_midnight() -> float:
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1.0, (tomorrow - now).total_seconds())

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
