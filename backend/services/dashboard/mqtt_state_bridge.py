"""MqttStateBridge — translate MQTT `vision/events` into legacy app state.

Phase B isolated the VisionService behind an MQTT topic. The rest of the
codebase (Flask polling API, AIService, VoiceService) was built for
Phase A and still listens on:

    - StateManager keys: ``visitors_today``, ``people_now``,
      ``last_visitor_b64``, ``last_alert_time``, ``snapshots_taken``
    - EventBus event: :data:`EV_VISITOR_DETECTED` with kwargs
      ``frame``, ``visitor_num``, ``mode``

This bridge subscribes to ``vision/events``, parses incoming payloads,
mutates StateManager (which already serializes via its own lock), and
republishes on the internal EventBus so existing handlers fire without
modification.

Thread safety
-------------
The paho MQTT client dispatches ``on_message`` from its network thread.
All mutations from that thread go through:
    - ``StateManager.update()`` / ``.increment()`` — internally locked
    - ``EventBus.publish()`` — internally locked, fans out on daemon threads
    - ``self._lock`` — guards only the bridge-local ``_last_visitor_at``
      timestamp used by the presence-decay sweeper.

No shared mutable state is touched without a lock.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from backend.config.settings import SNAPSHOT_DIR
from backend.core.constants import (
    EV_VISITOR_DETECTED,
    MODE_KARAOKE,
    MODE_NORMAL,
    MODE_SECURITY,
)

logger = logging.getLogger(__name__)

DEFAULT_TOPIC: str = "vision/events"
PRESENCE_TIMEOUT_SECONDS: float = 30.0
SWEEPER_INTERVAL_SECONDS: float = 5.0

# How long after the last received event we still consider the camera
# "Active". After this window we downgrade to "Idle" (broker connected
# but no traffic). If MQTT itself is disconnected we report "Offline".
CAMERA_ACTIVE_TIMEOUT_SECONDS: float = 60.0


class MqttStateBridge:
    """Subscribe to vision MQTT events, mirror them to StateManager + EventBus.

    Lifecycle:
        bridge = MqttStateBridge(mqtt_client, state, bus, db=db)
        bridge.start()   # safe to call any time; resubscribes on reconnect
        ...
        bridge.stop()    # idempotent
    """

    def __init__(
        self,
        mqtt_client: Any,
        state: Any,
        bus: Any,
        db: Optional[Any] = None,
        topic: str = DEFAULT_TOPIC,
    ) -> None:
        self._mqtt = mqtt_client
        self._state = state
        self._bus = bus
        self._db = db
        self._topic = topic

        self._lock: threading.Lock = threading.Lock()
        self._last_visitor_at: float = 0.0  # monotonic clock
        self._last_event_at: float = 0.0    # any event, used for camera_status
        self._stop_event: threading.Event = threading.Event()
        self._sweeper: Optional[threading.Thread] = None
        self._prev_on_connect = None
        self._started: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Wire callbacks, subscribe, and launch the presence sweeper."""
        if self._started:
            return
        if self._mqtt is None:
            logger.error("MqttStateBridge: no MQTT client — bridge disabled.")
            return

        # Route messages on our topic to our handler. This is independent of
        # connection state — registration is local to the client object.
        self._mqtt.message_callback_add(self._topic, self._on_vision_event)

        # Chain into the existing on_connect so we (re)subscribe automatically
        # every time paho re-establishes the broker connection. We do NOT
        # overwrite — main.py's logger callback still fires.
        self._prev_on_connect = self._mqtt.on_connect
        self._mqtt.on_connect = self._chained_on_connect

        # If we're already connected (start() called after connect_async +
        # loop_start has had time to handshake), subscribe immediately.
        # Otherwise the chained on_connect will handle it on connect.
        try:
            self._mqtt.subscribe(self._topic, qos=0)
        except Exception:
            pass

        self._stop_event.clear()
        self._sweeper = threading.Thread(
            target=self._presence_sweeper,
            name="mqtt-bridge-sweeper",
            daemon=True,
        )
        self._sweeper.start()
        self._started = True
        logger.info("MqttStateBridge started — topic=%s", self._topic)

    def stop(self) -> None:
        """Stop the sweeper and unsubscribe. Safe to call multiple times."""
        if not self._started:
            return
        self._stop_event.set()
        try:
            self._mqtt.unsubscribe(self._topic)
            self._mqtt.message_callback_remove(self._topic)
        except Exception:
            pass
        if self._prev_on_connect is not None:
            try:
                self._mqtt.on_connect = self._prev_on_connect
            except Exception:
                pass
        if self._sweeper is not None:
            self._sweeper.join(timeout=2.0)
        self._started = False
        logger.info("MqttStateBridge stopped.")

    # ── MQTT callbacks ───────────────────────────────────────────────────────

    def _chained_on_connect(self, client, userdata, flags, rc, *args):
        """Run the previously-installed on_connect, then (re)subscribe."""
        if self._prev_on_connect is not None:
            try:
                self._prev_on_connect(client, userdata, flags, rc, *args)
            except Exception:
                logger.exception("Previous on_connect handler raised.")
        if rc == 0:
            try:
                client.subscribe(self._topic, qos=0)
                logger.info("MqttStateBridge subscribed: %s", self._topic)
            except Exception:
                logger.exception("Subscribe to %s failed.", self._topic)

    def _on_vision_event(self, _client, _userdata, msg) -> None:
        """Dispatch on payload ``type``. Never raises out of this method."""
        try:
            raw = msg.payload.decode("utf-8", errors="replace")
            payload = json.loads(raw)
        except Exception:
            logger.exception("Bad MQTT payload on %s", getattr(msg, "topic", "?"))
            return

        # ── Camera liveness signal ──────────────────────────────────
        # Any event from `vision/events` proves the VisionService is
        # running and the RTSP grabber is producing frames. Flip the
        # dashboard's camera_status away from the stale "Connecting…"
        # default the moment we see proof of life.
        with self._lock:
            self._last_event_at = time.monotonic()
        if self._state.get("camera_status") != "Active":
            self._state.update(camera_status="Active")

        event_type = payload.get("type")
        if event_type == "new_visitor_detected":
            try:
                self._handle_new_visitor(payload)
            except Exception:
                logger.exception("new_visitor_detected handler failed.")
        elif event_type == "presence_update":
            try:
                self._handle_presence_update(payload)
            except Exception:
                logger.exception("presence_update handler failed.")
        else:
            logger.debug("Ignoring unknown vision event type=%r", event_type)

    # ── presence_update — keeps people_now honest ────────────────────────

    def _handle_presence_update(self, payload: dict) -> None:
        """Mirror the live-track count from VisionService into state.

        VisionService emits this ~1Hz from its inference loop. It carries
        the *current* number of people in frame regardless of whether
        they're "new" or already known — which is exactly what
        ``people_now`` needs to display correctly. Counting only NEW
        visitors (the old behavior) made the counter decay to 0 even
        while the same person was still standing in front of the camera.
        """
        try:
            count = int(payload.get("people_now", 0))
        except Exception:
            count = 0

        current = int(self._state.get("people_now", 0) or 0)
        if count != current:
            self._state.update(people_now=count)

        # If people are visible, refresh the visitor timestamp so the
        # decay sweeper (now mostly a safety net) doesn't override us.
        if count > 0:
            with self._lock:
                self._last_visitor_at = time.monotonic()

    # ── Event handlers ───────────────────────────────────────────────────────

    def _handle_new_visitor(self, payload: dict) -> None:
        """Mirror a single visitor event to state, DB, disk, and EventBus."""
        track_id = payload.get("track_id")
        visitor_id = payload.get("visitor_id")
        image_b64: str = payload.get("image_base64", "") or ""

        # ── Decode ONCE: the payload carries the full scene frame ────────
        # We reuse it for: (a) saving the gallery snapshot file,
        # (b) generating a small thumbnail for /api/status,
        # (c) feeding the AI greeting handler on the EventBus.
        frame = self._decode_face_crop(image_b64)

        # ── Generate a small thumbnail so /api/status doesn't ship the
        #    full 200 KB scene on every dashboard poll. ──────────────────
        thumb_b64 = self._encode_thumbnail(frame) or image_b64

        # ── StateManager: counters + thumbnail + timestamps ──────────────
        vnum = self._state.increment("visitors_today")
        self._state.increment("snapshots_taken")
        self._state.update(
            people_now=max(int(self._state.get("people_now", 0) or 0), 1),
            last_alert_time=datetime.datetime.now().strftime("%H:%M:%S"),
            last_visitor_b64=thumb_b64,
        )
        with self._lock:
            self._last_visitor_at = time.monotonic()

        # ── Persist the FULL-FRAME snapshot file ─────────────────────────
        # (side-effects layer — VisionService stays disk-free per Phase B).
        snapshot_filename = self._save_snapshot_file(frame)
        if snapshot_filename and self._db is not None:
            try:
                self._db.save_snapshot(snapshot_filename, "Visitor")
            except Exception:
                logger.exception("DB save_snapshot failed for %s", snapshot_filename)

        # ── DB log (best-effort) ─────────────────────────────────────────
        if self._db is not None:
            try:
                self._db.log_event(
                    f"Visitor #{vnum} arrived (track={track_id})",
                    "visitor",
                    person_count=1,
                )
            except Exception:
                logger.exception("DB log_event failed for visitor #%s.", vnum)

        # ── Republish on the internal EventBus ───────────────────────────
        # AIService._on_visitor_detected(frame, visitor_num, mode) is
        # already subscribed and triggers the TTS greeting.
        mode = self._current_mode()
        try:
            self._bus.publish(
                EV_VISITOR_DETECTED,
                frame=frame,
                visitor_num=vnum,
                mode=mode,
            )
        except Exception:
            logger.exception("EventBus publish of EV_VISITOR_DETECTED failed.")

        logger.info(
            "Visitor #%d bridged — visitor_id=%s track=%s mode=%s scene=%d KB file=%s",
            vnum, visitor_id, track_id, mode,
            len(image_b64) // 1024, snapshot_filename or "<none>",
        )

    # ── Snapshot persistence ─────────────────────────────────────────────

    @staticmethod
    def _save_snapshot_file(frame: Optional[np.ndarray]) -> Optional[str]:
        """Write the face crop to SNAPSHOT_DIR and return its basename.

        Returns ``None`` if the frame is empty or the write fails. The
        filename pattern matches what the legacy code used so existing
        gallery/snapshot routes keep working unchanged:
            visitor_HHMMSS.jpg
        Duplicate hits within the same second get a millisecond suffix
        so files never overwrite each other.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            now = datetime.datetime.now()
            fname = now.strftime("visitor_%H%M%S.jpg")
            fpath = os.path.join(SNAPSHOT_DIR, fname)
            if os.path.exists(fpath):
                # Collisions in the same second — add a millisecond suffix.
                fname = now.strftime("visitor_%H%M%S_") + f"{now.microsecond // 1000:03d}.jpg"
                fpath = os.path.join(SNAPSHOT_DIR, fname)
            ok = cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return fname if ok else None
        except Exception:
            logger.exception("Snapshot file write failed.")
            return None

    # ── Presence decay ───────────────────────────────────────────────────────

    def _presence_sweeper(self) -> None:
        """Decay ``people_now`` and maintain ``camera_status``.

        Two responsibilities:

        1.  Clear ``people_now`` once :data:`PRESENCE_TIMEOUT_SECONDS`
            has elapsed since the last new-visitor event. (VisionService
            never emits a "they left" signal, so we infer absence from
            silence — BoT-SORT inside vision is the authoritative source.)

        2.  Keep ``camera_status`` honest:
              - "Active"  → an event arrived within
                            :data:`CAMERA_ACTIVE_TIMEOUT_SECONDS`
              - "Idle"    → broker connected but no recent events
                            (camera up, nobody in frame)
              - "Offline" → broker is disconnected
        """
        while not self._stop_event.wait(SWEEPER_INTERVAL_SECONDS):
            now = time.monotonic()
            with self._lock:
                last_visitor = self._last_visitor_at
                last_event = self._last_event_at

            # ── Presence decay ────────────────────────────────────────
            if last_visitor > 0.0 and (now - last_visitor) > PRESENCE_TIMEOUT_SECONDS:
                if int(self._state.get("people_now", 0) or 0) > 0:
                    self._state.update(people_now=0)
                with self._lock:
                    self._last_visitor_at = 0.0

            # ── camera_status heartbeat ───────────────────────────────
            status = self._derive_camera_status(now, last_event)
            if self._state.get("camera_status") != status:
                self._state.update(camera_status=status)

    def _derive_camera_status(self, now: float, last_event: float) -> str:
        """Compute the current camera_status string for the dashboard."""
        if not self._is_broker_connected():
            return "Offline"
        if last_event > 0.0 and (now - last_event) <= CAMERA_ACTIVE_TIMEOUT_SECONDS:
            return "Active"
        return "Idle"

    def _is_broker_connected(self) -> bool:
        """Best-effort check across paho-mqtt 1.x and 2.x API surfaces."""
        client = self._mqtt
        if client is None:
            return False
        try:
            return bool(client.is_connected())
        except Exception:
            return False

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_face_crop(image_b64: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG into a BGR ndarray; ``None`` on failure."""
        if not image_b64:
            return None
        try:
            raw = base64.b64decode(image_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _encode_thumbnail(
        frame: Optional[np.ndarray],
        width: int = 320,
        quality: int = 65,
    ) -> Optional[str]:
        """Produce a small base64 JPEG for the dashboard's inline preview.

        Why this exists: the full scene snapshot is ~100–200 KB. Putting
        that into ``state.last_visitor_b64`` would mean every
        ``/api/status`` poll ships hundreds of KB. The dashboard's
        thumbnail tile only needs ~320 px wide, so we shrink + re-encode
        once here and let the gallery page serve the full file separately.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        try:
            h, w = frame.shape[:2]
            scale = width / float(w) if w > width else 1.0
            if scale < 1.0:
                frame = cv2.resize(frame, (width, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:
            return None

    def _current_mode(self) -> str:
        if self._state.get("security_mode"):
            return MODE_SECURITY
        if self._state.get("karaoke_mode"):
            return MODE_KARAOKE
        return MODE_NORMAL
