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
import json
import logging
import os
import queue
import threading
import time
from typing import Any, List, Optional

import cv2
import numpy as np

from backend.config.settings import SNAPSHOT_DIR
from backend.core.business_day import now_ist
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

        # ── SSE subscriber registry ──────────────────────────────────────
        # Each connected dashboard tab adds a Queue here via subscribe();
        # every state mutation fans out a small dict to all queues.
        # Dropped on full per-subscriber queue rather than blocking — a
        # slow tab must never back up the paho network thread.
        self._subscribers: List["queue.Queue[dict]"] = []
        self._subscribers_lock: threading.Lock = threading.Lock()

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
            self._broadcast("camera_status", camera_status="Active")

        event_type = payload.get("type")
        # Phase C Sprint 1 split the visitor event in two:
        #
        #   visitor_announce — instant-greet event. Fires at the
        #     first-sighting decision moment, before the snapshot
        #     buffer resolves. Handler bumps visitors_today and
        #     fires the voice greet via EventBus.
        #
        #   visitor_snapshot — fired after the SnapshotBuffer picks
        #     the best frame. Handler does the disk write, the DB
        #     row, and the thumbnail update. Never fires voice (the
        #     announce already did).
        #
        # The legacy name ``new_visitor_detected`` is still accepted
        # so any in-flight MQTT message from an older producer at the
        # moment of upgrade doesn't get dropped.
        if event_type == "visitor_announce":
            try:
                self._handle_visitor_announce(payload)
            except Exception:
                logger.exception("visitor_announce handler failed.")
        elif event_type in ("visitor_snapshot", "new_visitor_detected"):
            try:
                self._handle_new_visitor(payload)
            except Exception:
                logger.exception("visitor_snapshot handler failed.")
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
            # Push only on actual change — no need to spam every tick
            # with identical people_now values.
            self._broadcast("presence", people_now=count)

        # If people are visible, refresh the visitor timestamp so the
        # decay sweeper (now mostly a safety net) doesn't override us.
        if count > 0:
            with self._lock:
                self._last_visitor_at = time.monotonic()

    # ── Event handlers ───────────────────────────────────────────────────────

    def _handle_visitor_announce(self, payload: dict) -> None:
        """Instant-greet event — fires before the SnapshotBuffer resolves.

        Phase C Sprint 1 split the old single ``visitor_snapshot`` event
        into two so the voice greeting can fire instantly while the
        gallery photo waits for the best frame. This handler owns:

          • ``visitors_today`` increment (only here — not on snapshot)
          • ``last_alert_time`` update
          • EventBus :data:`EV_VISITOR_DETECTED` publish — fires voice

        It does NOT save anything to disk or DB. The follow-up
        ``visitor_snapshot`` event (handled in :meth:`_handle_new_visitor`)
        owns persistence and the thumbnail update.

        Refresh-snapshot events (cooldown + absent-duration both
        elapsed for a known visitor) do NOT emit visitor_announce —
        they only arrive at :meth:`_handle_new_visitor`.
        """
        visitor_id = payload.get("visitor_id")
        track_id = payload.get("track_id")
        image_b64: str = payload.get("image_base64", "") or ""
        in_zone: bool = bool(payload.get("in_zone", True))
        should_greet: bool = bool(payload.get("should_greet", in_zone))

        # ── Counter + timestamp ──────────────────────────────────────────
        # ``visitors_today`` is incremented exactly once per arrival —
        # here, on announce. The snapshot event NEVER bumps this counter.
        vnum = self._state.increment("visitors_today")
        self._state.update(
            people_now=max(int(self._state.get("people_now", 0) or 0), 1),
            last_alert_time=now_ist().strftime("%H:%M:%S"),
        )
        with self._lock:
            self._last_visitor_at = time.monotonic()

        # ── Voice greet via EventBus ────────────────────────────────────
        # AIService is the only consumer of EV_VISITOR_DETECTED. The
        # frame here is the FIRST resolving frame (not the SnapshotBuffer's
        # best one), accepted as the trade-off for instant voice. AI
        # captioning uses this frame for scene context.
        mode = self._current_mode()
        if should_greet:
            frame = self._decode_face_crop(image_b64) if image_b64 else None
            try:
                self._bus.publish(
                    EV_VISITOR_DETECTED,
                    frame=frame,
                    visitor_num=vnum,
                    mode=mode,
                    in_zone=in_zone,
                )
            except Exception:
                logger.exception("EventBus publish of EV_VISITOR_DETECTED failed.")

        logger.info(
            "Visitor announced — id=%s vnum=%d greet=%s zone=%s track=%s",
            visitor_id, vnum, should_greet, in_zone, track_id,
        )

        # ── SSE push: instant counter update for dashboards ─────────────
        # No thumbnail here — the snapshot event below pushes the
        # SHARPEST frame, which is what we want browsers to display.
        self._broadcast(
            "visitor",
            visitors_today=vnum,
            last_alert_time=self._state.get("last_alert_time", "—"),
            visitor_id=visitor_id,
            track_id=track_id,
            mode=mode,
        )

    def _handle_new_visitor(self, payload: dict) -> None:
        """Mirror a SnapshotBuffer-resolved snapshot to disk + DB + thumbnail.

        Phase C Sprint 1: this handler runs AFTER the buffer has picked
        the sharpest frame for the visitor. The voice greet and the
        ``visitors_today`` counter were already handled by the
        ``visitor_announce`` event for first-sighting visitors. Refresh
        snapshots skip announce and arrive only here.

        Responsibilities:
          • Always bump ``snapshots_taken``
          • Persist the JPEG to disk + insert a DB row
          • Update ``last_visitor_b64`` (the dashboard now shows the
            BEST frame instead of the first resolving one)
          • SSE push for live dashboard updates
          • Never fire voice — announce already did
        """
        track_id = payload.get("track_id")
        visitor_id = payload.get("visitor_id")
        image_b64: str = payload.get("image_base64", "") or ""
        is_first_sighting: bool = bool(payload.get("is_first_sighting", True))

        # ── Decode ONCE: the payload carries the full scene frame ────────
        frame = self._decode_face_crop(image_b64)
        thumb_b64 = self._encode_thumbnail(frame) or image_b64

        # ── Counters + thumbnail ─────────────────────────────────────────
        # snapshots_taken ALWAYS goes up (every committed buffer event is
        # a snapshot). visitors_today is NOT touched here — that lives
        # exclusively on visitor_announce so first-sighting and refresh
        # events can share this handler safely.
        self._state.increment("snapshots_taken")
        vnum = int(self._state.get("visitors_today", 0) or 0)

        self._state.update(
            people_now=max(int(self._state.get("people_now", 0) or 0), 1),
            last_visitor_b64=thumb_b64,
        )
        with self._lock:
            self._last_visitor_at = time.monotonic()

        # ── Persist the FULL-FRAME snapshot file ─────────────────────────
        snapshot_filename = self._save_snapshot_file(frame)
        if snapshot_filename and self._db is not None:
            try:
                self._db.save_snapshot(
                    snapshot_filename,
                    vtype="Visitor",
                    visitor_id=visitor_id,
                )
            except Exception:
                logger.exception("DB save_snapshot failed for %s", snapshot_filename)

        # ── DB log (best-effort) ─────────────────────────────────────────
        if self._db is not None:
            try:
                msg = (
                    f"Snapshot saved for visitor #{vnum} (track={track_id})"
                    if is_first_sighting
                    else f"Refresh snapshot for visitor #{visitor_id} (track={track_id})"
                )
                self._db.log_event(msg, "visitor", person_count=1)
            except Exception:
                logger.exception("DB log_event failed for visitor #%s.", vnum)

        logger.info(
            "Snapshot bridged — id=%s first=%s vnum=%d track=%s scene=%d KB file=%s",
            visitor_id, is_first_sighting, vnum, track_id,
            len(image_b64) // 1024, snapshot_filename or "<none>",
        )

        # ── SSE push: thumbnail + snapshot counter update ────────────────
        # The thumbnail is the SnapshotBuffer's best frame — browsers
        # see the sharpest available shot of the visitor here.
        self._broadcast(
            "visitor",
            visitors_today=vnum,
            snapshots_taken=int(self._state.get("snapshots_taken", 0) or 0),
            last_visitor_b64=thumb_b64,
            last_alert_time=self._state.get("last_alert_time", "—"),
            visitor_id=visitor_id,
            track_id=track_id,
            mode=self._current_mode(),
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
            now = now_ist()
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
                self._broadcast("camera_status", camera_status=status)

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

    # ── SSE subscriber API (consumed by /api/events) ─────────────────────

    def subscribe(self) -> "queue.Queue[dict]":
        """Register a dashboard tab as an SSE subscriber.

        Returns a Queue; the route handler reads from it in a streaming
        generator. The handler is responsible for calling
        :meth:`unsubscribe` on connection close so we don't leak queues.

        Per-subscriber maxsize=64: a tab that pauses (e.g. backgrounded)
        for a few seconds doesn't lose state because we keep up to 64
        pending events; beyond that we drop oldest events implicitly via
        the bounded-queue + put_nowait pattern in :meth:`_broadcast`.
        """
        q: "queue.Queue[dict]" = queue.Queue(maxsize=64)
        with self._subscribers_lock:
            self._subscribers.append(q)
        logger.debug("SSE subscriber added (total=%d)", len(self._subscribers))
        return q

    def unsubscribe(self, q: "queue.Queue[dict]") -> None:
        """Remove a subscriber. Idempotent."""
        with self._subscribers_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass
        logger.debug("SSE subscriber removed (total=%d)", len(self._subscribers))

    def _broadcast(self, event_type: str, **fields) -> None:
        """Fan out a state-change event to every connected SSE tab.

        The payload shape is intentionally tiny — only the changed keys
        — so a 1 Hz presence event costs a few hundred bytes per tab.
        New-visitor events do carry the thumbnail but only on that single
        message, never on the periodic presence ticks.

        Drop semantics on a full queue: skip the slow consumer rather
        than block. The dashboard's full /api/status remains the source
        of truth on reconnect, so a few dropped delta events are
        harmless.
        """
        msg = {"type": event_type, **fields}
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # slow consumer — drop, don't block

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
