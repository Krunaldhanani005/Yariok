"""SnapshotBuffer — best-frame holding pen for visitor snapshots.

Phase C Sprint 1, Issue #4.

When :meth:`VisionService._handle_face_job` decides a snapshot should be
committed (first sighting OR refresh after the cooldown elapsed) the
pipeline used to JPEG-encode the FIRST resolving frame and publish
``visitor_snapshot`` immediately. On CCTV streams the resolving frame
is almost always the moment the person turned side-on or glanced
toward the lens — i.e. the worst face quality the system will ever
see for them.

Instead, the decision opens a short-lived buffer keyed by
``visitor_id`` (default window 2.5 s). Subsequent frames for that
visitor's bound tracks are scored by Laplacian face-quality and the
buffer retains the sharpest one. When the window expires (or all
bound tracks vanish — see :meth:`commit_for_orphaned_visitors`) the
sharpest frame is committed via the ``on_commit`` callback — which
the vision service wires to publishing the real ``visitor_snapshot``
MQTT event.

Voice greeting is NOT routed through this buffer. The pipeline emits
``visitor_announce`` instantly at decision time so the bridge can
fire the greet without waiting on the buffer.

Threading
---------
Every public method takes an internal :class:`threading.RLock`. The
``commit_*`` methods invoke ``on_commit`` **outside** the lock so the
callback can safely take its own locks (MQTT publish, JPEG encoding,
EventBus publish).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _BufferedSnap:
    """Open buffer for one visitor's snapshot.

    ``payload_template`` is rendered to MQTT verbatim by the on_commit
    callback — it already carries visitor_id, track_id,
    is_first_sighting, in_zone, and the IST timestamp. The callback
    fills in ``image_base64`` + ``face_quality`` from
    :attr:`best_frame` / :attr:`best_quality` at commit time.

    ``track_id`` records the BoT-SORT id that OPENED the buffer; it is
    not used to gate submissions (any bound track for this visitor is
    free to feed it) but it is informational for logging.
    """
    visitor_id: int
    track_id: int
    opened_mono: float
    deadline_mono: float
    best_frame: np.ndarray
    best_quality: float
    payload_template: dict


class SnapshotBuffer:
    """Quality-aware holding pen for pending visitor snapshots."""

    def __init__(
        self,
        window_seconds: float = 2.5,
        on_commit: Optional[Callable[[dict, np.ndarray, float], None]] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._open: Dict[int, _BufferedSnap] = {}
        self._window: float = float(window_seconds)
        self._on_commit = on_commit

    # ── Read-only state ──────────────────────────────────────────────

    def is_open(self, visitor_id: int) -> bool:
        with self._lock:
            return int(visitor_id) in self._open

    def open_count(self) -> int:
        with self._lock:
            return len(self._open)

    # ── Mutations ────────────────────────────────────────────────────

    def open(
        self,
        visitor_id: int,
        track_id: int,
        frame: np.ndarray,
        face_quality: float,
        payload_template: dict,
    ) -> None:
        """Begin (or restart) a window for the given visitor.

        Restart semantics: if a buffer is already open (defensive
        case — e.g. a refresh snapshot triggered while a previous
        buffer was still draining), the new open overwrites with a
        fresh deadline and the current best frame.
        """
        vid = int(visitor_id)
        with self._lock:
            mono = time.monotonic()
            self._open[vid] = _BufferedSnap(
                visitor_id=vid,
                track_id=int(track_id),
                opened_mono=mono,
                deadline_mono=mono + self._window,
                best_frame=frame.copy(),
                best_quality=float(face_quality),
                payload_template=dict(payload_template),
            )

    def submit(
        self,
        visitor_id: int,
        frame: np.ndarray,
        face_quality: float,
    ) -> bool:
        """Offer a candidate frame; return True iff it became the new best.

        Silently no-ops if no buffer is open for the given visitor —
        callers (the face worker) race with the commit path and can
        land a submit microseconds after the buffer drained.
        """
        with self._lock:
            buf = self._open.get(int(visitor_id))
            if buf is None:
                return False
            if face_quality > buf.best_quality:
                buf.best_frame = frame.copy()
                buf.best_quality = float(face_quality)
                return True
        return False

    # ── Commit paths ─────────────────────────────────────────────────

    def commit_expired(self) -> List[int]:
        """Commit any buffers whose deadline has passed.

        Called from the inference loop on every frame. ``on_commit``
        is invoked outside the lock so MQTT publish / JPEG encoding
        can take their time without freezing other inference frames.
        """
        now = time.monotonic()
        to_commit: List[_BufferedSnap] = []
        with self._lock:
            for vid in list(self._open.keys()):
                if self._open[vid].deadline_mono <= now:
                    to_commit.append(self._open.pop(vid))
        return [self._fire_commit(b, reason="deadline") for b in to_commit]

    def commit_for_orphaned_visitors(self, alive_visitor_ids: Set[int]) -> List[int]:
        """Early-commit buffers whose visitor has no alive bound track.

        ``alive_visitor_ids`` is the set of visitor_ids currently bound
        to ANY in-frame track. If an open buffer's visitor is not in
        this set, the person has left the camera view — commit
        whatever the best frame currently is rather than waiting out
        the deadline on someone who is gone.

        This is more conservative than "commit when the OPENING track
        drops": a BoT-SORT track reset that spawns a fresh id for the
        same visitor keeps the buffer open across the transition.
        """
        to_commit: List[_BufferedSnap] = []
        with self._lock:
            for vid in list(self._open.keys()):
                if vid not in alive_visitor_ids:
                    to_commit.append(self._open.pop(vid))
        return [self._fire_commit(b, reason="visitor-left") for b in to_commit]

    def drain(self) -> List[int]:
        """Commit and remove every open buffer immediately.

        Used on cache reset (factory wipe, 9 AM rollover) and on
        service shutdown — we do not want a buffered visitor's photo
        landing on disk after the visitor_id no longer exists in the
        identity cache.
        """
        to_commit: List[_BufferedSnap] = []
        with self._lock:
            for vid in list(self._open.keys()):
                to_commit.append(self._open.pop(vid))
        return [self._fire_commit(b, reason="drain") for b in to_commit]

    # ── Internals ────────────────────────────────────────────────────

    def _fire_commit(self, buf: _BufferedSnap, reason: str) -> int:
        """Invoke ``on_commit`` outside the lock. Returns the visitor_id."""
        if self._on_commit is not None:
            try:
                self._on_commit(buf.payload_template, buf.best_frame, buf.best_quality)
            except Exception:
                logger.exception(
                    "SnapshotBuffer.on_commit failed — visitor=#%d reason=%s",
                    buf.visitor_id, reason,
                )
        else:
            logger.debug(
                "SnapshotBuffer commit (no callback) — visitor=#%d reason=%s quality=%.1f",
                buf.visitor_id, reason, buf.best_quality,
            )
        return buf.visitor_id
