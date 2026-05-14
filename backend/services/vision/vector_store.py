"""DailyFaceCache — daily-scoped HYBRID identity memory.

Phase B v2 — the original face-only design failed on CCTV-quality faces
(tiny crops, off-axis pose, lighting drift, occlusion). Same visitor was
embedded multiple times because their face never reached the cosine
threshold across track resets.

The fix: store TWO identity signals per visitor and accept a match from
either modality:

    1. ArcFace face embedding  (512-d, L2-normalized, FAISS IndexFlatIP)
       - High precision when a clear face is visible.
       - Often missing on CCTV frames (back of head, far away, occluded).

    2. HSV clothing histogram  (computed from the torso crop)
       - Always available — every YOLO person bbox has a torso.
       - Robust to face angle, head turn, partial occlusion.
       - Loses uniqueness if many people wear similar colors, but
         combined with the FAISS face index it is decisive in practice.

A new visitor is added once; subsequent track-IDs that match EITHER
signal are silently re-identified — no second snapshot, no second greet.

This module still touches NO disk, NO database, NO MQTT.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Set

import cv2
import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Project root: backend/services/vision/vector_store.py → ../../../
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
_DEFAULT_IDENTITY_FILE: str = os.path.join(_PROJECT_ROOT, "daily_identity.pkl")

# Recency window: a body-score that's below the strict threshold but
# above the recency threshold is accepted as a match if the candidate
# visitor was last seen within this many seconds. Rationale: real-world
# CCTV re-IDs of the same person within ~30s (same lighting, same room)
# tolerate a much looser body-color match than re-IDs across hours.
RECENCY_WINDOW_SECONDS: float = 120.0
RECENCY_BODY_THRESHOLD: float = 0.40


@dataclass(frozen=True)
class FaceMatch:
    """Result of an identity lookup.

    ``kind`` reports which modality decided the match (``"face"`` or
    ``"body"``) so the caller can log usefully and debug threshold
    tuning. ``index`` is the visitor id — stable across the day's
    session — not the FAISS row id.

    ``body_score`` / ``face_score`` are surfaced regardless of which
    (if any) modality crossed its threshold, for diagnostic logging.
    """
    matched: bool
    score: float
    index: int
    kind: str = "none"
    body_score: float = -1.0
    face_score: float = -1.0


class DailyFaceCache:
    """Hybrid (face + body) identity store for the current calendar day.

    Thread-safety: every public method takes an internal lock. Concurrent
    callers from the camera loop and the face-worker thread are safe.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        match_threshold: float = 0.40,
        body_threshold: float = 0.55,
        state_file: str = _DEFAULT_IDENTITY_FILE,
    ) -> None:
        self._dim: int = embedding_dim
        self._face_threshold: float = match_threshold
        self._body_threshold: float = body_threshold
        self._lock: threading.RLock = threading.RLock()
        self._state_file: str = state_file

        # ── Face index (FAISS) ───────────────────────────────────────────
        # Each row corresponds to ONE face observation. A visitor may have
        # no face entry (no face was ever extracted) or, in future, multiple.
        self._face_index: faiss.Index = faiss.IndexFlatIP(self._dim)
        self._face_row_to_visitor: List[int] = []

        # ── Per-visitor parallel data ────────────────────────────────────
        # Index == visitor id (stable for the day).
        # Each visitor owns a *library* of body histograms — one per
        # distinct angle / pose. Matching takes the max correlation
        # across all stored views, which copes with the natural HSV
        # drift the same person produces when turning around or moving
        # under varying lighting.
        self._body_hists: List[List[np.ndarray]] = []
        self._last_seen: List[float] = []   # monotonic timestamp per visitor
        self._max_views_per_visitor: int = 8

        # ── Session-wide track gate ──────────────────────────────────────
        self.known_bot_sort_ids: Set[int] = set()

        logger.info(
            "DailyFaceCache initialized — face_thresh=%.2f body_thresh=%.2f",
            self._face_threshold, self._body_threshold,
        )

        # Hydrate today's identity cache from disk if available. Same
        # daily-rollover semantics as ``daily_state.json``: yesterday's
        # file is ignored, a new one is written on the next mutation.
        with self._lock:
            self._load_from_disk_locked()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of distinct visitors currently stored today."""
        with self._lock:
            return len(self._body_hists)

    @property
    def face_index_size(self) -> int:
        """Number of face embeddings in the FAISS index (≤ size)."""
        with self._lock:
            return int(self._face_index.ntotal)

    @property
    def threshold(self) -> float:
        return self._face_threshold

    # ── Math helpers ─────────────────────────────────────────────────────

    @staticmethod
    def l2_normalize(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        if v.ndim == 1:
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else v
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        return (v / norms).astype(np.float32)

    # ── Identity ops ─────────────────────────────────────────────────────

    def search(
        self,
        face_embedding: Optional[np.ndarray] = None,
        body_hist: Optional[np.ndarray] = None,
    ) -> FaceMatch:
        """Look up an identity. Returns the first decisive match.

        Order of evaluation:
            1. Body histogram (linear scan, very cheap, always available).
               Most CCTV-induced false-new visitors are caught here.
            2. Face FAISS (sub-millisecond) — picks up cases where the
               person changed jacket / turned away from the camera.

        If neither modality crosses its threshold, returns an unmatched
        FaceMatch (caller treats as new visitor).
        """
        best_body_score: float = -1.0
        best_body_visitor: int = -1
        best_face_score: float = 0.0
        now = time.monotonic()

        with self._lock:
            # ── 1) Body match ────────────────────────────────────────────
            # Two-tier acceptance:
            #   strict :  ≥ self._body_threshold   (default 0.70)
            #   recency:  ≥ RECENCY_BODY_THRESHOLD (default 0.40)
            #             AND visitor was last seen < 45s ago
            # The recency tier catches the common CCTV failure mode
            # where the same person turns around within a few seconds
            # and their back-of-clothing histogram drops to ~0.5.
            if body_hist is not None and self._body_hists:
                q_body = body_hist.astype(np.float32)

                # Per visitor we keep a LIBRARY of body views (one per
                # distinct pose) — take the MAX correlation across all
                # stored views as that visitor's score. This handles
                # the natural HSV drift of the same person turning /
                # moving under varying light.
                best_recent_score, best_recent_visitor = -1.0, -1
                for vid, library in enumerate(self._body_hists):
                    if not library:
                        continue
                    v_best = max(
                        float(cv2.compareHist(q_body, h.astype(np.float32),
                                              cv2.HISTCMP_CORREL))
                        for h in library
                    )
                    if v_best > best_body_score:
                        best_body_score, best_body_visitor = v_best, vid
                    if (v_best >= RECENCY_BODY_THRESHOLD
                        and (now - self._last_seen[vid]) <= RECENCY_WINDOW_SECONDS
                        and v_best > best_recent_score):
                        best_recent_score, best_recent_visitor = v_best, vid

                # Strict match always wins.
                if best_body_visitor >= 0 and best_body_score >= self._body_threshold:
                    self._last_seen[best_body_visitor] = now
                    self._append_view(best_body_visitor, body_hist)
                    return FaceMatch(
                        True, best_body_score, best_body_visitor, "body",
                        body_score=best_body_score, face_score=best_face_score,
                    )
                # Otherwise: any recent visitor that crossed the relaxed bound.
                if best_recent_visitor >= 0:
                    self._last_seen[best_recent_visitor] = now
                    self._append_view(best_recent_visitor, body_hist)
                    return FaceMatch(
                        True, best_recent_score, best_recent_visitor, "body-recent",
                        body_score=best_recent_score, face_score=best_face_score,
                    )

            # ── 2) Face match (FAISS IP search) ──────────────────────────
            if face_embedding is not None and self._face_index.ntotal > 0:
                q = self.l2_normalize(face_embedding).reshape(1, -1).astype(np.float32)
                scores, rows = self._face_index.search(q, k=1)
                best_face_score = float(scores[0][0])
                top_row = int(rows[0][0])
                if best_face_score >= self._face_threshold:
                    visitor_id = self._face_row_to_visitor[top_row]
                    self._last_seen[visitor_id] = now
                    if body_hist is not None:
                        self._append_view(visitor_id, body_hist)
                    return FaceMatch(
                        True, best_face_score, visitor_id, "face",
                        body_score=best_body_score, face_score=best_face_score,
                    )

        # No decisive match. Surface both scores so the caller can log
        # body_score=X face_score=Y and tune thresholds from real data.
        miss_score = best_body_score if best_body_score >= 0 else best_face_score
        return FaceMatch(
            False, miss_score, -1, "miss",
            body_score=best_body_score, face_score=best_face_score,
        )

    def add_visitor(
        self,
        face_embedding: Optional[np.ndarray] = None,
        body_hist: Optional[np.ndarray] = None,
    ) -> int:
        """Register a new visitor. Returns the freshly assigned visitor id."""
        with self._lock:
            visitor_id = len(self._body_hists)
            library: List[np.ndarray] = []
            if body_hist is not None:
                library.append(body_hist.astype(np.float32).copy())
            self._body_hists.append(library)
            self._last_seen.append(time.monotonic())
            if face_embedding is not None:
                vec = self.l2_normalize(face_embedding).reshape(1, -1).astype(np.float32)
                self._face_index.add(vec)
                self._face_row_to_visitor.append(visitor_id)
            self._save_to_disk_locked()
            return visitor_id

    def _append_view(self, visitor_id: int, body_hist: np.ndarray) -> None:
        """Add another body-view to a visitor's library (caller holds lock).

        Caps the library at ``_max_views_per_visitor`` using a simple
        FIFO eviction — the oldest view is the most likely to be stale
        (e.g. early visit, different lighting). Cheap and effective.
        """
        if not (0 <= visitor_id < len(self._body_hists)):
            return
        lib = self._body_hists[visitor_id]
        lib.append(body_hist.astype(np.float32).copy())
        if len(lib) > self._max_views_per_visitor:
            del lib[0]
        self._save_to_disk_locked()

    def update_body_hist(self, visitor_id: int, body_hist: np.ndarray) -> None:
        """Public alias kept for backward-compat — appends a view."""
        with self._lock:
            self._append_view(visitor_id, body_hist)

    # ── Track-ID gate (Gate 1 in vision_service) ─────────────────────────

    def mark_track_known(self, track_id: int) -> None:
        with self._lock:
            self.known_bot_sort_ids.add(int(track_id))

    def is_track_known(self, track_id: int) -> bool:
        with self._lock:
            return int(track_id) in self.known_bot_sort_ids

    # ── Daily lifecycle ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Wipe everything. Designed for the midnight cron."""
        with self._lock:
            self._face_index = faiss.IndexFlatIP(self._dim)
            self._face_row_to_visitor.clear()
            self._body_hists.clear()
            self._last_seen.clear()
            self.known_bot_sort_ids.clear()
            # Persist the empty state so a restart inside the same day
            # doesn't load yesterday's identity back in.
            self._save_to_disk_locked()
        logger.info("DailyFaceCache reset — face index, body hists, and track set cleared")

    # ── Persistence (caller MUST hold self._lock) ────────────────────────

    def _save_to_disk_locked(self) -> None:
        """Atomic pickle write of today's identity cache.

        Same pattern as ``StateManager._save_to_disk_locked``:
        write a sibling temp file, fsync, ``os.replace`` it over the
        target. POSIX guarantees the rename is atomic, so a reader can
        only ever see the OLD file or the NEW file — never a half-
        flushed mix. Survives ``SIGKILL``, power loss, disk-full.

        ``_last_seen`` is a ``time.monotonic()`` value, which has no
        meaning across process restarts. We store each visitor's
        last-seen age (seconds-ago at save time) and reconstruct on
        load by subtracting from the fresh process's monotonic clock.

        Failures are logged and swallowed — we'd rather drop a write
        than crash the vision pipeline.
        """
        try:
            save_mono = time.monotonic()
            last_seen_age = [max(0.0, save_mono - t) for t in self._last_seen]
            payload = {
                "date": date.today().isoformat(),
                "face_index_bytes": bytes(faiss.serialize_index(self._face_index)),
                "face_to_visitor": list(self._face_row_to_visitor),
                "body_libraries": [
                    [h.copy() for h in lib] for lib in self._body_hists
                ],
                "last_seen_age": last_seen_age,
                "known_bot_sort_ids": list(self.known_bot_sort_ids),
            }
            dir_name = os.path.dirname(self._state_file) or "."
            os.makedirs(dir_name, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".daily_identity.", suffix=".tmp", dir=dir_name,
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._state_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            logger.exception(
                "DailyFaceCache: failed to persist %s", self._state_file,
            )

    def _load_from_disk_locked(self) -> None:
        """Restore today's identity cache from disk if date matches.

        Date mismatch → ignore the file (don't carry yesterday's
        visitors into today). Missing or corrupt file → start fresh
        and let the next mutation overwrite it.
        """
        today = date.today().isoformat()
        try:
            with open(self._state_file, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            logger.info(
                "DailyFaceCache: %s not found — fresh identity cache.",
                self._state_file,
            )
            return
        except Exception as exc:
            logger.warning(
                "DailyFaceCache: %s unreadable (%s) — fresh identity cache.",
                self._state_file, exc,
            )
            return

        if data.get("date") != today:
            logger.info(
                "DailyFaceCache: stored date=%s ≠ today=%s — fresh identity cache.",
                data.get("date"), today,
            )
            return

        try:
            idx_bytes = data.get("face_index_bytes")
            if idx_bytes:
                self._face_index = faiss.deserialize_index(
                    np.frombuffer(idx_bytes, dtype=np.uint8)
                )
            self._face_row_to_visitor = [int(v) for v in data.get("face_to_visitor", [])]
            libs = data.get("body_libraries", [])
            self._body_hists = [
                [np.asarray(h, dtype=np.float32).copy() for h in lib]
                for lib in libs
            ]
            ages = [float(a) for a in data.get("last_seen_age", [])]
            now_mono = time.monotonic()
            self._last_seen = [now_mono - a for a in ages]
            self.known_bot_sort_ids = {int(t) for t in data.get("known_bot_sort_ids", [])}
            logger.info(
                "DailyFaceCache: restored %d visitor(s), %d face entries from %s",
                len(self._body_hists), int(self._face_index.ntotal), self._state_file,
            )
        except Exception:
            logger.exception(
                "DailyFaceCache: failed to restore from %s — starting fresh.",
                self._state_file,
            )
            # Force a clean state and rewrite so the bad file is replaced.
            self._face_index = faiss.IndexFlatIP(self._dim)
            self._face_row_to_visitor = []
            self._body_hists = []
            self._last_seen = []
            self.known_bot_sort_ids = set()
            self._save_to_disk_locked()
