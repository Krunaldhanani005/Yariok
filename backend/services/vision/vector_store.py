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
from typing import Iterable, List, Optional

import cv2
import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Project root: backend/services/vision/vector_store.py → ../../../
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
_DEFAULT_IDENTITY_FILE: str = os.path.join(_PROJECT_ROOT, "daily_identity.pkl")

# ── Dual-signal agreement thresholds ─────────────────────────────────────
#
# Replaces the prior "any single signal above its own threshold wins"
# logic. The old scheme made false MERGES easy: two strangers in
# similar shirts could exceed RECENCY_BODY_THRESHOLD=0.40 alone and end
# up sharing a visitor_id. The new scheme requires that EITHER:
#
#   C1 — face alone is decisive            (face_cosine ≥ 0.55)
#   C2 — face confirms a strong body match (body ≥ 0.65 AND face ≥ 0.30)
#   C3 — body alone is exceptionally strong (body ≥ 0.85)
#
# Anything below all three → no match → new visitor_id → fresh gallery.
# Self-correcting: each false-split's library stays clean, so future
# matching becomes more reliable, not less.
FACE_SURE_THRESHOLD:   float = 0.45   # C1 — was 0.55; lowered because
                                       # forensic analysis of real CCTV
                                       # traffic showed same-person ArcFace
                                       # cosines reliably land in 0.45–0.55,
                                       # producing false-splits at 0.55.
BODY_CORROB_THRESHOLD: float = 0.65   # C2 body floor
FACE_HINT_THRESHOLD:   float = 0.30   # C2 face floor
BODY_SURE_THRESHOLD:   float = 0.85   # C3

# Legacy constants — kept for backward compat with tests and the
# DailyFaceCache constructor signature, but the search() method now
# applies the dual-signal rule above instead.
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
        # Wall-clock (Unix epoch) timestamp of the LAST snapshot we
        # actually saved for each visitor. Drives the 10-minute
        # per-visitor cooldown for refresh snapshots. Stored as wall
        # clock (not monotonic) so the cooldown survives process
        # restarts correctly — a visitor we snapshotted 8 minutes
        # before a crash is still on cooldown 2 minutes after recovery.
        self._last_snapshot_at: List[float] = []
        self._max_views_per_visitor: int = 8

        # ── Session-wide track → visitor binding ─────────────────────────
        # Was: a Set[int] of "known" track ids. Now a Dict[int, int]
        # mapping each resolved track_id to the visitor_id it belongs
        # to. That mapping is what powers the co-occurrence exclusion
        # rule: "if track A is bound to visitor #2 and is currently in
        # frame, track B (a different person in the same frame) MUST
        # NOT be allowed to match visitor #2."
        self.known_bot_sort_ids: dict = {}   # Dict[int, int]: track_id → visitor_id

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
        excluded_visitor_ids: Optional[Iterable[int]] = None,
    ) -> FaceMatch:
        """Look up an identity using the dual-signal agreement rule.

        Compared to the previous "any single signal above its own
        threshold wins" rule, this implementation requires that one of:

            C1 — face_cosine     ≥ FACE_SURE_THRESHOLD   (0.55)
            C2 — body_score      ≥ BODY_CORROB_THRESHOLD (0.65)
                 AND face_cosine ≥ FACE_HINT_THRESHOLD   (0.30)
            C3 — body_score      ≥ BODY_SURE_THRESHOLD   (0.85)

        be true for the BEST candidate. The old "recency body 0.40"
        path is gone — its false-merge risk outweighed its benefit.

        ``excluded_visitor_ids`` enforces the co-occurrence exclusivity
        rule: any visitor_id passed here is removed from the candidate
        pool BEFORE the rule check. The caller (vision worker) sets
        this to the union of visitor_ids currently bound to other
        active tracks, so two people simultaneously in frame can never
        be merged.
        """
        excluded: set = (
            {int(v) for v in excluded_visitor_ids}
            if excluded_visitor_ids else set()
        )

        # candidates[vid] = [body_score, face_score]
        candidates: dict = {}

        with self._lock:
            # ── Body scores (linear scan, max across each visitor's library) ──
            if body_hist is not None and self._body_hists:
                q_body = body_hist.astype(np.float32)
                for vid, library in enumerate(self._body_hists):
                    if vid in excluded or not library:
                        continue
                    v_best = max(
                        float(cv2.compareHist(q_body, h.astype(np.float32),
                                              cv2.HISTCMP_CORREL))
                        for h in library
                    )
                    candidates.setdefault(vid, [-1.0, -1.0])[0] = v_best

            # ── Face scores (FAISS top-k where k = total face rows so we
            #    get the best hit PER visitor and can exclude rows whose
            #    visitor is in the co-occurrence exclusion set) ────────
            if face_embedding is not None and self._face_index.ntotal > 0:
                q = self.l2_normalize(face_embedding).reshape(1, -1).astype(np.float32)
                k = int(self._face_index.ntotal)
                scores, rows = self._face_index.search(q, k=k)
                for s, r in zip(scores[0], rows[0]):
                    if r < 0:
                        continue
                    vid = self._face_row_to_visitor[int(r)]
                    if vid in excluded:
                        continue
                    cur = candidates.setdefault(vid, [-1.0, -1.0])
                    f = float(s)
                    if f > cur[1]:
                        cur[1] = f

            # ── Apply the three rules in priority order ──────────────────
            # C1 > C2 > C3 by tier; inside each tier, sort by the
            # dominant score (face for C1, body for C2/C3).
            c1: list = []
            c2: list = []
            c3: list = []
            for vid, (b, f) in candidates.items():
                if f >= FACE_SURE_THRESHOLD:
                    c1.append((vid, b, f))
                elif b >= BODY_CORROB_THRESHOLD and f >= FACE_HINT_THRESHOLD:
                    c2.append((vid, b, f))
                elif b >= BODY_SURE_THRESHOLD:
                    c3.append((vid, b, f))

            chosen = None  # (vid, body_score, face_score, kind, primary_score)
            if c1:
                c1.sort(key=lambda x: -x[2])   # highest face wins
                vid, b, f = c1[0]
                chosen = (vid, b, f, "face", f)
            elif c2:
                c2.sort(key=lambda x: -x[1])   # highest body wins
                vid, b, f = c2[0]
                chosen = (vid, b, f, "body+face", b)
            elif c3:
                c3.sort(key=lambda x: -x[1])
                vid, b, f = c3[0]
                chosen = (vid, b, f, "body-strict", b)

            if chosen is not None:
                vid, b, f, kind, primary = chosen
                self._last_seen[vid] = time.monotonic()
                if body_hist is not None:
                    self._append_view(vid, body_hist)
                return FaceMatch(
                    matched=True, score=primary, index=vid, kind=kind,
                    body_score=b, face_score=f,
                )

            # No candidate passed any rule. Surface the best-seen
            # scores so the caller can log them and tune thresholds.
            best_b = max((b for b, _ in candidates.values()), default=-1.0)
            best_f = max((f for _, f in candidates.values()), default=0.0)
            return FaceMatch(
                matched=False, score=max(best_b, best_f), index=-1, kind="miss",
                body_score=best_b, face_score=best_f,
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
            # The "first sighting" is itself a snapshot moment, so seed
            # last_snapshot_at to now. This means the 10-min cooldown
            # starts counting from the FIRST snapshot, not from zero.
            self._last_snapshot_at.append(time.time())
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

    # ── Snapshot cooldown ────────────────────────────────────────────────

    def time_since_snapshot(self, visitor_id: int) -> float:
        """Seconds elapsed since this visitor's last snapshot.

        Returns ``float("inf")`` if the visitor id is unknown, so the
        cooldown check naturally evaluates as "long past cooldown" for
        bogus ids — same behaviour as a brand-new visitor.
        """
        with self._lock:
            if 0 <= visitor_id < len(self._last_snapshot_at):
                return max(0.0, time.time() - self._last_snapshot_at[visitor_id])
        return float("inf")

    def mark_snapshot(self, visitor_id: int) -> None:
        """Record that we just took a snapshot for this visitor.

        Called by the vision worker when it commits a snapshot (either
        a first-sighting capture or a refresh after the cooldown has
        elapsed). Persists immediately so the cooldown survives a
        crash mid-day.
        """
        with self._lock:
            if 0 <= visitor_id < len(self._last_snapshot_at):
                self._last_snapshot_at[visitor_id] = time.time()
                self._save_to_disk_locked()

    # ── Track-ID gate (Gate 1 in vision_service) ─────────────────────────

    def mark_track_known(self, track_id: int, visitor_id: int) -> None:
        """Bind a BoT-SORT track id to its resolved visitor id.

        Used by the worker as the FINAL step of identity resolution.
        The mapping powers two things:
          - Gate 1 in the camera thread (is_track_known)
          - The co-occurrence exclusion in :meth:`visitor_ids_for_tracks`
        """
        with self._lock:
            self.known_bot_sort_ids[int(track_id)] = int(visitor_id)

    def is_track_known(self, track_id: int) -> bool:
        with self._lock:
            return int(track_id) in self.known_bot_sort_ids

    def known_track_ids_snapshot(self) -> set:
        """Return a lock-protected snapshot of all currently-bound track ids.

        Used by the inference loop to compute the co-visible "eligible"
        pool — a track that's already been resolved (bound to a
        visitor) is always trustworthy for exclusion, even if its
        stability count hasn't crossed the recent-frames threshold yet.
        Without this safeguard, two strangers walking in on the same
        frame could merge: neither track is "stable" until frame 2+,
        so the first-frame search for the second visitor wouldn't
        exclude the just-bound first visitor.
        """
        with self._lock:
            return set(self.known_bot_sort_ids.keys())

    def visitor_ids_for_tracks(self, track_ids: Iterable[int]) -> set:
        """Return the set of visitor_ids currently bound to the given
        track_ids.

        Used as the ``excluded_visitor_ids`` input to :meth:`search` —
        when track A and track B are visible in the same frame, track B
        is mathematically forbidden from matching whichever visitor_id
        track A is already bound to. That guarantees two simultaneously-
        visible people can never share a visitor_id / gallery.

        Unknown track_ids (not yet bound) contribute nothing to the
        exclusion set. The caller is expected to also be guarded by
        the in-flight set so a single unknown track doesn't get
        dispatched twice in parallel.
        """
        with self._lock:
            out = set()
            for tid in track_ids:
                vid = self.known_bot_sort_ids.get(int(tid))
                if vid is not None:
                    out.add(int(vid))
            return out

    # ── Daily lifecycle ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Wipe everything. Designed for the 9 AM IST business-day cron."""
        with self._lock:
            self._face_index = faiss.IndexFlatIP(self._dim)
            self._face_row_to_visitor.clear()
            self._body_hists.clear()
            self._last_seen.clear()
            self._last_snapshot_at.clear()
            self.known_bot_sort_ids.clear()   # dict.clear() — same call
            # Persist the empty state so a restart inside the same day
            # doesn't load yesterday's identity back in.
            self._save_to_disk_locked()
        logger.info(
            "DailyFaceCache reset — face index, body hists, "
            "snapshot timestamps, and track set cleared",
        )

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
                # Wall-clock per visitor — survives restart so the
                # 10-min snapshot cooldown stays accurate.
                "last_snapshot_at": list(self._last_snapshot_at),
                # Dict in v2 (Dict[track_id → visitor_id]). Legacy
                # files stored a list/set of ints — load handles both.
                "known_bot_sort_ids": dict(self.known_bot_sort_ids),
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
            # Snapshot timestamps are wall-clock — load directly, no
            # conversion needed. Pad to body-library length on legacy
            # files that didn't track this field yet (so the cooldown
            # check doesn't crash on missing entries).
            stored_snaps = [float(t) for t in data.get("last_snapshot_at", [])]
            self._last_snapshot_at = list(stored_snaps)
            while len(self._last_snapshot_at) < len(self._body_hists):
                self._last_snapshot_at.append(0.0)  # ancient → past cooldown
            # Accept both the new dict format and the legacy
            # list/set-of-track-ids format. Legacy entries lose their
            # visitor binding (no longer recoverable), so they're
            # effectively forgotten and will be reassigned on the
            # next sighting — safe behavior.
            raw_known = data.get("known_bot_sort_ids", {})
            if isinstance(raw_known, dict):
                self.known_bot_sort_ids = {
                    int(k): int(v) for k, v in raw_known.items()
                }
            else:
                # legacy list/set → can't recover the visitor binding;
                # drop these so the new exclusion rule has clean inputs.
                self.known_bot_sort_ids = {}
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
            self._last_snapshot_at = []
            self.known_bot_sort_ids = {}
            self._save_to_disk_locked()
