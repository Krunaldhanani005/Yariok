"""DailyFaceCache — daily-scoped HYBRID identity memory (OSNet edition).

Phase D Tier 1.1 — the prior HSV-clothing histogram has been replaced
by a 512-d OSNet-x0_25 body embedding. Same role in the pipeline,
fundamentally stronger representation:

    1. ArcFace face embedding  (512-d, L2-normalized, FAISS IndexFlatIP)
       - High precision when a clear face is visible.
       - Often missing on CCTV frames (back of head, far away, occluded).

    2. OSNet body embedding    (512-d, L2-normalized)
       - Always available — every YOLO person bbox has a torso.
       - SEPARABLE same- vs. different-person cosine distribution on
         surveillance-quality crops (the property HSV did not have).
       - Stored per-visitor as a small library of views (FIFO cap),
         scored by max cosine across the library.

A new visitor is added once; subsequent track-IDs that match EITHER
signal are silently re-identified — no second snapshot, no second greet.

This module still touches NO disk for images, NO database, NO MQTT.
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

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Project root: backend/services/vision/vector_store.py → ../../../
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
_DEFAULT_IDENTITY_FILE: str = os.path.join(_PROJECT_ROOT, "daily_identity.pkl")

# Bumped when the pickle payload schema changes in a non-backward-compatible
# way. Old files are silently ignored (treated as "no cache for today") so
# the next mutation overwrites them — same recovery path as a stale-date
# load failure.
_SCHEMA_VERSION: int = 2   # v1 = HSV histograms, v2 = OSNet body embeddings

# ── Dual-signal agreement thresholds — recalibrated for OSNet ───────────
#
# C1 (face_sure) is unchanged: the face engine itself is unchanged, so
# ArcFace's cosine distribution against today's visitor faces hasn't
# moved. 0.40 was empirically validated against CCTV-quality face
# crops during Phase B (visitors #38, #47-49, #57, #58, #63 same-
# person cosines clustering 0.30-0.45).
#
# C2 (body-corroborated) keeps its body floor at 0.55 and its face hint
# at 0.25. On OSNet's separable distribution a 0.55 body cosine is a
# MUCH stronger same-person signal than HSV-correlation 0.55 ever was —
# keeping the bar gives us automatic safety margin on the upgrade.
#
# C3 (body-strict) drops from 0.85 → 0.65. HSV needed 0.85 because its
# same- vs different-person correlation distributions overlapped heavily
# (any threshold below ~0.85 admitted false-merges from similar-coloured
# clothing). OSNet's distributions are separable: random-pair cosines
# typically stay below 0.40, while same-person cosines on torso crops
# routinely exceed 0.60. A 0.65 floor is comfortably above the
# different-person 99th percentile and below the same-person median —
# captures genuine re-IDs without inviting colour-collision merges.
# The face-disagreement guard remains as a belt-and-braces check.
FACE_SURE_THRESHOLD:   float = 0.40   # C1 — unchanged from Phase B/C
BODY_CORROB_THRESHOLD: float = 0.55   # C2 body floor (stronger on OSNet
                                       #  than the same number on HSV)
FACE_HINT_THRESHOLD:   float = 0.25   # C2 face floor — unchanged
BODY_SURE_THRESHOLD:   float = 0.65   # C3 — dropped from 0.85; OSNet's
                                       #  separable distribution makes
                                       #  body-only matches reliable
                                       #  well below the old HSV bar

# C3 face-disagreement guard — preserved from Phase B/C.
#
# When body-only matching fires AND faces are available on both sides,
# require the ArcFace cosine to NOT strongly disagree (≥0.10). Without
# this, two genuinely different people whose OSNet body embeddings
# happen to collide above C3 (rare but possible — twins, near-identical
# uniforms) could merge. 0.10 is "not provably different" — well below
# the noise floor for same-person on CCTV. With OSNet body now reliable,
# this guard fires much less often, but the safety margin is free.
FACE_DISAGREE_THRESHOLD: float = 0.10

# Legacy constants — retained only for backwards-compatible construction
# of DailyFaceCache from existing callers (they pass these by keyword).
# The current search() applies the dual-signal rule above instead.
RECENCY_WINDOW_SECONDS: float = 120.0
RECENCY_BODY_THRESHOLD: float = 0.40


@dataclass(frozen=True)
class FaceMatch:
    """Result of an identity lookup.

    ``kind`` reports which modality decided the match (``"face"``,
    ``"body+face"``, or ``"body-strict"``) so the caller can log
    usefully and debug threshold tuning. ``index`` is the visitor id —
    stable across the day's session — not the FAISS row id.

    ``body_score`` / ``face_score`` are surfaced regardless of which
    (if any) modality crossed its threshold, for diagnostic logging.
    Both are cosine similarities in [-1, 1] on OSNet (body) and
    ArcFace (face) respectively.

    ``top_candidates`` carries per-visitor (vid, body, face) tuples
    for the top few near-miss candidates — populated when ``matched``
    is False so the caller can log which visitor came closest to
    which rule.
    """
    matched: bool
    score: float
    index: int
    kind: str = "none"
    body_score: float = -1.0
    face_score: float = -1.0
    top_candidates: tuple = ()   # tuple[tuple[int, float, float], ...]


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

        # ── Per-visitor body embedding library ───────────────────────────
        # Index == visitor id (stable for the day).
        #
        # Each visitor owns a SMALL library of OSNet body embeddings —
        # one per distinct pose / angle / lighting moment. Matching takes
        # the MAX cosine across the library, which copes with the small
        # natural drift the same person produces when turning around or
        # walking under varying illumination.
        #
        # We deliberately use a per-visitor list (not a single FAISS
        # IndexFlatIP) because:
        #   - the library is tiny (≤ 8 vectors per visitor)
        #   - FIFO eviction (cap policy) is a one-line list slice, vs.
        #     IndexFlatIP requiring an IndexIDMap rebuild
        #   - numpy matmul of a (k, 512) matrix against a (512,) query is
        #     dominated by the dot product itself — FAISS adds setup cost
        #     without speedup at this scale
        # This is the "exact cosine similarity matrix" alternative noted
        # in the Tier 1 roadmap.
        self._body_embs: List[List[np.ndarray]] = []
        self._last_seen: List[float] = []   # monotonic timestamp per visitor
        # Wall-clock (Unix epoch) timestamp of the LAST snapshot we
        # actually saved for each visitor. Drives the 10-minute
        # per-visitor cooldown for refresh snapshots. Stored as wall
        # clock (not monotonic) so the cooldown survives process
        # restarts correctly.
        self._last_snapshot_at: List[float] = []
        # Wall-clock per visitor — last frame on which this visitor had
        # any bound track in view. Drives the absent-duration guard
        # (Issue #5). Wall-clock so the guard survives restarts.
        self._last_seen_wall: List[float] = []
        self._max_views_per_visitor: int = 8

        # ── Session-wide track → visitor binding ─────────────────────────
        # Maps each resolved BoT-SORT track_id to the visitor_id it
        # belongs to. Powers the co-occurrence exclusion rule: if track
        # A is bound to visitor #2 and is currently in frame, track B
        # (a different person in the same frame) MUST NOT match #2.
        self.known_bot_sort_ids: dict = {}   # Dict[int, int]: track_id → visitor_id

        logger.info(
            "DailyFaceCache initialized (OSNet body) — "
            "face_sure=%.2f body_corrob=%.2f body_sure=%.2f",
            FACE_SURE_THRESHOLD, BODY_CORROB_THRESHOLD, BODY_SURE_THRESHOLD,
        )

        # Hydrate today's identity cache from disk if available. Same
        # daily-rollover semantics as ``daily_state.json``: yesterday's
        # file is ignored, a new one is written on the next mutation.
        # v1 (HSV) pickles are also ignored — schema version mismatch.
        with self._lock:
            self._load_from_disk_locked()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of distinct visitors currently stored today."""
        with self._lock:
            return len(self._body_embs)

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
        body_emb: Optional[np.ndarray] = None,
        excluded_visitor_ids: Optional[Iterable[int]] = None,
    ) -> FaceMatch:
        """Look up an identity using the dual-signal agreement rule.

        Three accept rules, evaluated per visitor and applied in tier
        priority C1 > C2 > C3:

            C1 — face_cosine ≥ FACE_SURE_THRESHOLD       (0.40)
            C2 — body_cosine ≥ BODY_CORROB_THRESHOLD     (0.55)
                 AND face_cosine ≥ FACE_HINT_THRESHOLD   (0.25)
            C3 — body_cosine ≥ BODY_SURE_THRESHOLD       (0.65)
                 AND C3 face-disagreement guard passes

        ``body_emb`` is expected to be an L2-normalized 512-d OSNet
        embedding produced by :class:`BodyEngine`. The search computes
        per-visitor cosine = max over that visitor's library of
        ``visitor_emb · body_emb`` (both unit-norm, so the dot product
        IS the cosine).

        ``excluded_visitor_ids`` enforces the co-occurrence exclusivity
        rule: any visitor_id passed here is removed from the candidate
        pool BEFORE the rule check. The caller (vision worker) sets
        this to the union of visitor_ids currently bound to other
        active tracks, so two people simultaneously in frame can never
        be merged. The sibling-IoU exception in the inference loop
        carves out BoT-SORT track splits from this exclusion.
        """
        excluded: set = (
            {int(v) for v in excluded_visitor_ids}
            if excluded_visitor_ids else set()
        )

        # candidates[vid] = [body_score, face_score]
        candidates: dict = {}

        with self._lock:
            # ── Body scores (numpy matmul, max across each visitor's library) ──
            if body_emb is not None and self._body_embs:
                q_body = np.asarray(body_emb, dtype=np.float32).reshape(-1)
                # Normalize defensively in case the caller forgot — cheap.
                qn = float(np.linalg.norm(q_body))
                if qn > 1e-9:
                    q_body = q_body / qn
                    for vid, library in enumerate(self._body_embs):
                        if vid in excluded or not library:
                            continue
                        # Stack library views into a (k, 512) matrix; cosine
                        # with the query is a single matmul. Both sides are
                        # unit-norm (we L2-normalize on add), so the dot
                        # product equals the cosine similarity directly.
                        mat = np.stack(library, axis=0).astype(np.float32)
                        sims = mat @ q_body
                        v_best = float(np.max(sims))
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

            # Pre-compute which visitors have at least one face entry
            # in FAISS. Used by the C3 face-disagreement guard below
            # to decide whether body-only matching is safe.
            visitors_with_face: set = set(self._face_row_to_visitor)
            query_has_face: bool = face_embedding is not None

            # ── Apply the three rules in priority order ──────────────────
            c1: list = []
            c2: list = []
            c3: list = []
            for vid, (b, f) in candidates.items():
                if f >= FACE_SURE_THRESHOLD:
                    c1.append((vid, b, f))
                elif b >= BODY_CORROB_THRESHOLD and f >= FACE_HINT_THRESHOLD:
                    c2.append((vid, b, f))
                elif b >= BODY_SURE_THRESHOLD:
                    # C3 face-disagreement guard. With OSNet body now
                    # reliable, this fires rarely — but a uniform-clad
                    # twin remains a valid edge case to defend against.
                    visitor_has_face = vid in visitors_with_face
                    if query_has_face and visitor_has_face:
                        if f < FACE_DISAGREE_THRESHOLD:
                            continue   # faces strongly disagree → REFUSE
                    elif not query_has_face and visitor_has_face:
                        # Visitor was registered WITH a face but this
                        # query has none. Body alone is insufficient
                        # evidence — refuse so the worker waits for a
                        # frame with a face.
                        continue
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
                if body_emb is not None:
                    self._append_view(vid, body_emb)
                return FaceMatch(
                    matched=True, score=primary, index=vid, kind=kind,
                    body_score=b, face_score=f,
                )

            # No candidate passed any rule. Surface the best-seen
            # scores so the caller can log them and tune thresholds.
            best_b = max((b for b, _ in candidates.values()), default=-1.0)
            best_f = max((f for _, f in candidates.values()), default=0.0)
            ranked = sorted(
                candidates.items(),
                key=lambda kv: max(kv[1][0], kv[1][1]),
                reverse=True,
            )[:5]
            top_candidates = tuple(
                (int(vid), float(b), float(f)) for vid, (b, f) in ranked
            )
            return FaceMatch(
                matched=False, score=max(best_b, best_f), index=-1, kind="miss",
                body_score=best_b, face_score=best_f,
                top_candidates=top_candidates,
            )

    def add_visitor(
        self,
        face_embedding: Optional[np.ndarray] = None,
        body_emb: Optional[np.ndarray] = None,
    ) -> int:
        """Register a new visitor. Returns the freshly assigned visitor id.

        ``body_emb`` should be the L2-normalized 512-d OSNet embedding
        from :class:`BodyEngine.compute_body_embedding`. We re-normalize
        defensively so callers that bypass the engine still produce a
        unit-norm library row.
        """
        with self._lock:
            visitor_id = len(self._body_embs)
            library: List[np.ndarray] = []
            if body_emb is not None:
                v = np.asarray(body_emb, dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(v))
                if n > 1e-9:
                    library.append((v / n).astype(np.float32).copy())
            self._body_embs.append(library)
            self._last_seen.append(time.monotonic())
            # The "first sighting" is itself a snapshot moment, so seed
            # last_snapshot_at to now — the 10-min cooldown starts from
            # the FIRST snapshot, not from zero.
            self._last_snapshot_at.append(time.time())
            # Seed last-seen wall-clock to now. The visitor IS visible
            # right now, so the absent-duration guard correctly treats
            # them as "not absent".
            self._last_seen_wall.append(time.time())
            if face_embedding is not None:
                vec = self.l2_normalize(face_embedding).reshape(1, -1).astype(np.float32)
                self._face_index.add(vec)
                self._face_row_to_visitor.append(visitor_id)
            self._save_to_disk_locked()
            return visitor_id

    def _append_view(self, visitor_id: int, body_emb: np.ndarray) -> None:
        """Add another body view to a visitor's library (caller holds lock).

        Caps the library at ``_max_views_per_visitor`` using a simple
        FIFO eviction — the oldest view is the most likely to be stale
        (early visit, different lighting, partial occlusion). Cheap and
        effective for OSNet embeddings as it was for HSV histograms.
        """
        if not (0 <= visitor_id < len(self._body_embs)):
            return
        v = np.asarray(body_emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return
        v = (v / n).astype(np.float32)
        lib = self._body_embs[visitor_id]
        lib.append(v.copy())
        if len(lib) > self._max_views_per_visitor:
            del lib[0]
        self._save_to_disk_locked()

    def update_body_emb(self, visitor_id: int, body_emb: np.ndarray) -> None:
        """Public alias for appending a body view to a visitor's library."""
        with self._lock:
            self._append_view(visitor_id, body_emb)

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

    # ── Absent-duration tracking (Issue #5) ──────────────────────────────

    def touch_last_seen(self, visitor_ids: Iterable[int]) -> None:
        """Mark each visitor as seen RIGHT NOW.

        Called from the inference loop once per frame with the set of
        visitor_ids that currently have at least one bound BoT-SORT
        track in view. The wall-clock timestamp updated here drives
        the absent-duration guard.

        Unknown visitor_ids are silently skipped — the caller may pass
        in stale ids during shutdown / reset races.

        Persistence is NOT triggered per call; this method runs at
        camera FPS and would shred the disk. The next mutation that
        does persist will pick up the latest timestamps.
        """
        now = time.time()
        with self._lock:
            n = len(self._last_seen_wall)
            for vid in visitor_ids:
                i = int(vid)
                if 0 <= i < n:
                    self._last_seen_wall[i] = now

    def time_since_last_seen(self, visitor_id: int) -> float:
        """Seconds elapsed since this visitor's last bound-track view.

        Returns ``float("inf")`` for unknown visitor ids so callers
        treat the absent-duration check as "trivially satisfied".

        Read this BEFORE calling :meth:`mark_track_known` for a NEW
        track resolution: binding causes the next frame's
        :meth:`touch_last_seen` to wipe the absence we want to
        measure.
        """
        with self._lock:
            if 0 <= visitor_id < len(self._last_seen_wall):
                return max(0.0, time.time() - self._last_seen_wall[visitor_id])
        return float("inf")

    def track_to_visitor_snapshot(self) -> dict:
        """Return a stable copy of the track_id → visitor_id mapping.

        The inference loop uses this once per frame to (a) compute
        which visitors are currently visible (for touch_last_seen)
        and (b) decide whether a known-track dispatch should be a
        buffer-feed job.
        """
        with self._lock:
            return dict(self.known_bot_sort_ids)

    # ── Track-ID gate (Gate 1 in vision_service) ─────────────────────────

    def mark_track_known(self, track_id: int, visitor_id: int) -> None:
        """Bind a BoT-SORT track id to its resolved visitor id."""
        with self._lock:
            self.known_bot_sort_ids[int(track_id)] = int(visitor_id)

    def is_track_known(self, track_id: int) -> bool:
        with self._lock:
            return int(track_id) in self.known_bot_sort_ids

    def known_track_ids_snapshot(self) -> set:
        """Lock-protected snapshot of all currently-bound track ids.

        Used by the inference loop to compute the co-visible "eligible"
        pool — a track that's already been resolved is always
        trustworthy for exclusion, even if its stability count hasn't
        crossed the recent-frames threshold yet.
        """
        with self._lock:
            return set(self.known_bot_sort_ids.keys())

    def visitor_ids_for_tracks(self, track_ids: Iterable[int]) -> set:
        """Return the set of visitor_ids currently bound to the given
        track_ids.

        Used as the ``excluded_visitor_ids`` input to :meth:`search` —
        when track A and track B are visible in the same frame, track B
        is mathematically forbidden from matching whichever visitor_id
        track A is already bound to.
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
            self._body_embs.clear()
            self._last_seen.clear()
            self._last_snapshot_at.clear()
            self._last_seen_wall.clear()
            self.known_bot_sort_ids.clear()
            # Persist the empty state so a restart inside the same day
            # doesn't load yesterday's identity back in.
            self._save_to_disk_locked()
        logger.info(
            "DailyFaceCache reset — face index, body library, "
            "snapshot timestamps, and track set cleared",
        )

    # ── Persistence (caller MUST hold self._lock) ────────────────────────

    def _save_to_disk_locked(self) -> None:
        """Atomic pickle write of today's identity cache.

        ``_last_seen`` is a ``time.monotonic()`` value, which has no
        meaning across process restarts. We store each visitor's
        last-seen age (seconds-ago at save time) and reconstruct on
        load by subtracting from the fresh process's monotonic clock.

        Schema is versioned (``_SCHEMA_VERSION``): on load, a mismatch
        causes a fresh-start — the v1 (HSV) → v2 (OSNet) transition
        consequently wipes any stale histogram data automatically.

        Failures are logged and swallowed — we'd rather drop a write
        than crash the vision pipeline.
        """
        try:
            save_mono = time.monotonic()
            last_seen_age = [max(0.0, save_mono - t) for t in self._last_seen]
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "date": date.today().isoformat(),
                "face_index_bytes": bytes(faiss.serialize_index(self._face_index)),
                "face_to_visitor": list(self._face_row_to_visitor),
                # v2 schema: per-visitor list of 512-d OSNet embeddings.
                "body_libraries": [
                    [h.copy() for h in lib] for lib in self._body_embs
                ],
                "last_seen_age": last_seen_age,
                "last_snapshot_at": list(self._last_snapshot_at),
                "last_seen_wall": list(self._last_seen_wall),
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
        """Restore today's identity cache from disk if date AND schema match.

        Date mismatch OR schema mismatch → ignore the file. A v1 pickle
        from the HSV era is silently dropped; the next mutation will
        overwrite it with a v2 payload. This is the same safety net
        the daily-rollover already relied on for stale-date files.
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

        stored_schema = int(data.get("schema_version", 1))
        if stored_schema != _SCHEMA_VERSION:
            logger.info(
                "DailyFaceCache: stored schema=%d ≠ current=%d (v1 HSV → v2 "
                "OSNet) — discarding legacy cache and starting fresh.",
                stored_schema, _SCHEMA_VERSION,
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
            self._body_embs = [
                [np.asarray(h, dtype=np.float32).copy() for h in lib]
                for lib in libs
            ]
            ages = [float(a) for a in data.get("last_seen_age", [])]
            now_mono = time.monotonic()
            self._last_seen = [now_mono - a for a in ages]
            stored_snaps = [float(t) for t in data.get("last_snapshot_at", [])]
            self._last_snapshot_at = list(stored_snaps)
            while len(self._last_snapshot_at) < len(self._body_embs):
                self._last_snapshot_at.append(0.0)  # ancient → past cooldown
            stored_seen = [float(t) for t in data.get("last_seen_wall", [])]
            self._last_seen_wall = list(stored_seen)
            now_wall = time.time()
            while len(self._last_seen_wall) < len(self._body_embs):
                self._last_seen_wall.append(now_wall)
            raw_known = data.get("known_bot_sort_ids", {})
            if isinstance(raw_known, dict):
                self.known_bot_sort_ids = {
                    int(k): int(v) for k, v in raw_known.items()
                }
            else:
                self.known_bot_sort_ids = {}
            logger.info(
                "DailyFaceCache: restored %d visitor(s), %d face entries from %s",
                len(self._body_embs), int(self._face_index.ntotal), self._state_file,
            )
        except Exception:
            logger.exception(
                "DailyFaceCache: failed to restore from %s — starting fresh.",
                self._state_file,
            )
            # Force a clean state and rewrite so the bad file is replaced.
            self._face_index = faiss.IndexFlatIP(self._dim)
            self._face_row_to_visitor = []
            self._body_embs = []
            self._last_seen = []
            self._last_snapshot_at = []
            self._last_seen_wall = []
            self.known_bot_sort_ids = {}
            self._save_to_disk_locked()
