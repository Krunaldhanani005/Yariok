"""FaceEngine — InsightFace wrapper for face detection, quality, and embedding.

Pipeline contract (called from VisionService):
    person_bbox -> get_face(frame, bbox) -> face_crop, raw_face_obj
                -> check_quality(face_crop) -> Laplacian variance
                -> get_embedding(raw_face_obj) -> L2-normalized 512-d vector

The engine itself owns no detection state. It is a stateless transformer
that turns pixel regions into vectors. All decisions (gate thresholds,
"is this a new person?", etc.) live in VisionService.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """A face detected inside a person crop.

    Attributes:
        crop: BGR pixel ndarray of the face region (for quality + base64).
        embedding: Raw 512-d feature vector (NOT yet L2-normalized).
        det_score: InsightFace detector confidence.
    """
    crop: np.ndarray
    embedding: np.ndarray
    det_score: float


class FaceEngine:
    """Thin wrapper around InsightFace ``buffalo_l``.

    Defers the heavy ``FaceAnalysis`` initialization until ``initialize()``
    is called — this avoids paying the model-load cost at import time and
    lets the calling service surface a clean "loading AI" state.

    GPU selection:
        ctx_id=0 attempts CUDA. InsightFace silently falls back to CPU if
        the CUDAExecutionProvider is not installed, so a single ctx_id
        works on both GPU and CPU hosts.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: Tuple[int, int] = (640, 640),
        blur_threshold: float = 100.0,
    ) -> None:
        self._model_name: str = model_name
        self._ctx_id: int = ctx_id
        self._det_size: Tuple[int, int] = det_size
        self.blur_threshold: float = blur_threshold

        self._app = None
        self._ready: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Load the InsightFace model. Returns True on success.

        We restrict loading to just ``detection`` (SCRFD det_10g) and
        ``recognition`` (ArcFace w600k_r50, the 512-d embedder). The
        ``buffalo_l`` pack also ships ``landmark_3d_68`` (1k3d68.onnx,
        ~137 MB) and ``genderage`` (genderage.onnx) which this pipeline
        never uses — leaving them off saves ~138 MB of RAM and shaves
        a chunk off model-load time.
        """
        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=self._model_name,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
            self._ready = True
            logger.info(
                "FaceEngine ready — model=%s ctx_id=%d det_size=%s modules=[detection, recognition]",
                self._model_name, self._ctx_id, self._det_size,
            )
            return True
        except Exception as exc:
            self._ready = False
            logger.exception("FaceEngine failed to initialize: %s", exc)
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Quality filter (Gate 2) ──────────────────────────────────────────────

    @staticmethod
    def check_quality(face_crop: np.ndarray) -> float:
        """Return the Laplacian variance of ``face_crop``.

        Higher = sharper. A common threshold for "non-blurry" is ~100.
        Returns 0.0 for empty or invalid crops so they are unambiguously
        rejected by the caller's threshold check.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0
        if face_crop.ndim == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── Face detection (used before Gates 2-3) ───────────────────────────────

    def get_face(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[float, float, float, float],
        padding: float = 0.08,
        head_fraction: float = 0.40,
    ) -> Optional[DetectedFace]:
        """Detect the dominant face inside the HEAD region of a YOLO person bbox.

        Why we crop to the head region before running InsightFace:
            CCTV scenes routinely contain other faces inside a person's
            bbox — monitors, posters, photographs, reflections in glass,
            mirrors. InsightFace will dutifully detect those and return
            the largest one, which is often NOT the actual visitor's
            face. Each frame then picks a different "face", embeddings
            drift, and the same person is re-registered repeatedly.

            By restricting detection to the top ``head_fraction`` of the
            person bbox we eliminate that class of failure entirely:
            screens, posters, and reflections are almost always BELOW
            shoulder level in CCTV framing.

        Args:
            frame: Full BGR frame.
            person_bbox: (x1, y1, x2, y2) in pixel coordinates of ``frame``.
            padding: Fractional padding applied to the person bbox before
                cropping (helps when a head is right at the top edge).
            head_fraction: Fraction of the bbox height (from the top)
                kept for face detection. 0.55 covers head + neck + a bit
                of upper torso — enough slack for tall hats / hair but
                excludes laptop screens, table-top monitors, posters.

        Returns:
            DetectedFace or ``None`` if nothing valid is found.
        """
        if not self._ready or self._app is None:
            return None
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person_bbox

        # Pad horizontally, but constrain vertically to the head region.
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        px = padding * bw
        py = padding * bh
        cx1 = max(0, int(x1 - px))
        cx2 = min(w, int(x2 + px))
        # Top edge: small upward pad so the top of head is preserved.
        cy1 = max(0, int(y1 - py))
        # Bottom edge: cut at head_fraction of the bbox height.
        cy2 = min(h, int(y1 + bh * head_fraction))
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        person_crop = frame[cy1:cy2, cx1:cx2]
        if person_crop.size == 0:
            return None

        try:
            faces = self._app.get(person_crop)
        except Exception as exc:
            logger.debug("InsightFace.get() raised: %s", exc)
            return None

        if not faces:
            return None

        # Hard filters to reject screen / poster / reflection faces:
        #   - Minimum absolute face size (ArcFace embeddings degrade
        #     rapidly below 50px; small faces detected inside the head
        #     crop are typically monitors or photos, not the person).
        #   - Face top edge must be in the TOP 50% of the head crop
        #     (a real head touches the top; faces lower than that are
        #     almost always non-human pickup).
        crop_h = max(1.0, float(person_crop.shape[0]))
        crop_w = max(1.0, float(person_crop.shape[1]))
        crop_area = crop_h * crop_w
        MIN_FACE_SIZE_PX = 50.0
        MIN_FACE_AREA_FRAC = 0.02   # face must be ≥ 2 % of head-crop area
        MAX_FACE_TOP_FRAC = 0.50    # face top edge must be in top half of crop

        def _passes_filters(f) -> bool:
            fx1, fy1, fx2, fy2 = f.bbox
            fw, fh = max(0.0, fx2 - fx1), max(0.0, fy2 - fy1)
            if fw < MIN_FACE_SIZE_PX or fh < MIN_FACE_SIZE_PX:
                return False
            if (fw * fh) / crop_area < MIN_FACE_AREA_FRAC:
                return False
            if fy1 / crop_h > MAX_FACE_TOP_FRAC:
                return False
            return True

        valid = [f for f in faces if _passes_filters(f)]
        if not valid:
            return None

        # Among valid candidates, pick the largest. Topmost preference is
        # already covered by the position filter above.
        def _area(f) -> float:
            fx1, fy1, fx2, fy2 = f.bbox
            return max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)

        face = max(valid, key=_area)

        # Crop the face region from the person crop.
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        fx1 = max(0, fx1)
        fy1 = max(0, fy1)
        fx2 = min(person_crop.shape[1], fx2)
        fy2 = min(person_crop.shape[0], fy2)
        if fx2 <= fx1 or fy2 <= fy1:
            return None

        face_crop = person_crop[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            return None

        embedding = getattr(face, "embedding", None)
        if embedding is None:
            return None

        return DetectedFace(
            crop=face_crop.copy(),
            embedding=np.asarray(embedding, dtype=np.float32),
            det_score=float(getattr(face, "det_score", 0.0)),
        )

    # ── Body / clothing histogram (Gate 2.5) ─────────────────────────────────

    @staticmethod
    def compute_body_histogram(
        frame: np.ndarray,
        person_bbox: Tuple[float, float, float, float],
        bins: Tuple[int, int] = (30, 32),
    ) -> Optional[np.ndarray]:
        """Compute an HSV (hue, saturation) histogram of the torso region.

        Why a separate body signature on top of the ArcFace embedding:
            CCTV faces are tiny, often off-axis, and frequently occluded.
            The face embedding is *high precision when it fires* but
            doesn't fire often enough. Clothing color, in contrast, is
            available every single frame and is highly stable for the
            duration of a single visitor's day.

        We slice the bbox to keep only the torso strip (middle 80%
        horizontally, 20%–70% vertically) to avoid:
            - the head region (similar skin/hair colors between people),
            - the legs/floor region (background-dominated near the feet).

        Returns ``None`` for empty or clipped-to-zero crops.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person_bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None

        bw, bh = x2 - x1, y2 - y1
        tx1 = x1 + int(0.10 * bw)
        tx2 = x2 - int(0.10 * bw)
        ty1 = y1 + int(0.20 * bh)   # skip head
        ty2 = y1 + int(0.70 * bh)   # skip legs / feet
        if tx2 <= tx1 or ty2 <= ty1:
            return None

        torso = frame[ty1:ty2, tx1:tx2]
        if torso.size == 0:
            return None

        try:
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist.astype(np.float32)
        except Exception:
            logger.debug("compute_body_histogram failed", exc_info=True)
            return None

    # ── Embedding (Gate 3) ───────────────────────────────────────────────────

    @staticmethod
    def get_embedding(face: DetectedFace) -> np.ndarray:
        """Return the L2-normalized 512-d embedding from a DetectedFace.

        InsightFace ``buffalo_l`` already produces 512-d vectors, but does
        not guarantee unit norm — the FAISS IndexFlatIP requires unit norm
        for inner-product to equal cosine similarity, so we normalize here.
        """
        vec = np.asarray(face.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return vec
        return vec / norm
