"""BodyEngine — OSNet-x0_25 body re-identification embeddings.

Phase D Tier 1.1 — replaces the prior HSV-histogram body signal.

Why OSNet over HSV
------------------
HSV (hue, saturation) histograms of the torso were producing same-person
and different-person *correlation* distributions that overlapped heavily
on CCTV-quality crops: two people in similar light-coloured shirts
landed within 0.05 of the same-person mode. That overlap is the source
of the "tug-of-war" between false-splits and false-merges — no single
threshold on a 30×32 histogram can separate two overlapping
distributions.

OSNet-x0_25 (Zhou et al., 2019) was trained on Market-1501 and
DukeMTMC-reID — surveillance-quality body crops. Its 512-d embedding
produces *separable* same- vs. different-person cosine distributions
on the kind of crops this pipeline sees. The C2 / C3 rules in
vector_store can therefore ride a clean signal instead of a noisy one.

Inference cost
--------------
osnet_x0_25 is the smallest variant in the family (~0.2 M params).
Expected per-crop forward pass on CPU: ~25-50 ms. The face worker
queue absorbs this — the camera thread is unaffected.

Threading
---------
``torchreid.FeatureExtractor`` is a thin wrapper over a PyTorch model.
We hold a single instance and rely on the GIL to serialize calls from
the (single) face-worker thread. If a second worker thread is added in
the future the model still works (PyTorch eval-mode is reentrant) but
inference time per call may bunch under contention.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Default location of the OSNet weights. ``setup_env.sh`` / model-download
#: tooling should drop both Market-1501 and ImageNet variants into
#: ``<project_root>/models/``. The Market-1501 weights are preferred —
#: they were fine-tuned on surveillance-style person crops, which is the
#: deployment distribution.
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
DEFAULT_OSNET_WEIGHTS: str = os.path.join(
    _PROJECT_ROOT, "models", "osnet_x0_25_market1501.pth",
)

#: OSNet was trained at this canonical body-crop resolution
#: (height, width). FeatureExtractor handles the resize internally.
OSNET_INPUT_SIZE: Tuple[int, int] = (256, 128)

#: Embedding dimensionality. osnet_x0_25's penultimate FC layer
#: produces 512-d features (matching ArcFace) — parallel storage to
#: the face index falls out for free.
OSNET_EMBED_DIM: int = 512


class BodyEngine:
    """Body re-identification via OSNet-x0_25.

    Lifecycle mirrors :class:`FaceEngine`:
        * cheap construction (no model load)
        * :meth:`initialize` actually loads weights — call once from
          :meth:`VisionService.start`
        * :meth:`compute_body_embedding` is the per-frame hot path

    Returns ``None`` on degenerate crops (zero-area, too-narrow torso)
    so the caller can ``return`` without taking the worker further —
    identical contract to the old ``compute_body_histogram``.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_OSNET_WEIGHTS,
        device: str = "cpu",
    ) -> None:
        self._model_path: str = model_path
        self._device: str = device
        self._extractor = None
        self._lock: threading.Lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._extractor is not None

    @property
    def embed_dim(self) -> int:
        return OSNET_EMBED_DIM

    def initialize(self) -> bool:
        """Load the OSNet model weights. Returns True on success.

        Idempotent — repeat calls are no-ops once the model is loaded.
        Errors are caught and logged; on failure ``is_ready`` stays
        False so the caller can decide whether to run a degraded
        pipeline (face-only) or abort.
        """
        if self._extractor is not None:
            return True
        try:
            # Imported lazily so unit tests that don't exercise the
            # vision path don't pay the multi-second torchreid import
            # cost just to load the module.
            from torchreid.reid.utils import FeatureExtractor

            if not os.path.exists(self._model_path):
                logger.error(
                    "BodyEngine: weights not found at %s — body re-id disabled",
                    self._model_path,
                )
                return False

            self._extractor = FeatureExtractor(
                model_name="osnet_x0_25",
                model_path=self._model_path,
                device=self._device,
                image_size=OSNET_INPUT_SIZE,
                verbose=False,
            )
            logger.info(
                "BodyEngine ready — model=osnet_x0_25 device=%s weights=%s",
                self._device, os.path.basename(self._model_path),
            )
            return True
        except Exception:
            logger.exception("BodyEngine: initialize failed")
            self._extractor = None
            return False

    def compute_body_embedding(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Return a 512-d L2-normalized body embedding for the torso region.

        ``person_bbox`` is ``(x1, y1, x2, y2)`` in absolute frame
        coordinates — same format the old HSV path consumed. The torso
        is sliced as the middle 80 % horizontally and 20–70 %
        vertically to skip the head (face_engine owns face evidence)
        and the legs/feet (often background-dominated).

        Returns ``None`` on degenerate crops or if the model is not
        initialized — the caller treats this identically to the old
        "no body signal available" path.
        """
        if self._extractor is None:
            return None
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
        ty1 = y1 + int(0.20 * bh)
        ty2 = y1 + int(0.70 * bh)
        # OSNet's input is 256×128 (h×w). Below ~32×16 the crop has no
        # useful body content; reject rather than upsampling noise.
        if tx2 - tx1 < 16 or ty2 - ty1 < 32:
            return None

        torso_bgr = frame[ty1:ty2, tx1:tx2]
        if torso_bgr.size == 0:
            return None

        try:
            # torchreid's FeatureExtractor accepts a list of numpy
            # arrays in RGB order. Convert once here; the extractor
            # handles the resize + ImageNet normalization internally.
            torso_rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                feats = self._extractor([torso_rgb])
            emb = feats.detach().cpu().numpy().reshape(-1).astype(np.float32)
            if emb.size != OSNET_EMBED_DIM:
                logger.warning(
                    "BodyEngine: unexpected embedding dim %d (expected %d)",
                    emb.size, OSNET_EMBED_DIM,
                )
                return None
            n = float(np.linalg.norm(emb))
            if n < 1e-9:
                return None
            return emb / n
        except Exception:
            logger.debug("compute_body_embedding failed", exc_info=True)
            return None
