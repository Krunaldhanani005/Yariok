"""Drives the *decision tree* inside VisionService._handle_face_job.

The vision worker reduces to a fairly simple state machine once you
mock out the FAISS lookup and the face-engine I/O. These tests bolt
fake stand-ins onto a real VisionService instance and verify each
branch of the Policy B+ flow:

    1. First-sighting in-zone     → publish with should_greet=True
    2. First-sighting out-of-zone → publish with should_greet=False
    3. Faceless candidate         → defer, no publish, not marked known
    4. Known + within cooldown    → silent re-ID, no publish
    5. Known + cooldown elapsed   → publish, should_greet=False
"""

import unittest
from unittest import mock

import numpy as np

from backend.services.vision.vision_service import (
    SNAPSHOT_COOLDOWN_SECONDS,
    VisionService,
    _FaceJob,
)
from backend.services.vision.vector_store import FaceMatch


def _frame(w=1280, h=720):
    """Tiny stand-in for an RTSP frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _job(track_id=1, is_in_zone=True):
    return _FaceJob(
        frame=_frame(), track_id=track_id,
        bbox=(100.0, 100.0, 300.0, 600.0),
        is_in_zone=is_in_zone,
    )


def _make_service():
    """Build a VisionService that does NOT touch any real I/O.

    We bypass the constructor's side-effects (reload_zone reading from
    disk, FAISS init, MQTT init) by patching them, then replace the
    face engine + cache + publisher with mocks the tests control.
    """
    with mock.patch.object(VisionService, "reload_zone"):
        svc = VisionService(rtsp_url="rtsp://test", mqtt_client=mock.Mock())
    svc._face_engine = mock.Mock()
    svc._face_cache = mock.Mock()
    svc._publish_event = mock.Mock()
    # The face engine helpers used by _handle_face_job
    svc._face_engine.compute_body_histogram.return_value = np.ones(
        (30, 32), dtype=np.float32,
    )
    svc._face_engine.get_face.return_value = None     # default: no face
    svc._face_engine.check_quality.return_value = 200.0
    svc._face_engine.get_embedding.return_value = (
        np.ones(512, dtype=np.float32) / np.sqrt(512)
    )
    svc._face_cache.is_track_known.return_value = False
    return svc


class TestFaceJobPolicy(unittest.TestCase):

    # ── 1. First sighting in zone ────────────────────────────────────────

    def test_first_sighting_in_zone_publishes_with_greet(self):
        svc = _make_service()
        # Stand-in DetectedFace
        fake_face = mock.Mock(crop=np.zeros((80, 80, 3), dtype=np.uint8))
        svc._face_engine.get_face.return_value = fake_face
        # search → no match → first sighting
        svc._face_cache.search.return_value = FaceMatch(
            matched=False, score=0.0, index=-1, kind="miss",
            body_score=0.05, face_score=0.05,
        )
        svc._face_cache.add_visitor.return_value = 7   # arbitrary new visitor id

        svc._handle_face_job(_job(track_id=42, is_in_zone=True))

        svc._face_cache.add_visitor.assert_called_once()
        # Phase C v3: mark_track_known now requires the resolved
        # visitor_id so the cache can answer "which visitor is this
        # track bound to" for the co-occurrence guard.
        svc._face_cache.mark_track_known.assert_called_with(42, 7)
        svc._publish_event.assert_called_once()
        payload = svc._publish_event.call_args[0][0]
        self.assertEqual(payload["type"], "visitor_snapshot")
        self.assertEqual(payload["visitor_id"], 7)
        self.assertEqual(payload["track_id"], 42)
        self.assertTrue(payload["is_first_sighting"])
        self.assertTrue(payload["should_greet"])
        self.assertTrue(payload["in_zone"])

    # ── 2. First sighting OUTSIDE zone ───────────────────────────────────

    def test_first_sighting_outside_zone_no_greet(self):
        svc = _make_service()
        svc._face_engine.get_face.return_value = mock.Mock(
            crop=np.zeros((80, 80, 3), dtype=np.uint8),
        )
        svc._face_cache.search.return_value = FaceMatch(
            matched=False, score=0.0, index=-1, kind="miss",
        )
        svc._face_cache.add_visitor.return_value = 3

        svc._handle_face_job(_job(track_id=9, is_in_zone=False))

        svc._publish_event.assert_called_once()
        payload = svc._publish_event.call_args[0][0]
        self.assertTrue(payload["is_first_sighting"])
        self.assertFalse(payload["should_greet"])   # ← key: no greet outside zone
        self.assertFalse(payload["in_zone"])

    # ── 3. Faceless candidate — defer ───────────────────────────────────

    def test_faceless_candidate_defers(self):
        svc = _make_service()
        # No face detected at all
        svc._face_engine.get_face.return_value = None
        svc._face_cache.search.return_value = FaceMatch(
            matched=False, score=0.2, index=-1, kind="miss",
        )

        svc._handle_face_job(_job(track_id=11))

        # No visitor added, no track marked, no publish.
        svc._face_cache.add_visitor.assert_not_called()
        svc._face_cache.mark_track_known.assert_not_called()
        svc._publish_event.assert_not_called()

    # ── 4. Known visitor, within cooldown — silent ──────────────────────

    def test_known_within_cooldown_is_silent(self):
        svc = _make_service()
        svc._face_cache.search.return_value = FaceMatch(
            matched=True, score=0.8, index=2, kind="body",
            body_score=0.8, face_score=0.0,
        )
        # Last snapshot 5 seconds ago — well within the 600s cooldown
        svc._face_cache.time_since_snapshot.return_value = 5.0

        svc._handle_face_job(_job(track_id=21))

        # Phase C v3: mark_track_known now takes the resolved visitor_id
        # (match.index == 2 in this fixture) so the co-occurrence guard
        # can answer "which visitor is this track bound to?".
        svc._face_cache.mark_track_known.assert_called_with(21, 2)
        # Track marked known, but NO snapshot publish.
        svc._publish_event.assert_not_called()
        svc._face_cache.add_visitor.assert_not_called()
        svc._face_cache.mark_snapshot.assert_not_called()

    # ── 5. Known visitor, cooldown elapsed — refresh snapshot, no greet ─

    def test_known_after_cooldown_refreshes_snapshot_no_greet(self):
        svc = _make_service()
        svc._face_cache.search.return_value = FaceMatch(
            matched=True, score=0.7, index=2, kind="body",
            body_score=0.7, face_score=0.0,
        )
        # 11 minutes since last snapshot → well past 600s cooldown
        svc._face_cache.time_since_snapshot.return_value = SNAPSHOT_COOLDOWN_SECONDS + 60

        svc._handle_face_job(_job(track_id=30, is_in_zone=True))

        svc._face_cache.mark_snapshot.assert_called_with(2)
        svc._publish_event.assert_called_once()
        payload = svc._publish_event.call_args[0][0]
        self.assertFalse(payload["is_first_sighting"])
        self.assertFalse(payload["should_greet"])   # ← never greet a known visitor
        self.assertEqual(payload["visitor_id"], 2)

    # ── 6. Gate 1 — already-known track id short-circuits early ─────────

    def test_already_known_track_id_returns_immediately(self):
        svc = _make_service()
        svc._face_cache.is_track_known.return_value = True

        svc._handle_face_job(_job(track_id=55))

        # No engine work happened, no search, no publish.
        svc._face_engine.compute_body_histogram.assert_not_called()
        svc._face_cache.search.assert_not_called()
        svc._publish_event.assert_not_called()

    # ── 7. Gate 2 — empty body crop short-circuits ──────────────────────

    def test_empty_body_crop_returns(self):
        svc = _make_service()
        svc._face_engine.compute_body_histogram.return_value = None

        svc._handle_face_job(_job(track_id=66))

        # Face extraction is skipped, no search runs.
        svc._face_engine.get_face.assert_not_called()
        svc._face_cache.search.assert_not_called()
        svc._publish_event.assert_not_called()


class TestCoOccurrenceExclusion(unittest.TestCase):
    """Phase C v3 — simultaneously-visible people MUST NOT merge.

    The vision worker is supposed to:
      1. Ask the cache for the visitor_ids bound to OTHER active tracks.
      2. Pass that set as ``excluded_visitor_ids`` to ``search()``.
    These tests verify both halves of the contract.
    """

    def test_co_visible_track_ids_resolved_to_visitor_ids(self):
        """The worker queries visitor_ids_for_tracks(co_visible) and
        feeds the result into search()."""
        svc = _make_service()
        svc._face_engine.get_face.return_value = mock.Mock(
            crop=np.zeros((80, 80, 3), dtype=np.uint8),
        )
        # Stand-in: tracks 8 and 9 are already bound to visitors 1 and 5.
        svc._face_cache.visitor_ids_for_tracks.return_value = {1, 5}
        svc._face_cache.search.return_value = FaceMatch(
            matched=False, score=0.0, index=-1, kind="miss",
        )
        svc._face_cache.add_visitor.return_value = 12

        job = _FaceJob(
            frame=_frame(), track_id=99,
            bbox=(100.0, 100.0, 300.0, 600.0),
            is_in_zone=True, co_visible_track_ids=(8, 9),
        )
        svc._handle_face_job(job)

        # The cache was asked which visitors those co-visible tracks
        # belong to.
        svc._face_cache.visitor_ids_for_tracks.assert_called_with((8, 9))
        # And search() received that set as the exclusion list.
        svc._face_cache.search.assert_called_once()
        kwargs = svc._face_cache.search.call_args.kwargs
        self.assertEqual(kwargs.get("excluded_visitor_ids"), {1, 5})

    def test_no_co_visible_tracks_passes_empty_exclusion(self):
        svc = _make_service()
        svc._face_engine.get_face.return_value = mock.Mock(
            crop=np.zeros((80, 80, 3), dtype=np.uint8),
        )
        svc._face_cache.visitor_ids_for_tracks.return_value = set()
        svc._face_cache.search.return_value = FaceMatch(
            matched=False, score=0.0, index=-1, kind="miss",
        )
        svc._face_cache.add_visitor.return_value = 0

        svc._handle_face_job(_job(track_id=1))  # no co_visible

        svc._face_cache.search.assert_called_once()
        kwargs = svc._face_cache.search.call_args.kwargs
        self.assertEqual(kwargs.get("excluded_visitor_ids"), set())


class TestFaceJobInZoneFlag(unittest.TestCase):
    """Verify ``is_in_zone`` flows from camera → worker → MQTT payload."""

    def test_in_zone_true_propagates(self):
        svc = _make_service()
        svc._face_engine.get_face.return_value = mock.Mock(
            crop=np.zeros((80, 80, 3), dtype=np.uint8),
        )
        svc._face_cache.search.return_value = FaceMatch(
            matched=False, score=0.0, index=-1, kind="miss",
        )
        svc._face_cache.add_visitor.return_value = 1
        svc._handle_face_job(_job(is_in_zone=True))
        payload = svc._publish_event.call_args[0][0]
        self.assertTrue(payload["in_zone"])

    def test_in_zone_false_propagates(self):
        svc = _make_service()
        svc._face_engine.get_face.return_value = mock.Mock(
            crop=np.zeros((80, 80, 3), dtype=np.uint8),
        )
        svc._face_cache.search.return_value = FaceMatch(
            matched=False, score=0.0, index=-1, kind="miss",
        )
        svc._face_cache.add_visitor.return_value = 1
        svc._handle_face_job(_job(is_in_zone=False))
        payload = svc._publish_event.call_args[0][0]
        self.assertFalse(payload["in_zone"])


if __name__ == "__main__":
    unittest.main()
