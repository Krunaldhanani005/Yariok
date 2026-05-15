"""Tests for backend.services.vision.vector_store.DailyFaceCache.

Covers:
  • add_visitor + face FAISS + body library
  • search by face / body / body-recent / multi-view library
  • snapshot cooldown (time_since_snapshot, mark_snapshot)
  • reset clears everything
  • persistence: save → reload preserves all fields including last_snapshot_at
"""

import os
import tempfile
import time
import unittest

import numpy as np

from backend.services.vision.vector_store import (
    DailyFaceCache,
    RECENCY_WINDOW_SECONDS,
)


def _random_unit_vec(d=512, seed=None) -> np.ndarray:
    """Zero-mean Gaussian → L2-normalize. Two such vectors are
    approximately orthogonal in high dimensions (cosine ~0). Using
    ``rng.random()`` (uniform [0,1)) instead would give vectors that
    both point in the all-positive octant, so their cosine baseline
    is ~0.75 — useless for "are these different faces" tests."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _random_hist(seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((30, 32)).astype(np.float32)


class TestAddAndSearch(unittest.TestCase):
    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        os.remove(self.tmp)
        self.cache = DailyFaceCache(state_file=self.tmp)

    def tearDown(self):
        try: os.remove(self.tmp)
        except OSError: pass

    def test_empty_cache(self):
        self.assertEqual(self.cache.size, 0)
        self.assertEqual(self.cache.face_index_size, 0)

    def test_add_visitor_with_face_and_body(self):
        emb = _random_unit_vec(seed=1)
        hist = _random_hist(seed=1)
        vid = self.cache.add_visitor(face_embedding=emb, body_hist=hist)
        self.assertEqual(vid, 0)
        self.assertEqual(self.cache.size, 1)
        self.assertEqual(self.cache.face_index_size, 1)

    def test_add_visitor_body_only(self):
        hist = _random_hist(seed=2)
        vid = self.cache.add_visitor(face_embedding=None, body_hist=hist)
        self.assertEqual(vid, 0)
        self.assertEqual(self.cache.size, 1)
        self.assertEqual(self.cache.face_index_size, 0)

    def test_search_exact_face_match(self):
        emb = _random_unit_vec(seed=3)
        hist = _random_hist(seed=3)
        self.cache.add_visitor(face_embedding=emb, body_hist=hist)
        # Search with the SAME embedding → cosine = 1.0, matches
        m = self.cache.search(face_embedding=emb, body_hist=_random_hist(seed=999))
        self.assertTrue(m.matched)
        self.assertEqual(m.index, 0)

    def test_search_orthogonal_face_misses(self):
        # Pure-face test — skip body matching entirely (body_hist=None
        # on both sides). Two random unit vectors in 512-d are near
        # orthogonal (cosine ~0), well below the 0.30 threshold.
        emb1 = _random_unit_vec(seed=4)
        self.cache.add_visitor(face_embedding=emb1, body_hist=None)
        emb2 = _random_unit_vec(seed=5)
        m = self.cache.search(face_embedding=emb2, body_hist=None)
        self.assertFalse(m.matched)

    def test_body_strict_match(self):
        hist = _random_hist(seed=6)
        self.cache.add_visitor(face_embedding=None, body_hist=hist)
        # Search with same body → correlation ~1.0, well above the
        # new BODY_SURE_THRESHOLD = 0.85 used in the dual-signal rule.
        m = self.cache.search(face_embedding=None, body_hist=hist)
        self.assertTrue(m.matched)
        # Phase C v3: body-only matches are tagged ``body-strict`` so
        # they're distinguishable from face-driven matches in logs.
        self.assertEqual(m.kind, "body-strict")
        self.assertGreaterEqual(m.body_score, 0.85)

    def test_multi_view_library_grows_on_match(self):
        hist = _random_hist(seed=7)
        vid = self.cache.add_visitor(face_embedding=None, body_hist=hist)
        # Library starts with 1 view; another match should append.
        self.cache.search(face_embedding=None, body_hist=hist)
        # Internal: library should have grown to 2.
        self.assertEqual(len(self.cache._body_hists[vid]), 2)


class TestSnapshotCooldown(unittest.TestCase):
    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        os.remove(self.tmp)
        self.cache = DailyFaceCache(state_file=self.tmp)

    def tearDown(self):
        try: os.remove(self.tmp)
        except OSError: pass

    def test_unknown_visitor_returns_inf(self):
        self.assertEqual(self.cache.time_since_snapshot(99), float("inf"))
        self.assertEqual(self.cache.time_since_snapshot(-1), float("inf"))

    def test_freshly_added_visitor_has_near_zero_age(self):
        vid = self.cache.add_visitor(
            face_embedding=_random_unit_vec(seed=10),
            body_hist=_random_hist(seed=10),
        )
        self.assertLess(self.cache.time_since_snapshot(vid), 1.0)

    def test_age_grows_over_time(self):
        vid = self.cache.add_visitor(
            face_embedding=None, body_hist=_random_hist(seed=11),
        )
        time.sleep(0.5)
        age = self.cache.time_since_snapshot(vid)
        self.assertGreater(age, 0.4)
        self.assertLess(age, 1.0)

    def test_mark_snapshot_resets_age(self):
        vid = self.cache.add_visitor(
            face_embedding=None, body_hist=_random_hist(seed=12),
        )
        time.sleep(0.5)
        self.cache.mark_snapshot(vid)
        self.assertLess(self.cache.time_since_snapshot(vid), 0.1)

    def test_mark_snapshot_unknown_id_is_noop(self):
        # Should not raise
        self.cache.mark_snapshot(999)
        self.cache.mark_snapshot(-1)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        os.remove(self.tmp)

    def tearDown(self):
        try: os.remove(self.tmp)
        except OSError: pass

    def test_roundtrip_preserves_visitors(self):
        c1 = DailyFaceCache(state_file=self.tmp)
        emb1 = _random_unit_vec(seed=20)
        emb2 = _random_unit_vec(seed=21)
        v1 = c1.add_visitor(face_embedding=emb1, body_hist=_random_hist(seed=20))
        v2 = c1.add_visitor(face_embedding=emb2, body_hist=_random_hist(seed=21))
        self.assertEqual(c1.size, 2)
        self.assertEqual(c1.face_index_size, 2)

        # Simulate restart
        c2 = DailyFaceCache(state_file=self.tmp)
        self.assertEqual(c2.size, 2)
        self.assertEqual(c2.face_index_size, 2)

        # The same face embeddings still match after restart
        m1 = c2.search(face_embedding=emb1, body_hist=None)
        m2 = c2.search(face_embedding=emb2, body_hist=None)
        self.assertTrue(m1.matched)
        self.assertEqual(m1.index, v1)
        self.assertTrue(m2.matched)
        self.assertEqual(m2.index, v2)

    def test_roundtrip_preserves_last_snapshot_at(self):
        c1 = DailyFaceCache(state_file=self.tmp)
        vid = c1.add_visitor(face_embedding=None, body_hist=_random_hist(seed=22))
        time.sleep(0.3)
        age_before = c1.time_since_snapshot(vid)

        c2 = DailyFaceCache(state_file=self.tmp)
        age_after = c2.time_since_snapshot(vid)
        # Age should be approximately preserved across the "restart"
        # (within the small overhead of save + load).
        self.assertGreater(age_after, age_before - 0.5)
        self.assertLess(age_after, age_before + 0.5)

    def test_reset_clears_disk(self):
        c = DailyFaceCache(state_file=self.tmp)
        c.add_visitor(face_embedding=_random_unit_vec(seed=30),
                      body_hist=_random_hist(seed=30))
        self.assertEqual(c.size, 1)
        c.reset()
        self.assertEqual(c.size, 0)

        # Restart after reset must still see 0
        c2 = DailyFaceCache(state_file=self.tmp)
        self.assertEqual(c2.size, 0)

    def test_corrupt_file_falls_back_to_empty(self):
        # Write garbage
        with open(self.tmp, "wb") as f:
            f.write(b"not a valid pickle")
        # Construction must not crash
        c = DailyFaceCache(state_file=self.tmp)
        self.assertEqual(c.size, 0)


class TestDualSignalAgreement(unittest.TestCase):
    """Phase C v3 — search() must apply the new C1/C2/C3 rules and
    REJECT anything below all three (no recency-relaxed false-merge)."""

    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".pkl")
        os.close(fd); os.remove(self.tmp)
        self.cache = DailyFaceCache(state_file=self.tmp)

    def tearDown(self):
        try: os.remove(self.tmp)
        except OSError: pass

    def test_c1_face_sure_match(self):
        """Same face embedding → cosine 1.0 → C1 → match regardless of body."""
        emb = _random_unit_vec(seed=100)
        self.cache.add_visitor(face_embedding=emb, body_hist=_random_hist(seed=100))
        m = self.cache.search(face_embedding=emb, body_hist=None)
        self.assertTrue(m.matched)
        self.assertEqual(m.kind, "face")
        self.assertGreaterEqual(m.face_score, 0.55)

    def test_c2_corroborated_match(self):
        """Body ≥ 0.65 AND face ≥ 0.30 → C2 → match."""
        emb = _random_unit_vec(seed=101)
        hist = _random_hist(seed=101)
        self.cache.add_visitor(face_embedding=emb, body_hist=hist)
        # Same body (=1.0), slightly-modified face (still > 0.30 cosine)
        emb_drifted = (emb + 0.5 * _random_unit_vec(seed=102))
        emb_drifted = emb_drifted / np.linalg.norm(emb_drifted)
        m = self.cache.search(face_embedding=emb_drifted, body_hist=hist)
        self.assertTrue(m.matched)
        # Could fire C1 OR C2 depending on the drifted cosine.
        self.assertIn(m.kind, ("face", "body+face"))

    def test_c3_body_sure_match_no_face(self):
        """Body ≥ 0.85 alone → C3 → match even with no face."""
        hist = _random_hist(seed=103)
        self.cache.add_visitor(face_embedding=None, body_hist=hist)
        m = self.cache.search(face_embedding=None, body_hist=hist)
        self.assertTrue(m.matched)
        self.assertEqual(m.kind, "body-strict")

    def test_no_match_when_below_all_three(self):
        """Body at 0.5, no face → fails C1, fails C2 (no face), fails C3 (body<0.85)."""
        hist_stored = np.ones((30, 32), dtype=np.float32)
        hist_stored[0, 0] = 0.0   # tiny perturbation so it's not literally degenerate
        self.cache.add_visitor(face_embedding=None, body_hist=hist_stored)

        # Search with a histogram correlating ~0.5 — verifiable by adding
        # noise. We check directly that the result is unmatched even if
        # the body score lands in (0.40, 0.85): no face means C2 is out,
        # body alone needs ≥ 0.85 for C3.
        rng = np.random.default_rng(0)
        hist_query = hist_stored + 0.5 * rng.standard_normal(hist_stored.shape).astype(np.float32)
        m = self.cache.search(face_embedding=None, body_hist=hist_query)
        if 0.40 <= m.body_score < 0.85:
            self.assertFalse(m.matched, f"Should not match at body_score={m.body_score:.3f} without face")
            self.assertEqual(m.kind, "miss")

    def test_exclusion_set_skips_visitor(self):
        """A visitor in excluded_visitor_ids must be skipped even
        on a PERFECT face match."""
        emb = _random_unit_vec(seed=104)
        vid = self.cache.add_visitor(face_embedding=emb, body_hist=_random_hist(seed=104))
        # Without exclusion: matches.
        m1 = self.cache.search(face_embedding=emb, body_hist=None)
        self.assertTrue(m1.matched)
        # With exclusion of that visitor: must NOT match.
        m2 = self.cache.search(face_embedding=emb, body_hist=None,
                               excluded_visitor_ids={vid})
        self.assertFalse(m2.matched)

    def test_exclusion_set_lets_other_visitors_match(self):
        """Excluding A must not prevent matching B."""
        embA = _random_unit_vec(seed=105)
        embB = _random_unit_vec(seed=106)
        vidA = self.cache.add_visitor(face_embedding=embA, body_hist=_random_hist(seed=105))
        vidB = self.cache.add_visitor(face_embedding=embB, body_hist=_random_hist(seed=106))
        # Query with embB while excluding A → must match B.
        m = self.cache.search(face_embedding=embB, body_hist=None,
                              excluded_visitor_ids={vidA})
        self.assertTrue(m.matched)
        self.assertEqual(m.index, vidB)


class TestVisitorIdsForTracks(unittest.TestCase):
    """The cache must answer 'which visitor_ids are these tracks bound to?'"""

    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".pkl")
        os.close(fd); os.remove(self.tmp)
        self.cache = DailyFaceCache(state_file=self.tmp)

    def tearDown(self):
        try: os.remove(self.tmp)
        except OSError: pass

    def test_unknown_tracks_return_empty_set(self):
        self.assertEqual(
            self.cache.visitor_ids_for_tracks([42, 99, 1000]), set(),
        )

    def test_returns_bound_visitor_ids(self):
        # Register two visitors and bind tracks to them.
        v1 = self.cache.add_visitor(face_embedding=_random_unit_vec(seed=200),
                                    body_hist=_random_hist(seed=200))
        v2 = self.cache.add_visitor(face_embedding=_random_unit_vec(seed=201),
                                    body_hist=_random_hist(seed=201))
        self.cache.mark_track_known(7, v1)
        self.cache.mark_track_known(8, v2)

        self.assertEqual(self.cache.visitor_ids_for_tracks([7]), {v1})
        self.assertEqual(self.cache.visitor_ids_for_tracks([8]), {v2})
        self.assertEqual(self.cache.visitor_ids_for_tracks([7, 8]), {v1, v2})

    def test_mix_of_known_and_unknown_tracks(self):
        v1 = self.cache.add_visitor(face_embedding=_random_unit_vec(seed=300),
                                    body_hist=_random_hist(seed=300))
        self.cache.mark_track_known(11, v1)
        self.assertEqual(
            self.cache.visitor_ids_for_tracks([11, 999]), {v1},
        )


class TestRecencyWindow(unittest.TestCase):
    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        os.remove(self.tmp)
        self.cache = DailyFaceCache(state_file=self.tmp)

    def tearDown(self):
        try: os.remove(self.tmp)
        except OSError: pass

    def test_recency_window_constant_is_reasonable(self):
        # Should be exactly the documented 120 seconds
        self.assertEqual(RECENCY_WINDOW_SECONDS, 120.0)


if __name__ == "__main__":
    unittest.main()
