"""Tests for the static helpers on FaceEngine.

Only the pure-math methods are unit-tested here. The InsightFace
``get_face`` path needs the full model to be initialised so we leave
that to integration tests.
"""

import unittest

import cv2
import numpy as np

from backend.services.vision.face_engine import FaceEngine


class TestCheckQuality(unittest.TestCase):
    def test_empty_returns_zero(self):
        self.assertEqual(FaceEngine.check_quality(None), 0.0)
        self.assertEqual(
            FaceEngine.check_quality(np.zeros((0, 0, 3), dtype=np.uint8)),
            0.0,
        )

    def test_uniform_gray_low_variance(self):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Solid color has near-zero Laplacian variance
        self.assertLess(FaceEngine.check_quality(flat), 1.0)

    def test_high_contrast_high_variance(self):
        # Checkerboard-ish — lots of sharp edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, :] = 255
        self.assertGreater(FaceEngine.check_quality(img), 100.0)


class TestBodyHistogram(unittest.TestCase):
    def test_zero_area_bbox_returns_none(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIsNone(
            FaceEngine.compute_body_histogram(frame, (10.0, 10.0, 10.0, 10.0))
        )

    def test_negative_bbox_returns_none(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # x2 < x1
        self.assertIsNone(
            FaceEngine.compute_body_histogram(frame, (50.0, 50.0, 10.0, 100.0))
        )

    def test_valid_bbox_returns_correct_shape(self):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        hist = FaceEngine.compute_body_histogram(
            frame, (100.0, 100.0, 300.0, 500.0)
        )
        self.assertIsNotNone(hist)
        # Default bins = (30, 32) → shape (30, 32)
        self.assertEqual(hist.shape, (30, 32))
        self.assertEqual(hist.dtype, np.float32)

    def test_same_torso_produces_identical_hist(self):
        # Two identical frames + same bbox → identical histograms
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        bbox = (100.0, 100.0, 400.0, 600.0)
        h1 = FaceEngine.compute_body_histogram(frame, bbox)
        h2 = FaceEngine.compute_body_histogram(frame, bbox)
        # NORM_MINMAX normalises identically → exact equality
        np.testing.assert_array_equal(h1, h2)

    def test_different_torsos_low_correlation(self):
        """Two visually different torsos should have correlation < 1.0."""
        rng = np.random.default_rng(0)
        frame_a = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        # frame_b is a different random scene
        frame_b = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        bbox = (100.0, 100.0, 400.0, 600.0)
        h1 = FaceEngine.compute_body_histogram(frame_a, bbox)
        h2 = FaceEngine.compute_body_histogram(frame_b, bbox)
        corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        self.assertLess(corr, 1.0)
        self.assertGreater(corr, -1.0)


if __name__ == "__main__":
    unittest.main()
