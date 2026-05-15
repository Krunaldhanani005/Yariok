"""Integration tests for /api/zone and /api/snapshots/daywise.

Hits the LIVE Yaariok via HTTP (assumes the dev server is running on
``http://localhost:5001`` — same setup we use for manual testing).
Each test snapshots and restores the zone so the suite leaves the
system in the same state it found it.
"""

import json
import os
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = os.environ.get("YAARIOK_BASE", "http://localhost:5001")


def _get(path):
    with urlopen(BASE + path, timeout=5) as r:
        return r.getcode(), json.loads(r.read().decode("utf-8"))


def _post(path, body):
    req = Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=5) as r:
            return r.getcode(), json.loads(r.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body}


class _RequiresLiveServer(unittest.TestCase):
    """Skip the whole class if the server isn't reachable."""
    @classmethod
    def setUpClass(cls):
        try:
            _get("/api/status")
        except (URLError, OSError) as exc:
            raise unittest.SkipTest(f"Yaariok not running on {BASE}: {exc}")


class TestZoneEndpoint(_RequiresLiveServer):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Save whatever zone is currently set so we can restore later.
        _, data = _get("/api/zone")
        cls._saved_polygon = data.get("polygon", [])

    @classmethod
    def tearDownClass(cls):
        # Restore saved zone (or clear it)
        _post("/api/zone", {"polygon": cls._saved_polygon})

    def test_post_valid_4_point_polygon(self):
        polygon = [[0.1, 0.5], [0.9, 0.5], [0.9, 1.0], [0.1, 1.0]]
        code, data = _post("/api/zone", {"polygon": polygon})
        self.assertEqual(code, 200)
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("points"), 4)

        code, data = _get("/api/zone")
        self.assertEqual(code, 200)
        self.assertEqual(data["polygon"], polygon)

    def test_post_empty_polygon_clears_zone(self):
        code, data = _post("/api/zone", {"polygon": []})
        self.assertEqual(code, 200)
        self.assertTrue(data.get("ok"))
        code, data = _get("/api/zone")
        self.assertEqual(data["polygon"], [])

    def test_reject_one_point_polygon(self):
        code, data = _post("/api/zone", {"polygon": [[0.5, 0.5]]})
        self.assertEqual(code, 400)
        self.assertFalse(data.get("ok"))

    def test_reject_two_point_polygon(self):
        code, data = _post("/api/zone", {"polygon": [[0.0, 0.0], [1.0, 1.0]]})
        self.assertEqual(code, 400)

    def test_reject_out_of_range_coordinate(self):
        code, _ = _post(
            "/api/zone",
            {"polygon": [[1.5, 0.5], [0.0, 0.0], [0.0, 1.0]]},
        )
        self.assertEqual(code, 400)

        code, _ = _post(
            "/api/zone",
            {"polygon": [[-0.1, 0.5], [0.0, 0.0], [0.0, 1.0]]},
        )
        self.assertEqual(code, 400)

    def test_reject_non_numeric_coordinate(self):
        code, _ = _post(
            "/api/zone",
            {"polygon": [["a", 0.5], [0.0, 0.0], [0.0, 1.0]]},
        )
        self.assertEqual(code, 400)

    def test_reject_malformed_point(self):
        code, _ = _post("/api/zone", {"polygon": [[0.5], [0.0, 0.0], [0.0, 1.0]]})
        self.assertEqual(code, 400)

    def test_reject_non_list_polygon(self):
        code, _ = _post("/api/zone", {"polygon": "not a list"})
        self.assertEqual(code, 400)


class TestSnapshotsDaywise(_RequiresLiveServer):
    def test_response_shape(self):
        code, data = _get("/api/snapshots/daywise")
        self.assertEqual(code, 200)
        self.assertIn("grouped", data)
        self.assertIn("current_business_day", data)
        # current_business_day is ISO date format YYYY-MM-DD
        cbd = data["current_business_day"]
        self.assertRegex(cbd, r"^\d{4}-\d{2}-\d{2}$")

    def test_grouped_keys_are_business_day_dates(self):
        _, data = _get("/api/snapshots/daywise")
        for k in data.get("grouped", {}).keys():
            # Must be ISO date OR "Unknown" for garbage timestamps
            self.assertTrue(k == "Unknown" or len(k) == 10,
                            f"unexpected group key: {k}")

    def test_each_group_has_visitors_and_ungrouped(self):
        _, data = _get("/api/snapshots/daywise")
        for date, day in data.get("grouped", {}).items():
            self.assertIn("visitors", day, f"missing 'visitors' in {date}")
            self.assertIn("ungrouped", day, f"missing 'ungrouped' in {date}")
            self.assertIsInstance(day["visitors"], list)
            self.assertIsInstance(day["ungrouped"], list)


class TestStatusEndpoint(_RequiresLiveServer):
    def test_status_has_persistent_counters(self):
        code, data = _get("/api/status")
        self.assertEqual(code, 200)
        for key in ("visitors_today", "snapshots_taken", "alerts_today",
                    "people_now", "camera_status"):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
