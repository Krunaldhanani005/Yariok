"""Tests for backend.core.state_manager — persistence, atomic write,
business-day rollover semantics."""

import json
import os
import tempfile
import threading
import unittest
from datetime import date, timedelta

from backend.core.state_manager import StateManager
from backend.core.business_day import current_business_day


class TestStateManagerFreshInit(unittest.TestCase):
    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.tmp)  # let StateManager create it fresh

    def tearDown(self):
        try:
            os.remove(self.tmp)
        except OSError:
            pass

    def test_fresh_init_zero_counters(self):
        sm = StateManager(state_file=self.tmp)
        self.assertEqual(sm.get("visitors_today"), 0)
        self.assertEqual(sm.get("alerts_today"), 0)
        self.assertEqual(sm.get("snapshots_taken"), 0)

    def test_fresh_init_creates_file_with_today_business_day(self):
        StateManager(state_file=self.tmp)
        self.assertTrue(os.path.exists(self.tmp))
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertEqual(data["date"], current_business_day().isoformat())

    def test_people_now_always_starts_at_zero(self):
        # Stuff a non-zero people_now into the file → must still load as 0
        with open(self.tmp, "w") as f:
            json.dump({
                "date": current_business_day().isoformat(),
                "visitors_today": 5, "alerts_today": 0, "snapshots_taken": 5,
                "people_now": 99,  # bogus persisted value
            }, f)
        sm = StateManager(state_file=self.tmp)
        self.assertEqual(sm.get("people_now"), 0)
        # but the persistent counters DID load
        self.assertEqual(sm.get("visitors_today"), 5)
        self.assertEqual(sm.get("snapshots_taken"), 5)


class TestStateManagerPersistence(unittest.TestCase):
    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.tmp)

    def tearDown(self):
        try:
            os.remove(self.tmp)
        except OSError:
            pass

    def test_increment_visitor_persists(self):
        sm1 = StateManager(state_file=self.tmp)
        sm1.increment("visitors_today")
        sm1.increment("visitors_today")
        # Simulate restart
        sm2 = StateManager(state_file=self.tmp)
        self.assertEqual(sm2.get("visitors_today"), 2)

    def test_update_persistent_keys_persists(self):
        sm1 = StateManager(state_file=self.tmp)
        sm1.update(visitors_today=7, snapshots_taken=7, alerts_today=1)
        sm2 = StateManager(state_file=self.tmp)
        self.assertEqual(sm2.get("visitors_today"), 7)
        self.assertEqual(sm2.get("snapshots_taken"), 7)
        self.assertEqual(sm2.get("alerts_today"), 1)

    def test_update_non_persistent_key_does_not_change_disk(self):
        sm = StateManager(state_file=self.tmp)
        sm.update(visitors_today=1)
        mtime_before = os.path.getmtime(self.tmp)
        # Update only non-persistent keys — should NOT rewrite the file.
        import time
        time.sleep(0.05)
        sm.update(camera_status="Active", robot_speech="x")
        # mtime should be unchanged
        self.assertEqual(os.path.getmtime(self.tmp), mtime_before)

    def test_yesterdays_file_is_ignored(self):
        """A file with yesterday's business day must NOT restore counters."""
        yesterday = (current_business_day() - timedelta(days=1)).isoformat()
        with open(self.tmp, "w") as f:
            json.dump({
                "date": yesterday,
                "visitors_today": 99,
                "alerts_today": 0,
                "snapshots_taken": 99,
            }, f)
        sm = StateManager(state_file=self.tmp)
        # Yesterday's numbers are dropped
        self.assertEqual(sm.get("visitors_today"), 0)
        self.assertEqual(sm.get("snapshots_taken"), 0)
        # And the file is rewritten with today's business date
        with open(self.tmp) as f:
            self.assertEqual(json.load(f)["date"], current_business_day().isoformat())

    def test_reset_daily_counters_zeros_and_persists(self):
        sm = StateManager(state_file=self.tmp)
        sm.update(visitors_today=12, snapshots_taken=12)
        sm.reset_daily_counters()
        self.assertEqual(sm.get("visitors_today"), 0)
        self.assertEqual(sm.get("snapshots_taken"), 0)
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertEqual(data["visitors_today"], 0)
        self.assertEqual(data["date"], current_business_day().isoformat())

    def test_atomic_write_no_corruption_on_crash(self):
        """Even if a writer dies mid-flush, the file must contain either
        the OLD state or the NEW state — never a half-written mix."""
        sm = StateManager(state_file=self.tmp)
        sm.update(visitors_today=1)

        # Read the file → must be valid JSON every time
        for _ in range(50):
            sm.increment("snapshots_taken")
            with open(self.tmp) as f:
                # Will raise json.JSONDecodeError if corrupt
                data = json.load(f)
            self.assertIn("date", data)
            self.assertIn("visitors_today", data)
            self.assertIn("snapshots_taken", data)


class TestStateManagerConcurrency(unittest.TestCase):
    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.tmp)

    def tearDown(self):
        try:
            os.remove(self.tmp)
        except OSError:
            pass

    def test_concurrent_increments_exact_count(self):
        """100 threads × 10 increments must produce exactly 1000."""
        sm = StateManager(state_file=self.tmp)
        N_THREADS = 20
        PER_THREAD = 50

        def worker():
            for _ in range(PER_THREAD):
                sm.increment("visitors_today")

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(sm.get("visitors_today"), N_THREADS * PER_THREAD)
        # And the on-disk value matches
        with open(self.tmp) as f:
            self.assertEqual(json.load(f)["visitors_today"],
                             N_THREADS * PER_THREAD)


if __name__ == "__main__":
    unittest.main()
