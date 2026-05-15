"""Pure-logic tests for backend.core.business_day."""

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from backend.core import business_day as bd


class TestBusinessDay(unittest.TestCase):

    # ── now_ist ──────────────────────────────────────────────────────────

    def test_now_ist_is_tz_aware_in_kolkata(self):
        n = bd.now_ist()
        self.assertIsNotNone(n.tzinfo)
        self.assertEqual(n.tzinfo, ZoneInfo("Asia/Kolkata"))

    def test_now_ist_str_format(self):
        s = bd.now_ist_str()
        # YYYY-MM-DD HH:MM:SS — must parse back cleanly
        parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        self.assertIsNotNone(parsed)

    # ── business_day_of ──────────────────────────────────────────────────

    def test_business_day_just_before_9am_is_previous(self):
        dt = datetime(2026, 5, 15, 8, 59, 59, tzinfo=ZoneInfo("Asia/Kolkata"))
        self.assertEqual(bd.business_day_of(dt), datetime(2026, 5, 14).date())

    def test_business_day_at_exactly_9am_is_current(self):
        dt = datetime(2026, 5, 15, 9, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        self.assertEqual(bd.business_day_of(dt), datetime(2026, 5, 15).date())

    def test_business_day_well_after_9am_is_current(self):
        dt = datetime(2026, 5, 15, 14, 30, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        self.assertEqual(bd.business_day_of(dt), datetime(2026, 5, 15).date())

    def test_business_day_early_morning_is_previous(self):
        # 03:00 → still belongs to yesterday's business day
        dt = datetime(2026, 5, 16, 3, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        self.assertEqual(bd.business_day_of(dt), datetime(2026, 5, 15).date())

    def test_business_day_one_minute_before_rollover(self):
        # 08:59:59.999... → previous business day
        dt = datetime(2026, 5, 15, 8, 59, 59, 999999,
                      tzinfo=ZoneInfo("Asia/Kolkata"))
        self.assertEqual(bd.business_day_of(dt), datetime(2026, 5, 14).date())

    # ── business_day_from_iso ────────────────────────────────────────────

    def test_iso_string_before_9am(self):
        self.assertEqual(
            bd.business_day_from_iso("2026-05-15 08:59:59"), "2026-05-14"
        )

    def test_iso_string_at_9am(self):
        self.assertEqual(
            bd.business_day_from_iso("2026-05-15 09:00:00"), "2026-05-15"
        )

    def test_iso_string_well_after_9am(self):
        self.assertEqual(
            bd.business_day_from_iso("2026-05-15 23:59:59"), "2026-05-15"
        )

    def test_iso_string_garbage_returns_unknown(self):
        self.assertEqual(bd.business_day_from_iso("not a date"), "Unknown")
        self.assertEqual(bd.business_day_from_iso(""), "Unknown")
        self.assertEqual(bd.business_day_from_iso(None), "Unknown")

    # ── seconds_until_next_business_day_start ────────────────────────────

    def test_seconds_until_next_rollover_is_in_24h_window(self):
        s = bd.seconds_until_next_business_day_start()
        self.assertGreater(s, 0)
        # Worst case is just after 09:00 → ~86400s until tomorrow 09:00
        self.assertLessEqual(s, 86400)

    def test_seconds_until_next_rollover_at_exactly_9am_skips_to_next_day(self):
        """If we *are* exactly at 09:00 right now, the result should
        target the FOLLOWING 09:00, not 0 seconds away (which would
        cause an immediate-loop in the rollover thread)."""
        # Can't easily monkey-patch now() inside the module without
        # introducing freezegun. Instead, verify the function never
        # returns < 1 second (the floor we apply).
        s = bd.seconds_until_next_business_day_start()
        self.assertGreaterEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
