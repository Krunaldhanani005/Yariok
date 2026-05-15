"""Business-day clock — 09:00 IST → 09:00 IST.

A *business day* rolls over at **09:00:00 IST**, not at midnight.
Anyone seen at 08:59 belongs to *yesterday's* business day; anyone
seen at 09:01 is part of *today's*. All daily counters, snapshot
groupings, identity-cache resets, and scheduled rollovers respect
this boundary.

This module is the single source of truth — every other module
that needs to ask "what day is it?" should import from here so we
can't get inconsistent timezones drifting across the codebase.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

#: India Standard Time (UTC+05:30, no DST). All wall-clock timestamps
#: in the system — DB rows, snapshot filenames, log entries — are
#: written in this timezone.
IST: ZoneInfo = ZoneInfo("Asia/Kolkata")

#: Hour-of-day at which the business day rolls over. Configurable
#: down the road via env var if needed, but per spec this is 9 AM.
BUSINESS_DAY_START_HOUR: int = 9


def now_ist() -> datetime:
    """Return the current wall-clock time in IST."""
    return datetime.now(IST)


def now_ist_str() -> str:
    """``'YYYY-MM-DD HH:MM:SS'`` in IST — the canonical DB timestamp."""
    return now_ist().strftime("%Y-%m-%d %H:%M:%S")


def business_day_of(dt: datetime) -> date:
    """Return the business-day date that a given IST datetime belongs to.

    Times before 09:00 belong to the **previous** calendar day's
    business day; times at or after 09:00 belong to that same day.
    """
    if dt.hour < BUSINESS_DAY_START_HOUR:
        return dt.date() - timedelta(days=1)
    return dt.date()


def current_business_day() -> date:
    """Convenience: today's business-day date."""
    return business_day_of(now_ist())


def business_day_from_iso(ts_str: str) -> str:
    """Map a stored timestamp ``'YYYY-MM-DD HH:MM:SS'`` → business-day date.

    The timestamp is treated as a naive IST wall-clock value (which is
    what every part of the system now writes). Returns the ISO date
    string of the business day, or ``"Unknown"`` on a parse failure
    so the gallery doesn't blow up on legacy rows.
    """
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Unknown"
    if dt.hour < BUSINESS_DAY_START_HOUR:
        return (dt.date() - timedelta(days=1)).isoformat()
    return dt.date().isoformat()


def seconds_until_next_business_day_start() -> float:
    """Seconds from now until the next 09:00:00 IST tick.

    Used by the daily rollover thread to sleep in slices until the
    boundary, then fire the reset.
    """
    now = now_ist()
    target = now.replace(
        hour=BUSINESS_DAY_START_HOUR, minute=0, second=0, microsecond=0
    )
    if now >= target:
        target = target + timedelta(days=1)
    return max(1.0, (target - now).total_seconds())
