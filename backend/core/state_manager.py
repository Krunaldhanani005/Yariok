"""Thread-safe state holder for the live dashboard.

Daily counters (``visitors_today``, ``alerts_today``, ``snapshots_taken``)
are mirrored to ``daily_state.json`` at the project root so a process
restart does NOT silently zero them. The file is loaded on construction
only if its ``"date"`` field matches today — otherwise we start fresh.

Non-persistent keys (``people_now``, ``camera_status``, modes,
``robot_speech``, ``last_visitor_b64``, etc.) are always re-initialised
from defaults on startup. ``people_now`` in particular is transient by
design — it's a live presence signal, not a daily total.

Thread-safety
-------------
The same ``threading.Lock()`` that already guarded in-memory mutation
now also guards file I/O. Writes use the *atomic-replace* pattern
(``tempfile.mkstemp`` + ``os.replace``) so a half-written JSON can never
be observed even if the process is SIGKILL'd mid-flush. The lock is
held across the entire mutate-then-persist sequence, which makes the
on-disk file always consistent with whichever in-memory snapshot was
most recently published.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import date
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)

# Project root layout: backend/core/state_manager.py → ../../../
_PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_DEFAULT_STATE_FILE: str = os.path.join(_PROJECT_ROOT, "daily_state.json")


class StateManager:
    """Thread-safe replacement for the global mutable state dict.

    All reads and writes go through ``get()`` / ``update()`` /
    ``increment()`` so no caller can accidentally bypass the lock — and
    no caller has to know whether a key is persistent or not. The
    decision of "is this worth a disk write?" lives entirely here.
    """

    # Keys whose mutation triggers a save to ``daily_state.json``.
    # Deliberately narrow — saving on every camera_status /
    # robot_speech tick would dump hundreds of KB / minute for no gain.
    _PERSISTENT_KEYS: Set[str] = {
        "visitors_today",
        "alerts_today",
        "snapshots_taken",
    }

    def __init__(self, state_file: str = _DEFAULT_STATE_FILE) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._state_file: str = state_file

        # Ephemeral defaults — always reset on boot regardless of disk.
        self._state: Dict[str, Any] = {
            "system_running": False,
            "voice_running": False,
            "people_now": 0,                  # transient — never persisted
            "visitors_today": 0,              # persistent
            "alerts_today": 0,                # persistent
            "snapshots_taken": 0,             # persistent
            "security_mode": False,
            "karaoke_mode": False,
            "show_camera": False,
            "take_snapshot": False,
            "last_command": "—",
            "last_alert_time": "—",
            "last_live_alert": "",
            "last_live_alert_time": "—",
            "robot_status": "Starting...",
            "robot_speech": "System starting up...",
            "camera_status": "Connecting...",
            "running": True,
            "dash_command": "",
            "last_visitor_b64": "",
            "ai_alert": "",
            "voice_state": "listen",
        }

        # Hydrate the three persistent counters from disk (if today).
        # No other thread holds a reference to this instance yet, so
        # the lock isn't strictly necessary here — we take it anyway so
        # the rule "all file I/O happens under the lock" stays universal.
        with self._lock:
            self._load_from_disk_locked()

    # ── Public API (signatures unchanged — callers don't need to care) ──

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def update(self, **kwargs) -> None:
        """Atomic mutate + (optional) disk-persist.

        If any of the keyword arguments is a persistent counter, the
        in-memory write AND the disk write happen inside the same
        critical section — readers can never observe a state that is
        out of sync with the on-disk JSON.
        """
        with self._lock:
            self._state.update(kwargs)
            if any(k in self._PERSISTENT_KEYS for k in kwargs):
                self._save_to_disk_locked()

    def increment(self, key: str, by: int = 1) -> int:
        """Atomic add + (optional) disk-persist."""
        with self._lock:
            val = self._state.get(key, 0) + by
            self._state[key] = val
            if key in self._PERSISTENT_KEYS:
                self._save_to_disk_locked()
            return val

    def snapshot(self) -> dict:
        """Return a shallow copy of the full state dict (for API responses)."""
        with self._lock:
            return dict(self._state)

    # ── Daily reset — call from the existing midnight cron task ─────────

    def reset_daily_counters(self) -> None:
        """Zero today's persistent counters and stamp the new date.

        Wired in by ``main.py`` as the ``on_midnight_reset`` callback of
        ``VisionService``. Safe to call multiple times — idempotent in
        the same calendar day.
        """
        with self._lock:
            for k in self._PERSISTENT_KEYS:
                self._state[k] = 0
            # ``people_now`` is transient, but if midnight catches an
            # idle dashboard we may as well zero its display too.
            self._state["people_now"] = 0
            self._save_to_disk_locked()
        logger.info(
            "StateManager: daily counters reset to 0 for %s",
            date.today().isoformat(),
        )

    # ── Disk I/O (caller MUST hold self._lock) ─────────────────────────

    def _load_from_disk_locked(self) -> None:
        """Restore today's persistent counters from disk if date matches."""
        today = date.today().isoformat()
        try:
            with open(self._state_file) as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.info(
                "StateManager: %s not found — starting with fresh counters.",
                self._state_file,
            )
            self._save_to_disk_locked()
            return
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "StateManager: %s unreadable (%s) — starting with fresh counters.",
                self._state_file, exc,
            )
            self._save_to_disk_locked()
            return

        saved_date = data.get("date")
        if saved_date != today:
            # File is from yesterday (or some other date). Don't carry
            # those numbers over — the whole point of a daily counter
            # is that it starts each day at zero.
            logger.info(
                "StateManager: stored date=%s ≠ today=%s — fresh counters.",
                saved_date, today,
            )
            self._save_to_disk_locked()
            return

        for k in self._PERSISTENT_KEYS:
            v = data.get(k)
            if isinstance(v, int):
                self._state[k] = v
        logger.info(
            "StateManager: restored counters from %s — "
            "visitors=%d alerts=%d snapshots=%d",
            self._state_file,
            self._state["visitors_today"],
            self._state["alerts_today"],
            self._state["snapshots_taken"],
        )

    def _save_to_disk_locked(self) -> None:
        """Atomic JSON write of today's persistent counters.

        Pattern: write a temp file in the SAME directory, fsync it,
        then ``os.replace`` it over the target. POSIX guarantees that
        rename within a single filesystem is atomic, so a reader can
        only ever see the OLD file or the NEW file — never a half-
        flushed mix. This is the standard "atomic write" recipe and
        protects against power loss, SIGKILL, and disk-full mid-flush.

        Failures are logged and swallowed — losing a daily counter on
        a disk error is preferable to taking down the vision pipeline.
        """
        payload = {
            "date": date.today().isoformat(),
            **{k: int(self._state.get(k, 0)) for k in self._PERSISTENT_KEYS},
        }
        try:
            dir_name = os.path.dirname(self._state_file) or "."
            os.makedirs(dir_name, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".daily_state.", suffix=".tmp", dir=dir_name,
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, separators=(",", ":"))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._state_file)
            except Exception:
                # Clean up the orphan temp file on any write failure.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            logger.exception(
                "StateManager: failed to persist %s", self._state_file,
            )


# Module-level singleton — import this everywhere.
state = StateManager()
