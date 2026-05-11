import threading


class StateManager:
    """Thread-safe replacement for the global mutable state dict.

    All reads and writes go through get()/update()/increment() so no
    caller can accidentally bypass the lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = {
            "system_running": False,
            "voice_running": False,
            "people_now": 0,
            "visitors_today": 0,
            "alerts_today": 0,
            "snapshots_taken": 0,
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

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def update(self, **kwargs) -> None:
        with self._lock:
            self._state.update(kwargs)

    def increment(self, key: str, by: int = 1) -> int:
        with self._lock:
            val = self._state.get(key, 0) + by
            self._state[key] = val
            return val

    def snapshot(self) -> dict:
        """Return a shallow copy of the full state dict (for API responses)."""
        with self._lock:
            return dict(self._state)


# Module-level singleton
state = StateManager()
