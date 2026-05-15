import os
from dotenv import load_dotenv

load_dotenv()

# Project root is three levels up from this file:
# backend/config/settings.py → backend/config → backend → project root
_PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _abspath(env_key: str, default: str) -> str:
    """Resolve a path from env or default, always returning an absolute path.

    If the value is already absolute, return it unchanged.
    Otherwise, join it with the project root so Flask's send_from_directory
    and file I/O work correctly regardless of CWD.
    """
    raw = os.getenv(env_key, default)
    if os.path.isabs(raw):
        return raw
    return os.path.normpath(os.path.join(_PROJECT_ROOT, raw.lstrip("./")))


# --- API credentials ---
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")  # reserved for future use

# --- Camera ---
RTSP_URL: str = os.getenv("RTSP_URL", "")

# --- AI / LLM ---
OR_URL: str = "https://openrouter.ai/api/v1/chat/completions"
OR_MODEL: str = os.getenv("OR_MODEL", "anthropic/claude-sonnet-4-5")

# --- TTS ---
EDGE_VOICE: str = os.getenv("EDGE_VOICE", "en-US-AriaNeural")

# --- File paths — always absolute so Flask route handlers find them correctly ---
DB_PATH: str = _abspath("DB_PATH", "snapshots.db")
SNAPSHOT_DIR: str = _abspath("SNAPSHOT_DIR", "snapshots")
ALERTS_DIR: str = _abspath("ALERTS_DIR", "alerts")
LOG_FILE: str = _abspath("LOG_FILE", "visitor_log.txt")

# --- Vision ---
ENTRY_ZONE: tuple = (0.1, 0.2, 0.9, 1.0)

# --- Snapshot retention ---
# How long to keep snapshot JPEGs + their DB rows before pruning. Two
# limits combined: whichever hits first wins.
#   SNAPSHOT_RETENTION_DAYS — anything older than this is deleted.
#   SNAPSHOT_MAX_COUNT      — keep at most this many rows (newest first).
# Override via env if you need different retention on a per-deploy basis.
SNAPSHOT_RETENTION_DAYS: int = int(os.getenv("SNAPSHOT_RETENTION_DAYS", "7"))
SNAPSHOT_MAX_COUNT: int = int(os.getenv("SNAPSHOT_MAX_COUNT", "2000"))

# Per-visitor snapshot cooldown. After a visitor is first detected
# we take a fresh snapshot every time this many seconds have elapsed
# (subject to BoT-SORT track resets — same-track loiterers don't
# trigger refresh snapshots). Greetings still fire only once per
# business day per visitor regardless of cooldown.
SNAPSHOT_COOLDOWN_SECONDS: int = int(os.getenv("SNAPSHOT_COOLDOWN_SECONDS", "600"))

# --- Entry-zone polygon (greeting gate) ---
# Coordinates inside this file are stored as NORMALIZED [0.0, 1.0] x/y
# pairs so the same zone works regardless of the stream's actual pixel
# resolution. Missing or empty polygon → default-allow (greet anywhere).
ZONE_CONFIG_FILE: str = _abspath("ZONE_CONFIG_FILE", "zone_config.json")

# --- Dashboard ---
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "5000"))
