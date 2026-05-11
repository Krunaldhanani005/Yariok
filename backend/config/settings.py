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
VISION_CONFIG_FILE: str = _abspath("VISION_CONFIG_FILE", "vision_config.json")

# --- Vision ---
ENTRY_ZONE: tuple = (0.1, 0.2, 0.9, 1.0)

# --- Dashboard ---
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "5000"))
