"""Yaariok Smart System — Phase A entry point.

Run from the project root:
    python backend/main.py

Or as a module:
    python -m backend.main
"""

import os
import sys
import warnings

# ── Environment setup ────────────────────────────────────────────────────────────
# Silence OpenCV codec warnings before any cv2 import
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
warnings.filterwarnings("ignore")

# Ensure the project root (parent of backend/) is on sys.path so that
# "from backend.xxx import ..." works when this file is run directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Change CWD to project root so all relative file paths resolve correctly
os.chdir(_PROJECT_ROOT)

# ── Logging (must be configured before any other import that logs) ────────────────
from backend.config.logging_config import setup_logging
setup_logging(log_dir="./logs")

import logging
logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────────
from backend.core.event_bus import bus
from backend.core.state_manager import state
from backend.core.service_registry import registry
from backend.config.settings import DASHBOARD_PORT

# ── Service instantiation ─────────────────────────────────────────────────────────
from backend.services.database.db_service import DatabaseService
from backend.services.vision.camera_service import VisionService
from backend.services.voice.voice_service import VoiceService
from backend.services.ai.ai_service import AIService
from backend.services.automation.automation_service import AutomationService
from backend.services.dashboard.dashboard_service import create_app, start_dashboard

db      = DatabaseService()
vision  = VisionService(state, bus, db)
voice   = VoiceService(state, bus)
ai      = AIService(state, bus)
automation = AutomationService(state, bus, db, registry)

# ── Register services ─────────────────────────────────────────────────────────────
registry.register("state",      state)
registry.register("bus",        bus)
registry.register("db",         db)
registry.register("vision",     vision)
registry.register("voice",      voice)
registry.register("ai",         ai)
registry.register("automation", automation)


def main() -> None:
    # ── Startup banner ────────────────────────────────────────────────────────────
    print("\n\033[95m\033[1m")
    print("  ╔══════════════════════════════════════════╗")
    print("  ║     YAARIOK SMART SYSTEM  Phase A        ║")
    print("  ║  Voice  •  Camera  •  AI Detection       ║")
    print("  ╚══════════════════════════════════════════╝\033[0m\n")

    # ── Database ──────────────────────────────────────────────────────────────────
    db.init_db()
    db.log_event("=== System boot ===", "system")

    # ── TTS ───────────────────────────────────────────────────────────────────────
    voice.init()

    # ── LLM connectivity check ────────────────────────────────────────────────────
    ai.init()

    # ── Dashboard ─────────────────────────────────────────────────────────────────
    app = create_app(registry)
    start_dashboard(app, port=DASHBOARD_PORT)
    print(f"  Dashboard : \033[92mhttp://localhost:{DASHBOARD_PORT}\033[0m  (open in browser)\n")

    import threading, webbrowser
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")).start()

    # ── Start services ────────────────────────────────────────────────────────────
    automation.start_system()   # starts vision (camera) via event bus
    voice.start()               # starts voice listener

    print("  System ready. Say \033[96m'hey robot'\033[0m to begin.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────────
    try:
        while state.get("running"):
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        import cv2
        state.update(running=False, show_camera=False)
        cv2.destroyAllWindows()
        db.log_event("=== System shutdown ===", "system")
        print("\n\033[90m[Stopped]\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
