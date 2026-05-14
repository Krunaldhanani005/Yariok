"""Yaariok Smart System — Phase B entry point.

The Phase B cut-over:
    - Legacy `camera_service.VisionService` is no longer imported.
    - A single paho-mqtt client is owned by THIS module. Its lifecycle
      (connect, loop_start, loop_stop, disconnect) is managed here.
    - The new `backend.services.vision.vision_service.VisionService` is
      instantiated with that client injected. It spawns its own daemon
      threads (RTSP grabber, face worker, midnight reset) inside .start().
    - The dashboard's `/video_feed` endpoint (see
      `backend/api/routes/control.py`) consumes
      `vision.get_video_feed()` — a multipart MJPEG generator backed by
      the bounded stream queue. Do NOT iterate the queue elsewhere.

Run from the project root:
    python backend/main.py
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import warnings

# ── Environment setup (must run before any cv2 / ultralytics import) ──────────
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# ── Logging (configured before any other import that logs) ────────────────────
from backend.config.logging_config import setup_logging
setup_logging(log_dir="./logs")

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
from backend.core.event_bus import bus
from backend.core.state_manager import state
from backend.core.service_registry import registry
from backend.config.settings import DASHBOARD_PORT, RTSP_URL

# ── Services (legacy camera_service is INTENTIONALLY not imported) ────────────
from backend.services.database.db_service import DatabaseService
from backend.services.vision.vision_service import VisionService
from backend.services.voice.voice_service import VoiceService
from backend.services.ai.ai_service import AIService
from backend.services.automation.automation_service import AutomationService
from backend.services.dashboard.dashboard_service import create_app, start_dashboard
from backend.services.dashboard.mqtt_state_bridge import MqttStateBridge


# ── MQTT broker configuration ────────────────────────────────────────────────
MQTT_BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_CLIENT_ID:   str = os.getenv("MQTT_CLIENT_ID", "yaariok-main")
MQTT_KEEPALIVE:   int = 60


def _build_mqtt_client():
    """Create, connect, and start the paho-mqtt client.

    Uses ``connect_async`` + ``loop_start`` so the broker is contacted
    on a background thread — the main thread is never blocked, and
    paho handles reconnection automatically if the broker drops.
    """
    import paho.mqtt.client as mqtt

    # paho-mqtt 2.x requires CallbackAPIVersion; 1.x doesn't accept it.
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1,
            client_id=MQTT_CLIENT_ID,
        )
    except AttributeError:
        client = mqtt.Client(client_id=MQTT_CLIENT_ID)

    def _on_connect(_c, _ud, _flags, rc, *_args):
        if rc == 0:
            logger.info("MQTT connected to %s:%d", MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        else:
            logger.warning("MQTT connect rc=%s", rc)

    def _on_disconnect(_c, _ud, *_args):
        logger.warning("MQTT disconnected — paho will auto-reconnect.")

    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect

    try:
        client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=MQTT_KEEPALIVE)
        client.loop_start()
    except Exception as exc:
        logger.error("MQTT setup failed (%s) — vision events will be dropped.", exc)

    return client


# ── Build the shared MQTT client up-front ─────────────────────────────────────
mqtt_client = _build_mqtt_client()

# ── Service instantiation ─────────────────────────────────────────────────────
db = DatabaseService()

# New VisionService: only needs the RTSP URL and an injected MQTT client.
# All other coordination happens over MQTT (topic: vision/events).
# The midnight callback wires VisionService's existing daily reset
# thread into StateManager so the persistent counters in
# ``daily_state.json`` roll over to zero at the same moment the
# face/body identity cache does.
vision = VisionService(
    rtsp_url=RTSP_URL,
    mqtt_client=mqtt_client,
    on_midnight_reset=state.reset_daily_counters,
)

voice      = VoiceService(state, bus)
ai         = AIService(state, bus)
automation = AutomationService(state, bus, db, registry)

# MQTT → StateManager + EventBus bridge.
# Phase B vision events arrive on the MQTT topic `vision/events`. This
# bridge mirrors them onto the legacy StateManager (so the polling
# dashboard sees fresh counters) and the EventBus (so AIService /
# VoiceService greeting flows fire without code changes).
mqtt_bridge = MqttStateBridge(mqtt_client, state, bus, db=db)

# ── Register services so Flask routes can resolve them via the registry ───────
# The /video_feed route in backend/api/routes/control.py looks up
# registry.get("vision") and calls vision.get_video_feed() on it.
registry.register("state",       state)
registry.register("bus",         bus)
registry.register("db",          db)
registry.register("mqtt",        mqtt_client)
registry.register("mqtt_bridge", mqtt_bridge)
registry.register("vision",      vision)
registry.register("voice",       voice)
registry.register("ai",          ai)
registry.register("automation",  automation)


def main() -> None:
    # ── Startup banner ────────────────────────────────────────────────────────
    print("\n\033[95m\033[1m")
    print("  ╔══════════════════════════════════════════╗")
    print("  ║     YAARIOK SMART SYSTEM  Phase B        ║")
    print("  ║   YOLOv11 • BoT-SORT • ArcFace • MQTT    ║")
    print("  ╚══════════════════════════════════════════╝\033[0m\n")

    # ── Database ──────────────────────────────────────────────────────────────
    db.init_db()
    db.log_event("=== System boot (Phase B) ===", "system")

    # ── TTS / LLM init (unchanged from Phase A) ───────────────────────────────
    voice.init()
    ai.init()

    # ── MQTT → State bridge ───────────────────────────────────────────────────
    # Start the bridge BEFORE vision so a subscriber is attached the
    # moment the very first vision event is published. Bridge.start()
    # is non-blocking — it only installs callbacks and spawns one tiny
    # presence-decay sweeper thread.
    mqtt_bridge.start()

    # ── Vision service ────────────────────────────────────────────────────────
    # VisionService.start() is non-blocking: it spawns its own daemon
    # threads (RTSP grabber, face worker, midnight reset). We wrap it
    # in a dedicated daemon thread anyway so even the model-loading
    # phase can't accidentally block the web server.
    threading.Thread(
        target=vision.start,
        name="vision-bootstrap",
        daemon=True,
    ).start()

    # ── Dashboard / Flask app ─────────────────────────────────────────────────
    # The Flask app uses the registry to resolve `vision`. The
    # `/video_feed` route attaches to `vision.get_video_feed()` —
    # see backend/api/routes/control.py.
    app = create_app(registry)
    start_dashboard(app, port=DASHBOARD_PORT)
    print(f"  Dashboard : \033[92mhttp://localhost:{DASHBOARD_PORT}\033[0m\n")

    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")).start()

    # ── Start remaining services ──────────────────────────────────────────────
    # NOTE: automation.start_system() publishes EV_SYSTEM_STARTED. The
    # legacy vision service used to subscribe to that event; the new
    # VisionService is self-starting via vision.start() above and does
    # not listen to the EventBus. This is intentional.
    automation.start_system()
    voice.start()

    print("  System ready. Say \033[96m'hey robot'\033[0m to begin.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while state.get("running"):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — beginning graceful shutdown.")
    finally:
        _shutdown()


def _shutdown() -> None:
    """Graceful shutdown — stop vision threads, then disconnect MQTT.

    Order matters: stop the publisher (vision) before tearing down the
    transport (mqtt) so we don't drop a final in-flight event.
    """
    state.update(running=False, show_camera=False)

    try:
        vision.stop()
    except Exception as exc:
        logger.warning("vision.stop() raised: %s", exc)

    try:
        mqtt_bridge.stop()
    except Exception as exc:
        logger.warning("mqtt_bridge.stop() raised: %s", exc)

    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except Exception as exc:
        logger.warning("MQTT teardown raised: %s", exc)

    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass

    try:
        db.log_event("=== System shutdown ===", "system")
    except Exception:
        pass

    print("\n\033[90m[Stopped]\033[0m")
    sys.exit(0)


if __name__ == "__main__":
    main()
