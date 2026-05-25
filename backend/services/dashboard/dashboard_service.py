"""DashboardService — creates the Flask application and wires all blueprints.

The Flask app reads state via StateManager and gets frames from VisionService
through the ServiceRegistry. No direct coupling to other services.
"""

import logging
import os
import threading

from flask import Flask

logger = logging.getLogger(__name__)

# Locate project root (two levels above this file: backend/services/dashboard → root)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def create_app(registry) -> Flask:
    """Construct the Flask application. Call once at startup."""
    app = Flask(
        __name__,
        template_folder=os.path.join(_PROJECT_ROOT, "templates"),
        static_folder=os.path.join(_PROJECT_ROOT, "static"),
    )
    app.config["registry"] = registry

    # Flask 3.x's DefaultJSONProvider sets sort_keys=True, which
    # alphabetically reorders dict keys on serialization. That breaks
    # any endpoint that returns a dict whose key order is semantic —
    # e.g. /api/snapshots/daywise, where we want today's date first
    # and previous days following. Disable it globally; routes that
    # need specific ordering can rely on Python's dict insertion
    # ordering.
    app.json.sort_keys = False

    # Register all route blueprints
    from backend.api.routes.control import bp as control_bp
    from backend.api.routes.snapshots import bp as snapshots_bp
    from backend.api.routes.alerts import bp as alerts_bp
    from backend.api.routes.logs import bp as logs_bp

    app.register_blueprint(control_bp)
    app.register_blueprint(snapshots_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(logs_bp)

    return app


def start_dashboard(app: Flask, port: int = 5000) -> None:
    """Start Flask in a daemon thread (non-blocking)."""
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        ),
        daemon=True,
    ).start()
    logger.info("Dashboard started on http://0.0.0.0:%d", port)
