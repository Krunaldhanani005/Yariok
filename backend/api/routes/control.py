"""Control routes — system, voice, mode, camera, video feed, and main page."""

import json
import queue
import time

import cv2
import numpy as np
from flask import Blueprint, Response, current_app, jsonify, render_template, request

from backend.core.constants import EV_VOICE_COMMAND

bp = Blueprint("control", __name__)

# How long the SSE generator waits before sending a keep-alive comment.
# Some HTTP intermediaries close connections idle for >30s; 15s is a
# safe interval that's invisible to the browser.
SSE_KEEPALIVE_SECONDS: float = 15.0


def _registry():
    return current_app.config["registry"]


def _state():
    return _registry().get("state")


def _bus():
    return _registry().get("bus")


# ── Placeholder frame (shown while camera / YOLO is loading) ─────────────────────

def _make_placeholder() -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera loading...", (140, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 160, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Please wait", (210, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
    return frame


# ── MJPEG camera stream ──────────────────────────────────────────────────────────

@bp.route("/video_feed")
def video_feed():
    # Phase B: stream comes from the new VisionService's bounded queue.
    # vision.get_video_feed() yields already-multipart-encoded chunks
    # (boundary + JPEG bytes). We must NOT poll get_annotated_frame()
    # any more — that method belonged to the legacy camera_service.
    reg = current_app.config["registry"]
    vision = reg.get("vision")
    placeholder = _make_placeholder()

    def _generator():
        if vision is None:
            # Vision service not registered — serve a single placeholder so the
            # <img> tag doesn't show a broken icon. Browser will retry.
            ok, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes() + b"\r\n"
                )
            return
        # Delegate directly to the VisionService generator. It already
        # produces "--frame\r\nContent-Type: image/jpeg\r\n\r\n<bytes>\r\n"
        # framing and respects the bounded queue (maxsize=5, drop-oldest).
        yield from vision.get_video_feed()

    return Response(
        _generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


# ── Main page ────────────────────────────────────────────────────────────────────

@bp.route("/")
def index():
    return render_template("index.html")


# ── Real-time push (Server-Sent Events) ──────────────────────────────────────────
#
# The dashboard's polling /api/status is still served — it stays as a
# fallback. /api/events is the new push channel:
#
#   • On connect, we send one ``snapshot`` event containing the full
#     current state so the UI is populated immediately.
#   • Every time the bridge mutates state (visitor, presence, camera_status),
#     we receive a small delta dict via our subscriber queue and stream it
#     out as a single ``data: {...}\n\n`` line.
#   • Every 15 s of silence we emit a ``:`` comment line as a keep-alive
#     so HTTP intermediaries don't close the idle connection.
#
# The subscriber queue is registered/unregistered via bridge.subscribe()
# and bridge.unsubscribe(); the ``finally`` block guarantees cleanup even
# when the browser tab is closed.

@bp.route("/api/events")
def api_events():
    reg = _registry()
    bridge = reg.get("mqtt_bridge")
    state = reg.get("state")
    if bridge is None or state is None:
        return Response("bridge not available", status=503)

    q = bridge.subscribe()

    def _stream():
        try:
            # Initial full snapshot so the dashboard renders immediately
            # without waiting for the first delta event.
            initial = {"type": "snapshot", **state.snapshot()}
            yield f"data: {json.dumps(initial, default=str)}\n\n"

            while True:
                try:
                    msg = q.get(timeout=SSE_KEEPALIVE_SECONDS)
                    yield f"data: {json.dumps(msg, default=str)}\n\n"
                except queue.Empty:
                    # Idle keep-alive — colon-prefixed comment, ignored
                    # by EventSource. Defeats proxy idle timeouts.
                    yield f": ping {int(time.time())}\n\n"
        except GeneratorExit:
            # Browser closed the tab — exit cleanly.
            pass
        finally:
            bridge.unsubscribe(q)

    return Response(
        _stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",   # disable proxy buffering (nginx)
            "Connection": "keep-alive",
        },
    )


# ── Status API ────────────────────────────────────────────────────────────────────

@bp.route("/api/status")
def api_status():
    s = _state()
    sec = s.get("security_mode", False)
    kar = s.get("karaoke_mode", False)
    return jsonify({
        "system_running":       s.get("system_running", False),
        "voice_running":        s.get("voice_running", False),
        "camera_status":        s.get("camera_status", "Off"),
        "security_mode":        sec,
        "karaoke_mode":         kar,
        "current_mode":         "security" if sec else ("karaoke" if kar else "normal"),
        "people_now":           s.get("people_now", 0),
        "visitors_today":       s.get("visitors_today", 0),
        "alerts_today":         s.get("alerts_today", 0),
        "snapshots_taken":      s.get("snapshots_taken", 0),
        "last_live_alert":      s.get("last_live_alert", ""),
        "last_live_alert_time": s.get("last_live_alert_time", ""),
        "robot_status":         s.get("robot_status", ""),
        "robot_speech":         s.get("robot_speech", ""),
        "last_command":         s.get("last_command", "—"),
        "last_visitor_b64":     s.get("last_visitor_b64", ""),
        "ai_alert":             s.get("ai_alert", ""),
    })


# ── Control API ───────────────────────────────────────────────────────────────────

_ACTION_MAP = {
    "security_on":   "security mode",
    "security_off":  "normal mode",
    "karaoke_on":    "karaoke mode",
    "karaoke_off":   "normal mode",
    "normal_mode":   "normal mode",
    "take_snapshot": "snapshot",
    "status_report": "status",
}


@bp.route("/api/control", methods=["POST"])
def api_control():
    data = request.json or {}
    action = data.get("action", "")

    # Special actions handled here
    if action in ("reset_visitors", "__reset__"):
        automation = _registry().get("automation")
        if automation:
            automation.reset_stats()
        return jsonify({"ok": True})

    if action == "__text__":
        text = (data.get("text") or "").strip()
        if text:
            _bus().publish(EV_VOICE_COMMAND, command=text)
        return jsonify({"ok": bool(text)})

    # Map action key to a voice command string
    cmd = _ACTION_MAP.get(action, "")
    if cmd:
        _bus().publish(EV_VOICE_COMMAND, command=cmd)
    return jsonify({"ok": bool(cmd), "cmd": cmd})


# ── System routes ─────────────────────────────────────────────────────────────────

@bp.route("/api/system/start", methods=["POST"])
def api_sys_start():
    automation = _registry().get("automation")
    if automation:
        automation.start_system()
    return jsonify({"ok": True})


@bp.route("/api/system/stop", methods=["POST"])
def api_sys_stop():
    automation = _registry().get("automation")
    if automation:
        automation.stop_system()
    return jsonify({"ok": True})


# ── Voice routes ──────────────────────────────────────────────────────────────────

@bp.route("/api/voice/toggle", methods=["POST"])
def api_voice_toggle():
    voice = _registry().get("voice")
    if voice:
        if _state().get("voice_running"):
            voice.stop()
        else:
            voice.start()
    return jsonify({"ok": True})


@bp.route("/api/voice/on", methods=["POST"])
def api_voice_on():
    voice = _registry().get("voice")
    if voice:
        voice.start()
    return jsonify({"ok": True})


@bp.route("/api/voice/off", methods=["POST"])
def api_voice_off():
    voice = _registry().get("voice")
    if voice:
        voice.stop()
    return jsonify({"ok": True})


# ── Mode routes ───────────────────────────────────────────────────────────────────

@bp.route("/api/mode/normal", methods=["POST"])
def api_mode_normal():
    automation = _registry().get("automation")
    if automation:
        automation.set_mode("normal")
    return jsonify({"ok": True})


@bp.route("/api/mode/security", methods=["POST"])
def api_mode_security():
    automation = _registry().get("automation")
    if automation:
        automation.set_mode("security")
    return jsonify({"ok": True})


@bp.route("/api/mode/karaoke", methods=["POST"])
def api_mode_karaoke():
    automation = _registry().get("automation")
    if automation:
        automation.set_mode("karaoke")
    return jsonify({"ok": True})


# ── Camera routes ─────────────────────────────────────────────────────────────────

@bp.route("/api/camera/snapshot", methods=["POST"])
def api_camera_snapshot():
    automation = _registry().get("automation")
    if automation:
        automation.take_snapshot()
    return jsonify({"ok": True})


# ── Entry-zone polygon (greeting gate) ───────────────────────────────────────────
#
# GET  /api/zone — returns the currently-stored normalized polygon (or
#                  ``{"polygon": []}`` if none has been configured yet).
# POST /api/zone — body: ``{"polygon": [[x,y], [x,y], ...]}`` with all
#                  coords in [0.0, 1.0]. Writes to ``zone_config.json``
#                  atomically and tells VisionService to hot-reload so
#                  the change is live with no restart.
#
# The polygon is stored normalized so it survives stream resolution
# changes — the very point of the spec. Coords are validated server-side;
# a malformed payload returns 400 without touching the on-disk file.

@bp.route("/api/zone", methods=["GET"])
def api_zone_get():
    import os
    from backend.config.settings import ZONE_CONFIG_FILE
    if not os.path.exists(ZONE_CONFIG_FILE):
        return jsonify({"polygon": []})
    try:
        with open(ZONE_CONFIG_FILE) as f:
            data = json.load(f)
        polygon = data.get("polygon", [])
        if not isinstance(polygon, list):
            polygon = []
        return jsonify({"polygon": polygon})
    except Exception:
        return jsonify({"polygon": []})


@bp.route("/api/zone", methods=["POST"])
def api_zone_post():
    import os
    import tempfile
    from backend.config.settings import ZONE_CONFIG_FILE

    data = request.get_json(silent=True) or {}
    polygon_in = data.get("polygon", [])
    if not isinstance(polygon_in, list):
        return jsonify({"ok": False, "error": "polygon must be a list"}), 400

    # Validate every point: must be [x, y] with x,y in [0, 1].
    cleaned = []
    for pt in polygon_in:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            return jsonify({"ok": False, "error": "each point must be [x, y]"}), 400
        x, y = pt
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return jsonify({"ok": False, "error": "coords must be numeric"}), 400
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return jsonify(
                {"ok": False, "error": "coords must be normalized [0, 1]"}
            ), 400
        cleaned.append([float(x), float(y)])

    # An empty polygon is allowed — it means "clear the zone, greet
    # everywhere". 1- or 2-point polygons are rejected because they
    # can't enclose anything; VisionService would treat them as empty
    # anyway but the server-side validation gives a clearer 400.
    if 0 < len(cleaned) < 3:
        return jsonify(
            {"ok": False, "error": "polygon must have at least 3 points (or be empty)"}
        ), 400

    payload = {"polygon": cleaned, "version": 1}

    # Atomic write — same recipe as StateManager / DailyFaceCache so a
    # crash mid-flush can't leave a half-written JSON.
    try:
        dir_name = os.path.dirname(ZONE_CONFIG_FILE) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".zone_config.", suffix=".tmp", dir=dir_name,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, ZONE_CONFIG_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:
        current_app.logger.exception("zone save failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

    # Hot-reload VisionService so the change is live immediately.
    vision = _registry().get("vision")
    if vision is not None:
        try:
            vision.reload_zone()
        except Exception:
            current_app.logger.exception("vision.reload_zone failed")

    return jsonify({"ok": True, "points": len(cleaned)})


# ── First-time wipe (dashboard "START SYSTEM FIRST TIME" button) ─────────────────

@bp.route("/api/system/reset_all", methods=["POST"])
def api_system_reset_all():
    """Wipe ALL persisted + in-memory state so the system behaves as a
    brand-new install. Used by the "Start System First Time" button on
    the dashboard for repeated testing.

    Wipe order is deliberate:
      1. Vision identity cache FIRST — so any in-flight face job that
         finishes mid-wipe sees an empty FAISS index and treats every
         track as new.
      2. StateManager counters — resets the on-disk daily_state.json too.
      3. SQLite tables — snapshots / logs / alerts.
      4. Files on disk — JPEGs in snapshots/ and alerts/, visitor_log.txt.
      5. (Optionally) start the system afterwards via the existing route.

    NOT safe in production: there is no auth, and a concurrent vision
    event may write a row that survives the wipe by milliseconds.
    """
    import glob
    import os
    import sqlite3
    from backend.config.settings import ALERTS_DIR, DB_PATH, LOG_FILE, SNAPSHOT_DIR

    reg = _registry()
    summary = {
        "vision_cache": False,
        "state_counters": False,
        "db_rows_deleted": 0,
        "snapshot_files_deleted": 0,
        "alert_files_deleted": 0,
        "visitor_log_deleted": False,
    }

    # 1) In-memory identity (FAISS index, body library, known_bot_sort_ids).
    vision = reg.get("vision")
    if vision is not None:
        try:
            vision.factory_reset()
            summary["vision_cache"] = True
        except Exception:
            current_app.logger.exception("vision.factory_reset failed")

    # 2) StateManager + daily_state.json.
    state = reg.get("state")
    if state is not None:
        try:
            state.reset_daily_counters()
            # Also clear cosmetic fields the dashboard reads so the UI
            # doesn't briefly show a stale "last_visitor_b64" image after
            # the wipe.
            state.update(
                last_visitor_b64="",
                last_alert_time="—",
                last_live_alert="",
                last_live_alert_time="—",
                ai_alert="",
                people_now=0,
            )
            summary["state_counters"] = True
        except Exception:
            current_app.logger.exception("state.reset_daily_counters failed")

    # 3) SQLite tables.
    try:
        conn = sqlite3.connect(DB_PATH)
        for tbl in ("snapshots", "logs", "alerts"):
            cur = conn.execute(f"DELETE FROM {tbl}")
            summary["db_rows_deleted"] += cur.rowcount or 0
            conn.execute("DELETE FROM sqlite_sequence WHERE name=?", (tbl,))
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
    except Exception:
        current_app.logger.exception("DB truncate failed")

    # 4) Files on disk.
    for d, key in ((SNAPSHOT_DIR, "snapshot_files_deleted"),
                   (ALERTS_DIR, "alert_files_deleted")):
        try:
            for f in glob.glob(os.path.join(d, "*.jpg")):
                try:
                    os.unlink(f)
                    summary[key] += 1
                except OSError:
                    pass
        except Exception:
            current_app.logger.exception("failed to clean %s", d)

    try:
        if os.path.exists(LOG_FILE):
            os.unlink(LOG_FILE)
            summary["visitor_log_deleted"] = True
    except OSError:
        pass

    # 5b) Identity pickle. vision.factory_reset() above already
    # overwrites it with empty state, but explicitly removing it
    # protects against a stale file lingering if the cache write
    # failed silently.
    try:
        identity_file = os.path.join(os.path.dirname(DB_PATH), "daily_identity.pkl")
        if os.path.exists(identity_file):
            os.unlink(identity_file)
            summary["identity_cache_deleted"] = True
    except OSError:
        pass

    return jsonify({"ok": True, "summary": summary})


# ── Stats reset ───────────────────────────────────────────────────────────────────

@bp.route("/api/reset/stats", methods=["POST"])
def api_reset_stats():
    automation = _registry().get("automation")
    if automation:
        automation.reset_stats()
    db = _registry().get("db")
    if db:
        try:
            from backend.config.settings import DB_PATH
            import sqlite3
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM logs")
            conn.commit()
            conn.close()
        except Exception:
            pass
    return jsonify({"ok": True})
