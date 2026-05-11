"""Control routes — system, voice, mode, camera, video feed, and main page."""

import datetime
import time

import cv2
import numpy as np
from flask import Blueprint, Response, current_app, jsonify, render_template, request

from backend.core.constants import EV_VOICE_COMMAND

bp = Blueprint("control", __name__)


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
    # Capture the vision service reference HERE inside the request context,
    # then pass it into the generator as a closure. Calling current_app inside
    # the while-True loop of a streaming generator is unreliable in Flask.
    reg = current_app.config["registry"]
    placeholder = _make_placeholder()

    def _generator():
        last_id = None
        while True:
            vision = reg.get("vision")
            frame = vision.get_annotated_frame() if vision else None

            if frame is None:
                # Show a loading screen instead of a blank / broken image
                frame = placeholder
                time.sleep(0.5)

            fid = id(frame)
            if fid == last_id:
                time.sleep(0.008)
                continue
            last_id = fid

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )

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


@bp.route("/api/log")
def api_log():
    try:
        from backend.config.settings import LOG_FILE
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        lines = [l.strip() for l in open(LOG_FILE) if today in l][-30:]
        return jsonify({"lines": list(reversed(lines))})
    except Exception:
        return jsonify({"lines": []})


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
