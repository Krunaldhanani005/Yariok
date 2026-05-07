"""Yaariok Dashboard — Flask web server with live camera and real-time controls."""

from flask import Flask, Response, render_template, jsonify, request, send_from_directory
import cv2, threading, datetime, time, sqlite3, os

app   = Flask(__name__)
_s    = {}
_f    = [None]
_lock = threading.Lock()
_log  = ""

def init(state_dict, frame_ref, lock, log_file):
    global _s, _f, _lock, _log
    _s, _f, _lock, _log = state_dict, frame_ref, lock, log_file

# ── MJPEG stream ───────────────────────────────────────────────────────────────
def _gen():
    last_id = None
    while True:
        with _lock:
            frame = _f[0]
        if frame is None:
            time.sleep(0.02)
            continue
        fid = id(frame)
        if fid == last_id:
            time.sleep(0.008)
            continue
        last_id = fid
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes()
                   + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(
        _gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
            "X-Accel-Buffering": "no",
        },
    )

# ── Status API ─────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    sec  = _s.get("security_mode", False)
    kar  = _s.get("karaoke_mode", False)
    mode = "security" if sec else ("karaoke" if kar else "normal")
    return jsonify({
        "system_running":      _s.get("system_running", False),
        "voice_running":       _s.get("voice_running", False),
        "camera_status":       _s.get("camera_status", "Off"),
        "security_mode":       sec,
        "karaoke_mode":        kar,
        "current_mode":        mode,
        "people_now":          _s.get("people_now", 0),
        "visitors_today":      _s.get("visitors_today", 0),
        "alerts_today":        _s.get("alerts_today", 0),
        "snapshots_taken":     _s.get("snapshots_taken", 0),
        "last_live_alert":     _s.get("last_live_alert", ""),
        "last_live_alert_time":_s.get("last_live_alert_time", ""),
        "robot_status":        _s.get("robot_status", ""),
        "robot_speech":        _s.get("robot_speech", ""),
        "last_command":        _s.get("last_command", "—"),
        "last_visitor_b64":    _s.get("last_visitor_b64", ""),
        "ai_alert":            _s.get("ai_alert", ""),
    })

@app.route("/api/log")
def api_log():
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        lines = [l.strip() for l in open(_log).readlines() if today in l][-30:]
        return jsonify({"lines": list(reversed(lines))})
    except Exception:
        return jsonify({"lines": []})

# ── Control API ────────────────────────────────────────────────────────────────
ACTION_COMMANDS = {
    "security_on":   "enable security mode",
    "security_off":  "disable security",
    "karaoke_on":    "karaoke mode",
    "karaoke_off":   "stop karaoke",
    "normal_mode":   "normal mode",
    "take_snapshot": "take snapshot",
    "status_report": "status",
    "reset_visitors": "__reset__",
}

@app.route("/api/control", methods=["POST"])
def api_control():
    action = request.json.get("action", "")
    if action in ("__reset__", "reset_visitors"):
        _s["visitors_today"]  = 0
        _s["alerts_today"]    = 0
        _s["snapshots_taken"] = 0
        return jsonify({"ok": True})
    if action == "__text__":
        text = (request.json.get("text") or "").strip()
        if text:
            _s["dash_command"] = text
        return jsonify({"ok": bool(text)})
    cmd = ACTION_COMMANDS.get(action, "")
    if cmd:
        _s["dash_command"] = cmd
    return jsonify({"ok": bool(cmd), "cmd": cmd})

# ── System / Voice routes ──────────────────────────────────────────────────────
@app.route("/api/system/start", methods=["POST"])
def api_sys_start():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["start_system"]()
    return jsonify({"ok": True})

@app.route("/api/system/stop", methods=["POST"])
def api_sys_stop():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["stop_system"]()
    return jsonify({"ok": True})

@app.route("/api/voice/toggle", methods=["POST"])
def api_voice_toggle():
    if hasattr(app, "smart_funcs"):
        if _s.get("voice_running"):
            app.smart_funcs["stop_voice"]()
        else:
            app.smart_funcs["start_voice"]()
    return jsonify({"ok": True})

@app.route("/api/voice/on", methods=["POST"])
def api_voice_on():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["start_voice"]()
    return jsonify({"ok": True})

@app.route("/api/voice/off", methods=["POST"])
def api_voice_off():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["stop_voice"]()
    return jsonify({"ok": True})

# ── Mode routes ────────────────────────────────────────────────────────────────
@app.route("/api/mode/normal", methods=["POST"])
def api_mode_normal():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["set_mode"]("normal")
    return jsonify({"ok": True})

@app.route("/api/mode/security", methods=["POST"])
def api_mode_security():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["set_mode"]("security")
    return jsonify({"ok": True})

@app.route("/api/mode/karaoke", methods=["POST"])
def api_mode_karaoke():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["set_mode"]("karaoke")
    return jsonify({"ok": True})

# ── Camera routes ──────────────────────────────────────────────────────────────
@app.route("/api/camera/snapshot", methods=["POST"])
def api_camera_snapshot():
    if hasattr(app, "smart_funcs"):
        app.smart_funcs["take_snapshot"]()
    else:
        _s["take_snapshot"] = True
    return jsonify({"ok": True})

@app.route("/api/reset/stats", methods=["POST"])
def api_reset_stats():
    _s["visitors_today"]  = 0
    _s["alerts_today"]    = 0
    _s["snapshots_taken"] = 0
    try:
        open(_log, "w").close()
    except Exception:
        pass
    try:
        conn = sqlite3.connect("snapshots.db")
        c = conn.cursor()
        c.execute("DELETE FROM logs")
        conn.commit()
        conn.close()
    except Exception:
        pass
    return jsonify({"ok": True})

# ── Snapshots ──────────────────────────────────────────────────────────────────
@app.route("/snapshots")
def page_snapshots():
    return render_template("snapshots.html")

@app.route("/api/snapshots")
def api_snapshots():
    try:
        conn = sqlite3.connect("snapshots.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM snapshots ORDER BY id DESC")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify({"snapshots": rows})
    except Exception as e:
        return jsonify({"snapshots": [], "error": str(e)})

@app.route("/api/snapshots/daywise")
def api_snapshots_daywise():
    try:
        conn = sqlite3.connect("snapshots.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM snapshots ORDER BY timestamp DESC")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        grouped = {}
        for r in rows:
            date = r["timestamp"].split(" ")[0] if r.get("timestamp") else "Unknown"
            if date not in grouped:
                grouped[date] = []
            grouped[date].append(r)
        return jsonify({"grouped": grouped})
    except Exception as e:
        return jsonify({"grouped": {}, "error": str(e)})

@app.route("/api/snapshot/<int:sid>", methods=["DELETE"])
def api_del_snapshot(sid):
    try:
        conn = sqlite3.connect("snapshots.db")
        c = conn.cursor()
        c.execute("SELECT filename FROM snapshots WHERE id=?", (sid,))
        row = c.fetchone()
        if row:
            fpath = os.path.join("./snapshots/", row[0])
            if os.path.exists(fpath):
                os.remove(fpath)
        c.execute("DELETE FROM snapshots WHERE id=?", (sid,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/snapshots/day/<date>", methods=["DELETE"])
def api_del_snapshots_day(date):
    try:
        conn = sqlite3.connect("snapshots.db")
        c = conn.cursor()
        c.execute("SELECT filename FROM snapshots WHERE timestamp LIKE ?", (f"{date}%",))
        rows = c.fetchall()
        for row in rows:
            fpath = os.path.join("./snapshots/", row[0])
            if os.path.exists(fpath):
                os.remove(fpath)
        c.execute("DELETE FROM snapshots WHERE timestamp LIKE ?", (f"{date}%",))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return send_from_directory('./snapshots', filename)

# ── Alerts ─────────────────────────────────────────────────────────────────────
@app.route("/alerts")
def page_alerts():
    return render_template("alerts.html")

@app.route("/api/alerts")
def api_alerts():
    try:
        conn = sqlite3.connect("snapshots.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY id DESC")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify({"alerts": rows})
    except Exception as e:
        return jsonify({"alerts": [], "error": str(e)})

@app.route("/api/alerts/daywise")
def api_alerts_daywise():
    try:
        conn = sqlite3.connect("snapshots.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        grouped = {}
        for r in rows:
            date = r["timestamp"].split(" ")[0] if r.get("timestamp") else "Unknown"
            if date not in grouped:
                grouped[date] = []
            grouped[date].append(r)
        return jsonify({"grouped": grouped})
    except Exception as e:
        return jsonify({"grouped": {}, "error": str(e)})

@app.route("/api/alert/<int:sid>", methods=["DELETE"])
def api_del_alert(sid):
    try:
        conn = sqlite3.connect("snapshots.db")
        c = conn.cursor()
        c.execute("SELECT image_path FROM alerts WHERE id=?", (sid,))
        row = c.fetchone()
        if row:
            fpath = os.path.join("./alerts/", row[0])
            if os.path.exists(fpath):
                os.remove(fpath)
        c.execute("DELETE FROM alerts WHERE id=?", (sid,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False})

@app.route('/alerts/<path:filename>')
def serve_alert(filename):
    return send_from_directory('./alerts', filename)

# ── Logs ───────────────────────────────────────────────────────────────────────
@app.route("/logs")
def page_logs():
    return render_template("logs.html")

@app.route("/api/logs")
def api_logs():
    try:
        conn = sqlite3.connect("snapshots.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM logs ORDER BY id DESC")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        grouped = {}
        for r in rows:
            date = r.get("date") or (r["timestamp"].split(" ")[0] if r.get("timestamp") else "Unknown")
            if date not in grouped:
                grouped[date] = {"voice": [], "visitor": [], "alert": [], "system": []}
            ltype = r.get("log_type", "system")
            if ltype not in grouped[date]:
                grouped[date][ltype] = []
            grouped[date][ltype].append(r)
        return jsonify({"grouped": grouped})
    except Exception as e:
        return jsonify({"grouped": {}, "error": str(e)})

# ── Main page ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

def start_dashboard(state, frame_ref, lock, log_file, port=5000):
    init(state, frame_ref, lock, log_file)
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=port,
            debug=False, use_reloader=False, threaded=True,
        ),
        daemon=True,
    ).start()
    return port
