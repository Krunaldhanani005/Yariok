"""Alert routes — alert gallery, delete, file serving."""

import os
import sqlite3

from flask import Blueprint, jsonify, render_template, send_from_directory

from backend.config.settings import ALERTS_DIR, DB_PATH

bp = Blueprint("alerts", __name__)


@bp.route("/alerts")
def page_alerts():
    return render_template("alerts.html")


@bp.route("/api/alerts")
def api_alerts():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM alerts ORDER BY id DESC")]
        conn.close()
        return jsonify({"alerts": rows})
    except Exception as e:
        return jsonify({"alerts": [], "error": str(e)})


@bp.route("/api/alerts/daywise")
def api_alerts_daywise():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC")]
        conn.close()
        grouped: dict = {}
        for r in rows:
            date = r["timestamp"].split(" ")[0] if r.get("timestamp") else "Unknown"
            grouped.setdefault(date, []).append(r)
        return jsonify({"grouped": grouped})
    except Exception as e:
        return jsonify({"grouped": {}, "error": str(e)})


@bp.route("/api/alert/<int:sid>", methods=["DELETE"])
def api_del_alert(sid):
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT image_path FROM alerts WHERE id=?", (sid,)).fetchone()
        if row:
            fpath = os.path.join(ALERTS_DIR, row[0])
            if os.path.exists(fpath):
                os.remove(fpath)
        conn.execute("DELETE FROM alerts WHERE id=?", (sid,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False})


@bp.route("/alerts/<path:filename>")
def serve_alert(filename):
    return send_from_directory(ALERTS_DIR, filename)
