"""Log routes — event log viewer."""

import sqlite3

from flask import Blueprint, jsonify, render_template

from backend.config.settings import DB_PATH

bp = Blueprint("logs", __name__)


@bp.route("/logs")
def page_logs():
    return render_template("logs.html")


@bp.route("/api/logs")
def api_logs():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM logs ORDER BY id DESC")]
        conn.close()
        grouped: dict = {}
        for r in rows:
            date = r.get("date") or (r["timestamp"].split(" ")[0] if r.get("timestamp") else "Unknown")
            if date not in grouped:
                grouped[date] = {"voice": [], "visitor": [], "alert": [], "system": [], "activity": []}
            ltype = r.get("log_type", "system")
            grouped[date].setdefault(ltype, []).append(r)
        return jsonify({"grouped": grouped})
    except Exception as e:
        return jsonify({"grouped": {}, "error": str(e)})
