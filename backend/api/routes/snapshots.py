"""Snapshot routes — gallery, individual delete, day delete, file serving."""

import os
import sqlite3

from flask import Blueprint, jsonify, render_template, send_from_directory

from backend.config.settings import DB_PATH, SNAPSHOT_DIR

bp = Blueprint("snapshots", __name__)


@bp.route("/snapshots")
def page_snapshots():
    return render_template("snapshots.html")


@bp.route("/api/snapshots")
def api_snapshots():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM snapshots ORDER BY id DESC")]
        conn.close()
        return jsonify({"snapshots": rows})
    except Exception as e:
        return jsonify({"snapshots": [], "error": str(e)})


@bp.route("/api/snapshots/daywise")
def api_snapshots_daywise():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM snapshots ORDER BY timestamp DESC")]
        conn.close()
        grouped: dict = {}
        for r in rows:
            date = r["timestamp"].split(" ")[0] if r.get("timestamp") else "Unknown"
            grouped.setdefault(date, []).append(r)
        return jsonify({"grouped": grouped})
    except Exception as e:
        return jsonify({"grouped": {}, "error": str(e)})


@bp.route("/api/snapshot/<int:sid>", methods=["DELETE"])
def api_del_snapshot(sid):
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT filename FROM snapshots WHERE id=?", (sid,)).fetchone()
        if row:
            fpath = os.path.join(SNAPSHOT_DIR, row[0])
            if os.path.exists(fpath):
                os.remove(fpath)
        conn.execute("DELETE FROM snapshots WHERE id=?", (sid,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/api/snapshots/day/<date>", methods=["DELETE"])
def api_del_snapshots_day(date):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT filename FROM snapshots WHERE timestamp LIKE ?",
                            (f"{date}%",)).fetchall()
        for (fn,) in rows:
            fpath = os.path.join(SNAPSHOT_DIR, fn)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
        conn.execute("DELETE FROM snapshots WHERE timestamp LIKE ?", (f"{date}%",))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)
