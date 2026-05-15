"""Snapshot routes — gallery, individual delete, day delete, file serving."""

import os
import sqlite3

from flask import Blueprint, jsonify, render_template, send_from_directory

from backend.config.settings import DB_PATH, SNAPSHOT_DIR
from backend.core.business_day import business_day_from_iso, current_business_day

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
    """Snapshots nested by BUSINESS DAY (09:00 → 09:00 IST) AND by visitor_id.

    Response shape::

        {
          "grouped": {
            "2026-05-14": {
              "visitors": [
                {"visitor_id": 0, "label": "Visitor #1", "count": 3,
                 "first_seen": "...", "last_seen": "...",
                 "snaps": [<row>, <row>, <row>]},
                ...
              ],
              "ungrouped": [<row>, ...]    # manual / legacy without visitor_id
            },
            ...
          },
          "current_business_day": "2026-05-15"
        }

    A snapshot taken at 08:50 IST is bucketed under the previous date
    (yesterday's business day); 09:01 IST starts a new bucket. Rows
    inside each visitor's ``snaps`` are oldest-first so the UI can
    show how that person looked over the course of the day.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [
            dict(r) for r in conn.execute(
                "SELECT * FROM snapshots ORDER BY timestamp DESC"
            )
        ]
        conn.close()

        # First pass: bucket by BUSINESS DAY → by visitor_id.
        by_date: dict = {}
        for r in rows:
            date = business_day_from_iso(r.get("timestamp") or "")
            day = by_date.setdefault(date, {"_by_vid": {}, "ungrouped": []})
            vid = r.get("visitor_id")
            if vid is None:
                day["ungrouped"].append(r)
                continue
            v = day["_by_vid"].setdefault(int(vid), {
                "visitor_id": int(vid),
                "label": f"Visitor #{int(vid) + 1}",
                "snaps": [],
            })
            v["snaps"].append(r)

        # Second pass: sort each visitor's snaps oldest→newest, compute
        # the timestamps, then sort visitors most-recent-activity first.
        grouped: dict = {}
        for date, day in by_date.items():
            visitors = []
            for vid_data in day["_by_vid"].values():
                vid_data["snaps"].sort(key=lambda s: s.get("timestamp") or "")
                vid_data["count"] = len(vid_data["snaps"])
                vid_data["first_seen"] = vid_data["snaps"][0].get("timestamp", "")
                vid_data["last_seen"] = vid_data["snaps"][-1].get("timestamp", "")
                visitors.append(vid_data)
            visitors.sort(key=lambda v: v["last_seen"], reverse=True)
            grouped[date] = {"visitors": visitors, "ungrouped": day["ungrouped"]}

        return jsonify({
            "grouped": grouped,
            # The UI labels "Today" against the BUSINESS day, not the
            # calendar date — at 08:50 IST we're still in yesterday's
            # business day and the gallery should reflect that.
            "current_business_day": current_business_day().isoformat(),
        })
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
