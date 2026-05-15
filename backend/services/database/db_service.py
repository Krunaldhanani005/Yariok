"""DatabaseService — sole owner of all SQLite I/O.

No other service connects to the DB directly. All reads/writes go through
this class to keep the data access layer isolated and testable.
"""

import datetime
import logging
import os
import sqlite3

from backend.config.settings import DB_PATH, SNAPSHOT_DIR
from backend.core.business_day import now_ist

logger = logging.getLogger(__name__)


class DatabaseService:

    def init_db(self) -> None:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT, timestamp TEXT,
                visitor_type TEXT, notes TEXT,
                visitor_id INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT, timestamp TEXT, date TEXT,
                alert_type TEXT, status TEXT, person_count INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_type TEXT, message TEXT, person_count INTEGER,
                timestamp TEXT, date TEXT
            )
        """)
        # Safe schema migrations — idempotent
        for stmt in [
            "ALTER TABLE logs ADD COLUMN person_count INTEGER DEFAULT 0",
            "ALTER TABLE alerts ADD COLUMN date TEXT",
            "ALTER TABLE alerts ADD COLUMN person_count INTEGER DEFAULT 0",
            # visitor_id added in Phase B so the gallery can cluster
            # multiple snapshots of the same person.
            "ALTER TABLE snapshots ADD COLUMN visitor_id INTEGER",
        ]:
            try:
                c.execute(stmt)
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()
        logger.info("Database initialised: %s", DB_PATH)

    # ── Write operations ────────────────────────────────────────────────────────

    def save_alert(
        self,
        filename: str,
        alert_type: str = "Person Detected",
        status: str = "Active",
        person_count: int = 0,
    ) -> None:
        now = now_ist()
        self._execute(
            "INSERT INTO alerts (image_path, timestamp, date, alert_type, status, person_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (filename, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
             alert_type, status, person_count),
        )

    def save_snapshot(
        self,
        filename: str,
        vtype: str = "",
        notes: str = "",
        visitor_id: int = None,
    ) -> None:
        """Persist a snapshot row.

        ``visitor_id`` is the daily-cache visitor index (0-based) for
        this snapshot, used by the gallery to cluster multiple frames of
        the same person under one card. ``None`` for manual/legacy
        snapshots so the gallery can render them as ungrouped tiles.
        """
        ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        self._execute(
            "INSERT INTO snapshots (filename, timestamp, visitor_type, notes, visitor_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (filename, ts, vtype, notes,
             int(visitor_id) if visitor_id is not None else None),
        )

    def save_log(self, log_type: str, message: str, person_count: int = 0) -> None:
        now = now_ist()
        self._execute(
            "INSERT INTO logs (log_type, message, person_count, timestamp, date) "
            "VALUES (?, ?, ?, ?, ?)",
            (log_type, message, person_count,
             now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d")),
        )

    def log_event(self, event: str, log_type: str = "system", person_count: int = 0) -> None:
        """Write to both the flat text log and the DB logs table."""
        from backend.config.settings import LOG_FILE
        ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"[{ts}] {event}\n")
        except Exception:
            pass
        self.save_log(log_type, event, person_count)

    def prune_old_snapshots(self, days: int, max_count: int) -> dict:
        """Delete snapshot files + DB rows that exceed retention limits.

        Two filters combine — a row is deleted if **either** is true:

          - its ``timestamp`` is older than ``days`` days, **or**
          - it falls outside the newest ``max_count`` rows.

        For each deleted row we also try to unlink the file in
        ``SNAPSHOT_DIR``; a missing file is logged at DEBUG and not a
        failure. The DB delete commits even if some files couldn't be
        removed, so an unrelated permissions issue can't permanently
        wedge retention.

        Returns:
            ``{"deleted_rows": N, "deleted_files": M, "kept": K}``.
        """
        summary = {"deleted_rows": 0, "deleted_files": 0, "kept": 0}
        try:
            cutoff = (now_ist()
                      - datetime.timedelta(days=int(days))
                      ).strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()

            # 1) Time-based: anything older than cutoff.
            c.execute(
                "SELECT id, filename FROM snapshots WHERE timestamp < ?",
                (cutoff,),
            )
            old = c.fetchall()

            # 2) Count-based: rows ranked beyond max_count (newest first).
            #    SQLite needs OFFSET combined with a LIMIT; ``-1`` means
            #    "no limit" so the OFFSET selects the entire tail.
            c.execute(
                "SELECT id, filename FROM snapshots "
                "ORDER BY id DESC LIMIT -1 OFFSET ?",
                (int(max_count),),
            )
            extras = c.fetchall()

            # Combine while preserving uniqueness.
            doomed: dict = {}
            for rid, fn in old + extras:
                doomed[rid] = fn

            # Delete files first (best-effort), then DB rows.
            for rid, fn in doomed.items():
                if not fn:
                    continue
                fpath = os.path.join(SNAPSHOT_DIR, fn)
                try:
                    if os.path.exists(fpath):
                        os.unlink(fpath)
                        summary["deleted_files"] += 1
                except OSError:
                    logger.debug("prune: could not unlink %s", fpath)

            if doomed:
                # Single bulk delete keeps it atomic.
                placeholders = ",".join("?" for _ in doomed)
                c.execute(
                    f"DELETE FROM snapshots WHERE id IN ({placeholders})",
                    tuple(doomed.keys()),
                )
                summary["deleted_rows"] = c.rowcount or 0

            c.execute("SELECT COUNT(*) FROM snapshots")
            summary["kept"] = int(c.fetchone()[0])

            conn.commit()
            conn.close()
            if summary["deleted_rows"] or summary["deleted_files"]:
                logger.info(
                    "Snapshot retention: pruned %d rows / %d files "
                    "(kept %d, cutoff=%s, max_count=%d)",
                    summary["deleted_rows"], summary["deleted_files"],
                    summary["kept"], cutoff, max_count,
                )
        except Exception:
            logger.exception("prune_old_snapshots failed")
        return summary

    def delete_today_snapshots(self) -> int:
        today = now_ist().strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT filename FROM snapshots WHERE timestamp LIKE ?", (f"{today}%",))
        rows = c.fetchall()
        for (fn,) in rows:
            fpath = os.path.join(SNAPSHOT_DIR, fn)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
        c.execute("DELETE FROM snapshots WHERE timestamp LIKE ?", (f"{today}%",))
        conn.commit()
        conn.close()
        return len(rows)

    # ── Internal helper ─────────────────────────────────────────────────────────

    def _execute(self, sql: str, params: tuple = ()) -> None:
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(sql, params)
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("DB write failed: %s", sql[:60])
