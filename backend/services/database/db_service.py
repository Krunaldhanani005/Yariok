"""DatabaseService — sole owner of all SQLite I/O.

No other service connects to the DB directly. All reads/writes go through
this class to keep the data access layer isolated and testable.
"""

import datetime
import logging
import os
import sqlite3

from backend.config.settings import DB_PATH, SNAPSHOT_DIR

logger = logging.getLogger(__name__)


class DatabaseService:

    def init_db(self) -> None:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT, timestamp TEXT,
                visitor_type TEXT, notes TEXT
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
        now = datetime.datetime.now()
        self._execute(
            "INSERT INTO alerts (image_path, timestamp, date, alert_type, status, person_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (filename, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
             alert_type, status, person_count),
        )

    def save_snapshot(self, filename: str, vtype: str = "", notes: str = "") -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._execute(
            "INSERT INTO snapshots (filename, timestamp, visitor_type, notes) VALUES (?, ?, ?, ?)",
            (filename, ts, vtype, notes),
        )

    def save_log(self, log_type: str, message: str, person_count: int = 0) -> None:
        now = datetime.datetime.now()
        self._execute(
            "INSERT INTO logs (log_type, message, person_count, timestamp, date) "
            "VALUES (?, ?, ?, ?, ?)",
            (log_type, message, person_count,
             now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d")),
        )

    def log_event(self, event: str, log_type: str = "system", person_count: int = 0) -> None:
        """Write to both the flat text log and the DB logs table."""
        from backend.config.settings import LOG_FILE
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"[{ts}] {event}\n")
        except Exception:
            pass
        self.save_log(log_type, event, person_count)

    def delete_today_snapshots(self) -> int:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
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
