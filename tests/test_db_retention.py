"""Tests for DatabaseService.prune_old_snapshots — time AND count based."""

import datetime as dt
import os
import sqlite3
import tempfile
import unittest
from unittest import mock


class TestPrune(unittest.TestCase):

    def setUp(self):
        # Each test gets a fresh DB + snapshot dir on tmpfs.
        self.dbfd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.dbfd)
        os.remove(self.db_path)
        self.snap_dir = tempfile.mkdtemp(prefix="snaps_")

        # Patch the module-level constants the service reads.
        self._patches = [
            mock.patch("backend.services.database.db_service.DB_PATH", self.db_path),
            mock.patch("backend.services.database.db_service.SNAPSHOT_DIR", self.snap_dir),
        ]
        for p in self._patches:
            p.start()

        from backend.services.database.db_service import DatabaseService
        self.db = DatabaseService()
        self.db.init_db()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        try: os.remove(self.db_path)
        except OSError: pass
        for f in os.listdir(self.snap_dir):
            try: os.unlink(os.path.join(self.snap_dir, f))
            except OSError: pass
        try: os.rmdir(self.snap_dir)
        except OSError: pass

    def _insert(self, filename: str, timestamp: dt.datetime, visitor_id=None):
        """Backdoor INSERT bypassing the normal save_snapshot (which uses
        the current time) so we can simulate aged rows."""
        # Also create the actual file on disk
        with open(os.path.join(self.snap_dir, filename), "wb") as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 32)  # fake JPEG header
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO snapshots (filename, timestamp, visitor_type, notes, visitor_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (filename, timestamp.strftime("%Y-%m-%d %H:%M:%S"),
             "Visitor", "", visitor_id),
        )
        conn.commit()
        conn.close()

    def test_prune_empty_db_returns_zero(self):
        result = self.db.prune_old_snapshots(days=7, max_count=100)
        self.assertEqual(result["deleted_rows"], 0)
        self.assertEqual(result["deleted_files"], 0)
        self.assertEqual(result["kept"], 0)

    def test_prune_deletes_old_files(self):
        now = dt.datetime.now()
        # 3 fresh rows + 2 old rows (10 days back)
        for i in range(3):
            self._insert(f"fresh_{i}.jpg", now - dt.timedelta(minutes=i))
        for i in range(2):
            self._insert(f"old_{i}.jpg", now - dt.timedelta(days=10))

        result = self.db.prune_old_snapshots(days=7, max_count=100)
        self.assertEqual(result["deleted_rows"], 2)
        self.assertEqual(result["deleted_files"], 2)
        self.assertEqual(result["kept"], 3)

        # The OLD files should be gone from disk
        for i in range(2):
            self.assertFalse(os.path.exists(os.path.join(self.snap_dir, f"old_{i}.jpg")))
        # The fresh ones should still be there
        for i in range(3):
            self.assertTrue(os.path.exists(os.path.join(self.snap_dir, f"fresh_{i}.jpg")))

    def test_prune_enforces_max_count(self):
        now = dt.datetime.now()
        # 10 fresh rows — but max_count = 3
        for i in range(10):
            self._insert(f"r_{i}.jpg", now - dt.timedelta(seconds=i))
        result = self.db.prune_old_snapshots(days=999, max_count=3)
        self.assertEqual(result["deleted_rows"], 7)
        self.assertEqual(result["kept"], 3)

    def test_prune_handles_missing_files_gracefully(self):
        """If the JPG was already deleted (e.g. manual cleanup), pruning
        the DB row must still succeed and not raise."""
        now = dt.datetime.now()
        self._insert("ghost.jpg", now - dt.timedelta(days=10))
        os.remove(os.path.join(self.snap_dir, "ghost.jpg"))
        result = self.db.prune_old_snapshots(days=7, max_count=100)
        # Row is deleted, file count is 0 (file was already gone)
        self.assertEqual(result["deleted_rows"], 1)
        self.assertEqual(result["deleted_files"], 0)


if __name__ == "__main__":
    unittest.main()
