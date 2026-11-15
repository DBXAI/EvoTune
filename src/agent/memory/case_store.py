"""SQLite persistence for memory cases; retrievers still use in-memory cases."""

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path


logger = logging.getLogger(__name__)


class SQLiteCaseStore:
    def __init__(self, sqlite_path, legacy_json_path):
        self.sqlite_path = Path(sqlite_path)
        self.legacy_json_path = Path(legacy_json_path)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.sqlite_path), timeout=30)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self):
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS cases ("
                    "case_id TEXT PRIMARY KEY, "
                    "position INTEGER NOT NULL UNIQUE, "
                    "payload TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS metadata ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                migrated = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'migration_complete'"
                ).fetchone()
                if migrated is None:
                    if connection.execute("SELECT 1 FROM cases LIMIT 1").fetchone():
                        raise RuntimeError(
                            f"SQLite case store {self.sqlite_path} has cases but no migration marker"
                        )
                    if self.legacy_json_path.exists():
                        with self.legacy_json_path.open("r", encoding="utf-8") as handle:
                            legacy_cases = json.load(handle)
                        if not isinstance(legacy_cases, dict):
                            raise ValueError(
                                f"Legacy case memory {self.legacy_json_path} must contain a JSON object"
                            )
                    else:
                        legacy_cases = {}
                    connection.executemany(
                        "INSERT INTO cases (case_id, position, payload) VALUES (?, ?, ?)",
                        (
                            (case_id, position, json.dumps(case_info))
                            for position, (case_id, case_info) in enumerate(legacy_cases.items())
                        ),
                    )
                    connection.execute(
                        "INSERT INTO metadata (key, value) VALUES ('migration_complete', '1')"
                    )
                    if legacy_cases:
                        logger.info(
                            "Migrated %d memory cases from %s to %s; the JSON is retained as a backup",
                            len(legacy_cases), self.legacy_json_path, self.sqlite_path,
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def count(self):
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]

    def iter_cases(self):
        with closing(self._connect()) as connection:
            for case_id, payload in connection.execute(
                "SELECT case_id, payload FROM cases ORDER BY position"
            ):
                yield case_id, json.loads(payload)

    def upsert(self, case_id, case_info):
        payload = json.dumps(case_info)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO cases (case_id, position, payload) "
                    "VALUES (?, (SELECT COALESCE(MAX(position), -1) + 1 FROM cases), ?) "
                    "ON CONFLICT(case_id) DO UPDATE SET payload = excluded.payload",
                    (case_id, payload),
                )

    def replace_all(self, cases):
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM cases")
                connection.executemany(
                    "INSERT INTO cases (case_id, position, payload) VALUES (?, ?, ?)",
                    (
                        (case_id, position, json.dumps(case_info))
                        for position, (case_id, case_info) in enumerate(cases.items())
                    ),
                )
