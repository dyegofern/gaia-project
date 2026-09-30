import contextlib
import sqlite3
import time

DEFAULT_DB_PATH = "gaia_runs.db"

PENDING, RUNNING, DONE, ERROR = "pending", "running", "done", "error"
INCOMPLETE_STATUSES = (PENDING, RUNNING, ERROR)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    backend TEXT
);
CREATE TABLE IF NOT EXISTS results (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    task_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    status TEXT NOT NULL,
    started_at REAL,
    finished_at REAL,
    PRIMARY KEY (run_id, task_id)
);
"""


class ResultsDB:
    """Every method opens its own short-lived connection, so it is safe to
    call from many worker threads at once (SQLite serialises the writes)."""

    def __init__(self, path=DEFAULT_DB_PATH):
        self.path = path
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_run(self, backend=None):
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (started_at, backend) VALUES (?, ?)", (time.time(), backend)
            )
            return cur.lastrowid

    def latest_run_id(self):
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(run_id) AS run_id FROM runs").fetchone()
            return row["run_id"]

    def add_questions(self, run_id, questions):
        """Register questions as pending; questions already in the run are left untouched."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO results (run_id, task_id, question, status) VALUES (?, ?, ?, ?)",
                [(run_id, q["task_id"], q["question"], PENDING) for q in questions],
            )

    def incomplete(self, run_id):
        """Tasks with no usable answer: never finished, interrupted, errored, or an empty answer."""
        placeholders = ",".join("?" * len(INCOMPLETE_STATUSES))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT task_id, question FROM results WHERE run_id = ? AND "
                f"(status IN ({placeholders}) OR answer IS NULL OR TRIM(answer) = '') "
                f"ORDER BY rowid",
                (run_id, *INCOMPLETE_STATUSES),
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_running(self, run_id, task_id):
        with self._connect() as conn:
            conn.execute(
                "UPDATE results SET status = ?, started_at = ?, finished_at = NULL "
                "WHERE run_id = ? AND task_id = ?",
                (RUNNING, time.time(), run_id, task_id),
            )

    def save_answer(self, run_id, task_id, answer, ok=True):
        with self._connect() as conn:
            conn.execute(
                "UPDATE results SET answer = ?, status = ?, finished_at = ? "
                "WHERE run_id = ? AND task_id = ?",
                (answer, DONE if ok else ERROR, time.time(), run_id, task_id),
            )

    def reset_running(self, run_id):
        with self._connect() as conn:
            conn.execute(
                "UPDATE results SET status = ? WHERE run_id = ? AND status = ?",
                (PENDING, run_id, RUNNING),
            )

    def results(self, run_id):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task_id, question, answer, status FROM results WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def summary(self, run_id):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM results WHERE run_id = ? GROUP BY status",
                (run_id,),
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}
