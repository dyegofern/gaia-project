import contextlib
import json
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
CREATE TABLE IF NOT EXISTS tool_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    task_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    duration_ms INTEGER NOT NULL,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS transcripts (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    task_id TEXT NOT NULL,
    messages TEXT NOT NULL,
    answer TEXT,
    saved_at REAL NOT NULL,
    PRIMARY KEY (run_id, task_id)
);
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    username TEXT NOT NULL,
    agent_code TEXT NOT NULL,
    score REAL,
    correct_count INTEGER,
    total_attempted INTEGER,
    message TEXT,
    submitted_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_run_status ON results(run_id, status);
CREATE INDEX IF NOT EXISTS idx_results_status ON results(status);
CREATE INDEX IF NOT EXISTS idx_tool_stats_run ON tool_usage(run_id);
"""


class ResultsDB:
    """Every method opens its own short-lived connection, so it is safe to
    call from many worker threads at once (SQLite serialises the writes)."""

    def __init__(self, path=DEFAULT_DB_PATH):
        self.path = path
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            self._migrate_tool_usage(conn)
            conn.executescript(_SCHEMA)
            self._migrate_runs(conn)

    @staticmethod
    def _migrate_runs(conn):
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
        for col, ddl in (("pid", "INTEGER"), ("log_path", "TEXT")):
            if col not in cols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {ddl}")

    @staticmethod
    def _migrate_tool_usage(conn):
        # Older DBs keyed tool_usage on (run_id, task_id, tool_name), so a
        # second call to the same tool within one task violated the key.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(tool_usage)")]
        if cols and "id" not in cols:
            conn.execute("ALTER TABLE tool_usage RENAME TO tool_usage_old")
            conn.execute("DROP INDEX IF EXISTS idx_tool_stats_run")
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT INTO tool_usage (run_id, task_id, tool_name, success, duration_ms, error_message) "
                "SELECT run_id, task_id, tool_name, success, duration_ms, error_message FROM tool_usage_old"
            )
            conn.execute("DROP TABLE tool_usage_old")

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

    def set_run_process(self, run_id, pid, log_path=None):
        """Record which OS process is executing a run (so it can be cancelled
        and so a dead process can be told apart from a live one) and where its
        log file lives."""
        with self._connect() as conn:
            conn.execute("UPDATE runs SET pid = ?, log_path = COALESCE(?, log_path) WHERE run_id = ?",
                         (pid, log_path, run_id))

    def save_transcript(self, run_id, task_id, messages, answer=None, max_chars=6000):
        """Store the conversation (system prompt excluded by the caller if
        desired), truncating very long tool outputs."""
        def clip(m):
            m = dict(m)
            if isinstance(m.get("content"), str) and len(m["content"]) > max_chars:
                m["content"] = m["content"][:max_chars] + f"\n...[truncated {len(m['content']) - max_chars} chars]"
            return m
        blob = json.dumps([clip(m) for m in messages], default=str)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO transcripts (run_id, task_id, messages, answer, saved_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, task_id, blob, answer, time.time()))

    def get_transcript(self, run_id, task_id):
        with self._connect() as conn:
            row = conn.execute("SELECT messages, answer, saved_at FROM transcripts WHERE run_id = ? AND task_id = ?",
                               (run_id, task_id)).fetchone()
        if row is None:
            return None
        return {"messages": json.loads(row["messages"]), "answer": row["answer"], "saved_at": row["saved_at"]}

    def save_submission(self, run_id, username, agent_code, outcome):
        """Remember what the leaderboard replied, so scores show next to runs."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO submissions (run_id, username, agent_code, score, correct_count, total_attempted, message, submitted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, username, agent_code, outcome.get("score"), outcome.get("correct_count"),
                 outcome.get("total_attempted"), outcome.get("message"), time.time()))

    def submissions(self, run_id=None):
        with self._connect() as conn:
            if run_id is None:
                rows = conn.execute("SELECT * FROM submissions ORDER BY id DESC").fetchall()
            else:
                rows = conn.execute("SELECT * FROM submissions WHERE run_id = ? ORDER BY id DESC", (run_id,)).fetchall()
        return [dict(r) for r in rows]

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

    def record_tool_call(self, run_id, task_id, tool_name, success, duration_ms, error_message=None):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_usage (run_id, task_id, tool_name, success, duration_ms, error_message) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, task_id, tool_name, success, duration_ms, error_message)
            )

    def get_tool_stats(self, run_id=None):
        with self._connect() as conn:
            if run_id:
                rows = conn.execute(
                    """SELECT tool_name,
                              COUNT(*) as call_count,
                              SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                              SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as error_count,
                              SUM(duration_ms) as total_duration_ms
                       FROM tool_usage WHERE run_id = ?
                       GROUP BY tool_name""",
                    (run_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT tool_name,
                              COUNT(*) as call_count,
                              SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                              SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as error_count,
                              SUM(duration_ms) as total_duration_ms
                       FROM tool_usage
                       GROUP BY tool_name
                       ORDER BY call_count DESC"""
                ).fetchall()
            return [dict(r) for r in rows]
