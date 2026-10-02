import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import initSqlJs from 'sql.js';
import cors from 'cors';
import { execFile, spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join, resolve, sep } from 'path';
import { existsSync, statSync, readFileSync, mkdirSync, openSync, closeSync, readSync, writeFileSync, readdirSync } from 'fs';
import http from 'http';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: { origin: '*', methods: ['GET', 'POST'] }
});

app.use(cors());
app.use(express.json());
app.use(express.static(join(__dirname, 'public'), { etag: false, lastModified: false, setHeaders: res => res.set('Cache-Control', 'no-store') }));

// Database paths
const GAIA_DB_PATH = join(__dirname, '..', 'gaia_runs.db');
const WEB_DB_PATH = join(__dirname, 'gaia_web.db');

let SQL = null;
let webDb = null;

// Initialize SQL.js and web database
async function initDatabase() {
  SQL = await initSqlJs();
  webDb = new SQL.Database();

  webDb.run(`
    CREATE TABLE IF NOT EXISTS run_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id INTEGER,
      backend TEXT,
      started_at REAL,
      ended_at REAL,
      total_questions INTEGER,
      completed INTEGER DEFAULT 0,
      failed INTEGER DEFAULT 0,
      status TEXT DEFAULT 'pending'
    );

    CREATE TABLE IF NOT EXISTS tool_stats (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id INTEGER,
      tool_name TEXT,
      call_count INTEGER DEFAULT 1,
      success_count INTEGER DEFAULT 0,
      error_count INTEGER DEFAULT 0,
      total_duration_ms INTEGER DEFAULT 0,
      FOREIGN KEY (run_id) REFERENCES run_history(run_id)
    );

    CREATE TABLE IF NOT EXISTS config (
      key TEXT PRIMARY KEY,
      value TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_tool_stats_run ON tool_stats(run_id);
    CREATE INDEX IF NOT EXISTS idx_run_history_status ON run_history(status);
  `);

  console.log('Web database initialized');
}

initDatabase().catch(err => {
  console.error('Failed to initialize database:', err);
});

// Health check for backends (using native http module)
async function checkBackendHealth(backend) {
  if (backend === 'lemonade') {
    return new Promise((resolve) => {
      const req = http.request({
        hostname: 'localhost',
        port: 13305,
        path: '/api/v0/models',
        method: 'GET',
        timeout: 5000
      }, (res) => {
        resolve(res.statusCode === 200);
      });
      req.on('error', () => resolve(false));
      req.on('timeout', () => { req.destroy(); resolve(false); });
      req.end();
    });
  } else if (backend === 'groq' || backend === 'hf' || backend === 'gemini') {
    const envVar = backend === 'groq' ? 'GROQ_API_KEY' :
                   backend === 'hf' ? 'HF_TOKEN' : 'GEMINI_API_KEY';
    return Boolean(process.env[envVar]);
  }
  return false;
}

// Helper functions for SQL.js
function dbQuery(sql, params = []) {
  if (!webDb) return [];
  try {
    const stmt = webDb.prepare(sql);
    if (params.length > 0) {
      params.forEach((param, i) => stmt.bind(i + 1, param));
    }
    const results = [];
    while (stmt.step()) {
      results.push(stmt.getAsObject());
    }
    stmt.free();
    return results;
  } catch (e) {
    console.error('DB Query error:', e);
    return [];
  }
}

function dbRun(sql, params = []) {
  if (!webDb) return;
  try {
    const stmt = webDb.prepare(sql);
    if (params.length > 0) {
      params.forEach((param, i) => stmt.bind(i + 1, param));
    }
    stmt.run();
    stmt.free();
  } catch (e) {
    console.error('DB Run error:', e);
  }
}

// Read-only queries against the agent's own SQLite DB (gaia_runs.db).
// The DB is tiny and written by a separate Python process, so load a fresh
// copy per query instead of holding a stale in-memory snapshot.
function gaiaQuery(sql, params = []) {
  if (!SQL || !existsSync(GAIA_DB_PATH)) return [];
  let db;
  try {
    db = new SQL.Database(readFileSync(GAIA_DB_PATH));
    const stmt = db.prepare(sql);
    stmt.bind(params);
    const rows = [];
    while (stmt.step()) rows.push(stmt.getAsObject());
    stmt.free();
    return rows;
  } catch (e) {
    console.error('GAIA DB query error:', e.message);
    return [];
  } finally {
    if (db) db.close();
  }
}

// One row per run, shaped like what the dashboard expects.
const RUNS_SQL = `
  SELECT
    r.run_id,
    r.backend,
    r.started_at,
    r.pid,
    r.log_path,
    (SELECT s.score FROM submissions s WHERE s.run_id = r.run_id ORDER BY s.id DESC LIMIT 1) AS score,
    (SELECT s.correct_count || '/' || s.total_attempted FROM submissions s WHERE s.run_id = r.run_id ORDER BY s.id DESC LIMIT 1) AS score_detail,
    COUNT(x.task_id) AS total_questions,
    COALESCE(SUM(x.status = 'done'), 0) AS completed,
    COALESCE(SUM(x.status = 'error'), 0) AS failed,
    CASE
      WHEN COALESCE(SUM(x.status = 'running'), 0) > 0 THEN 'running'
      WHEN COALESCE(SUM(x.status = 'pending'), 0) > 0 THEN 'pending'
      WHEN COALESCE(SUM(x.status = 'error'), 0) > 0 THEN 'failed'
      ELSE 'completed'
    END AS status,
    MAX(x.finished_at) AS last_activity,
    CASE WHEN COUNT(x.task_id) > 0
      THEN COALESCE(SUM(x.status = 'done'), 0) * 100.0 / COUNT(x.task_id)
      ELSE 0 END AS progress_pct
  FROM runs r
  LEFT JOIN results x ON x.run_id = r.run_id
  GROUP BY r.run_id
`;

const LOGS_DIR = join(__dirname, '..', 'logs');
const FILES_DIR = join(__dirname, '..', 'files');
const SCORING_API = 'https://agents-course-unit4-scoring.hf.space';
const GAIA_DATASET_URL = 'https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/';

async function questionsWithFiles() {
  const res = await fetch(`${SCORING_API}/questions`, { signal: AbortSignal.timeout(15000) });
  if (!res.ok) throw new Error(`scoring API returned ${res.status}`);
  return (await res.json()).filter(q => q.file_name);
}

function hasLocalAttachment(taskId) {
  return existsSync(FILES_DIR) &&
    readdirSync(FILES_DIR).some(n => n.startsWith(taskId) && statSync(join(FILES_DIR, n)).size > 0);
}

function isRunEvalProcess(pid) {
  // Guard against pid reuse: only treat it as ours if it is really run_eval.py.
  try {
    return readFileSync(`/proc/${pid}/cmdline`, 'utf8').includes('run_eval.py');
  } catch {
    return false;
  }
}

// The agent side owns the schema; open it once at startup so newly added
// tables (e.g. submissions) exist before the dashboard queries them.
execFile('/home/dyego/rocm10-test/bin/python', ['-c', 'from gaia_agent.results_db import ResultsDB; ResultsDB()'],
  { cwd: join(__dirname, '..') }, (error) => { if (error) console.error('Schema check failed:', error.message); });

// A run whose process is gone but which still has unfinished questions is
// "interrupted" (crashed, killed, or cancelled) rather than "running".
function withLiveness(run) {
  if (!run) return run;
  const unfinished = ['running', 'pending'].includes(run.status);
  const alive = run.pid ? isRunEvalProcess(run.pid) : null;
  return { ...run, alive, status: unfinished && alive === false ? 'interrupted' : run.status };
}

function tailFile(path, maxBytes) {
  const size = statSync(path).size;
  const start = Math.max(0, size - maxBytes);
  const fd = openSync(path, 'r');
  try {
    const buf = Buffer.alloc(size - start);
    readSync(fd, buf, 0, buf.length, start);
    return { text: buf.toString('utf8'), truncated: start > 0, size };
  } finally {
    closeSync(fd);
  }
}

// API Routes

app.get('/api/status', async (req, res) => {
  try {
    const backends = ['lemonade', 'groq', 'hf', 'gemini'];
    const health = {};
    for (const backend of backends) {
      health[backend] = await checkBackendHealth(backend);
    }

    const latestRun = withLiveness(gaiaQuery(RUNS_SQL + ' ORDER BY r.run_id DESC LIMIT 1')[0]) || null;
    const recentRuns = gaiaQuery(RUNS_SQL + ' ORDER BY r.run_id DESC LIMIT 10').map(withLiveness);
    const totalRunsRow = gaiaQuery('SELECT COUNT(*) AS count FROM runs')[0];

    res.json({
      health,
      latestRun,
      recentRuns,
      totalRuns: totalRunsRow?.count || 0,
      gaiaDbExists: existsSync(GAIA_DB_PATH)
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/run/:runId', (req, res) => {
  try {
    const id = Number.parseInt(req.params.runId, 10);
    const run = withLiveness(gaiaQuery(`SELECT * FROM (${RUNS_SQL}) WHERE run_id = ?`, [id])[0]) || null;
    if (!run) {
      return res.status(404).json({ error: 'Run not found' });
    }
    const questions = gaiaQuery(
      'SELECT task_id, status, question, answer FROM results WHERE run_id = ? ORDER BY rowid', [id]);
    const tools = gaiaQuery(`
      SELECT tool_name,
             COUNT(*) AS call_count,
             SUM(success) AS success_count,
             COUNT(*) - SUM(success) AS error_count,
             SUM(duration_ms) AS total_duration_ms
      FROM tool_usage WHERE run_id = ? GROUP BY tool_name ORDER BY call_count DESC`, [id]);
    const submissions = gaiaQuery('SELECT username, agent_code, score, correct_count, total_attempted, message, submitted_at FROM submissions WHERE run_id = ? ORDER BY id DESC', [id]);
    res.json({ run, questions, tools, submissions });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/run/:runId/log', (req, res) => {
  try {
    const id = Number.parseInt(req.params.runId, 10);
    const row = gaiaQuery('SELECT log_path FROM runs WHERE run_id = ?', [id])[0];
    if (!row || !row.log_path) {
      return res.json({ text: '', note: 'No log recorded for this run (it was not started from the dashboard).' });
    }
    // Only ever serve files inside the logs directory.
    const path = resolve(row.log_path);
    if (!path.startsWith(resolve(LOGS_DIR) + sep) || !existsSync(path)) {
      return res.json({ text: '', note: 'Log file not available.' });
    }
    const maxBytes = Math.min(Number.parseInt(req.query.bytes, 10) || 20000, 200000);
    res.json(tailFile(path, maxBytes));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/run/:runId/task/:taskId', (req, res) => {
  try {
    const id = Number.parseInt(req.params.runId, 10);
    const task = gaiaQuery(
      'SELECT task_id, status, question, answer FROM results WHERE run_id = ? AND task_id = ?',
      [id, req.params.taskId])[0];
    if (!task) return res.status(404).json({ error: 'Task not found' });
    const tr = gaiaQuery('SELECT messages, saved_at FROM transcripts WHERE run_id = ? AND task_id = ?',
      [id, req.params.taskId])[0];
    const tools = gaiaQuery(
      'SELECT tool_name, success, duration_ms, error_message FROM tool_usage WHERE run_id = ? AND task_id = ? ORDER BY id',
      [id, req.params.taskId]);
    res.json({ task, tools, messages: tr ? JSON.parse(tr.messages) : null });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PYTHON_BIN = '/home/dyego/rocm10-test/bin/python';

app.get('/api/submit-defaults', (req, res) => {
  const last = gaiaQuery('SELECT username, agent_code FROM submissions ORDER BY id DESC LIMIT 1')[0] || {};
  res.json({ username: last.username || process.env.GAIA_SUBMIT_USERNAME || '',
             agent_code: last.agent_code || process.env.GAIA_AGENT_CODE || '' });
});

// What would be sent: the answered tasks, after the same cleaning submit() applies.
app.get('/api/run/:runId/submit-preview', (req, res) => {
  const id = Number.parseInt(req.params.runId, 10);
  const code = [
    'import json, sys',
    'from gaia_agent.answers import clean_answer',
    'from gaia_agent.results_db import ResultsDB',
    'rows = ResultsDB().results(int(sys.argv[1]))',
    'done = [r for r in rows if r["status"] == "done"]',
    'print(json.dumps({"total": len(rows), "answers": [{"task_id": r["task_id"], "question": r["question"][:90], "answer": r["answer"], "cleaned": clean_answer(r["answer"])} for r in done]}))',
  ].join('\n');
  execFile(PYTHON_BIN, ['-c', code, String(id)], { cwd: join(__dirname, '..'), timeout: 30000 }, (error, stdout) => {
    if (error) return res.status(500).json({ error: error.message });
    try {
      const data = JSON.parse(stdout.trim().split('\n').pop());
      const run = withLiveness(gaiaQuery(`SELECT * FROM (${RUNS_SQL}) WHERE run_id = ?`, [id])[0]);
      res.json({ ...data, unanswered: data.total - data.answers.length, alive: run ? run.alive : null,
                 previous: gaiaQuery('SELECT username, score, correct_count, total_attempted, submitted_at FROM submissions WHERE run_id = ? ORDER BY id DESC', [id]) });
    } catch {
      res.status(500).json({ error: 'Unreadable preview output' });
    }
  });
});

// Public leaderboard submission: needs an explicit confirm flag, refuses a run
// that is still executing, and refuses partial runs unless told otherwise.
app.post('/api/run/:runId/submit', (req, res) => {
  const id = Number.parseInt(req.params.runId, 10);
  const { username, agent_code, confirm, allowPartial } = req.body || {};
  if (confirm !== true) return res.status(400).json({ error: 'Submission must be explicitly confirmed.' });
  if (!/^[A-Za-z0-9_.-]{1,64}$/.test(username || '')) return res.status(400).json({ error: 'Invalid Hugging Face username.' });
  if (!/^https?:\/\/[^\s]{3,300}$/.test(agent_code || '')) return res.status(400).json({ error: 'Agent code must be an http(s) URL.' });
  const run = withLiveness(gaiaQuery(`SELECT * FROM (${RUNS_SQL}) WHERE run_id = ?`, [id])[0]);
  if (!run) return res.status(404).json({ error: 'Run not found' });
  if (run.alive) return res.status(409).json({ error: 'This run is still executing; wait for it to finish.' });
  if (run.completed === 0) return res.status(409).json({ error: 'This run has no answered questions.' });
  if (run.completed < run.total_questions && allowPartial !== true) {
    return res.status(409).json({ error: `Only ${run.completed} of ${run.total_questions} questions are answered; confirm partial submission to continue.` });
  }
  execFile(PYTHON_BIN, [join(__dirname, '..', 'run_eval.py'), '--from-db', '--run-id', String(id),
    '--submit', '--username', username, '--agent-code', agent_code],
    { cwd: join(__dirname, '..'), timeout: 120000 }, (error, stdout, stderr) => {
      const marker = '=== Submission Result ===';
      if (error || !stdout.includes(marker)) {
        return res.status(502).json({ error: ((stderr || error?.message || 'Submission failed').trim().split('\n').slice(-4).join(' | ')).slice(0, 500) });
      }
      try {
        res.json({ success: true, result: JSON.parse(stdout.split(marker)[1]) });
      } catch {
        res.status(502).json({ error: 'Submitted, but the response could not be read.', raw: stdout.slice(-300) });
      }
    });
});

app.post('/api/run/:runId/cancel', (req, res) => {
  try {
    const id = Number.parseInt(req.params.runId, 10);
    const row = gaiaQuery('SELECT pid FROM runs WHERE run_id = ?', [id])[0];
    if (!row || !row.pid || !isRunEvalProcess(row.pid)) {
      return res.status(409).json({ error: 'This run has no live process to cancel.' });
    }
    // SIGTERM: run_eval resets in-flight questions to pending, so --continue resumes cleanly.
    process.kill(row.pid, 'SIGTERM');
    res.json({ success: true, message: `Cancelling run ${id} (pid ${row.pid})` });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/attachments', async (req, res) => {
  try {
    const questions = await questionsWithFiles();
    const token = process.env.HF_TOKEN;
    const rows = await Promise.all(questions.map(async (q) => {
      const probe = async (url, headers = {}) => {
        try {
          const r = await fetch(url, { headers, signal: AbortSignal.timeout(15000) });
          r.body?.cancel();
          return r.ok;
        } catch { return null; }
      };
      return {
        task_id: q.task_id,
        file_name: q.file_name,
        question: q.question.slice(0, 100),
        server: await probe(`${SCORING_API}/files/${q.task_id}`),
        local: hasLocalAttachment(q.task_id),
        dataset: token ? await probe(GAIA_DATASET_URL + q.file_name, { Authorization: `Bearer ${token}` }) : null,
      };
    }));
    res.json(rows);
  } catch (error) {
    res.status(502).json({ error: error.message });
  }
});

// Upload an attachment by hand (the agent looks in files/ first). The stored
// name comes from the question list, never from the client.
app.post('/api/attachments/:taskId', express.raw({ type: () => true, limit: '50mb' }), async (req, res) => {
  try {
    const q = (await questionsWithFiles()).find(x => x.task_id === req.params.taskId);
    if (!q) return res.status(404).json({ error: 'No attachment question with that task id' });
    if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
      return res.status(400).json({ error: 'Empty upload' });
    }
    mkdirSync(FILES_DIR, { recursive: true });
    writeFileSync(join(FILES_DIR, q.file_name), req.body);
    res.json({ success: true, saved_as: q.file_name, bytes: req.body.length });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/tool-stats', (req, res) => {
  try {
    res.json(gaiaQuery(`
      SELECT
        tool_name,
        COUNT(*) AS total_calls,
        SUM(success) AS total_success,
        COUNT(*) - SUM(success) AS total_errors,
        SUM(duration_ms) AS total_duration_ms
      FROM tool_usage
      GROUP BY tool_name
      ORDER BY total_calls DESC
    `));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/run-progress', (req, res) => {
  try {
    res.json(gaiaQuery(RUNS_SQL + ' ORDER BY r.run_id DESC').map(withLiveness));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/probe-backend', (req, res) => {
  const backend = req.body?.backend;
  if (!BACKENDS.includes(backend)) {
    return res.status(400).json({ error: `Unknown backend '${backend}'` });
  }
  const code = 'import json; from gaia_agent.llm import probe_backend; ok, reason = probe_backend(); print(json.dumps({"ok": ok, "reason": reason}))';
  execFile('/home/dyego/rocm10-test/bin/python', ['-c', code],
    { cwd: join(__dirname, '..'), timeout: 660000, env: { ...process.env, GAIA_LLM_BACKEND: backend } },
    (error, stdout, stderr) => {
      if (error) return res.status(500).json({ ok: false, reason: (stderr || error.message).slice(-300) });
      try {
        res.json(JSON.parse(stdout.trim().split('\n').pop()));
      } catch {
        res.status(500).json({ ok: false, reason: 'Unreadable probe output' });
      }
    });
});

const BACKEND_ENV_KEYS = { groq: 'GROQ_API_KEY', hf: 'HF_TOKEN', gemini: 'GEMINI_API_KEY' };
const BACKENDS = ['lemonade', ...Object.keys(BACKEND_ENV_KEYS)];

app.post('/api/trigger-run', async (req, res) => {
  try {
    const { type, workers, backend = 'lemonade' } = req.body;
    const workersArg = Number.parseInt(workers, 10);
    if (!Number.isInteger(workersArg) || workersArg < 1 || workersArg > 64) {
      return res.status(400).json({ error: 'workers must be an integer between 1 and 64' });
    }
    if (!BACKENDS.includes(backend)) {
      return res.status(400).json({ error: `Unknown backend '${backend}'` });
    }
    if (!(await checkBackendHealth(backend))) {
      const why = backend === 'lemonade'
        ? 'Lemonade is not reachable on localhost:13305'
        : `${BACKEND_ENV_KEYS[backend]} is not set on the server`;
      return res.status(400).json({ error: `Backend '${backend}' unavailable: ${why}` });
    }

    const typeFlags = { random: ['--random'], all: [], continue: ['--continue'] };
    if (!(type in typeFlags)) {
      return res.status(400).json({ error: 'Invalid run type' });
    }

    const pythonBin = '/home/dyego/rocm10-test/bin/python';
    const scriptPath = join(__dirname, '..', 'run_eval.py');
    const args = [scriptPath, ...typeFlags[type], '--workers', String(workersArg)];

    // Run in the background with output going to its own log file; the backend
    // is chosen via the env var the agent reads.
    mkdirSync(LOGS_DIR, { recursive: true });
    const logPath = join(LOGS_DIR, `run-${new Date().toISOString().replace(/[:.]/g, '-')}-${backend}.log`);
    const fd = openSync(logPath, 'a');
    const child = spawn(pythonBin, [...args, '--log-file', logPath], {
      env: { ...process.env, GAIA_LLM_BACKEND: backend, PYTHONUNBUFFERED: '1' },
      stdio: ['ignore', fd, fd],
    });
    closeSync(fd);
    child.on('error', (error) => console.error('Run error:', error.message));
    child.on('exit', (code, signal) => console.log(`Run process ${child.pid} exited (code ${code}, signal ${signal})`));

    res.json({ success: true, message: `Run started in background on ${backend}`, pid: child.pid });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/config', (req, res) => {
  try {
    const rows = dbQuery('SELECT key, value FROM config');
    const configs = {};
    rows.forEach(row => configs[row.key] = row.value);
    res.json(configs);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/config', (req, res) => {
  try {
    const { key, value } = req.body;
    dbRun('INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)', [key, String(value)]);
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Socket.IO for real-time updates
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('subscribe-run', (runId) => {
    socket.join(`run-${runId}`);
    console.log(`Client ${socket.id} subscribed to run ${runId}`);
  });

  socket.on('unsubscribe-run', (runId) => {
    socket.leave(`run-${runId}`);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

app.get('/api/environment', async (req, res) => {
  try {
    const checks = {
      pythonBin: {
        path: '/home/dyego/rocm10-test/bin/python',
        exists: existsSync('/home/dyego/rocm10-test/bin/python'),
        version: null
      },
      gaiaDb: {
        path: GAIA_DB_PATH,
        exists: existsSync(GAIA_DB_PATH),
        size: existsSync(GAIA_DB_PATH) ? statSync(GAIA_DB_PATH).size : 0
      },
      scratchDir: {
        path: join(__dirname, '..', '.scratch'),
        exists: existsSync(join(__dirname, '..', '.scratch'))
      },
      lemonade: {
        url: 'http://localhost:13305',
        reachable: false
      },
      envVars: {
        HF_TOKEN: !!process.env.HF_TOKEN,
        GROQ_API_KEY: !!process.env.GROQ_API_KEY,
        GEMINI_API_KEY: !!process.env.GEMINI_API_KEY
      }
    };

    // Check Python version
    try {
      const { execSync } = await import('child_process');
      checks.pythonBin.version = execSync('/home/dyego/rocm10-test/bin/python --version').toString().trim();
    } catch (e) {
      checks.pythonBin.error = e.message;
    }

    // Check Lemonade
    checks.lemonade.reachable = await checkBackendHealth('lemonade');

    res.json(checks);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// The agent runs in separate processes, so watch the DB file and tell
// connected clients to refresh whenever it changes.
let lastDbMtime = 0;
setInterval(() => {
  try {
    const m = statSync(GAIA_DB_PATH).mtimeMs;
    if (m !== lastDbMtime) {
      lastDbMtime = m;
      io.emit('run-progress', 'database updated');
    }
  } catch { /* DB not created yet */ }
}, 2000);

const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
  console.log(`GAIA Web Monitor running on http://localhost:${PORT}`);
});
