import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import initSqlJs from 'sql.js';
import cors from 'cors';
import { exec } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, statSync, readFileSync } from 'fs';
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
app.use(express.static(join(__dirname, 'public')));

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

// API Routes

app.get('/api/status', async (req, res) => {
  try {
    const backends = ['lemonade', 'groq', 'hf', 'gemini'];
    const health = {};
    for (const backend of backends) {
      health[backend] = await checkBackendHealth(backend);
    }

    const latestRun = gaiaQuery(RUNS_SQL + ' ORDER BY r.run_id DESC LIMIT 1')[0] || null;
    const recentRuns = gaiaQuery(RUNS_SQL + ' ORDER BY r.run_id DESC LIMIT 10');
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
    const run = gaiaQuery(`SELECT * FROM (${RUNS_SQL}) WHERE run_id = ?`, [id])[0] || null;
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
    res.json({ run, questions, tools });
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
    res.json(gaiaQuery(RUNS_SQL + ' ORDER BY r.run_id DESC'));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/trigger-run', (req, res) => {
  try {
    const { type, workers } = req.body;
    const workersArg = Number.parseInt(workers, 10);
    if (!Number.isInteger(workersArg) || workersArg < 1 || workersArg > 64) {
      return res.status(400).json({ error: 'workers must be an integer between 1 and 64' });
    }

    const pythonBin = '/home/dyego/rocm10-test/bin/python';
    const scriptPath = join(__dirname, '..', 'run_eval.py');

    let command;
    if (type === 'random') {
      command = `${pythonBin} ${scriptPath} --random --workers ${workersArg}`;
    } else if (type === 'all') {
      command = `${pythonBin} ${scriptPath} --workers ${workersArg}`;
    } else if (type === 'continue') {
      command = `${pythonBin} ${scriptPath} --continue --workers ${workersArg}`;
    } else {
      return res.status(400).json({ error: 'Invalid run type' });
    }

    exec(command, { timeout: 3600000 }, (error, stdout, stderr) => {
      // Run is in background - don't send response here
      if (error) {
        console.error('Run error:', error.message);
      }
    });

    res.json({ success: true, message: 'Run started in background' });
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
