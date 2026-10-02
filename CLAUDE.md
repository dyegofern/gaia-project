# GAIA Agent - Claude Code Configuration

## Project Overview

This project is a **GAIA benchmark evaluation agent** that answers AI benchmark questions using tool-augmented LLMs. It fetches questions from the HF Agents Course Unit 4 scoring API, runs an autonomous agent with various tools, records answers to SQLite, and optionally submits results to the official leaderboard.

## Key Files

### Core Agent
- `gaia_agent/agent.py` - `GaiaAgent` class with tool-calling loop
- `gaia_agent/llm.py` - LLM client supporting multiple backends (Lemonade, HF, Groq, Gemini)
- `gaia_agent/tools.py` - 11 tool implementations (web search, file reading, Python exec, etc.)
- `gaia_agent/chess_vision.py` - Deterministic chess board reader using template matching
- `gaia_agent/results_db.py` - SQLite database for storing runs and tool statistics

### Runner
- `run_eval.py` - Main entry point for running evaluations

### Tests
- `tests/test_agent.py` - Agent logic tests
- `tests/test_llm.py` - LLM client tests  
- `tests/test_tools.py` - Tool function tests
- `tests/test_llm_new.py` - New: rate limit and health check tests
- `tests/test_chess_vision.py` - New: chess confidence threshold tests

### Web Monitor
- `web/server.js` - Express.js + Socket.IO server
- `web/public/index.html` - Dashboard UI

## Running the Agent

```bash
# Install dependencies
/home/dyego/rocm10-test/bin/pip install -r requirements.txt

# Run on random question
/home/dyego/rocm10-test/bin/python run_eval.py --random

# Run on all questions (parallel)
/home/dyego/rocm10-test/bin/python run_eval.py --workers 10

# Resume interrupted run
/home/dyego/rocm10-test/bin/python run_eval.py --continue

# Submit to leaderboard
/home/dyego/rocm10-test/bin/python run_eval.py --submit \
  --username YOUR_HF_USERNAME \
  --agent-code https://huggingface.co/spaces/YOUR_SPACE/tree/main
```

## Running the Web Monitor

```bash
cd web
npm install
npm start
# Open http://localhost:3000
```

## LLM Backends

### Lemonade (default, local, free)
```bash
# Must have Lemonade Server running on localhost:13305
export GAIA_LLM_BACKEND=lemonade
```

### Hugging Face (cloud, free credits)
```bash
export GAIA_LLM_BACKEND=h
export HF_TOKEN=your_hf_token
```

### Groq (cloud, genuinely free)
```bash
export GAIA_LLM_BACKEND=groq
export GROQ_API_KEY=your_groq_key
```

## Environment Validation

Before running, validate the environment:
```bash
/home/dyego/rocm10-test/bin/python -c "from run_eval import validate_environment; validate_environment()"
```

## New Features Implemented

1. **Rate Limit Retry** - Exponential backoff for API rate limits
2. **Backend Health Check** - Verify backend availability before runs
3. **Chess Confidence Threshold** - 0.2 IoU threshold for piece detection
4. **Tool Usage Statistics** - Track tool calls, success rates, durations
5. **Answer Verification** - Validate answer format and quality
6. **SQLite Optimizations** - WAL mode, indexes, caching
7. **Environment Validation** - Pre-flight checks for Python, tools, backends
8. **Better Error Messages** - Actionable error messages for common failures
9. **Unit Tests** - Tests for new features

## Debugging & Operations

- **Transcripts** - every question's full conversation (tool calls + results) is stored in the `transcripts` table; click a question in the dashboard's run modal.
- **Run logs** - runs started from the dashboard write to `logs/run-*.log` (path stored in `runs.log_path`); `--log-file` does the same from the CLI.
- **Cancel** - the dashboard's Cancel sends SIGTERM to the run (`runs.pid`); in-flight questions return to pending, resume with Continue.
- **Attachments** - the scoring API's `/files/<task_id>` currently returns 404. The agent looks in `files/<task_id>.<ext>` first (upload via the dashboard's Attachments card), then the scoring API, then the gated `gaia-benchmark/GAIA` dataset (needs an HF account with access).
- **Preflight** - `probe_backend()` makes a real 1-token request before a run (catches exhausted credits that `check_backend_health` misses); billing/auth errors (401/402/403) stop a run and leave the rest pending.
- `GAIA_LLM_TIMEOUT` overrides the per-request timeout (default 300s Lemonade, 120s cloud); `GAIA_FILES_DIR` overrides `files/`.

## Configuration

- `GAIA_LLM_BACKEND` - LLM backend (lemonade, hf, groq, gemini)
- `GAIA_PYTHON_EXEC_INTERPRETER` - Python interpreter for code execution
- `GAIA_STOCKFISH_PATH` - Path to Stockfish chess engine
- `GAIA_LLM_CONCURRENCY` - Max concurrent LLM requests

## Architecture Patterns

- **Tool-calling loop** - Agent iterates up to MAX_ITERATIONS (10) using tools
- **Parallel execution** - Multiple workers process questions concurrently
- **Persistent state** - SQLite stores all runs, answers, tool stats
- **Graceful degradation** - Handles API errors, rate limits, interruptions

## Known Issues & Workarounds

- **IPv6 routing broken** - DNS patch forces IPv4-only resolution
- **Lemonade concurrency** - Server runs with --parallel 1, client limits to 1
- **YouTube downloads** - Cached per-process, re-download on restart
