# GAIA Agent - Quick Start Guide

## Prerequisites

### 1. Python Environment
```bash
# Use the ROCm-enabled venv
/home/dyego/rocm10-test/bin/pip install -r requirements.txt
```

### 2. System Tools
```bash
# Install ffmpeg and stockfish (if not already installed)
sudo apt-get install ffmpeg stockfish
```

### 3. LLM Backend (choose one)

**Option A: Lemonade Server (recommended - local, free)**
```bash
# Start Lemonade Server with Qwen3.5-35B-A3B-GGUF
lemonade-server load Qwen3.5-35B-A3B-GGUF

# Verify it's running
curl -s http://localhost:13305/api/v0/models
```

**Option B: Hugging Face Inference Providers**
```bash
export HF_TOKEN=your_hf_token_here
export GAIA_LLM_BACKEND=hf
```

**Option C: Groq**
```bash
export GROQ_API_KEY=your_groq_key_here
export GAIA_LLM_BACKEND=groq
```

## Quick Tests

### 1. Environment Validation
```bash
/home/dyego/rocm10-test/bin/python -c "
from run_eval import validate_environment
validate_environment()
"
```

### 2. Run a Single Question (Smoke Test)
```bash
/home/dyego/rocm10-test/bin/python run_eval.py --random
```

### 3. Run Web Monitor
```bash
cd web
npm install
npm start
# Then open http://localhost:3000 in your browser
```

### 4. Run All Questions
```bash
# Run 10 questions in parallel
/home/dyego/rocm10-test/bin/python run_eval.py --workers 10

# Save output to file
/home/dyego/rocm10-test/bin/python run_eval.py > answers.txt
```

### 5. Resume Interrupted Run
```bash
/home/dyego/rocm10-test/bin/python run_eval.py --continue
```

## New Features Quick Reference

### Rate Limit Handling
- Automatic exponential backoff on rate limits
- Up to 5 retries over 1 hour
- Clear error messages for daily vs per-minute limits

### Backend Health Check
- Pre-flight check before starting runs
- Fails fast if backend is unavailable
- Helpful error messages for each backend

### Chess Vision
- 0.2 IoU threshold for piece detection
- Low-confidence matches return empty square
- More reliable board reading

### Tool Usage Statistics
- Tracks all tool calls with timing
- Success/error counts per tool
- Available via web dashboard

### Answer Verification
- Checks for FINAL ANSWER marker
- Rejects empty or too-long answers
- Flags error indicators

### SQLite Optimizations
- WAL mode for concurrency
- Indexes on run_id, status
- 64MB cache

## Web Dashboard

The web monitor provides:
- **Real-time progress** - See runs as they happen
- **Tool statistics** - Which tools are used most
- **Health indicators** - Backend status at a glance
- **Environment checks** - Validate your setup
- **Run control** - Start new runs from the UI

### Dashboard Features
1. **Environment Health** - Shows which backends are available
2. **Quick Stats** - Total runs, completed, failed
3. **Start New Run** - Random, All Questions, or Continue
4. **Tool Usage Statistics** - Per-tool call counts and success rates
5. **Recent Runs** - Progress bar for each run
6. **Live Logs** - Real-time output from running agent

## Running Tests

```bash
# Run all tests
/home/dyego/rocm10-test/bin/python -m pytest tests/ -v

# Run new tests only
/home/dyego/rocm10-test/bin/python -m pytest tests/test_llm_new.py tests/test_chess_vision.py -v
```

## Troubleshooting

### "Backend unavailable" error
- **Lemonade**: Check `curl http://localhost:13305/api/v0/models`
- **Groq**: Check `GROQ_API_KEY` is set
- **HF**: Check `HF_TOKEN` is set

### "Python interpreter not found"
```bash
# Use the correct path
export GAIA_PYTHON_EXEC_INTERPRETER=/home/dyego/rocm10-test/bin/python
```

### Chess detection issues
- Ensure board image has clear 8x8 grid
- Good lighting and contrast
- No rotation (or use flipped parameter)

### Rate limit errors
- Wait for quota to reset (daily: ~24h, per-minute: use --workers 1)
- Switch to a different backend
- Reduce concurrency with `--workers 1`

## Next Steps

1. **Review answers** - Check `gaia_runs.db` for run history
2. **Submit to leaderboard** - Use `--submit` flag when ready
3. **Optimize** - Use tool statistics to improve agent performance
4. **Extend** - Add custom tools or modify existing ones
