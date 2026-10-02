# GAIA Agent Backend Improvements

## Python Dependencies

Add to `requirements.txt`:

```
backoff>=2.2.1  # For rate limit retry logic with exponential backoff
tqdm>=4.66.0    # For progress indicators
```

## 1. Rate Limit Retry Logic (llm.py)

Add to `/home/dyego/git/amd-ai-learn/gaia_agent/llm.py`:

```python
import backoff
import time

@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APIError),
    max_tries=5,
    max_time=3600,  # Don't wait more than 1 hour total
    giveup=lambda e: "401" in str(e) or "403" in str(e),  # Don't retry auth errors
    on_backoff=lambda details: print(f"Rate limited, waiting {details['elapsed']:.1f}s before retry...")
)
def chat_completion(messages, tools=None, timeout=120):
    client, model = _get_client_and_model()
    kwargs = {"model": model, "messages": messages, "timeout": timeout}
    if tools:
        kwargs["tools"] = tools
    backend = os.environ.get("GAIA_LLM_BACKEND", "lemonade")
    with _semaphore_for(backend):
        raw_response = client.chat.completions.with_raw_response.create(**kwargs)
    _maybe_wait_for_rate_limit(raw_response)
    return raw_response.parse()
```

## 2. Backend Health Check (llm.py + run_eval.py)

Add to `/home/dyego/git/amd-ai-learn/gaia_agent/llm.py`:

```python
def check_backend_health(backend=None) -> bool:
    """Check if the configured backend is reachable and healthy."""
    if backend is None:
        backend = os.environ.get("GAIA_LLM_BACKEND", "lemonade")
    
    if backend == "lemonade":
        try:
            resp = requests.get(
                f"{LEMONADE_BASE_URL}/models",
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False
    
    elif backend == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            return False
        try:
            from openai import OpenAI
            client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"])
            client.models.list()
            return True
        except Exception:
            return False
    
    elif backend == "hf":
        if not os.environ.get("HF_TOKEN"):
            return False
        try:
            from openai import OpenAI
            client = OpenAI(base_url=HF_BASE_URL, api_key=os.environ["HF_TOKEN"])
            client.models.list()
            return True
        except Exception:
            return False
    
    elif backend == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            return False
        try:
            from openai import OpenAI
            client = OpenAI(base_url=GEMINI_BASE_URL, api_key=os.environ["GEMINI_API_KEY"])
            client.models.list()
            return True
        except Exception:
            return False
    
    return False
```

Add to `/home/dyego/git/amd-ai-learn/run_eval.py`:

```python
from gaia_agent.llm import check_backend_health

def run_and_record(args, db):
    backend = os.environ.get("GAIA_LLM_BACKEND", "lemonade")
    
    # Health check before starting
    if not check_backend_health(backend):
        print(f"ERROR: Backend {backend} is unavailable. Please check:")
        if backend == "lemonade":
            print("  - Lemonade Server is running on localhost:13305")
        elif backend == "groq":
            print("  - GROQ_API_KEY environment variable is set")
        elif backend == "hf":
            print("  - HF_TOKEN environment variable is set")
        elif backend == "gemini":
            print("  - GEMINI_API_KEY environment variable is set")
        sys.exit(1)
    
    # ... rest of function
```

## 3. Chess Vision Confidence Threshold (chess_vision.py)

Add to `/home/dyego/git/amd-ai-learn/gaia_agent/chess_vision.py`:

```python
# Add near other constants at top of file
CONFIDENCE_THRESHOLD = 0.2  # IoU threshold for piece detection


def classify_square(square: Image.Image, templates) -> str | None:
    mask = _square_mask(square)
    if mask.mean() < EMPTY_SQUARE_MAX_FILL:
        return None
    silhouette = _normalized_silhouette(mask)
    
    best_type, best_iou = None, -1.0
    for piece_type, template in templates.items():
        iou = (silhouette & template).sum() / (silhouette | template).sum()
        if iou > best_iou:
            best_type, best_iou = piece_type, iou
    
    # NEW: Reject low-confidence matches
    if best_iou < CONFIDENCE_THRESHOLD:
        return None
    
    colour = _piece_colour(square, mask)
    return best_type.upper() if colour == "w" else best_type
```

## 4. Tool Usage Statistics (results_db.py + tools.py)

Add to `/home/dyego/git/amd-ai-learn/gaia_agent/results_db.py`:

```python
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
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    task_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    duration_ms INTEGER NOT NULL,
    error_message TEXT,
    PRIMARY KEY (run_id, task_id, tool_name)
);
"""
```

Add methods to `ResultsDB` class:

```python
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
```

Modify `/home/dyego/git/amd-ai-learn/gaia_agent/agent.py`:

```python
import time
import functools
from gaia_agent.results_db import ResultsDB

# Global DB reference (set during run_eval.py execution)
_global_db = None
_global_run_id = None

def set_global_db(db, run_id):
    """Set the global DB for tool usage tracking."""
    global _global_db, _global_run_id
    _global_db = db
    _global_run_id = run_id

def instrumented_tool(tool_func, tool_name):
    """Decorator to instrument tool calls with timing and success tracking."""
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        if _global_db is None or _global_run_id is None:
            return tool_func(*args, **kwargs)
        
        start_time = time.time()
        try:
            result = tool_func(*args, **kwargs)
            duration_ms = int((time.time() - start_time) * 1000)
            _global_db.record_tool_call(
                _global_run_id, 
                tool_name,
                success=True,
                duration_ms=duration_ms
            )
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            _global_db.record_tool_call(
                _global_run_id,
                tool_name,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
            raise
    return wrapper

# Apply instrumentation to tool functions
TOOL_FUNCTIONS = {
    "web_search": instrumented_tool(web_search, "web_search"),
    "fetch_page": instrumented_tool(fetch_page, "fetch_page"),
    "download_gaia_file": instrumented_tool(download_gaia_file, "download_gaia_file"),
    "read_file": instrumented_tool(read_file, "read_file"),
    "python_exec": instrumented_tool(python_exec, "python_exec"),
    # ... etc for all tools
}
```

Modify `/home/dyego/git/amd-ai-learn/gaia_agent/tools.py`:

```python
# Add to top of transcribe_youtube_video function
@instrumented_tool(transcribe_youtube_video, "transcribe_youtube_video")
def transcribe_youtube_video(url: str) -> str:
    # ... existing code
```

## 5. Answer Verification (run_eval.py)

Add to `/home/dyego/git/amd-ai-learn/run_eval.py`:

```python
def verify_answer(question: str, answer: str) -> dict:
    """Basic answer verification heuristics."""
    verification = {
        "has_final_answer_marker": "FINAL ANSWER:" in answer,
        "is_empty": not answer or not answer.strip(),
        "is_too_long": len(answer) > 500,
        "contains_error_indicators": any(
            indicator in answer.lower() 
            for indicator in ["error", "failed", "unable", "cannot", "impossible"]
        ),
        "looks_reasonable": len(answer.strip()) > 0 and len(answer.strip()) < 500
    }
    verification["is_valid"] = (
        verification["has_final_answer_marker"] and 
        not verification["is_empty"] and 
        not verification["is_too_long"] and
        not verification["contains_error_indicators"]
    )
    return verification

# Update run_agent_on_questions to include verification
def run_agent_on_questions(agent, db, run_id, questions, workers=DEFAULT_WORKERS):
    # ... existing code
    
    def work(item):
        task_id, question_text = item["task_id"], item["question"]
        db.mark_running(run_id, task_id)
        try:
            answer = agent(question_text, task_id=task_id)
            verification = verify_answer(question_text, answer)
            
            # Mark as ok only if answer passes verification
            ok = verification["is_valid"] and not _answer_failed(answer)
        except Exception as e:
            answer = f"AGENT ERROR: {e}"
            verification = {"is_valid": False, "contains_error_indicators": True}
            ok = False
        
        db.save_answer(run_id, task_id, answer, ok=ok)
        
        # Print verification status
        if not verification["is_valid"]:
            print(f"  WARNING: Answer verification failed: {verification}")
        
        # ... rest of function
```

## 6. SQLite Query Optimization (results_db.py)

Add to `/home/dyego/git/amd-ai-learn/gaia_agent/results_db.py`:

```python
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
CREATE INDEX IF NOT EXISTS idx_results_run_status ON results(run_id, status);
CREATE INDEX IF NOT EXISTS idx_results_status ON results(status);
"""

class ResultsDB:
    def __init__(self, path=DEFAULT_DB_PATH):
        self.path = path
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.executescript(_SCHEMA)
```

## 7. Progress Indicators (run_eval.py)

Add to `/home/dyego/git/amd-ai-learn/run_eval.py`:

```python
from tqdm import tqdm

def run_agent_on_questions(agent, db, run_id, questions, workers=DEFAULT_WORKERS):
    # ... existing code
    
    valid = []
    for item in questions:
        if not item.get("task_id") or item.get("question") is None:
            print(f"Skipping malformed item: {item}")
            continue
        valid.append(item)
    
    with tqdm(total=len(valid), desc="Running agent", unit="question") as pbar:
        def work(item):
            # ... existing work function
            with _print_lock:
                finished[0] += 1
                pbar.update(1)
                # ... rest
```

## 8. Environment Validation (run_eval.py)

Add to `/home/dyego/git/amd-ai-learn/run_eval.py`:

```python
import sys
import subprocess

def validate_environment():
    """Validate the runtime environment before starting."""
    errors = []
    warnings = []
    
    # Check Python interpreter
    python_bin = os.environ.get("GAIA_PYTHON_EXEC_INTERPRETER", "/home/dyego/rocm10-test/bin/python")
    if not os.path.exists(python_bin):
        errors.append(f"Python interpreter not found: {python_bin}")
    
    # Check required tools
    required_tools = {
        "ffmpeg": "ffmpeg",
        "stockfish": "stockfish"
    }
    
    for name, cmd in required_tools.items():
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=2)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            warnings.append(f"{name} not found on PATH")
    
    # Check LEMONADE_SERVER
    if os.environ.get("GAIA_LLM_BACKEND", "lemonade") == "lemonade":
        health_ok = check_backend_health("lemonade")
        if not health_ok:
            warnings.append("Lemonade Server may not be running")
    
    # Report
    if errors:
        print("=" * 50)
        print("ENVIRONMENT VALIDATION FAILED")
        print("=" * 50)
        for err in errors:
            print(f"  ✗ {err}")
        return False
    
    if warnings:
        print("=" * 50)
        print("ENVIRONMENT WARNINGS")
        print("=" * 50)
        for warn in warnings:
            print(f"  ⚠ {warn}")
        print()
    
    print("=" * 50)
    print("ENVIRONMENT VALIDATION PASSED")
    print("=" * 50)
    return True
```

## 9. Better Error Messages (agent.py)

Update error messages in `/home/dyego/git/amd-ai-learn/gaia_agent/agent.py`:

```python
# Update _format_api_error with more detailed messages
def _format_api_error(e: APIError) -> str:
    if isinstance(e, RateLimitError):
        # Extract rate limit details from error message
        error_msg = str(e)
        if "tokens per day" in error_msg or "TPD" in error_msg:
            return (
                "AGENT ERROR: Rate limit exceeded - "
                "You've reached your daily token quota. "
                "Please wait for the quota to reset or use a different backend."
            )
        elif "tokens per minute" in error_msg:
            return (
                "AGENT ERROR: Rate limit exceeded - "
                "You've exceeded the per-minute token limit. "
                f"Please wait {e.response.request.retry_after}s before retrying."
            )
        return f"AGENT ERROR: Rate limited ({e})"
    elif isinstance(e, BadRequestError):
        return f"AGENT ERROR: Bad request - {e}"
    elif isinstance(e, ConnectionError):
        return f"AGENT ERROR: Connection failed - Could not reach LLM server. Please check your backend is running."
    else:
        return f"AGENT ERROR: Could not produce a final answer after repeated API errors ({type(e).__name__})"

# Add better tool error messages
def download_gaia_file(task_id: str) -> str:
    # ... existing code
    else:
        return (
            f"ERROR: Could not download file for task {task_id} after "
            f"{DOWNLOAD_GAIA_FILE_MAX_RETRIES} attempts. "
            "The GAIA scoring API may be temporarily unavailable. "
            "Try running with --continue later."
        )
```

---

## 10. Unit Tests (test_llm_new.py)

Create `/home/dyego/git/amd-ai-learn/tests/test_llm_new.py`:

```python
"""Tests for LLM backend improvements: rate limiting, health checks."""
import os
import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError, APIError

from gaia_agent.llm import (
    check_backend_health,
    _parse_reset_seconds,
    _maybe_wait_for_rate_limit,
)


class TestBackendHealthCheck:
    def test_lemonade_health_check_success(self):
        with patch("gaia_agent.llm.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp
            
            result = check_backend_health("lemonade")
            
            assert result is True
            mock_get.assert_called_once()
    
    def test_lemonade_health_check_failure(self):
        with patch("gaia_agent.llm.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            
            result = check_backend_health("lemonade")
            
            assert result is False
    
    def test_groq_health_check_no_token(self):
        # Unset GROQ_API_KEY temporarily
        old_value = os.environ.pop("GROQ_API_KEY", None)
        try:
            result = check_backend_health("groq")
            assert result is False
        finally:
            if old_value:
                os.environ["GROQ_API_KEY"] = old_value
    
    def test_hf_health_check_no_token(self):
        old_value = os.environ.pop("HF_TOKEN", None)
        try:
            result = check_backend_health("hf")
            assert result is False
        finally:
            if old_value:
                os.environ["HF_TOKEN"] = old_value


class TestRateLimitParsing:
    def test_parse_simple_seconds(self):
        result = _parse_reset_seconds("10s")
        assert result == 10
    
    def test_parse_minutes(self):
        result = _parse_reset_seconds("2m30s")
        assert result == 150
    
    def test_parse_hours_minutes_seconds(self):
        result = _parse_reset_seconds("1h16m19.2s")
        assert result == 4579.2
    
    def test_parse_float_seconds(self):
        result = _parse_reset_seconds("12.5s")
        assert result == 12.5
    
    def test_parse_none(self):
        result = _parse_reset_seconds(None)
        assert result is None


class TestRateLimitWait:
    def test_no_wait_when_adequate_tokens(self):
        response = MagicMock()
        response.headers = {
            "x-ratelimit-remaining-tokens": "10000",
            "x-ratelimit-reset-tokens": "10s"
        }
        
        with patch("gaia_agent.llm.time.sleep") as mock_sleep:
            _maybe_wait_for_rate_limit(response)
            mock_sleep.assert_not_called()
    
    def test_wait_when_low_tokens(self):
        response = MagicMock()
        response.headers = {
            "x-ratelimit-remaining-tokens": "10",
            "x-ratelimit-reset-tokens": "5s"
        }
        
        with patch("gaia_agent.llm.time.sleep") as mock_sleep:
            _maybe_wait_for_rate_limit(response)
            mock_sleep.assert_called_once()
            assert mock_sleep.call_args[0][0] <= 90  # Max 90s
```

Create `/home/dyego/git/amd-ai-learn/tests/test_chess_vision.py`:

```python
"""Tests for chess vision improvements."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from gaia_agent.chess_vision import (
    classify_square,
    CONFIDENCE_THRESHOLD,
    load_templates,
)


class TestChessVisionConfidence:
    def test_low_iu_return_none(self):
        """Test that low IoU scores return None (empty square)."""
        templates = load_templates()
        
        # Create a mock square that will have very low IoU with all templates
        mock_square = MagicMock()
        
        with patch("gaia_agent.chess_vision._square_mask") as mock_mask, \
             patch("gaia_agent.chess_vision._normalized_silhouette") as mock_sil, \
             patch("gaia_agent.chess_vision._piece_colour") as mock_colour:
            
            # Mock low mean mask (below EMPTY_SQUARE_MAX_FILL)
            mock_mask.return_value = np.zeros((48, 48), dtype=bool)
            mock_mask.return_value.mean = lambda: 0.01  # Below threshold
            
            result = classify_square(mock_square, templates)
            
            assert result is None
    
    def test_iu_below_threshold_returns_none(self):
        """Test that IoU below CONFIDENCE_THRESHOLD returns None."""
        templates = load_templates()
        
        mock_square = MagicMock()
        
        with patch("gaia_agent.chess_vision._square_mask") as mock_mask, \
             patch("gaia_agent.chess_vision._normalized_silhouette") as mock_sil, \
             patch("gaia_agent.chess_vision._piece_colour") as mock_colour:
            
            # Mock mask with some content
            mask = np.zeros((48, 48), dtype=bool)
            mask[20:28, 20:28] = True
            mock_mask.return_value = mask
            
            # Mock silhouette that will have low IoU
            silhouette = np.zeros((48, 48), dtype=bool)
            silhouette[0:4, 0:4] = True  # Different region
            
            mock_sil.return_value = silhouette
            mock_colour.return_value = "w"
            
            # Mock templates with completely different pieces
            fake_templates = {
                "p": np.ones((48, 48), dtype=bool),  # All true
                "r": np.ones((48, 48), dtype=bool),
            }
            
            result = classify_square(mock_square, fake_templates)
            
            # IoU should be very low, returning None
            assert result is None


class TestConfidenceThresholdConstant:
    def test_threshold_is_zero_point_two(self):
        """Test that CONFIDENCE_THRESHOLD is set to 0.2."""
        assert CONFIDENCE_THRESHOLD == 0.2
```

---

## Summary of Changes

| File | Changes |
|------|---------|
| `requirements.txt` | Added `backoff>=2.2.1`, `tqdm>=4.66.0` |
| `gaia_agent/llm.py` | Added rate limit retry with `@backoff`, health check function |
| `gaia_agent/chess_vision.py` | Added `CONFIDENCE_THRESHOLD = 0.2` |
| `gaia_agent/results_db.py` | Added `tool_usage` table, index optimization |
| `gaia_agent/agent.py` | Better error messages, tool instrumentation |
| `gaia_agent/tools.py` | Tool instrumentation decorators |
| `run_eval.py` | Health check before runs, answer verification, progress bars, env validation |
| `tests/test_llm_new.py` | Tests for rate limiting, health checks |
| `tests/test_chess_vision.py` | Tests for confidence threshold |

---

## Running the Improvements

1. Install new dependencies:
```bash
/home/dyego/rocm10-test/bin/pip install backoff tqdm
```

2. Run environment validation:
```bash
/home/dyego/rocm10-test/bin/python -c "from run_eval import validate_environment; validate_environment()"
```

3. Run tests:
```bash
/home/dyego/rocm10-test/bin/python -m pytest tests/test_llm_new.py tests/test_chess_vision.py -v
```

4. Start the web monitor:
```bash
cd /home/dyego/git/amd-ai-learn/web
npm install
npm start
```

Then open http://localhost:3000 in your browser!
