import os
import re
import subprocess
import sys
import tempfile

import pandas as pd
import requests
import trafilatura
from ddgs import DDGS
from pypdf import PdfReader


def web_search(query: str) -> str:
    try:
        results = list(DDGS().text(query, max_results=5))
    except Exception as e:
        return f"ERROR: web search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title')}\n  {r.get('href')}\n  {r.get('body')}")
    return "\n".join(lines)


WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web via DuckDuckGo and return the top results (title, URL, snippet).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
}


MAX_PAGE_CHARS = 30000


def fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR: could not fetch {url}: {e}"
    text = trafilatura.extract(resp.text, with_metadata=True) or ""
    if not text.strip():
        text = resp.text
    return text[:MAX_PAGE_CHARS]


FETCH_PAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": "Download a web page and return its main text content (HTML stripped).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."}
            },
            "required": ["url"],
        },
    },
}


SCORING_API_URL = "https://agents-course-unit4-scoring.hf.space"
SCRATCH_DIR = os.path.join(os.path.dirname(__file__), "..", ".scratch")


def download_gaia_file(task_id: str) -> str:
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    url = f"{SCORING_API_URL}/files/{task_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR: could not download file for task {task_id}: {e}"

    content_disp = resp.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', content_disp)
    filename = match.group(1) if match else task_id

    path = os.path.join(SCRATCH_DIR, filename)
    with open(path, "wb") as f:
        f.write(resp.content)
    return os.path.abspath(path)


DOWNLOAD_GAIA_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "download_gaia_file",
        "description": "Download the file attached to the current GAIA question (by task_id) and return its local file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "The GAIA task_id for the current question."}
            },
            "required": ["task_id"],
        },
    },
}


MAX_FILE_CHARS = 10000


def read_file(path: str) -> str:
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, sheet_name=None)
            text = "\n\n".join(f"Sheet: {name}\n{sheet.to_string()}" for name, sheet in df.items())
        elif ext == ".csv":
            df = pd.read_csv(path)
            text = df.to_string()
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
    except Exception as e:
        return f"ERROR: could not read file {path}: {e}"

    return text[:MAX_FILE_CHARS]


READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a local file (text, code, .pdf, .csv, .xlsx) and return its extracted text content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local filesystem path to the file."}
            },
            "required": ["path"],
        },
    },
}


LEMONADE_TRANSCRIPTION_URL = "http://localhost:13305/api/v0/audio/transcriptions"
TRANSCRIPTION_MODEL = "Whisper-Large-v3-Turbo"


def transcribe_audio(path: str) -> str:
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                LEMONADE_TRANSCRIPTION_URL,
                files={"file": (os.path.basename(path), f)},
                data={"model": TRANSCRIPTION_MODEL},
                timeout=120,
            )
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR: could not transcribe {path}: {e}"
    return resp.json().get("text", "").strip() or "(no speech detected)"


TRANSCRIBE_AUDIO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "transcribe_audio",
        "description": "Transcribe a local audio file (e.g. .mp3, .wav) to text using speech-to-text.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local filesystem path to the audio file."}
            },
            "required": ["path"],
        },
    },
}


_LOCAL_ROCM10_PYTHON = "/home/dyego/rocm10-test/bin/python"
ROCM10_PYTHON = os.environ.get(
    "GAIA_PYTHON_EXEC_INTERPRETER",
    _LOCAL_ROCM10_PYTHON if os.path.exists(_LOCAL_ROCM10_PYTHON) else sys.executable,
)
PYTHON_EXEC_TIMEOUT = 30


def python_exec(code: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.run(
            [ROCM10_PYTHON, script_path],
            capture_output=True,
            text=True,
            timeout=PYTHON_EXEC_TIMEOUT,
        )
        output = proc.stdout
        if proc.returncode != 0:
            output += "\n" + proc.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: code did not finish within {PYTHON_EXEC_TIMEOUT}s"
    finally:
        os.remove(script_path)


PYTHON_EXEC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": "Run a Python script (with numpy/pandas/torch available) and return its stdout/stderr. Use for calculations or data processing.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python source code to execute."}
            },
            "required": ["code"],
        },
    },
}
