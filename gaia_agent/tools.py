import base64
import io
import os
import re
import subprocess
import sys
import tempfile
import time

import pandas as pd
import requests
import trafilatura
import yt_dlp
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


DOWNLOAD_GAIA_FILE_MAX_RETRIES = 3
DOWNLOAD_GAIA_FILE_RETRY_DELAY = 3
GAIA_DATASET_FILE_URL = "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/{name}"
# Optional folder of manually supplied attachments, named "<task_id>.<ext>".
LOCAL_FILES_DIR = os.environ.get("GAIA_FILES_DIR", os.path.join(os.path.dirname(__file__), "..", "files"))
_question_file_names = None


def _attachment_name(task_id: str):
    """File name of the task's attachment, as listed by the scoring API."""
    global _question_file_names
    if _question_file_names is None:
        try:
            resp = requests.get(f"{SCORING_API_URL}/questions", timeout=15)
            resp.raise_for_status()
            _question_file_names = {q["task_id"]: q.get("file_name") for q in resp.json()}
        except Exception:
            return None
    return _question_file_names.get(task_id) or None


def _find_local_attachment(task_id: str):
    for folder in (LOCAL_FILES_DIR, SCRATCH_DIR):
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                path = os.path.join(folder, name)
                if name.startswith(task_id) and os.path.isfile(path) and os.path.getsize(path) > 0:
                    return os.path.abspath(path)
    return None


def _download_from_gaia_dataset(task_id: str):
    """Fallback: the gated gaia-benchmark/GAIA dataset (needs an HF_TOKEN whose
    account has accepted the dataset's terms). Returns (path, None) or
    (None, reason)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None, "HF_TOKEN is not set"
    name = _attachment_name(task_id)
    if not name:
        return None, "attachment file name unknown"
    try:
        resp = requests.get(GAIA_DATASET_FILE_URL.format(name=name), timeout=60,
                            headers={"Authorization": f"Bearer {token}"})
    except Exception as e:
        return None, str(e)
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code} (the HF account likely lacks access to gaia-benchmark/GAIA)"
    path = os.path.join(SCRATCH_DIR, name)
    with open(path, "wb") as f:
        f.write(resp.content)
    return os.path.abspath(path), None


def download_gaia_file(task_id: str) -> str:
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    cached = _find_local_attachment(task_id)
    if cached:
        return cached

    url = f"{SCORING_API_URL}/files/{task_id}"
    last_error = None
    for attempt in range(DOWNLOAD_GAIA_FILE_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404 and "No file path associated" in resp.text:
                # The server itself has no file for this task; retrying can't help.
                last_error = "scoring API has no file for this task (404 'No file path associated')"
                break
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            # The file endpoint is intermittently flaky (the same task_id 404s
            # then succeeds seconds later), so retry other HTTP errors briefly.
            last_error = e
            if attempt < DOWNLOAD_GAIA_FILE_MAX_RETRIES - 1:
                time.sleep(DOWNLOAD_GAIA_FILE_RETRY_DELAY)
            continue
        except Exception as e:
            last_error = e
            break
    else:
        resp = None
    if resp is None or not resp.ok:
        path, reason = _download_from_gaia_dataset(task_id)
        if path:
            return path
        return (
            f"ERROR: the attachment for task {task_id} is unavailable ({last_error}; "
            f"GAIA dataset fallback: {reason}). Do not retry. Answer from the question text "
            "alone only if that is genuinely sufficient; otherwise reply exactly "
            "'Unable to access the attached file'."
        )

    content_disp = resp.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', content_disp)
    filename = match.group(1) if match else (_attachment_name(task_id) or task_id)

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


YOUTUBE_MAX_FRAMES = 6
_YOUTUBE_DOWNLOAD_CACHE = {}


def _download_youtube_video(url: str) -> str:
    """Download a YouTube video once per URL within this process and cache
    the local path -- both YouTube tools below need the video file, and
    downloading is the slowest, most failure-prone step, so it should only
    happen once even if the agent calls both tools for the same video.
    """
    if url in _YOUTUBE_DOWNLOAD_CACHE:
        return _YOUTUBE_DOWNLOAD_CACHE[url]

    tmpdir = tempfile.mkdtemp()
    with yt_dlp.YoutubeDL({
        "outtmpl": os.path.join(tmpdir, "video.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        # Frames get downscaled for the vision model and audio is all
        # Whisper needs, so full-resolution video is wasted download time.
        "format": "bv*[height<=480]+ba/b[height<=480]/b",
        # Without an explicit JS runtime, yt-dlp only enables deno by
        # default (not installed here) and falls back to a degraded
        # extraction path with a loud warning. node is already present on
        # this system, so use it instead of leaving extraction degraded.
        "js_runtimes": {"node": {}},
    }) as ydl:
        info = ydl.extract_info(url, download=True)

    video_files = [f for f in os.listdir(tmpdir) if f.startswith("video.")]
    if not video_files:
        raise RuntimeError(f"download reported success but no video file was found for {url}")
    video_path = os.path.join(tmpdir, video_files[0])
    _YOUTUBE_DOWNLOAD_CACHE[url] = (video_path, info)
    return _YOUTUBE_DOWNLOAD_CACHE[url]


def transcribe_youtube_video(url: str) -> str:
    try:
        video_path, info = _download_youtube_video(url)
    except Exception as e:
        return f"ERROR: could not download video {url}: {e}"

    audio_path = video_path + ".audio.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame",
             "-ar", "16000", "-ac", "1", audio_path, "-y"],
            capture_output=True, timeout=60, check=True,
        )
    except Exception as e:
        return f"ERROR: could not extract audio from {url}: {e}"

    return transcribe_audio(audio_path)


TRANSCRIBE_YOUTUBE_VIDEO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "transcribe_youtube_video",
        "description": (
            "Download a YouTube video and transcribe its spoken audio to "
            "text. Use for questions about dialogue or narration in a video "
            "(what someone says). Faster than analyze_youtube_frames -- "
            "prefer this first if the question is about speech/dialogue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The YouTube video URL."}
            },
            "required": ["url"],
        },
    },
}


DEFAULT_IMAGE_ANALYSIS_PROMPT = (
    "Describe everything visible in this image in detail: people, animals, "
    "objects, text, actions, counts of things if relevant. Be factual and "
    "specific."
)

# Full-resolution images (e.g. a 3480x2160 wallpaper) were observed to time
# out the vision model entirely; downscaling to a reasonable max dimension
# keeps inference fast and reliable without losing content a vision model
# needs (verified live: a 341x512 image analyzed quickly and accurately).
IMAGE_ANALYSIS_MAX_DIMENSION = 1024


def _describe_frame(image_path: str, question: str = "") -> str:
    return analyze_image(image_path, question)


def analyze_image(path: str, question: str = "") -> str:
    from PIL import Image
    from gaia_agent.llm import chat_completion

    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"

    prompt = question.strip() or DEFAULT_IMAGE_ANALYSIS_PROMPT
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((IMAGE_ANALYSIS_MAX_DIMENSION, IMAGE_ANALYSIS_MAX_DIMENSION))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return f"ERROR: could not read image {path}: {e}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }
    ]
    try:
        # Vision inference on this local model has been observed to take
        # 60-90s+ even for moderate prompts; the default 120s timeout isn't
        # enough headroom for more detailed/structured analysis requests.
        response = chat_completion(messages, timeout=240)
    except Exception as e:
        return f"ERROR: could not analyze image {path}: {e}"
    return response.choices[0].message.content or ""


ANALYZE_IMAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_image",
        "description": (
            "Describe or answer a question about a local image file using a "
            "vision-capable model. Use this for any image file (e.g. a "
            "downloaded chess position, diagram, photo, or screenshot) -- "
            "for a chess position, ask it to describe the exact position of "
            "every piece on the board."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local filesystem path to the image file."},
                "question": {
                    "type": "string",
                    "description": "Optional specific question to ask about the image; if omitted, gives a general description.",
                },
            },
            "required": ["path"],
        },
    },
}


def analyze_youtube_frames(url: str, question: str = "") -> str:
    """Describe several frames sampled evenly across the video using a
    vision-capable model -- slow (one model call per frame) since it's only
    needed for questions about what is visually shown, not said.
    """
    try:
        video_path, info = _download_youtube_video(url)
    except Exception as e:
        return f"ERROR: could not download video {url}: {e}"

    duration = info.get("duration") or 0
    frames_dir = video_path + ".frames"
    os.makedirs(frames_dir, exist_ok=True)
    try:
        fps = max(YOUTUBE_MAX_FRAMES / duration, 0.05) if duration else 0.2
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}",
             os.path.join(frames_dir, "frame_%03d.jpg"), "-y"],
            capture_output=True, timeout=60, check=True,
        )
    except Exception as e:
        return f"ERROR: could not extract frames from {url}: {e}"

    frame_files = sorted(os.listdir(frames_dir))[:YOUTUBE_MAX_FRAMES]
    if not frame_files:
        return f"ERROR: no frames could be extracted from {url}"

    descriptions = []
    for i, fname in enumerate(frame_files):
        try:
            desc = _describe_frame(os.path.join(frames_dir, fname), question)
            descriptions.append(f"Frame {i + 1} (of {len(frame_files)}, sampled across the video): {desc}")
        except Exception as e:
            descriptions.append(f"Frame {i + 1}: ERROR describing frame: {e}")
    return "\n\n".join(descriptions)


ANALYZE_YOUTUBE_FRAMES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_youtube_frames",
        "description": (
            "Download a YouTube video and describe several frames sampled "
            "across it using a vision model. Use for questions about what "
            "is visually shown (objects, people, counts, on-screen text). "
            "Slower than transcribe_youtube_video -- only use this when the "
            "question is about visual content, not dialogue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The YouTube video URL."},
                "question": {"type": "string", "description": "Optional: what to look for in each frame (e.g. 'how many bird species are visible?')."},
            },
            "required": ["url"],
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


STOCKFISH_PATH = os.environ.get("GAIA_STOCKFISH_PATH", "/usr/games/stockfish")
CHESS_ANALYSIS_DEPTH = 20
CHESS_ANALYSIS_LINES = 5


def best_chess_moves(fen: str) -> str:
    import chess
    import chess.engine

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return f"ERROR: invalid FEN {fen!r}: {e}"
    if not board.is_valid():
        return f"ERROR: FEN {fen!r} describes an illegal position: {board.status()!r}"
    if board.is_game_over():
        return f"ERROR: game is already over ({board.result()})"

    side = "White" if board.turn == chess.WHITE else "Black"
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except (FileNotFoundError, PermissionError) as e:
        return f"ERROR: could not start Stockfish at {STOCKFISH_PATH}: {e}"
    try:
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=CHESS_ANALYSIS_DEPTH),
            multipv=CHESS_ANALYSIS_LINES,
        )
    finally:
        engine.quit()

    lines = [f"{side} to move. Top moves (Stockfish depth {CHESS_ANALYSIS_DEPTH}):"]
    for info in infos:
        score = info["score"].pov(board.turn)
        if score.is_mate():
            evaluation = f"forced mate in {abs(score.mate())}" if score.mate() > 0 else f"gets mated in {abs(score.mate())}"
        else:
            evaluation = f"{score.score() / 100:+.2f} pawns"
        lines.append(f"{board.san(info['pv'][0])}: {evaluation}")
    return "\n".join(lines)


BEST_CHESS_MOVES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "best_chess_moves",
        "description": "Analyze a chess position with the Stockfish engine and return the best candidate moves in standard algebraic notation with evaluations. Requires a FEN string; the side to move is taken from the FEN.",
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {"type": "string", "description": "The position in FEN notation, e.g. '3r2k1/pp3pp1/4b2p/7Q/3n4/PqBBR2P/5PP1/6K1 b - - 0 1'."}
            },
            "required": ["fen"],
        },
    },
}


def read_chess_board_image(path: str, flipped=None, side_to_move: str = "w") -> str:
    import chess

    from gaia_agent.chess_vision import board_image_to_placement

    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    if side_to_move not in ("w", "b"):
        return "ERROR: side_to_move must be 'w' or 'b'"
    try:
        placement = board_image_to_placement(path, flipped=flipped)
    except Exception as e:
        return f"ERROR: could not read chess board from {path}: {e}"
    fen = f"{placement} {side_to_move} - - 0 1"
    return f"FEN: {fen}\n{chess.Board(fen)}"


READ_CHESS_BOARD_IMAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_chess_board_image",
        "description": "Deterministically read a chess board screenshot (8x8 grid image) into a FEN string plus an ASCII diagram. Much more reliable than asking analyze_image to list pieces. The result FEN can be passed to best_chess_moves. Orientation is auto-detected; only pass flipped if the coordinate labels clearly show the opposite (rank 1 at the top / file h at the left means flipped=true).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local path to the board image."},
                "flipped": {"type": "boolean", "description": "Optional override: true if the board is displayed rotated 180 degrees (a8 NOT at top-left). Omit to auto-detect."},
                "side_to_move": {"type": "string", "enum": ["w", "b"], "description": "Whose turn it is, from the question text. Default 'w'."},
            },
            "required": ["path"],
        },
    },
}
