import json
import re
import time
import functools
import contextvars
from openai import APIError, RateLimitError, BadRequestError

from gaia_agent.llm import chat_completion
from gaia_agent.tools import (
    web_search, WEB_SEARCH_SCHEMA,
    fetch_page, FETCH_PAGE_SCHEMA,
    download_gaia_file, DOWNLOAD_GAIA_FILE_SCHEMA,
    read_file, READ_FILE_SCHEMA,
    python_exec, PYTHON_EXEC_SCHEMA,
    transcribe_audio, TRANSCRIBE_AUDIO_SCHEMA,
    transcribe_youtube_video, TRANSCRIBE_YOUTUBE_VIDEO_SCHEMA,
    analyze_youtube_frames, ANALYZE_YOUTUBE_FRAMES_SCHEMA,
    analyze_image, ANALYZE_IMAGE_SCHEMA,
    read_chess_board_image, READ_CHESS_BOARD_IMAGE_SCHEMA,
    best_chess_moves, BEST_CHESS_MOVES_SCHEMA,
)

SYSTEM_PROMPT = """You are a general-purpose research assistant answering benchmark questions.

You have access to tools: web_search, fetch_page, download_gaia_file, read_file,
python_exec, transcribe_audio, transcribe_youtube_video, analyze_youtube_frames,
analyze_image, read_chess_board_image, best_chess_moves.
Use them as needed to research and compute the answer. If the question references an
attached file and a task_id is provided, call download_gaia_file first, then read_file
for text/PDF/CSV/XLSX, transcribe_audio for .mp3/.wav files, or analyze_image for any
image file (e.g. .png/.jpg). For a chess position image, do NOT ask analyze_image to list
pieces (unreliable): call read_chess_board_image (orientation is auto-detected;
side_to_move from the question), then call best_chess_moves with the resulting FEN and answer with the move in the notation the
question asks for. If the question references a YouTube video, use transcribe_youtube_video
for questions about dialogue/speech, or analyze_youtube_frames for questions about visual
content (objects, people, counts) -- only use analyze_youtube_frames if actually needed,
since it is much slower.

Never search the web for the task_id itself or for "GAIA benchmark answer" or similar --
answer keys or discussions of this exact question may be indexed online, but using them
would not be a genuine answer. Always derive the answer yourself from the question's own
attached file or the sources it points to.

For any task involving character-level string manipulation (reversing text, counting
letters, checking palindromes, rearranging characters, etc.), use python_exec to compute
the exact result rather than doing it by eye -- this is exactly the kind of task language
models get wrong by "reasoning" about it instead of just running the code.

Report your thoughts, and finish your answer with a line in exactly this format:
FINAL ANSWER: [YOUR FINAL ANSWER]

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated
list of numbers and/or strings. If you are asked for a number, don't use commas to write
your number, and don't use units such as $ or % unless specified otherwise. If you are
asked for a string, don't use articles ("a", "the"), don't use abbreviations (e.g. write
city names in full), and write digits in plain text (e.g. "seven" not "7") unless
specified otherwise. If you are asked for a comma separated list, apply the rules above
to each element, with exactly one space after each comma."""

TOOLS = [
    WEB_SEARCH_SCHEMA,
    FETCH_PAGE_SCHEMA,
    DOWNLOAD_GAIA_FILE_SCHEMA,
    READ_FILE_SCHEMA,
    PYTHON_EXEC_SCHEMA,
    TRANSCRIBE_AUDIO_SCHEMA,
    TRANSCRIBE_YOUTUBE_VIDEO_SCHEMA,
    ANALYZE_YOUTUBE_FRAMES_SCHEMA,
    ANALYZE_IMAGE_SCHEMA,
    READ_CHESS_BOARD_IMAGE_SCHEMA,
    BEST_CHESS_MOVES_SCHEMA,
]

_TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "download_gaia_file": download_gaia_file,
    "read_file": read_file,
    "python_exec": python_exec,
    "transcribe_audio": transcribe_audio,
    "transcribe_youtube_video": transcribe_youtube_video,
    "analyze_youtube_frames": analyze_youtube_frames,
    "analyze_image": analyze_image,
    "read_chess_board_image": read_chess_board_image,
    "best_chess_moves": best_chess_moves,
}

_global_db = None
_global_run_id = None
_task_id_cache = None


def set_global_db(db, run_id):
    """Set the global DB for tool usage tracking."""
    global _global_db, _global_run_id
    _global_db = db
    _global_run_id = run_id


_current_task_id = contextvars.ContextVar("gaia_task_id", default=None)


def _record(tool_name, success, duration_ms, error_message=None):
    # Bookkeeping must never break (or change the result of) a tool call.
    if _global_db is None or _global_run_id is None:
        return
    try:
        _global_db.record_tool_call(
            _global_run_id, _current_task_id.get() or "unknown",
            tool_name, success=success, duration_ms=duration_ms,
            error_message=error_message,
        )
    except Exception as e:
        print(f"WARNING: could not record tool call {tool_name}: {e}", flush=True)


def _instrument_tool(tool_func, tool_name):
    """Wrap a tool to record timing and success. Tools report most failures
    by returning an "ERROR: ..." string rather than raising, so that counts
    as a failure too, and the message is stored for later debugging."""
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = tool_func(*args, **kwargs)
        except Exception as e:
            _record(tool_name, False, int((time.time() - start_time) * 1000), str(e)[:500])
            raise
        failed = isinstance(result, str) and result.lstrip().startswith("ERROR")
        _record(tool_name, not failed, int((time.time() - start_time) * 1000),
                result[:500] if failed else None)
        return result
    return wrapper


# Instrumented tool functions
TOOL_FUNCTIONS = {
    name: _instrument_tool(func, name) for name, func in _TOOL_FUNCTIONS.items()
}


MAX_ITERATIONS = 10


def _format_api_error(e: APIError) -> str:
    if isinstance(e, RateLimitError):
        error_msg = str(e)
        if "tokens per day" in error_msg or "TPD" in error_msg:
            return (
                "AGENT ERROR: Rate limit exceeded - "
                "You've reached your daily token quota. "
                f"Please wait for the quota to reset or use a different backend. ({error_msg})"
            )
        elif "tokens per minute" in error_msg:
            return (
                "AGENT ERROR: Rate limit exceeded - "
                "You've exceeded the per-minute token limit. "
                f"Please wait before retrying. ({error_msg})"
            )
        return f"AGENT ERROR: Rate limited ({e})"
    elif isinstance(e, BadRequestError):
        return f"AGENT ERROR: Bad request - {e}"
    elif isinstance(e, ConnectionError):
        return f"AGENT ERROR: Connection failed - Could not reach LLM server. Please check your backend is running."
    else:
        return f"AGENT ERROR: Could not produce a final answer after repeated API errors ({type(e).__name__})"


# Some models (e.g. Qwen3.5 on Lemonade) sometimes write tool calls as
# informal XML-like text directly into the message `content` instead of
# using the structured `tool_calls` API field. Detect and parse that
# pattern so it can be executed like a real tool call rather than being
# mistaken for the final answer.
INFORMAL_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
INFORMAL_PARAMETER_RE = re.compile(
    r"<parameter=([\w.-]+)>(.*?)</parameter>",
    re.DOTALL,
)


def _parse_informal_tool_call(content: str):
    match = INFORMAL_TOOL_CALL_RE.search(content or "")
    if not match:
        return None
    name = match.group(1)
    params_blob = match.group(2)
    args = {
        pname: pvalue.strip()
        for pname, pvalue in INFORMAL_PARAMETER_RE.findall(params_blob)
    }
    return name, args


_COMMON_ENGLISH_WORDS = {
    "the", "is", "you", "if", "and", "of", "to", "a", "in", "that", "this",
    "as", "it", "for", "on", "with", "answer", "word", "sentence", "opposite",
    "write", "understand",
}


def _looks_like_reversed_text(text: str) -> bool:
    # Heuristic: if reversing the whole string turns it into recognizable
    # English (several common short words appear), the original is almost
    # certainly presented reversed/scrambled on purpose (a common GAIA
    # question pattern) -- worth pre-computing deterministically rather
    # than relying on the model to notice and choose to use python_exec.
    reversed_text = text[::-1]
    words = re.findall(r"[a-zA-Z']+", reversed_text.lower())
    if len(words) < 4:
        return False
    hits = sum(1 for w in words if w in _COMMON_ENGLISH_WORDS)
    return hits / len(words) >= 0.3


class GaiaAgent:
    def __init__(self):
        print("GaiaAgent initialized.")

    def __call__(self, question: str, task_id: str | None = None) -> str:
        _current_task_id.set(task_id)
        user_content = question
        if _looks_like_reversed_text(question):
            user_content += (
                f"\n\n(This text appears to be reversed. Read it character-by-character "
                f"reversed, it says: {question[::-1]!r})"
            )
        if task_id:
            user_content += f"\n\n(task_id: {task_id})"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        last_content = ""
        nudged = False
        stopped_early = False
        for i in range(MAX_ITERATIONS):
            if not nudged and i >= MAX_ITERATIONS - 2:
                messages.append({
                    "role": "user",
                    "content": (
                        "You are running low on tool calls. Synthesize a FINAL ANSWER "
                        "from the information you already have unless one more targeted "
                        "lookup is truly necessary."
                    ),
                })
                nudged = True

            try:
                response = chat_completion(messages, tools=TOOLS)
            except APIError as e:
                # The server rejected the model's own generation (e.g. a
                # malformed tool-call name with leaked internal formatting
                # tokens) before returning any response object -- there is
                # no valid assistant message to append and no tool_call_id
                # to respond to. Tell the model what went wrong and ask it
                # to retry with a valid tool call, using the same backend
                # (no need to switch models -- a corrective nudge is usually
                # enough for the model to self-correct on the next sample).
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last tool call was rejected as invalid "
                        f"({e}). Retry with a valid tool call using exactly "
                        "one of the available tool names."
                    ),
                })
                try:
                    response = chat_completion(messages, tools=TOOLS)
                except APIError:
                    # Two consecutive failures: give up on the tool-calling
                    # loop for this question and fall through to the
                    # forced-final-answer fallback below, using whatever
                    # conversation history (including prior successful tool
                    # calls) has accumulated so far.
                    break

            message = response.choices[0].message

            if message.tool_calls:
                messages.append(message.model_dump(exclude_none=True))
                for tool_call in message.tool_calls:
                    result = self._run_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                continue

            informal_call = _parse_informal_tool_call(message.content)
            if informal_call:
                name, args = informal_call
                messages.append(message.model_dump(exclude_none=True))
                result = self._run_tool_by_name(name, args)
                messages.append({
                    "role": "user",
                    "content": (
                        f"(Note: your tool call for {name} was written as plain text "
                        "instead of using the tool-calling API -- I ran it anyway. "
                        "Please use the structured tool-calling mechanism going forward.)\n\n"
                        f"Result: {result}"
                    ),
                })
                continue

            last_content = message.content or ""
            stopped_early = True
            break

        if not stopped_early:
            messages.append({
                "role": "user",
                "content": (
                    "No more tool calls are allowed. Based on everything gathered so "
                    "far, give your best-effort FINAL ANSWER now."
                ),
            })
            try:
                response = chat_completion(messages, tools=None)
                last_content = response.choices[0].message.content or ""
            except APIError as e:
                return _format_api_error(e)

            # Even with tools=None, some models persistently write an
            # informal tool-call pattern into content out of habit. No more
            # tool calls are allowed at this point, so don't execute it --
            # retry a bounded number of times to get a clean answer instead.
            retries = 0
            while _parse_informal_tool_call(last_content) and retries < 2:
                retries += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "Tool calls are no longer available. Do not write a tool call -- "
                        "just answer directly with FINAL ANSWER: <answer> based on what "
                        "you already know."
                    ),
                })
                try:
                    response = chat_completion(messages, tools=None)
                    last_content = response.choices[0].message.content or ""
                except APIError as e:
                    return _format_api_error(e)

        return self._extract_final_answer(last_content)

    def _run_tool(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            return f"ERROR: could not parse arguments for {name}: {e}"
        return self._run_tool_by_name(name, args)

    def _run_tool_by_name(self, name: str, args: dict) -> str:
        func = TOOL_FUNCTIONS.get(name)
        if func is None:
            return f"ERROR: unknown tool {name}"
        try:
            return func(**args)
        except Exception as e:
            return f"ERROR: tool {name} raised an exception: {e}"

    def _extract_final_answer(self, content: str) -> str:
        marker = "FINAL ANSWER:"
        if marker in content:
            # The model sometimes restates its answer after rambling, so
            # prefer the last marker that is actually followed by text.
            candidates = [part.strip() for part in content.split(marker)[1:]]
            content = next((c for c in reversed(candidates) if c), "")
        # Last-resort safety net: never surface a leaked, unexecuted
        # informal tool call as if it were an answer, however it got here.
        content = INFORMAL_TOOL_CALL_RE.sub("", content)
        return content.strip() or "AGENT ERROR: model produced no usable answer"
