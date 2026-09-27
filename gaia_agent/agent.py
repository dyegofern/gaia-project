import json
import re

from openai import APIError

from gaia_agent.llm import chat_completion
from gaia_agent.tools import (
    web_search, WEB_SEARCH_SCHEMA,
    fetch_page, FETCH_PAGE_SCHEMA,
    download_gaia_file, DOWNLOAD_GAIA_FILE_SCHEMA,
    read_file, READ_FILE_SCHEMA,
    python_exec, PYTHON_EXEC_SCHEMA,
    transcribe_audio, TRANSCRIBE_AUDIO_SCHEMA,
)

SYSTEM_PROMPT = """You are a general-purpose research assistant answering benchmark questions.

You have access to tools: web_search, fetch_page, download_gaia_file, read_file,
python_exec, transcribe_audio.
Use them as needed to research and compute the answer. If the question references an
attached file and a task_id is provided, call download_gaia_file first, then read_file
(or transcribe_audio for .mp3/.wav files).

When you know the final answer, respond with a line in exactly this format and nothing else:
FINAL ANSWER: <answer>

The answer must be as short as possible: a number, a word, a short phrase, or a
comma-separated list — with no explanation, units unless asked, or extra text."""

TOOLS = [
    WEB_SEARCH_SCHEMA,
    FETCH_PAGE_SCHEMA,
    DOWNLOAD_GAIA_FILE_SCHEMA,
    READ_FILE_SCHEMA,
    PYTHON_EXEC_SCHEMA,
    TRANSCRIBE_AUDIO_SCHEMA,
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "download_gaia_file": download_gaia_file,
    "read_file": read_file,
    "python_exec": python_exec,
    "transcribe_audio": transcribe_audio,
}

MAX_ITERATIONS = 10

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


class GaiaAgent:
    def __init__(self):
        print("GaiaAgent initialized.")

    def __call__(self, question: str, task_id: str | None = None) -> str:
        user_content = question
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
            except APIError:
                return "AGENT ERROR: could not produce a final answer after repeated API errors"

            # Even with tools=None, some models still write an informal
            # tool-call pattern into content out of habit. No more tool
            # calls are allowed at this point, so don't execute it --
            # just strip it out and force one more genuinely-final call.
            if _parse_informal_tool_call(last_content):
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
                except APIError:
                    return "AGENT ERROR: could not produce a final answer after repeated API errors"

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
            return content.split(marker, 1)[1].strip()
        return content.strip()
