import json

from gaia_agent.llm import chat_completion
from gaia_agent.tools import (
    web_search, WEB_SEARCH_SCHEMA,
    fetch_page, FETCH_PAGE_SCHEMA,
    download_gaia_file, DOWNLOAD_GAIA_FILE_SCHEMA,
    read_file, READ_FILE_SCHEMA,
    python_exec, PYTHON_EXEC_SCHEMA,
)

SYSTEM_PROMPT = """You are a general-purpose research assistant answering benchmark questions.

You have access to tools: web_search, fetch_page, download_gaia_file, read_file, python_exec.
Use them as needed to research and compute the answer. If the question references an
attached file and a task_id is provided, call download_gaia_file first, then read_file.

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
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "download_gaia_file": download_gaia_file,
    "read_file": read_file,
    "python_exec": python_exec,
}

MAX_ITERATIONS = 10


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
        for _ in range(MAX_ITERATIONS):
            response = chat_completion(messages, tools=TOOLS)
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

            last_content = message.content or ""
            break

        return self._extract_final_answer(last_content)

    def _run_tool(self, tool_call) -> str:
        name = tool_call.function.name
        func = TOOL_FUNCTIONS.get(name)
        if func is None:
            return f"ERROR: unknown tool {name}"
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            return f"ERROR: could not parse arguments for {name}: {e}"
        try:
            return func(**args)
        except Exception as e:
            return f"ERROR: tool {name} raised an exception: {e}"

    def _extract_final_answer(self, content: str) -> str:
        marker = "FINAL ANSWER:"
        if marker in content:
            return content.split(marker, 1)[1].strip()
        return content.strip()
