# gaia_agent/llm.py
from openai import OpenAI

LEMONADE_BASE_URL = "http://localhost:13305/api/v0"
MODEL = "Qwen3.5-35B-A3B-GGUF"

_client = OpenAI(base_url=LEMONADE_BASE_URL, api_key="not-needed")


def chat_completion(messages, tools=None, timeout=120):
    kwargs = {"model": MODEL, "messages": messages, "timeout": timeout}
    if tools:
        kwargs["tools"] = tools
    return _client.chat.completions.create(**kwargs)
