# gaia_agent/llm.py
import os

from openai import OpenAI

LEMONADE_BASE_URL = "http://localhost:13305/api/v0"
LEMONADE_MODEL = "Qwen3.5-35B-A3B-GGUF"

HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = os.environ.get("GAIA_HF_MODEL", "openai/gpt-oss-120b")

_clients = {}


def _build_backend(backend):
    if backend == "hf":
        client = OpenAI(base_url=HF_BASE_URL, api_key=os.environ["HF_TOKEN"])
        return client, HF_MODEL

    client = OpenAI(base_url=LEMONADE_BASE_URL, api_key="not-needed")
    return client, LEMONADE_MODEL


def _get_client_and_model():
    backend = os.environ.get("GAIA_LLM_BACKEND", "lemonade")

    if backend == "hf" and not os.environ.get("HF_TOKEN"):
        # Never serve a cached client for a token-less request: the token
        # may have been unset since the cache entry was built (e.g. across
        # tests), and a stale client would silently use a stale/no token.
        raise RuntimeError(
            "GAIA_LLM_BACKEND is set to 'hf' but the HF_TOKEN environment "
            "variable is not set. Set HF_TOKEN to a Hugging Face access "
            "token to use the Hugging Face Inference Providers backend."
        )

    if backend not in _clients:
        _clients[backend] = _build_backend(backend)
    return _clients[backend]


def chat_completion(messages, tools=None, timeout=120):
    client, model = _get_client_and_model()
    kwargs = {"model": model, "messages": messages, "timeout": timeout}
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)
