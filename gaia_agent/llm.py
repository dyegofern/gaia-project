# gaia_agent/llm.py
import os
import threading
import re
import socket
import time

from openai import OpenAI

_ipv4_patch_applied = False


def _force_ipv4_dns():
    """Force IPv4-only DNS resolution, process-wide, exactly once.

    This machine's outbound IPv6 routing is broken: SYN packets to IPv6
    addresses go out but never get a response (confirmed via `ss -tnp`
    showing connections stuck in SYN-SENT to api.groq.com's IPv6
    address). Hosts like api.groq.com resolve to both A and AAAA
    records, and httpx/httpcore (used by the `openai` SDK) do not
    implement Happy-Eyeballs-style racing/fallback between address
    families the way a browser would -- they just try one address
    (often IPv6 first) and can hang far past any configured request
    timeout, since a SYN that gets no response at all outlasts normal
    socket read timeouts.

    Monkeypatching `socket.getaddrinfo` to only ever return IPv4
    results is the simplest reliable fix, and doing it process-wide
    (rather than scoped to one httpx transport) is desirable here: it
    protects any other library in this process that might hit a
    dual-stack host over this same broken network path.
    """
    global _ipv4_patch_applied
    if _ipv4_patch_applied:
        return

    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only_getaddrinfo
    _ipv4_patch_applied = True


_force_ipv4_dns()

LEMONADE_BASE_URL = "http://localhost:13305/api/v0"
LEMONADE_MODEL = "Qwen3.5-35B-A3B-GGUF"

HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = os.environ.get("GAIA_HF_MODEL", "openai/gpt-oss-120b")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GAIA_GROQ_MODEL", "openai/gpt-oss-120b")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.environ.get("GAIA_GEMINI_MODEL", "gemini-2.5-flash")

_clients = {}


def _build_backend(backend):
    if backend == "hf":
        client = OpenAI(base_url=HF_BASE_URL, api_key=os.environ["HF_TOKEN"])
        return client, HF_MODEL

    if backend == "groq":
        client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"])
        return client, GROQ_MODEL

    if backend == "gemini":
        client = OpenAI(base_url=GEMINI_BASE_URL, api_key=os.environ["GEMINI_API_KEY"])
        return client, GEMINI_MODEL

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

    if backend == "groq" and not os.environ.get("GROQ_API_KEY"):
        # Never serve a cached client for a token-less request: the token
        # may have been unset since the cache entry was built (e.g. across
        # tests), and a stale client would silently use a stale/no token.
        raise RuntimeError(
            "GAIA_LLM_BACKEND is set to 'groq' but the GROQ_API_KEY "
            "environment variable is not set. Set GROQ_API_KEY to a Groq "
            "API key to use the Groq backend."
        )

    if backend == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        # Never serve a cached client for a token-less request: the token
        # may have been unset since the cache entry was built (e.g. across
        # tests), and a stale client would silently use a stale/no token.
        raise RuntimeError(
            "GAIA_LLM_BACKEND is set to 'gemini' but the GEMINI_API_KEY "
            "environment variable is not set. Set GEMINI_API_KEY to a Gemini "
            "API key to use the Gemini backend."
        )

    if backend not in _clients:
        _clients[backend] = _build_backend(backend)
    return _clients[backend]


_RESET_DURATION_RE = re.compile(
    r"(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m(?!s))?(?:(?P<s>\d+(?:\.\d+)?)s)?"
)


def _parse_reset_seconds(value):
    # Groq-style headers give durations like "12.5s", "1m2.5s", or "1h16m19.2s".
    if value is None:
        return None
    value = value.strip().lower()
    match = _RESET_DURATION_RE.fullmatch(value)
    if not match or not any(match.groups()):
        return None
    hours = float(match.group("h") or 0)
    minutes = float(match.group("m") or 0)
    seconds = float(match.group("s") or 0)
    return hours * 3600 + minutes * 60 + seconds


def _maybe_wait_for_rate_limit(response, low_watermark=1000):
    """Proactively pace requests when a token-based rate limit is close to
    exhausted, using the server's own reported remaining/reset values
    (Groq-style headers) rather than guessing -- avoids repeatedly hitting
    429s and burning the OpenAI SDK's blind exponential-backoff retries.
    Backends without these headers (e.g. Lemonade) are unaffected.
    """
    headers = getattr(response, "headers", None)
    if not headers:
        return

    remaining = headers.get("x-ratelimit-remaining-tokens")
    reset = headers.get("x-ratelimit-reset-tokens")
    if remaining is None or reset is None:
        return

    try:
        remaining = float(remaining)
    except ValueError:
        return

    if remaining >= low_watermark:
        return

    wait_seconds = _parse_reset_seconds(reset)
    if wait_seconds is not None and wait_seconds > 0:
        # Cap the wait: this is meant to smooth over a rolling per-minute
        # token window, not to block indefinitely if a header is ever
        # misread or a much longer (e.g. daily) limit is reported here.
        time.sleep(min(wait_seconds, 90))


_semaphores = {}
_semaphores_lock = threading.Lock()


def _concurrency_limit(backend):
    override = os.environ.get("GAIA_LLM_CONCURRENCY")
    if override:
        return max(1, int(override))
    # Lemonade's llama-server runs with --parallel 1: extra requests just
    # queue server-side while the client timeout keeps ticking, so hold
    # them back here instead. Cloud backends handle concurrent requests.
    return 1 if backend == "lemonade" else 4


def _semaphore_for(backend):
    with _semaphores_lock:
        if backend not in _semaphores:
            _semaphores[backend] = threading.Semaphore(_concurrency_limit(backend))
        return _semaphores[backend]


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
