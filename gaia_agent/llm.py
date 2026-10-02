# gaia_agent/llm.py
import os
import threading
import re
import socket
import time
import requests
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
import backoff

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


# Auth/billing failures are deterministic: retrying (or running the rest of
# the questions) can't help, so callers should stop and report them.
FATAL_STATUS_CODES = (401, 402, 403)


def is_fatal_backend_error(e) -> bool:
    return getattr(e, "status_code", None) in FATAL_STATUS_CODES


def _default_timeout(backend):
    """Seconds to wait for one completion. Lemonade is a local, single-slot
    server whose first request may also have to load the model."""
    override = os.environ.get("GAIA_LLM_TIMEOUT")
    if override:
        return float(override)
    return 300 if backend == "lemonade" else 120


def _giveup(e):
    # Fatal auth/billing errors can't be fixed by retrying; timeouts get their
    # own (single) retry in chat_completion instead of the long backoff.
    return is_fatal_backend_error(e) or isinstance(e, APITimeoutError)


@backoff.on_exception(backoff.constant, APITimeoutError, max_tries=2, interval=1)
@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APIError),
    max_tries=5,
    max_time=3600,
    giveup=_giveup,
    on_backoff=lambda details: print(f"Rate limited, waiting {details['elapsed']:.1f}s before retry...")
)
def chat_completion(messages, tools=None, timeout=None):
    client, model = _get_client_and_model()
    backend = os.environ.get("GAIA_LLM_BACKEND", "lemonade")
    kwargs = {"model": model, "messages": messages,
              "timeout": timeout if timeout is not None else _default_timeout(backend)}
    if tools:
        kwargs["tools"] = tools
    with _semaphore_for(backend):
        raw_response = client.chat.completions.with_raw_response.create(**kwargs)
    _maybe_wait_for_rate_limit(raw_response)
    return raw_response.parse()


def probe_backend(backend=None, timeout=None):
    """Make one real 1-token request and return (ok, reason).

    Listing models is not enough: a Hugging Face account with no credits left
    still lists models fine, then answers every completion with 402. On
    Lemonade this also loads the model, which can take minutes the first time.
    """
    backend = backend or os.environ.get("GAIA_LLM_BACKEND", "lemonade")
    previous = os.environ.get("GAIA_LLM_BACKEND")
    os.environ["GAIA_LLM_BACKEND"] = backend
    try:
        client, model = _get_client_and_model()
        # Two requests for cloud backends: an account that is nearly out of
        # credits can pass one request and fail the next.
        for _ in range(1 if backend == "lemonade" else 2):
            client.chat.completions.create(
                model=model, max_tokens=1, messages=[{"role": "user", "content": "Reply with OK"}],
                timeout=timeout if timeout is not None else (600 if backend == "lemonade" else 30),
            )
        return True, "ok"
    except RateLimitError as e:
        return True, f"reachable but currently rate limited ({str(e)[:120]})"
    except APITimeoutError:
        return False, "request timed out" + (" (the model may still be loading)" if backend == "lemonade" else "")
    except APIError as e:
        if is_fatal_backend_error(e):
            hint = {401: "API key rejected", 402: "credits/quota exhausted", 403: "access forbidden"}[e.status_code]
            return False, f"{hint} (HTTP {e.status_code}): {str(getattr(e, 'message', e))[:200]}"
        return False, f"{type(e).__name__}: {str(e)[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if previous is None:
            os.environ.pop("GAIA_LLM_BACKEND", None)
        else:
            os.environ["GAIA_LLM_BACKEND"] = previous


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
