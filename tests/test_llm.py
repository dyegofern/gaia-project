# tests/test_llm.py
from unittest.mock import MagicMock

import pytest

from gaia_agent.llm import chat_completion, _maybe_wait_for_rate_limit

def test_chat_completion_returns_content():
    messages = [{"role": "user", "content": "Reply with exactly the word: PONG"}]
    response = chat_completion(messages)
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0


def test_chat_completion_with_hf_backend_returns_content(monkeypatch):
    monkeypatch.setenv("GAIA_LLM_BACKEND", "hf")
    messages = [{"role": "user", "content": "Reply with exactly the word: PONG"}]
    response = chat_completion(messages)
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0


def test_hf_backend_raises_clear_error_without_token(monkeypatch):
    monkeypatch.setenv("GAIA_LLM_BACKEND", "hf")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    messages = [{"role": "user", "content": "Reply with exactly the word: PONG"}]
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        chat_completion(messages)


def test_chat_completion_with_groq_backend_returns_content(monkeypatch):
    monkeypatch.setenv("GAIA_LLM_BACKEND", "groq")
    messages = [{"role": "user", "content": "Reply with exactly the word: PONG"}]
    response = chat_completion(messages)
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0


def test_groq_backend_raises_clear_error_without_token(monkeypatch):
    monkeypatch.setenv("GAIA_LLM_BACKEND", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    messages = [{"role": "user", "content": "Reply with exactly the word: PONG"}]
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        chat_completion(messages)


def _make_response_with_headers(headers):
    response = MagicMock()
    response.headers = headers
    return response


def test_maybe_wait_for_rate_limit_sleeps_when_tokens_nearly_exhausted(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("gaia_agent.llm.time.sleep", lambda s: sleep_calls.append(s))

    response = _make_response_with_headers({
        "x-ratelimit-remaining-tokens": "500",
        "x-ratelimit-reset-tokens": "12.5s",
    })
    _maybe_wait_for_rate_limit(response, low_watermark=1000)

    assert sleep_calls == [12.5]


def test_maybe_wait_for_rate_limit_does_not_sleep_when_tokens_plentiful(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("gaia_agent.llm.time.sleep", lambda s: sleep_calls.append(s))

    response = _make_response_with_headers({
        "x-ratelimit-remaining-tokens": "7000",
        "x-ratelimit-reset-tokens": "12.5s",
    })
    _maybe_wait_for_rate_limit(response, low_watermark=1000)

    assert sleep_calls == []


def test_maybe_wait_for_rate_limit_handles_missing_headers_gracefully(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("gaia_agent.llm.time.sleep", lambda s: sleep_calls.append(s))

    response = _make_response_with_headers({})
    _maybe_wait_for_rate_limit(response, low_watermark=1000)

    assert sleep_calls == []


def test_lemonade_calls_are_serialized_across_threads(monkeypatch):
    # Lemonade's llama-server runs with --parallel 1, so concurrent requests
    # just queue server-side -- and the client timeout keeps ticking while
    # queued, turning parallel eval runs into timeouts. chat_completion must
    # hold concurrent lemonade calls back in-process instead.
    import threading
    import time

    from gaia_agent import llm

    monkeypatch.delenv("GAIA_LLM_BACKEND", raising=False)
    monkeypatch.delenv("GAIA_LLM_CONCURRENCY", raising=False)
    llm._semaphores.clear()

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def slow_create(**kwargs):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        raw = MagicMock()
        raw.headers = {}
        return raw

    fake_client = MagicMock()
    fake_client.chat.completions.with_raw_response.create.side_effect = slow_create
    monkeypatch.setattr(llm, "_get_client_and_model", lambda: (fake_client, "m"))

    threads = [threading.Thread(target=chat_completion, args=([{"role": "user", "content": "hi"}],)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max_active"] == 1


def test_llm_concurrency_env_override(monkeypatch):
    from gaia_agent import llm

    monkeypatch.setenv("GAIA_LLM_CONCURRENCY", "3")
    assert llm._concurrency_limit("lemonade") == 3
    monkeypatch.delenv("GAIA_LLM_CONCURRENCY")
    assert llm._concurrency_limit("lemonade") == 1
    assert llm._concurrency_limit("groq") > 1
