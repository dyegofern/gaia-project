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
