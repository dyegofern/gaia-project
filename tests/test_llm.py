# tests/test_llm.py
import pytest

from gaia_agent.llm import chat_completion

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
