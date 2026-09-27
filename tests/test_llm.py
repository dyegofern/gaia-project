# tests/test_llm.py
from gaia_agent.llm import chat_completion

def test_chat_completion_returns_content():
    messages = [{"role": "user", "content": "Reply with exactly the word: PONG"}]
    response = chat_completion(messages)
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0
