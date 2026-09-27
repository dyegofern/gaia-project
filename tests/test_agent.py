# tests/test_agent.py
from gaia_agent.agent import GaiaAgent

def test_agent_answers_simple_question_without_tools():
    agent = GaiaAgent()
    answer = agent("What is 2 + 2? Reply with just the number.")
    assert "4" in answer
    assert "FINAL ANSWER" not in answer

def test_agent_can_use_web_search_tool():
    agent = GaiaAgent()
    answer = agent(
        "Use the web_search tool to find what year the Eiffel Tower was completed. "
        "Reply with just the year as a number."
    )
    assert "1889" in answer
