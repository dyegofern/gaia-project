# tests/test_agent.py
from unittest.mock import patch, MagicMock

from gaia_agent.agent import GaiaAgent, MAX_ITERATIONS


def _make_tool_call_response():
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "web_search"
    tool_call.function.arguments = '{"query": "test query"}'

    message = MagicMock()
    message.tool_calls = [tool_call]
    message.content = None
    message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "test query"}'},
            }
        ],
    }

    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def _make_final_response(content):
    message = MagicMock()
    message.tool_calls = None
    message.content = content

    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def test_agent_never_returns_empty_string_when_max_iterations_exhausted():
    responses = [_make_tool_call_response() for _ in range(MAX_ITERATIONS)]
    responses.append(_make_final_response("Paris"))

    with patch("gaia_agent.agent.chat_completion", side_effect=responses) as mock_chat, \
         patch("gaia_agent.agent.web_search", return_value="some search result snippet"):
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert answer.strip() == "Paris"
    assert answer != ""
    # One forced extra call beyond MAX_ITERATIONS, made without tools.
    assert mock_chat.call_count == MAX_ITERATIONS + 1
    last_call = mock_chat.call_args
    args, kwargs = last_call
    tools_arg = kwargs.get("tools", None) if "tools" in kwargs else (args[1] if len(args) > 1 else None)
    assert tools_arg is None


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
