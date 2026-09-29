# tests/test_agent.py
from unittest.mock import patch, MagicMock

from openai import BadRequestError, RateLimitError

from gaia_agent.agent import GaiaAgent, MAX_ITERATIONS


def _make_fake_rate_limit_error():
    fake_response = MagicMock()
    fake_response.request = MagicMock()
    fake_response.status_code = 429
    return RateLimitError(
        "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day "
        "(TPD): Limit 200000, Used 199711, Requested 1534. Please try again "
        "in 8m58.272s.",
        response=fake_response,
        body=None,
    )


def _make_fake_api_error():
    # BadRequestError (a subclass of openai.APIError, the class agent.py
    # catches) requires a `response` object with `.request` at construction
    # time; a MagicMock satisfies that without needing a real httpx response.
    fake_response = MagicMock()
    fake_response.request = MagicMock()
    fake_response.status_code = 400
    return BadRequestError(
        "Tool call validation failed: tool call validation failed: attempted "
        "to call tool 'fetch_page<|channel|>commentary' which was not in "
        "request.tools",
        response=fake_response,
        body=None,
    )


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


def test_agent_never_injects_a_system_message_after_the_first():
    # Some backends (e.g. Qwen3.5's chat template on Lemonade) reject any
    # conversation where a "system" role message appears anywhere but first,
    # raising a 500 "System message must be at the beginning" error. The
    # turn-budget nudge must use role "user", not "system".
    responses = [_make_tool_call_response() for _ in range(MAX_ITERATIONS)]
    responses.append(_make_final_response("Paris"))

    captured_messages = []

    def fake_chat_completion(messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        return responses[len(captured_messages) - 1]

    with patch("gaia_agent.agent.chat_completion", side_effect=fake_chat_completion), \
         patch("gaia_agent.agent.web_search", return_value="some search result snippet"):
        agent = GaiaAgent()
        agent("What is the capital of France?")

    final_messages = captured_messages[-1]
    system_message_indices = [i for i, m in enumerate(final_messages) if m["role"] == "system"]
    assert system_message_indices == [0]


def test_agent_injects_corrective_message_before_retrying_after_api_error():
    # After an API error, the retry should not be a blind identical call --
    # the conversation should include a corrective user message describing
    # the failure, so the model has a real chance to self-correct instead of
    # just re-sampling the same broken generation.
    responses = [
        _make_fake_api_error(),
        _make_final_response("FINAL ANSWER: Paris"),
    ]

    captured_messages = []

    def fake_chat_completion(messages, tools=None, **kwargs):
        captured_messages.append([dict(m) for m in messages])
        response = responses[len(captured_messages) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    with patch("gaia_agent.agent.chat_completion", side_effect=fake_chat_completion):
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert answer.strip() == "Paris"
    retry_messages = captured_messages[-1]
    assert retry_messages[-1]["role"] == "user"
    assert "invalid" in retry_messages[-1]["content"].lower() or "rejected" in retry_messages[-1]["content"].lower()


def test_agent_retries_once_on_api_error_then_continues():
    # First call raises the API error; the retry (second call) succeeds
    # with a normal tool-call response; then the loop proceeds normally to
    # a final answer.
    responses = [
        _make_fake_api_error(),
        _make_tool_call_response(),
        _make_final_response("FINAL ANSWER: Paris"),
    ]

    with patch("gaia_agent.agent.chat_completion", side_effect=responses) as mock_chat, \
         patch("gaia_agent.agent.web_search", return_value="some search result snippet"):
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert answer.strip() == "Paris"
    assert answer != ""
    assert mock_chat.call_count == 3


def test_agent_falls_back_to_final_answer_after_two_consecutive_api_errors():
    # The first call and its retry both raise the API error; the agent
    # should then abandon the tool-calling loop and go straight to the
    # forced-final-answer fallback call (made with tools=None), which
    # succeeds here.
    responses = [
        _make_fake_api_error(),
        _make_fake_api_error(),
        _make_final_response("FINAL ANSWER: Paris"),
    ]

    with patch("gaia_agent.agent.chat_completion", side_effect=responses) as mock_chat:
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert answer.strip() == "Paris"
    # Original call + one retry + the forced-final-answer fallback call.
    assert mock_chat.call_count == 3
    last_call = mock_chat.call_args
    args, kwargs = last_call
    tools_arg = kwargs.get("tools", None) if "tools" in kwargs else (args[1] if len(args) > 1 else None)
    assert tools_arg is None


def test_agent_returns_error_string_if_final_answer_call_also_fails():
    # Every call, including the forced-final-answer fallback, raises the
    # API error. __call__ must never raise -- it should return an
    # informative error string instead.
    def always_fail(*args, **kwargs):
        raise _make_fake_api_error()

    with patch("gaia_agent.agent.chat_completion", side_effect=always_fail):
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert isinstance(answer, str)
    assert "error" in answer.lower()


def test_agent_surfaces_rate_limit_details_instead_of_generic_error():
    # A RateLimitError (e.g. Groq's daily token quota) carries a specific,
    # actionable message from the server -- the agent should surface it
    # rather than hiding it behind the generic "repeated API errors"
    # string, so it's clear this isn't the same kind of failure as a
    # malformed tool call and no amount of retrying right now will help.
    def always_rate_limited(*args, **kwargs):
        raise _make_fake_rate_limit_error()

    with patch("gaia_agent.agent.chat_completion", side_effect=always_rate_limited):
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert "rate limit" in answer.lower()
    assert "tokens per day" in answer.lower() or "tpd" in answer.lower()


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


def _make_informal_tool_call_response(function_name, params, trailing_content=""):
    # Some models (e.g. Qwen3.5 on Lemonade) sometimes write tool calls as
    # informal XML-like text directly into `content` instead of using the
    # structured `tool_calls` API field. The agent must detect and execute
    # these rather than treating the raw XML text as the final answer.
    param_lines = "".join(
        f"<parameter={name}>\n{value}\n</parameter>\n" for name, value in params.items()
    )
    content = (
        f"<tool_call>\n<function={function_name}>\n{param_lines}</function>\n</tool_call>"
        f"{trailing_content}"
    )
    message = MagicMock()
    message.tool_calls = None
    message.content = content
    message.model_dump.return_value = {"role": "assistant", "content": content}

    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def test_agent_executes_informal_tool_call_written_into_content():
    responses = [
        _make_informal_tool_call_response("web_search", {"query": "test query"}),
        _make_final_response("FINAL ANSWER: Paris"),
    ]

    with patch("gaia_agent.agent.TOOL_FUNCTIONS", {"web_search": MagicMock(return_value="some search result snippet")}) as tool_functions, \
         patch("gaia_agent.agent.chat_completion", side_effect=responses) as mock_chat:
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert answer.strip() == "Paris"
    tool_functions["web_search"].assert_called_once_with(query="test query")
    assert mock_chat.call_count == 2


def test_agent_strips_informal_tool_call_from_forced_final_answer():
    # If the model still writes an informal tool call into content on the
    # very last, tools=None call (after MAX_ITERATIONS is exhausted), the
    # agent must not return that raw XML as the final answer -- there are
    # no more tool calls available to execute it, so it must force one
    # more plain-answer call instead.
    responses = [_make_tool_call_response() for _ in range(MAX_ITERATIONS)]
    responses.append(_make_informal_tool_call_response("web_search", {"query": "test query"}))
    responses.append(_make_final_response("FINAL ANSWER: Paris"))

    with patch("gaia_agent.agent.chat_completion", side_effect=responses) as mock_chat, \
         patch("gaia_agent.agent.TOOL_FUNCTIONS", {"web_search": MagicMock(return_value="some search result snippet")}):
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert answer.strip() == "Paris"
    assert "<tool_call>" not in answer
    # MAX_ITERATIONS calls + the informal-tool-call attempt + one more forced call.
    assert mock_chat.call_count == MAX_ITERATIONS + 2


def test_agent_never_returns_raw_informal_tool_call_xml_as_final_answer():
    # Regression guard: if parsing/execution of the informal tool call ever
    # regresses, the agent must not silently return the raw XML as if it
    # were a real answer.
    responses = [
        _make_informal_tool_call_response("web_search", {"query": "test query"}),
        _make_final_response("FINAL ANSWER: Paris"),
    ]

    with patch("gaia_agent.agent.chat_completion", side_effect=responses), \
         patch("gaia_agent.agent.TOOL_FUNCTIONS", {"web_search": MagicMock(return_value="some search result snippet")}):
        agent = GaiaAgent()
        answer = agent("What is the capital of France?")

    assert "<tool_call>" not in answer
    assert "<function=" not in answer
