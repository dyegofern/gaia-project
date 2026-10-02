"""Tests for LLM backend improvements: rate limiting, health checks."""
import os
import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError, APIError, BadRequestError

from gaia_agent.agent import _format_api_error
from gaia_agent.llm import (
    check_backend_health,
    _parse_reset_seconds,
    _maybe_wait_for_rate_limit,
)


class TestBackendHealthCheck:
    def test_lemonade_health_check_success(self):
        with patch("gaia_agent.llm.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            result = check_backend_health("lemonade")

            assert result is True
            mock_get.assert_called_once()

    def test_lemonade_health_check_failure(self):
        with patch("gaia_agent.llm.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            result = check_backend_health("lemonade")

            assert result is False

    def test_groq_health_check_no_token(self):
        # Unset GROQ_API_KEY temporarily
        old_value = os.environ.pop("GROQ_API_KEY", None)
        try:
            result = check_backend_health("groq")
            assert result is False
        finally:
            if old_value:
                os.environ["GROQ_API_KEY"] = old_value

    def test_hf_health_check_no_token(self):
        old_value = os.environ.pop("HF_TOKEN", None)
        try:
            result = check_backend_health("hf")
            assert result is False
        finally:
            if old_value:
                os.environ["HF_TOKEN"] = old_value


class TestRateLimitParsing:
    def test_parse_simple_seconds(self):
        result = _parse_reset_seconds("10s")
        assert result == 10

    def test_parse_minutes(self):
        result = _parse_reset_seconds("2m30s")
        assert result == 150

    def test_parse_hours_minutes_seconds(self):
        result = _parse_reset_seconds("1h16m19.2s")
        assert result == 4579.2

    def test_parse_float_seconds(self):
        result = _parse_reset_seconds("12.5s")
        assert result == 12.5

    def test_parse_none(self):
        result = _parse_reset_seconds(None)
        assert result is None

    def test_parse_empty_string(self):
        result = _parse_reset_seconds("")
        assert result is None


class TestRateLimitWait:
    def test_no_wait_when_adequate_tokens(self):
        response = MagicMock()
        response.headers = {
            "x-ratelimit-remaining-tokens": "10000",
            "x-ratelimit-reset-tokens": "10s"
        }

        with patch("gaia_agent.llm.time.sleep") as mock_sleep:
            _maybe_wait_for_rate_limit(response)
            mock_sleep.assert_not_called()

    def test_wait_when_low_tokens(self):
        response = MagicMock()
        response.headers = {
            "x-ratelimit-remaining-tokens": "10",
            "x-ratelimit-reset-tokens": "5s"
        }

        with patch("gaia_agent.llm.time.sleep") as mock_sleep:
            _maybe_wait_for_rate_limit(response)
            mock_sleep.assert_called_once()
            assert mock_sleep.call_args[0][0] <= 90  # Max 90s

    def test_no_wait_when_no_headers(self):
        response = MagicMock()
        response.headers = {}

        with patch("gaia_agent.llm.time.sleep") as mock_sleep:
            _maybe_wait_for_rate_limit(response)
            mock_sleep.assert_not_called()


class TestFormatApiError:
    def test_rate_limit_daily_quota(self):
        fake_response = MagicMock()
        fake_response.request = MagicMock()
        fake_response.status_code = 429
        error = RateLimitError(
            "Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 199711",
            response=fake_response,
            body=None,
        )
        result = _format_api_error(error)
        assert "rate limit" in result.lower()
        assert ("daily token quota" in result.lower() or "tpd" in result.lower())

    def test_rate_limit_per_minute(self):
        fake_response = MagicMock()
        fake_response.request = MagicMock()
        fake_response.status_code = 429
        error = RateLimitError(
            "Rate limit reached for model on tokens per minute",
            response=fake_response,
            body=None,
        )
        result = _format_api_error(error)
        assert "rate limit" in result.lower()

    def test_bad_request_error(self):
        fake_response = MagicMock()
        fake_response.request = MagicMock()
        fake_response.status_code = 400
        error = BadRequestError(
            "Tool call validation failed",
            response=fake_response,
            body=None,
        )
        result = _format_api_error(error)
        assert "bad request" in result.lower() or "400" in result

    def test_generic_api_error(self):
        fake_response = MagicMock()
        fake_response.request = MagicMock()
        fake_response.status_code = 500
        error = APIError(
            "Generic API error",
            request=fake_response.request,
            body=None,
        )
        result = _format_api_error(error)
        assert "error" in result.lower()


class TestFatalBackendErrors:
    def _err(self, cls, status, msg):
        resp = MagicMock(); resp.request = MagicMock(); resp.status_code = status
        return cls(msg, response=resp, body=None)

    def test_402_is_fatal_and_message_explains_credits(self):
        from openai import APIStatusError
        from gaia_agent.llm import is_fatal_backend_error
        e = self._err(APIStatusError, 402, "You have depleted your monthly included credits")
        assert is_fatal_backend_error(e)
        out = _format_api_error(e)
        assert out.startswith("AGENT ERROR: BACKEND UNAVAILABLE") and "credits" in out and "402" in out

    def test_429_and_500_are_not_fatal(self):
        from gaia_agent.llm import is_fatal_backend_error
        assert not is_fatal_backend_error(self._err(RateLimitError, 429, "slow down"))
        assert not is_fatal_backend_error(ValueError("x"))


class TestProbeAndTimeouts:
    def _status_err(self, status, msg):
        from openai import APIStatusError
        resp = MagicMock(); resp.request = MagicMock(); resp.status_code = status
        return APIStatusError(msg, response=resp, body=None)

    def test_probe_reports_exhausted_credits_even_when_listing_models_works(self):
        from gaia_agent.llm import probe_backend
        client = MagicMock()
        client.chat.completions.create.side_effect = self._status_err(402, "depleted your monthly included credits")
        with patch("gaia_agent.llm._get_client_and_model", return_value=(client, "m")):
            ok, reason = probe_backend("hf")
        assert not ok and "402" in reason and "credits" in reason

    def test_probe_ok_sends_two_requests_for_cloud_one_for_lemonade(self):
        from gaia_agent.llm import probe_backend
        client = MagicMock()
        with patch("gaia_agent.llm._get_client_and_model", return_value=(client, "m")):
            assert probe_backend("groq") == (True, "ok")
            assert client.chat.completions.create.call_count == 2
            client.reset_mock()
            assert probe_backend("lemonade") == (True, "ok")
            assert client.chat.completions.create.call_count == 1

    def test_probe_restores_backend_env(self, monkeypatch):
        from gaia_agent.llm import probe_backend
        monkeypatch.setenv("GAIA_LLM_BACKEND", "lemonade")
        with patch("gaia_agent.llm._get_client_and_model", return_value=(MagicMock(), "m")):
            probe_backend("groq")
        assert os.environ["GAIA_LLM_BACKEND"] == "lemonade"

    def test_timeouts_are_retried_once_not_five_times(self, monkeypatch):
        from openai import APITimeoutError
        from gaia_agent import llm
        monkeypatch.setenv("GAIA_LLM_BACKEND", "lemonade")
        client = MagicMock()
        client.chat.completions.with_raw_response.create.side_effect = APITimeoutError(request=MagicMock())
        with patch.object(llm, "_get_client_and_model", return_value=(client, "m")), patch("time.sleep"):
            with pytest.raises(APITimeoutError):
                llm.chat_completion([{"role": "user", "content": "hi"}])
        assert client.chat.completions.with_raw_response.create.call_count == 2

    def test_default_timeout_is_longer_for_lemonade(self, monkeypatch):
        from gaia_agent.llm import _default_timeout
        monkeypatch.delenv("GAIA_LLM_TIMEOUT", raising=False)
        assert _default_timeout("lemonade") > _default_timeout("groq")
        monkeypatch.setenv("GAIA_LLM_TIMEOUT", "42")
        assert _default_timeout("groq") == 42


def test_404_model_not_found_is_fatal_not_retried():
    from openai import APIStatusError
    from gaia_agent.llm import is_fatal_backend_error
    resp = MagicMock(); resp.request = MagicMock(); resp.status_code = 404
    e = APIStatusError("models/x is no longer available", response=resp, body=None)
    assert is_fatal_backend_error(e)
    assert "not found" in _format_api_error(e)
