"""Regression tests: tool-usage tracking and attachment download handling."""
import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

import gaia_agent.agent as agent_mod
import gaia_agent.tools as tools
from gaia_agent.results_db import ResultsDB
from run_eval import _answer_failed


@pytest.fixture
def db(tmp_path):
    return ResultsDB(str(tmp_path / "t.db"))


def test_same_tool_can_be_recorded_twice_per_task(db):
    run = db.create_run("lemonade")
    db.record_tool_call(run, "t1", "web_search", True, 5)
    db.record_tool_call(run, "t1", "web_search", False, 7, "boom")
    stats = {r["tool_name"]: r for r in db.get_tool_stats(run)}
    assert stats["web_search"]["call_count"] == 2
    assert stats["web_search"]["error_count"] == 1


def test_old_tool_usage_table_is_migrated_keeping_rows(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL NOT NULL, backend TEXT);
        CREATE TABLE tool_usage (
            run_id INTEGER NOT NULL, task_id TEXT NOT NULL, tool_name TEXT NOT NULL,
            success BOOLEAN NOT NULL, duration_ms INTEGER NOT NULL, error_message TEXT,
            PRIMARY KEY (run_id, task_id, tool_name));
        INSERT INTO runs (started_at, backend) VALUES (1, 'x');
        INSERT INTO tool_usage VALUES (1, 't', 'web_search', 1, 10, NULL);
    """)
    conn.commit(); conn.close()
    db = ResultsDB(path)
    db.record_tool_call(1, "t", "web_search", True, 5)  # would have violated the old key
    assert db.get_tool_stats(1)[0]["call_count"] == 2


def test_returned_error_string_counts_as_failure_and_keeps_message(db):
    run = db.create_run("lemonade")
    agent_mod.set_global_db(db, run)
    agent_mod._current_task_id.set("t1")
    wrapped = agent_mod._instrument_tool(lambda: "ERROR: nope", "download_gaia_file")
    assert wrapped() == "ERROR: nope"
    row = db.get_tool_stats(run)[0]
    assert row["error_count"] == 1
    with db._connect() as c:
        assert c.execute("SELECT error_message FROM tool_usage").fetchone()[0] == "ERROR: nope"


def test_recording_failure_never_breaks_the_tool(db):
    broken = MagicMock()
    broken.record_tool_call.side_effect = RuntimeError("db down")
    agent_mod.set_global_db(broken, 1)
    assert agent_mod._instrument_tool(lambda: "fine", "x")() == "fine"


def _resp(status, text="", ok=None):
    r = MagicMock(status_code=status, text=text, content=b"data", headers={})
    r.ok = status < 400 if ok is None else ok
    r.raise_for_status.side_effect = None if r.ok else __import__("requests").exceptions.HTTPError(str(status))
    return r


def test_missing_server_file_fails_fast_without_retries(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "SCRATCH_DIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(tools, "LOCAL_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = []
    def fake_get(url, **kw):
        calls.append(url)
        return _resp(404, '{"detail":"No file path associated with task_id x."}')
    with patch.object(tools.requests, "get", fake_get), patch.object(tools.time, "sleep") as sleep:
        out = tools.download_gaia_file("x")
    assert out.startswith("ERROR") and "Do not retry" in out
    assert len(calls) == 1 and not sleep.called


def test_local_attachment_is_used_when_present(monkeypatch, tmp_path):
    files = tmp_path / "files"; files.mkdir()
    (files / "abc123.xlsx").write_bytes(b"x")
    monkeypatch.setattr(tools, "SCRATCH_DIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(tools, "LOCAL_FILES_DIR", str(files))
    with patch.object(tools.requests, "get", side_effect=AssertionError("network used")):
        assert tools.download_gaia_file("abc123").endswith("abc123.xlsx")


def test_youtube_frames_accepts_question():
    with patch.object(tools, "_download_youtube_video", side_effect=Exception("no net")):
        assert tools.analyze_youtube_frames("http://y", question="how many birds?").startswith("ERROR")


@pytest.mark.parametrize("answer,failed", [
    ("I’m unable to retrieve the attached file", True),
    ("Data not available", True),
    ("Claus", False),
])
def test_give_up_answers_are_flagged(answer, failed):
    assert _answer_failed(answer) is failed


def test_agent_stops_immediately_on_billing_error():
    from openai import APIStatusError
    resp = MagicMock(); resp.request = MagicMock(); resp.status_code = 402
    err = APIStatusError("You have depleted your monthly included credits", response=resp, body=None)
    with patch.object(agent_mod, "chat_completion", side_effect=err) as cc:
        out = agent_mod.GaiaAgent()("What is 2+2?")
    assert out.startswith(agent_mod.BACKEND_FATAL_PREFIX)
    assert cc.call_count == 1  # no corrective-nudge retry, no forced-answer fallback


def test_transcript_roundtrip_truncates_long_tool_output(db):
    run = db.create_run("lemonade")
    msgs = [{"role": "user", "content": "q"}, {"role": "tool", "content": "x" * 9000}]
    db.save_transcript(run, "t1", msgs, answer="42", max_chars=100)
    t = db.get_transcript(run, "t1")
    assert t["answer"] == "42" and t["messages"][0]["content"] == "q"
    assert len(t["messages"][1]["content"]) < 200 and "truncated" in t["messages"][1]["content"]
    assert db.get_transcript(run, "missing") is None


def test_set_run_process_and_old_runs_table_migrates(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL NOT NULL, backend TEXT);"
                       "INSERT INTO runs (started_at, backend) VALUES (1, 'x');")
    conn.commit(); conn.close()
    db = ResultsDB(path)
    db.set_run_process(1, 4242, "/tmp/x.log")
    with db._connect() as c:
        row = c.execute("SELECT pid, log_path FROM runs WHERE run_id = 1").fetchone()
    assert (row["pid"], row["log_path"]) == (4242, "/tmp/x.log")


def test_agent_saves_transcript_and_hints_attachment(db):
    run = db.create_run("lemonade")
    agent_mod.set_global_db(db, run)
    reply = MagicMock()
    reply.choices = [MagicMock(message=MagicMock(tool_calls=None, content="FINAL ANSWER: 4"))]
    with patch.object(agent_mod, "chat_completion", return_value=reply), \
         patch.object(agent_mod, "attachment_hint", return_value="(no file is attached to this question)"):
        assert agent_mod.GaiaAgent()("2+2?", task_id="tid") == "4"
    t = db.get_transcript(run, "tid")
    assert t["answer"] == "4"
    assert t["messages"][0]["role"] == "user" and "no file is attached" in t["messages"][0]["content"]
    assert t["messages"][-1] == {"role": "assistant", "content": "FINAL ANSWER: 4"}


def _reply(content):
    r = MagicMock()
    r.choices = [MagicMock(message=MagicMock(tool_calls=None, content=content))]
    return r


def test_prose_answer_is_reformatted_into_final_answer_line():
    prose = 'Based on the transcription, Teal\'c responds with: **"Extremely."**'
    with patch.object(agent_mod, "chat_completion", side_effect=[_reply(prose), _reply("FINAL ANSWER: Extremely")]) as cc:
        assert agent_mod.GaiaAgent()("What does Teal'c say?") == "Extremely"
    assert cc.call_count == 2


def test_prose_answer_is_kept_if_reformat_fails():
    from openai import APIError
    prose = "The answer is 4."
    with patch.object(agent_mod, "chat_completion", side_effect=[_reply(prose), APIError("x", request=MagicMock(), body=None)]):
        assert agent_mod.GaiaAgent()("2+2?") == prose


def test_marked_answer_needs_no_extra_call():
    with patch.object(agent_mod, "chat_completion", return_value=_reply("FINAL ANSWER: 4")) as cc:
        agent_mod.GaiaAgent()("2+2?")
    assert cc.call_count == 1


@pytest.mark.parametrize("answer", [
    "I attempted to access the attached audio file, but the file download failed.",
    "Sorry, I cannot access the spreadsheet.",
])
def test_attachment_failure_prose_is_flagged_as_failed(answer):
    assert _answer_failed(answer)
