import pytest

from gaia_agent.answers import clean_answer


@pytest.mark.parametrize("raw,expected", [
    ("Extremely.", "Extremely"),
    ("**Extremely**", "Extremely"),
    ('"Claus"', "Claus"),
    ("`Rd5`", "Rd5"),
    ("FINAL ANSWER: 42", "42"),
    ("  broccoli, celery, fresh basil.  ", "broccoli, celery, fresh basil"),
    ("89706.00**", "89706.00"),
    ("**89706.00", "89706.00"),
    ("3.5", "3.5"),            # decimals untouched
    ("U.S.A.", "U.S.A."),      # abbreviation period kept
    ("Saint Petersburg", "Saint Petersburg"),
    ("Yamasaki, Uehara", "Yamasaki, Uehara"),
    ("it's", "it's"),          # inner apostrophes untouched
    ("", ""),
])
def test_clean_answer(raw, expected):
    assert clean_answer(raw) == expected


def test_agent_extraction_cleans_answers():
    from gaia_agent.agent import GaiaAgent
    assert GaiaAgent()._extract_final_answer("Thoughts...\nFINAL ANSWER: **Extremely.**") == "Extremely"


def test_export_and_submit_clean_stored_answers(tmp_path, monkeypatch):
    import json
    import run_eval
    out = tmp_path / "r.txt"
    run_eval.export_results([{"task_id": "t", "answer": "Extremely."}], str(out))
    assert json.loads(out.read_text().split("=== Results ===")[1])["submitted_answer"] == "Extremely"
    sent = {}
    class R:
        def raise_for_status(self): pass
        def json(self): return {}
    monkeypatch.setattr(run_eval.requests, "post", lambda url, json, timeout: sent.update(json) or R())
    run_eval.submit("u", "http://code", [{"task_id": "t", "submitted_answer": "**Claus.**"}])
    assert sent["answers"] == [{"task_id": "t", "submitted_answer": "Claus"}]
