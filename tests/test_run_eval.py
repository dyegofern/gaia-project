import threading
import time

import run_eval
from gaia_agent.results_db import ResultsDB

QUESTIONS = [{"task_id": f"t{i}", "question": f"q{i}"} for i in range(6)]


class FakeAgent:
    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, question, task_id=None):
        with self._lock:
            self.calls.append(task_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        if task_id in self.fail_ids:
            raise RuntimeError("boom")
        return f"answer-{task_id}"


def test_runs_questions_in_parallel_and_records_answers(tmp_path):
    db = ResultsDB(str(tmp_path / "runs.db"))
    run_id = db.create_run("lemonade")
    db.add_questions(run_id, QUESTIONS)
    agent = FakeAgent()

    run_eval.run_agent_on_questions(agent, db, run_id, db.incomplete(run_id), workers=4)

    assert agent.max_active > 1
    assert {r["task_id"]: r["answer"] for r in db.results(run_id)} == {q["task_id"]: f"answer-{q['task_id']}" for q in QUESTIONS}
    assert db.incomplete(run_id) == []


def test_failed_tasks_are_incomplete_and_only_they_rerun_on_continue(tmp_path):
    db = ResultsDB(str(tmp_path / "runs.db"))
    run_id = db.create_run()
    db.add_questions(run_id, QUESTIONS)
    run_eval.run_agent_on_questions(FakeAgent(fail_ids={"t1", "t4"}), db, run_id, db.incomplete(run_id), workers=3)

    assert {t["task_id"] for t in db.incomplete(run_id)} == {"t1", "t4"}

    retry_agent = FakeAgent()
    run_eval.run_agent_on_questions(retry_agent, db, run_id, db.incomplete(run_id), workers=3)

    assert sorted(retry_agent.calls) == ["t1", "t4"]
    assert db.incomplete(run_id) == []


def test_agent_error_string_and_empty_answers_count_as_unanswered(tmp_path):
    db = ResultsDB(str(tmp_path / "runs.db"))
    run_id = db.create_run()
    db.add_questions(run_id, QUESTIONS[:3])
    db.save_answer(run_id, "t0", "AGENT ERROR: x", ok=False)
    db.save_answer(run_id, "t1", "  ", ok=True)
    db.save_answer(run_id, "t2", "fine", ok=True)

    assert {t["task_id"] for t in db.incomplete(run_id)} == {"t0", "t1"}


def test_add_questions_does_not_overwrite_existing_answers(tmp_path):
    db = ResultsDB(str(tmp_path / "runs.db"))
    run_id = db.create_run()
    db.add_questions(run_id, QUESTIONS[:2])
    db.save_answer(run_id, "t0", "kept")
    db.add_questions(run_id, QUESTIONS[:3])

    answers = {r["task_id"]: r["answer"] for r in db.results(run_id)}
    assert answers["t0"] == "kept"
    assert "t2" in answers


def test_runs_are_tracked_separately_and_export_round_trips(tmp_path):
    db = ResultsDB(str(tmp_path / "runs.db"))
    first, second = db.create_run(), db.create_run()
    assert db.latest_run_id() == second
    db.add_questions(first, QUESTIONS[:1])
    db.save_answer(first, "t0", "one")
    db.add_questions(second, QUESTIONS[:1])
    db.save_answer(second, "t0", "two")

    export_path = tmp_path / "out.txt"
    run_eval.export_results(run_eval.answers_from_run(db, first), str(export_path))

    assert run_eval.load_results_from_file(str(export_path)) == [{"task_id": "t0", "submitted_answer": "one"}]
