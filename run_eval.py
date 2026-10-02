# run_eval.py
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from gaia_agent.agent import BACKEND_FATAL_PREFIX, GaiaAgent, set_global_db
from gaia_agent.results_db import DEFAULT_DB_PATH, DONE, ResultsDB
from gaia_agent.llm import check_backend_health

DEFAULT_WORKERS = 10

SCORING_API_URL = "https://agents-course-unit4-scoring.hf.space"


def fetch_questions():
    resp = requests.get(f"{SCORING_API_URL}/questions", timeout=15)
    resp.raise_for_status()
    return resp.json()


_print_lock = threading.Lock()


_give_up_prefixes = ("unable to", "cannot determine", "can't determine", "could not", "couldn't", "i cannot", "i can't", "i was unable", "i am unable", "i'm unable", "i'm not able", "i am not able", "no answer", "data not available", "not available", "insufficient information")


def _answer_failed(answer):
    # "Unable to access audio file"-style answers come from flaky attachment
    # downloads, not real answers -- keep them retryable via --continue.
    if not answer or not answer.strip() or answer.startswith("AGENT ERROR"):
        return True
    normalized = answer.strip().lower().replace("\u2019", "'").replace("\u2018", "'")
    return normalized.startswith(_give_up_prefixes)


def verify_answer(question: str, answer: str) -> dict:
    """Basic answer verification heuristics."""
    verification = {
        "has_final_answer_marker": "FINAL ANSWER:" in answer,
        "is_empty": not answer or not answer.strip(),
        "is_too_long": len(answer) > 500,
        "contains_error_indicators": any(
            indicator in answer.lower()
            for indicator in ["error", "failed", "unable", "cannot", "impossible"]
        ),
        "looks_reasonable": len(answer.strip()) > 0 and len(answer.strip()) < 500
    }
    # The agent returns the already-extracted answer, so the FINAL ANSWER marker
    # and error-word heuristics are informational only, not validity criteria.
    verification["is_valid"] = (
        not verification["is_empty"] and not verification["is_too_long"]
    )
    return verification


def run_agent_on_questions(agent, db, run_id, questions, workers=DEFAULT_WORKERS):
    """Run the agent on each question in a thread pool, recording every
    answer in the DB the moment it finishes (so an interrupted run loses
    nothing that was already answered)."""
    valid = []
    for item in questions:
        if not item.get("task_id") or item.get("question") is None:
            print(f"Skipping malformed item: {item}")
            continue
        valid.append(item)
    total = len(valid)
    finished = [0]

    # Use tqdm for progress bar if available, otherwise use simple print
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    backend_dead = threading.Event()

    def work(item):
        if backend_dead.is_set():
            return  # leave it pending so --continue picks it up
        task_id, question_text = item["task_id"], item["question"]
        db.mark_running(run_id, task_id)
        try:
            answer = agent(question_text, task_id=task_id)
            # Verify answer
            verification = verify_answer(question_text, answer)
            if not verification["is_valid"]:
                msg = f"  WARNING: Answer verification failed: {verification}"
                tqdm.write(msg) if use_tqdm else print(msg, flush=True)
            ok = verification["is_valid"] and not _answer_failed(answer)
        except Exception as e:
            answer = f"AGENT ERROR: {e}"
            ok = False
        if answer.startswith(BACKEND_FATAL_PREFIX):
            db.save_answer(run_id, task_id, answer, ok=False)
            if not backend_dead.is_set():
                backend_dead.set()
                msg = f"\nSTOPPING: {answer}\nRemaining questions left pending; fix the backend and run with --continue."
                tqdm.write(msg) if use_tqdm else print(msg, flush=True)
            return
        db.save_answer(run_id, task_id, answer, ok=ok)
        with _print_lock:
            finished[0] += 1
            if use_tqdm:
                pbar.update(1)
            msg = f"\n--- Task {task_id} ({finished[0]}/{total}) ---\n{question_text}\nAnswer: {answer}"
            tqdm.write(msg) if use_tqdm else print(msg, flush=True)

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(work, item) for item in valid]

    try:
        if use_tqdm:
            with tqdm(total=len(valid), desc="Running agent", unit="question") as pbar:
                for f in as_completed(futures):
                    f.result()
        else:
            for f in futures:
                f.result()
        executor.shutdown()
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        db.reset_running(run_id)
        print(f"\nInterrupted. Resume with: run_eval.py --continue (run {run_id}).", flush=True)
        os._exit(130)


def export_results(results, path=None):
    lines = ["=== Results ==="]
    for r in results:
        lines.append(json.dumps({"task_id": r["task_id"], "submitted_answer": r["answer"]}, indent=2))
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "w") as f:
            f.write(text)
    else:
        print("\n" + text, end="")


def answers_from_run(db, run_id):
    rows = db.results(run_id)
    done = [r for r in rows if r["status"] == DONE]
    if len(done) < len(rows):
        print(f"WARNING: run {run_id} has {len(rows) - len(done)} unanswered task(s); only {len(done)} answered task(s) are included.")
    return done


def load_results_from_file(path):
    with open(path) as f:
        text = f.read()
    marker = "=== Results ==="
    if marker not in text:
        raise ValueError(f"{path} does not contain a '{marker}' section")
    results_blob = text.split(marker, 1)[1]
    # The results are printed as consecutive pretty-printed JSON objects
    # (via json.dumps(r, indent=2)), not a single JSON array -- split them
    # back out by matching each top-level {...} block.
    results = []
    depth = 0
    start = None
    for i, ch in enumerate(results_blob):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                results.append(json.loads(results_blob[start:i + 1]))
                start = None
    if not results:
        raise ValueError(f"No results could be parsed from {path}")
    return results


def submit(username, agent_code, answers):
    payload = {"username": username, "agent_code": agent_code, "answers": answers}
    resp = requests.post(f"{SCORING_API_URL}/submit", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    import signal

    def _terminate(signum, frame):
        raise KeyboardInterrupt  # the dashboard's Cancel (SIGTERM) takes the same graceful path as Ctrl-C

    signal.signal(signal.SIGTERM, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Run on a single random question instead of the full set")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Questions to run in parallel (default {DEFAULT_WORKERS})")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite file holding run history (default {DEFAULT_DB_PATH})")
    parser.add_argument("--continue", dest="resume", action="store_true", help="Resume the most recent run: re-run tasks that were interrupted, errored or have no answer, and add any missing questions")
    parser.add_argument("--run-id", type=int, help="Run to use with --from-db/--export (default: most recent)")
    parser.add_argument("--from-db", action="store_true", help="Don't run the agent; use the answers stored in the DB (with --submit and/or --export)")
    parser.add_argument("--export", metavar="FILE", help="Write a run's answers to FILE in the '=== Results ===' format (usable with --from-file)")
    parser.add_argument("--submit", action="store_true", help="Submit answers to the real scoring API")
    parser.add_argument("--username", help="HF username, required with --submit")
    parser.add_argument("--agent-code", help="Public URL to this code, required with --submit")
    parser.add_argument("--log-file", help="Path of the file this process's output is being written to (recorded with the run so the dashboard can show it)")
    parser.add_argument("--from-file", help="Submit previously-saved results from a run_eval.py output file instead of running the agent again")
    args = parser.parse_args()

    if args.from_file:
        if not args.submit:
            parser.error("--from-file requires --submit")
        results = load_results_from_file(args.from_file)
        print(f"Loaded {len(results)} previously-saved result(s) from {args.from_file}.")
    else:
        db = ResultsDB(args.db)
        if args.from_db:
            run_id = args.run_id or db.latest_run_id()
            if run_id is None:
                parser.error(f"no runs found in {args.db}")
        else:
            run_id = run_and_record(args, db)
        answered = answers_from_run(db, run_id)
        print(f"\nRun {run_id}: {db.summary(run_id)}")
        if args.export or not args.from_db:
            export_results(answered, args.export)
        results = [{"task_id": r["task_id"], "submitted_answer": r["answer"]} for r in answered]

    if args.submit:
        if not args.username or not args.agent_code:
            parser.error("--submit requires --username and --agent-code")
        outcome = submit(args.username, args.agent_code, results)
        print("\n=== Submission Result ===")
        print(json.dumps(outcome, indent=2))


def validate_environment():
    """Validate the runtime environment before starting."""
    import subprocess

    errors = []
    warnings = []

    # Check Python interpreter
    python_bin = os.environ.get("GAIA_PYTHON_EXEC_INTERPRETER", "/home/dyego/rocm10-test/bin/python")
    if not os.path.exists(python_bin):
        errors.append(f"Python interpreter not found: {python_bin}")

    # Check required tools
    required_tools = {
        "ffmpeg": "ffmpeg",
        "stockfish": "stockfish"
    }

    for name, cmd in required_tools.items():
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=2)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            warnings.append(f"{name} not found on PATH")

    # Check LEMONADE_SERVER
    if os.environ.get("GAIA_LLM_BACKEND", "lemonade") == "lemonade":
        health_ok = check_backend_health("lemonade")
        if not health_ok:
            warnings.append("Lemonade Server may not be running")

    # Report
    if errors:
        print("=" * 50)
        print("ENVIRONMENT VALIDATION FAILED")
        print("=" * 50)
        for err in errors:
            print(f"  x {err}")
        return False

    if warnings:
        print("=" * 50)
        print("ENVIRONMENT WARNINGS")
        print("=" * 50)
        for warn in warnings:
            print(f"  ! {warn}")
        print()

    print("=" * 50)
    print("ENVIRONMENT VALIDATION PASSED")
    print("=" * 50)
    return True


def run_and_record(args, db):

    backend = os.environ.get("GAIA_LLM_BACKEND", "lemonade")

    # Health check before starting
    if not check_backend_health(backend):
        print(f"ERROR: Backend {backend} is unavailable. Please check:")
        if backend == "lemonade":
            print("  - Lemonade Server is running on localhost:13305")
        elif backend == "groq":
            print("  - GROQ_API_KEY environment variable is set")
        elif backend == "hf":
            print("  - HF_TOKEN environment variable is set")
        elif backend == "gemini":
            print("  - GEMINI_API_KEY environment variable is set")
        sys.exit(1)

    if args.random:
        resp = requests.get(f"{SCORING_API_URL}/random-question", timeout=15)
        resp.raise_for_status()
        questions = [resp.json()]
        run_id = db.create_run(backend)
    else:
        questions = fetch_questions()
        run_id = db.latest_run_id() if args.resume else None
        if run_id is None:
            if args.resume:
                print("No previous run to continue; starting a new one.")
            run_id = db.create_run(backend)
        else:
            print(f"Continuing run {run_id}.")
    db.add_questions(run_id, questions)
    todo = db.incomplete(run_id)
    print(f"Run {run_id}: {len(questions)} question(s), {len(todo)} to run with {args.workers} worker(s).")

    # Record who is running this (for cancel / stalled-run detection) and its log.
    db.set_run_process(run_id, os.getpid(), os.path.abspath(args.log_file) if args.log_file else None)

    # Set global DB for tool instrumentation
    set_global_db(db, run_id)

    run_agent_on_questions(GaiaAgent(), db, run_id, todo, workers=args.workers)
    return run_id


if __name__ == "__main__":
    main()
