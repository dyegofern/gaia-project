# run_eval.py
import argparse
import json

import requests

from gaia_agent.agent import GaiaAgent

SCORING_API_URL = "https://agents-course-unit4-scoring.hf.space"


def fetch_questions():
    resp = requests.get(f"{SCORING_API_URL}/questions", timeout=15)
    resp.raise_for_status()
    return resp.json()


def run_agent_on_questions(agent, questions):
    results = []
    for item in questions:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping malformed item: {item}")
            continue
        print(f"\n--- Task {task_id} ---\n{question_text}")
        try:
            answer = agent(question_text, task_id=task_id)
        except Exception as e:
            answer = f"AGENT ERROR: {e}"
        print(f"Answer: {answer}")
        results.append({"task_id": task_id, "submitted_answer": answer})
    return results


def submit(username, agent_code, answers):
    payload = {"username": username, "agent_code": agent_code, "answers": answers}
    resp = requests.post(f"{SCORING_API_URL}/submit", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Run on a single random question instead of the full set")
    parser.add_argument("--submit", action="store_true", help="Submit answers to the real scoring API")
    parser.add_argument("--username", help="HF username, required with --submit")
    parser.add_argument("--agent-code", help="Public URL to this code, required with --submit")
    args = parser.parse_args()

    agent = GaiaAgent()

    if args.random:
        resp = requests.get(f"{SCORING_API_URL}/random-question", timeout=15)
        resp.raise_for_status()
        questions = [resp.json()]
    else:
        questions = fetch_questions()

    print(f"Fetched {len(questions)} question(s).")
    results = run_agent_on_questions(agent, questions)

    print("\n=== Results ===")
    for r in results:
        print(json.dumps(r, indent=2))

    if args.submit:
        if not args.username or not args.agent_code:
            parser.error("--submit requires --username and --agent-code")
        outcome = submit(args.username, args.agent_code, results)
        print("\n=== Submission Result ===")
        print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
