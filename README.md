# GAIA Agent — HF Agents Course Unit 4

A local agent that answers GAIA benchmark questions via the HF Agents Course
Unit 4 scoring API, using tool-calling with a choice of LLM backends.

## Prerequisites

- Python environment: `/home/dyego/rocm10-test` (ROCm-enabled venv), with
  dependencies installed via `pip install -r requirements.txt`.
- One of the two LLM backends:
  - **Lemonade Server** (default, local, free, no usage limits) running on
    port 13305 with `Qwen3.5-35B-A3B-GGUF` loaded, and `Whisper-Large-v3-Turbo`
    loaded for the `transcribe_audio` tool
    (`curl -s http://localhost:13305/api/v0/models` to check; load with
    `lemonade-server load Whisper-Large-v3-Turbo` if missing).
  - **Hugging Face Inference Providers** (cloud, free monthly credits, faster) —
    set `GAIA_LLM_BACKEND=hf` and `HF_TOKEN` (a Hugging Face access token with
    "Make calls to Inference Providers" permission). Note the free tier has a
    monthly credit cap; once exhausted, requests fail with a 402 error until
    the next billing cycle or until you add paid credits.
  - **Groq** (cloud, genuinely free tier, fast) — set `GAIA_LLM_BACKEND=groq`
    and `GROQ_API_KEY` (a Groq API key). The free tier for
    `openai/gpt-oss-120b` allows 1000 requests/min and 8000 tokens/min, no
    credit card or monthly credit cap involved.

## Choosing a backend

```bash
# Local Lemonade Server (default — no env vars needed)
/home/dyego/rocm10-test/bin/python run_eval.py --random

# Hugging Face Inference Providers
export HF_TOKEN=your_hf_token
export GAIA_LLM_BACKEND=hf
export GAIA_HF_MODEL=openai/gpt-oss-120b   # optional, this is the default
/home/dyego/rocm10-test/bin/python run_eval.py --random

# Groq
export GROQ_API_KEY=your_groq_api_key
export GAIA_LLM_BACKEND=groq
export GAIA_GROQ_MODEL=openai/gpt-oss-120b   # optional, this is the default
/home/dyego/rocm10-test/bin/python run_eval.py --random
```

## Usage

Run on a single random question (fast smoke test):

    /home/dyego/rocm10-test/bin/python run_eval.py --random

Run on the full question set (no submission):

    /home/dyego/rocm10-test/bin/python run_eval.py

Submit answers to the real leaderboard (requires a public HF Space URL hosting
this code):

    /home/dyego/rocm10-test/bin/python run_eval.py --submit \
        --username YOUR_HF_USERNAME \
        --agent-code https://huggingface.co/spaces/YOUR_SPACE/tree/main

## Running tests

    /home/dyego/rocm10-test/bin/python -m pytest tests/ -v

Note: `tests/test_llm.py` includes a test for the HF backend that requires
`HF_TOKEN` to be set to pass; it's skipped implicitly failing if unset, so
export `HF_TOKEN` before running the full suite if you want it to pass too.

## Architecture

- `gaia_agent/llm.py` — LLM client supporting three OpenAI-compatible backends
  (local Lemonade Server, Hugging Face Inference Providers, or Groq), selected
  via the `GAIA_LLM_BACKEND` env var (`"lemonade"` default, `"hf"`, or `"groq"`).
- `gaia_agent/tools.py` — web_search, fetch_page, download_gaia_file, read_file,
  python_exec, transcribe_audio (speech-to-text via Lemonade's local Whisper model).
- `gaia_agent/agent.py` — `GaiaAgent`, the tool-calling loop, matches the course's
  `BasicAgent.__call__(question) -> str` interface (plus an optional `task_id`).
  Guarantees a non-empty best-effort answer even if the tool-calling budget
  (`MAX_ITERATIONS`) runs out before the model naturally stops.
- `run_eval.py` — fetches questions from the scoring API, runs the agent, prints
  results, and optionally submits.

See `docs/superpowers/specs/2026-09-26-gaia-agent-design.md` for full design rationale.

## Known limitations

- No image/video understanding tools (audio is now supported via
  `transcribe_audio`) — GAIA questions requiring those modalities will likely
  be answered incorrectly.
- `python_exec` runs with no sandboxing beyond a subprocess timeout; fine for
  personal/trusted use, not suitable for untrusted input.
- Hard, multi-hop research questions can still produce wrong (but non-empty)
  answers if the agent doesn't find or recognize the right information within
  its tool-call budget.
