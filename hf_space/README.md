# Deploying to the HF Space

This directory holds the Space-specific files (`app.py`, `requirements.txt`).
The Space repo root also needs a copy of `gaia_agent/` (the agent package)
alongside `app.py` for the import in app.py to work, since HF Spaces don't
share this repo's Python path.

## Before pushing, decide the LLM backend

The Space runs in HF's cloud and cannot reach `localhost:13305` (Lemonade
Server) or this machine's `/home/dyego/rocm10-test` venv. Set these as
**Space secrets** (Settings -> Repository secrets) before running the Space:

    GAIA_LLM_BACKEND=hf
    HF_TOKEN=<a Hugging Face access token with "Make calls to Inference Providers">

(`GAIA_HF_MODEL` is optional, defaults to `openai/gpt-oss-120b`.)

`python_exec` falls back to `sys.executable` automatically when the local
ROCm venv path doesn't exist, so no extra config is needed for that tool.

`transcribe_audio` currently hits `http://localhost:13305` directly (the
local Lemonade Whisper endpoint) -- this will NOT work from the Space as-is
and needs to be pointed at a reachable transcription backend before relying
on it there.

## Steps to actually deploy (not yet done)

1. Ensure the Space (`dyegofern/gaia`) is set to **Public** visibility --
   required so `agent_code` (`https://huggingface.co/spaces/{SPACE_ID}/tree/main`)
   is inspectable, per the course's submission rules.
2. Copy `gaia_agent/` into this directory (or push the whole repo with
   `app.py` + `requirements.txt` moved to the root and `gaia_agent/` kept
   alongside them).
3. Set the `GAIA_LLM_BACKEND` / `HF_TOKEN` secrets in the Space settings.
4. Push to the Space's git remote (`https://huggingface.co/spaces/dyegofern/gaia`).
5. Open the Space, log in via the LoginButton, click "Run Evaluation & Submit
   All Answers".
