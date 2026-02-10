#!/bin/bash

# Set environment variables
export MODEL_PATH="${MODEL_PATH:-microsoft/VibeVoice}"
export MODEL_DEVICE="${MODEL_DEVICE:-cuda}"
export VOICE_PRESET="${VOICE_PRESET:-en-Carter_man}"

# Launch FastAPI app
cd "$(dirname "$0")"
uv run uvicorn app:app --host 0.0.0.0 --port 8000
