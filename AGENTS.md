# VibeVoice Agent Guidelines

This document provides coding standards and workflows for agentic agents working in the VibeVoice repository.

## Build & Test Commands

### Installation
```bash
uv sync                    # Install all dependencies
uv add <package>           # Add new dependency
uv add --no-build-isolation flash-attn  # For packages needing torch first
```

### Running Tests
```bash
# Run single test file (vLLM tests)
python vllm_plugin/tests/test_api.py audio.wav

# Run with hotwords and custom URL
python vllm_plugin/tests/test_api_auto_recover.py audio.wav --hotwords "Microsoft,Azure" --url http://localhost:8000

# Run demo applications
uv run --dir demo/web uvicorn app:app --host 0.0.0.0 --port 8000
python demo/vibevoice_realtime_demo.py
```

### Build Commands
```bash
uv build                   # Build wheel package
uv publish                 # Publish to package registry
```

## Code Style Guidelines

### General Principles
- **Minimalism First**: Code must be concise, clear, and minimal
- **Research-focused**: This is academic research code, not commercial software
- **High Readability**: Prioritize clarity over cleverness
- **No Over-Engineering**: Reject unnecessary encapsulation or excessive abstraction

### Python Style
- **Imports**: Group as: stdlib, third-party, local (`vibevoice`, `vllm_plugin`)
- **Type Hints**: Use full type hints for all public functions
  ```python
  def process_audio(path: str, hotwords: Optional[str] = None) -> str:
  ```
- **Error Handling**: Use specific exceptions; avoid bare `except`
- **Documentation**: All public functions need docstrings with Args/Returns/Raises

### Naming Conventions
- **Classes**: PascalCase (`VibeVoiceForCausalLM`, `AudioMediaIO`)
- **Functions**: snake_case (`load_audio_use_ffmpeg`, `test_transcription`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_TOKENS = 32768`)
- **Variables**: snake_case

### File Organization
- `vibevoice/` - Core TTS/ASR model code
- `vllm_plugin/` - vLLM integration and API
- `demo/` - Demo applications and scripts
- `finetuning-asr/` - ASR fine-tuning code

### Audio Processing
- Use FFmpeg for all audio loading (not librosa)
- Always resample to 24000 Hz
- Normalize audio to [-1.0, 1.0] range

### Testing Requirements
- Tests must be self-contained and executable
- Include example usage in docstrings
- Test both success and error paths
- For API tests: include hotwords parameter testing

### LLM-Generated Code Policy
- **Strictly Scrutinize**: Large AI-generated chunks require human verification
- **Remove Redundancy**: Eliminate duplicate or unnecessary logic
- **Verify Every Line**: Ensure each line has absolute necessity

## Documentation Standards
- **Precise & Concise**: Maximize information density
- **English Only**: No non-English comments or docs
- **Accurate**: All claims must be verified and testable

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Add package | `uv add <pkg>` |
| Run TTS app | `uv run --dir demo/web uvicorn app:app` |
| Test ASR API | `python vllm_plugin/tests/test_api.py audio.wav` |
| Build wheel | `uv build` |

---

**Remember**: This is research code. Prioritize correctness, clarity, and minimalism over all else.
