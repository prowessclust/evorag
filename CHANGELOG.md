## [2026-04-04] Phase 4 — AI Assistant
### Files Modified
- retriever.py — Added `system_override` to `build_prompt` to allow dynamic prompt generation.
- api.py — Updated the `/query` endpoint and Pydantic models to return a list of persona responses instead of a single string.
- CHANGELOG.md — Logged Phase 4 milestone.

### Files Created
- personas.py — Orchestrates `asyncio.gather` across 8 unique system prompts targeting the retrieved FAISS chunks.

### What was working before my changes
- Phase 3 was complete. System returned 1 unified LLM answer per query.

### What is working after my changes
- Phase 4 is complete. System executes 8 parallel LLM streams mimicking specific personas asynchronously, mapped through a list-based JSON schema.

### What the next person needs to know
- Our `QueryResponse` model changed heavily. Frontend mapping in Phase 8 must now iterate over `responses`. Phase 5 will introduce peer voting logic into this flow.

## [2026-04-04] Environment Fixes — AI Assistant
### Files Modified
- config.py — Added TF_CPP_MIN_LOG_LEVEL and TF_ENABLE_ONEDNN_OPTS suppressions to hide TensorFlow logs.
- embedder.py — Reordered imports to ensure TensorFlow logs are actually suppressed before sentence-transformers loads.

### Files Created
- CHANGELOG.md — Created file as it was incorrectly omitted by the previous teammate.

### What was working before my changes
- Phases 1, 2, and 3 logic.
- API keys, dependencies, and local Ollama instance fully running.
- However, building the semantic index was flooding the terminal with noisy and alarming red TensorFlow warnings.
- No central changelog existed.

### What is working after my changes
- Phases 1, 2, and 3 logic works perfectly.
- TensorFlow logs are suppressed globally, cleanly outputting success messages.
- CHANGELOG.md is now established.

### What the next person needs to know
- Currently on Phase 4. We will need to define global constants in `config.py` for our 8 distinct AI personas to build out the `asyncio` parallel evaluation engine!
