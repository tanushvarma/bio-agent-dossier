---
status: accepted
---

# Standardize on the `google-genai` SDK

All Gemini calls—structured JSON extraction, dossier synthesis, and multi-turn tool calling—go through the official `google-genai` SDK (`from google import genai`) via a shared lazy singleton in `gemini_client.py`, rather than the deprecated `google-generativeai` library. `google-genai` natively supports newer `AQ.` API keys, `response_mime_type="application/json"`, and typed function declarations (`types.FunctionDeclaration`) through one client interface.

## Consequences

- Model selection is controlled centrally via `GEMINI_MODEL` (default `gemini-flash-latest`), so switching models does not require editing individual modules.
- Code must not re-introduce `google.generativeai` imports.
