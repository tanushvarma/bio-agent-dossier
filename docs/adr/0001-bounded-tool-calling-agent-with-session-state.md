---
status: accepted
---

# Bounded Tool-Calling Agent with Explicit Session State

Research orchestration runs inside a bounded Gemini function-calling loop (capped at 8 tool rounds per user turn) that reads from and mutates a single `Research Session` object (`AgentState`), rather than a hardcoded linear pipeline or an unbounded autonomous agent. Researchers routinely ask follow-up questions ("mine more papers on hydrogels", "compare curing times", "regenerate the dossier") without re-fetching gene metadata; explicit session state lets the model selectively invoke only the missing tools while the round cap prevents runaway API loops and cost spikes.

## Consequences

- Every tool handler must treat `AgentState` as the shared store of record (caching `gene_data`, `papers`, and `dossier`) and return compact JSON summaries to the LLM context rather than dumping raw abstracts repeatedly.
- If a turn hits the 8-round ceiling without a final model reply, the agent exits gracefully with a status notice instead of looping indefinitely.
