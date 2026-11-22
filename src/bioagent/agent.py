"""Bounded structured tool-calling loop.

    pre-flight (deterministic) -> up to max_steps-1 tool steps -> 1 finalize step -> citation verifier

Every LLM call counts as one step, so a run never makes more than `max_steps` model calls.
The final step must return a DossierDraft (validated JSON); deterministic fields (IDs,
evidence list) are filled in from the trace, never by the model.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from pydantic import ValidationError

from . import guardrails
from .clients.llm import LLM, ToolResult, drop_properties, inline_refs, make_llm
from .config import Settings, get_settings
from .schemas import Abstention, DossierDraft, LLMCall, Resolution, RunResult, TargetDossier, ToolTrace
from .tools import Toolbox

SYSTEM_PROMPT = """You are a biomedical target-dossier agent. You gather evidence with tools, then write a \
short, factual dossier about one human drug target.

Rules:
- Call tools to gather evidence. Typical plan: fetch_opentargets_drugs (with the Ensembl ID given), \
fetch_ncbi_gene, then one or two focused search_literature queries on mechanism and drug response.
- Cite only IDs that appeared in tool results in this conversation: ChEMBL IDs from \
fetch_opentargets_drugs, PMIDs from search_literature. Never invent or recall IDs from memory.
- Every drug claim and mechanism claim needs evidence_ids. If you have no evidence for a claim, leave it out.
- Drug names must be written as the tool returned them.
- Stop calling tools once you have enough; you have a small, fixed number of steps."""

FINALIZE_INSTRUCTION = """Write the final dossier now as JSON matching the schema. Use only ChEMBL IDs \
and PMIDs that appeared in the tool results above (PMIDs as digits only). Keep the summary under \
150 words. Omit any claim you cannot support with those IDs."""


def _user_prompt(res: Resolution) -> str:
    ids = ", ".join(f"{k} {v}" for k, v in (("Ensembl", res.ensembl_id), ("NCBI Gene", res.entrez_id)) if v)
    return f"Build a target dossier for {res.approved_symbol} ({ids}). The identity is already verified."


class BioAgent:
    def __init__(self, llm: LLM, toolbox: Toolbox, *, max_steps: int = 6, max_tokens_per_run: int = 60_000,
                 citation_mode: Literal["strip", "flag"] = "strip", max_calls_per_step: int = 4,
                 clock: Callable[[], float] = time.perf_counter):
        if max_steps < 2:
            raise ValueError("max_steps must allow at least one tool step and the final step")
        self.llm, self.toolbox = llm, toolbox
        self.max_steps, self.max_tokens_per_run = max_steps, max_tokens_per_run
        self.citation_mode, self.max_calls_per_step = citation_mode, max_calls_per_step
        self._clock = clock
        self._draft_schema = drop_properties(inline_refs(DossierDraft.model_json_schema()), {"verified"})

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> BioAgent:
        s = settings or get_settings()
        return cls(make_llm(s), Toolbox.from_settings(s), max_steps=s.max_steps,
                   max_tokens_per_run=s.max_tokens_per_run, citation_mode=s.citation_mode)

    def _llm_call(self, trace: ToolTrace, step: int, kind: str, t0: float, tin: int, tout: int,
                  n_calls: int = 0) -> None:
        trace.llm_calls.append(LLMCall(step=step, kind=kind, provider=self.llm.provider,  # type: ignore[arg-type]
                                       model=self.llm.model, input_tokens=tin, output_tokens=tout,
                                       latency_ms=round((self._clock() - t0) * 1000, 2), n_tool_calls=n_calls))

    def run(self, gene_symbol: str) -> RunResult:
        t_start = self._clock()
        trace = ToolTrace()
        budget = guardrails.TokenBudget(self.max_tokens_per_run)

        def result(status: str, **kw) -> RunResult:
            return RunResult(status=status, query=gene_symbol, trace=trace, steps_used=len(trace.llm_calls),  # type: ignore[arg-type]
                             input_tokens=trace.input_tokens, output_tokens=trace.output_tokens,
                             total_latency_ms=round((self._clock() - t_start) * 1000, 2), **kw)

        pre = guardrails.preflight(gene_symbol, self.toolbox, trace)
        if isinstance(pre, Abstention):
            return result("abstained", abstention=pre, message=pre.message)

        session = self.llm.session(SYSTEM_PROMPT, _user_prompt(pre), self.toolbox.specs())
        try:
            # ---- tool steps 1 .. max_steps-1
            for step in range(1, self.max_steps):
                t0 = self._clock()
                turn = session.next()
                self._llm_call(trace, step, "tool_step", t0, turn.input_tokens, turn.output_tokens,
                               len(turn.tool_calls))
                budget.add(turn.input_tokens, turn.output_tokens)
                if budget.exceeded:
                    return result("aborted_budget", message=budget.message())
                if not turn.tool_calls:
                    break
                results = []
                for i, call in enumerate(turn.tool_calls):
                    if i >= self.max_calls_per_step:     # every requested call still gets an answer
                        content = {"status": "error", "error": f"skipped: max {self.max_calls_per_step} calls per step"}
                    else:
                        content = self.toolbox.execute(call.name, call.args, step, trace).llm_content()
                    results.append(ToolResult(id=call.id, name=call.name, content=content))
                session.add_tool_results(results)

            # ---- final step: structured output
            t0 = self._clock()
            draft, tin, tout = session.finalize(DossierDraft, FINALIZE_INSTRUCTION, self._draft_schema)
            self._llm_call(trace, len(trace.llm_calls) + 1, "finalize", t0, tin, tout)
            budget.add(tin, tout)
            if budget.exceeded:
                return result("aborted_budget", message=budget.message())
        except ValidationError as exc:
            return result("error", message=f"Model output did not match the dossier schema: {exc.error_count()} error(s)")
        except Exception as exc:  # provider/network failure: report, don't crash the caller
            return result("error", message=f"{type(exc).__name__}: {exc}")

        dossier = TargetDossier(
            query=gene_symbol, approved_symbol=pre.approved_symbol or gene_symbol.upper(),
            ensembl_id=pre.ensembl_id, entrez_id=pre.entrez_id, summary=draft.summary,
            drug_claims=draft.drug_claims, mechanism_claims=draft.mechanism_claims,
            provider=self.llm.provider, model=self.llm.model,
        )
        verified, report = guardrails.verify_citations(dossier, trace, self.citation_mode)
        return result("ok", dossier=verified, verification=report)
