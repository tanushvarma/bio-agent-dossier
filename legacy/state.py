"""Shared agent state for the conversational research agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Mutable state persisted across a chat session."""

    target: str | None = None
    gene_data: dict[str, Any] | None = None
    papers: list[dict[str, Any]] = field(default_factory=list)
    dossier: dict[str, Any] | None = None
    queries_tried: list[str] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)

    def log_tool(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.tool_log.append({"tool": name, "args": args, "result": result})

    def to_context(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "has_gene_data": self.gene_data is not None,
            "paper_count": len(self.papers),
            "has_dossier": self.dossier is not None,
            "queries_tried": self.queries_tried,
        }
