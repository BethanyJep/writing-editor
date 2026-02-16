"""Result models returned by agents and the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.observability import EvaluationMetrics


@dataclass
class Correction:
    """An inline text correction."""

    original: str  # exact text to replace
    replacement: str  # corrected text
    agent: str  # which agent suggested this


@dataclass
class Comment:
    """A margin comment attached to a text span."""

    anchor_text: str  # text in the content this comment references
    comment: str  # the feedback
    agent: str  # which agent left this
    category: str = ""  # e.g. "grammar", "fact-check", "style"


@dataclass
class AnnotatedContent:
    """The original content with corrections and comments overlaid."""

    corrections: list[Correction] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corrections": [
                {"original": c.original, "replacement": c.replacement, "agent": c.agent}
                for c in self.corrections
            ],
            "comments": [
                {
                    "anchor_text": c.anchor_text,
                    "comment": c.comment,
                    "agent": c.agent,
                    "category": c.category,
                }
                for c in self.comments
            ],
        }


@dataclass
class AgentResult:
    """Output produced by a single agent."""

    agent_name: str
    findings: str  # markdown-formatted analysis
    suggestions: list[str] = field(default_factory=list)
    cross_references: dict[str, str] = field(default_factory=dict)
    # cross_references maps other agent names → relevance notes

    def summary_line(self) -> str:
        return f"[{self.agent_name}] {len(self.suggestions)} suggestion(s)"


@dataclass
class TeamSynthesis:
    """Combined output from all agents after team synthesis."""

    individual_results: list[AgentResult]
    refined_results: list[AgentResult]  # after cross-agent refinement
    unified_summary: str  # final synthesised recommendations
    content_type: str
    annotations: AnnotatedContent = field(default_factory=AnnotatedContent)
    original_content: str = ""  # kept for rendering the editing view
    trace_id: str = ""  # OpenTelemetry trace ID for observability
    evaluation_metrics: Any = None  # EvaluationMetrics for tracking

    def format_report(self) -> str:
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"  WRITING EDITOR REPORT — {self.content_type.upper()}")
        lines.append("=" * 60)

        for result in self.refined_results:
            lines.append("")
            lines.append(f"── {result.agent_name} {'─' * (45 - len(result.agent_name))}")
            lines.append(result.findings)
            if result.suggestions:
                lines.append("")
                lines.append("  Suggestions:")
                for i, s in enumerate(result.suggestions, 1):
                    lines.append(f"    {i}. {s}")
            if result.cross_references:
                lines.append("")
                lines.append("  Cross-agent connections:")
                for agent, note in result.cross_references.items():
                    lines.append(f"    ↔ {agent}: {note}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("  UNIFIED RECOMMENDATIONS")
        lines.append("=" * 60)
        lines.append(self.unified_summary)
        lines.append("")

        # Editing-mode view (ANSI colours for terminal)
        if self.annotations and (self.annotations.corrections or self.annotations.comments):
            RED = "\033[31m"
            STRIKE = "\033[9m"
            RESET = "\033[0m"
            CYAN = "\033[36m"
            DIM = "\033[2m"

            lines.append("=" * 60)
            lines.append("  EDITING VIEW")
            lines.append("=" * 60)

            if self.annotations.corrections:
                lines.append("")
                lines.append("  Corrections:")
                for c in self.annotations.corrections:
                    lines.append(
                        f"    {STRIKE}{RED}{c.original}{RESET}  →  "
                        f"{RED}{c.replacement}{RESET}  {DIM}[{c.agent}]{RESET}"
                    )

            if self.annotations.comments:
                lines.append("")
                lines.append("  Comments:")
                for c in self.annotations.comments:
                    cat = f" ({c.category})" if c.category else ""
                    lines.append(
                        f"    {CYAN}💬 \"{c.anchor_text}\"{RESET}"
                    )
                    lines.append(
                        f"       {c.comment}  {DIM}[{c.agent}{cat}]{RESET}"
                    )
            lines.append("")

        return "\n".join(lines)
