"""EditorOrchestrator — coordinates the multi-agent editing workflow using Microsoft Agent Framework."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from agent_framework import Agent, AgentExecutorResponse, Message
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.orchestrations import ConcurrentBuilder
from opentelemetry.trace import SpanKind
from opentelemetry.trace.span import format_trace_id

from src.agents.base import AGENT_CONFIGS, create_agent, create_refinement_agent
from src.models.context import EditingContext
from src.models.results import AgentResult, AnnotatedContent, Comment, Correction, TeamSynthesis
from src.observability import (
    EvaluationTracker,
    get_tracer,
    record_suggestions,
    trace_agent,
    trace_phase,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregatedAnalysis:
    """Aggregated analysis from all agents."""
    results: list[AgentResult]


def parse_agent_response(agent_name: str, response_text: str) -> AgentResult:
    """Parse an agent's response into an AgentResult."""
    try:
        data: dict[str, Any] = json.loads(response_text)
    except json.JSONDecodeError:
        data = {"findings": response_text, "suggestions": [], "cross_references": {}}
    
    return AgentResult(
        agent_name=agent_name,
        findings=data.get("findings", response_text),
        suggestions=data.get("suggestions", []),
        cross_references=data.get("cross_references", {}),
    )


class EditorOrchestrator:
    """Coordinates five specialist agents through a 3-phase workflow using Microsoft Agent Framework."""

    def __init__(self, client: AzureOpenAIChatClient, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model
        self.agent_names = list(AGENT_CONFIGS.keys())

    def run(self, content: str, content_type: str) -> TeamSynthesis:
        """Execute the full 3-phase editing workflow."""
        return asyncio.run(self._run_async(content, content_type))

    async def _run_async(self, content: str, content_type: str) -> TeamSynthesis:
        """Async implementation of the editing workflow."""
        context = EditingContext(content=content, content_type=content_type)
        tracer = get_tracer()
        
        # Initialize evaluation tracker for metrics collection
        eval_tracker = EvaluationTracker(content, content_type)
        
        # Start root span for the entire workflow
        with tracer.start_as_current_span(
            "WritingAgentWorkflow",
            kind=SpanKind.CLIENT,
            attributes={
                "workflow.name": "WritingAgentEditor",
                "content.type": context.label,
                "content.length": len(content),
                "agent.count": len(self.agent_names),
            },
        ) as workflow_span:
            trace_id = format_trace_id(workflow_span.get_span_context().trace_id)
            logger.info(f"🔍 Trace ID: {trace_id}")
            
            eval_tracker.start_workflow()
            print(f"\n📝 Analyzing {context.label} content… (Trace: {trace_id[:8]}...)")

            # Phase 1 — parallel independent analysis using ConcurrentBuilder
            with trace_phase("Independent Analysis", 1, context.label) as phase1_span:
                eval_tracker.start_phase(1)
                print("  Phase 1: Independent agent analysis…")
                phase1_results = await self._run_parallel_analysis(context, eval_tracker)
                for r in phase1_results:
                    print(f"    ✓ {r.summary_line()}")
                phase1_span.set_attribute("phase.result_count", len(phase1_results))
                phase1_span.set_attribute(
                    "phase.total_suggestions",
                    sum(len(r.suggestions) for r in phase1_results),
                )
                eval_tracker.end_phase(1)

            # Phase 2 — team synthesis (each agent refines with awareness of others)
            with trace_phase("Team Synthesis", 2, context.label) as phase2_span:
                eval_tracker.start_phase(2)
                print("  Phase 2: Team synthesis…")
                phase2_results = await self._run_team_synthesis(phase1_results, context, eval_tracker)
                for r in phase2_results:
                    print(f"    ✓ {r.summary_line()} (refined)")
                phase2_span.set_attribute("phase.result_count", len(phase2_results))
                phase2_span.set_attribute(
                    "phase.total_suggestions",
                    sum(len(r.suggestions) for r in phase2_results),
                )
                eval_tracker.end_phase(2)

            # Phase 3 — unified summary + annotations
            with trace_phase("Unified Summary", 3, context.label) as phase3_span:
                eval_tracker.start_phase(3)
                print("  Phase 3: Generating unified recommendations…")
                unified = await self._generate_unified_summary(phase2_results, context)
                print("  Phase 3b: Generating editing annotations…")
                annotations = await self._generate_annotations(phase2_results, context)
                phase3_span.set_attribute("summary.length", len(unified))
                phase3_span.set_attribute("annotations.corrections", len(annotations.corrections))
                phase3_span.set_attribute("annotations.comments", len(annotations.comments))
                eval_tracker.end_phase(3)

            # Finalize evaluation
            eval_tracker.end_workflow()
            eval_tracker.calculate_refinement_ratio()
            
            # Add evaluation metrics to workflow span
            workflow_span.set_attributes(eval_tracker.to_span_attributes())
            
            # Log evaluation summary
            eval_tracker.log_summary()

            return TeamSynthesis(
                individual_results=phase1_results,
                refined_results=phase2_results,
                unified_summary=unified,
                content_type=context.label,
                annotations=annotations,
                original_content=content,
                trace_id=trace_id,
                evaluation_metrics=eval_tracker.metrics,
            )

    async def _run_parallel_analysis(
        self, context: EditingContext, eval_tracker: EvaluationTracker
    ) -> list[AgentResult]:
        """Run parallel analysis using ConcurrentBuilder fan-out/fan-in pattern."""
        # Create agents for Phase 1 analysis
        agents: list[Agent] = []
        for name, config in AGENT_CONFIGS.items():
            instructions = config["get_instructions"](context)
            agent = create_agent(
                self.client,
                name=name,
                role_description=config["role_description"],
                content_type_instructions=instructions,
                context=context,
            )
            agents.append(agent)

        # Build concurrent workflow
        workflow = ConcurrentBuilder(participants=agents).build()

        # Run the workflow with the content as input
        user_prompt = f"Analyze the following {context.label} content:\n\n{context.content}"
        
        results: list[AgentResult] = []
        async for event in workflow.run(user_prompt, stream=True):
            if event.type == "output":
                # Process aggregated outputs
                output_data = event.data
                if isinstance(output_data, list):
                    for i, msg in enumerate(output_data):
                        if hasattr(msg, 'text'):
                            agent_name = self.agent_names[i] if i < len(self.agent_names) else f"Agent {i}"
                            result = parse_agent_response(agent_name, msg.text or "")
                            results.append(result)
                            # Record metrics
                            eval_tracker.record_agent_result(
                                agent_name=agent_name,
                                phase=1,
                                suggestion_count=len(result.suggestions),
                                findings_length=len(result.findings),
                                cross_reference_count=len(result.cross_references),
                            )
                        elif isinstance(msg, Message):
                            agent_name = self.agent_names[i] if i < len(self.agent_names) else f"Agent {i}"
                            result = parse_agent_response(agent_name, msg.text or "")
                            results.append(result)
                            # Record metrics
                            eval_tracker.record_agent_result(
                                agent_name=agent_name,
                                phase=1,
                                suggestion_count=len(result.suggestions),
                                findings_length=len(result.findings),
                                cross_reference_count=len(result.cross_references),
                            )

        # If we didn't get results from the workflow output, run agents individually
        if not results:
            results = await self._run_agents_individually(agents, context, eval_tracker)

        return results

    async def _run_agents_individually(
        self, agents: list[Agent], context: EditingContext, eval_tracker: EvaluationTracker
    ) -> list[AgentResult]:
        """Fallback: run agents individually if concurrent workflow doesn't return expected output."""
        user_prompt = f"Analyze the following {context.label} content:\n\n{context.content}"
        results: list[AgentResult] = []
        
        async def run_single_agent(agent: Agent, name: str) -> AgentResult:
            start_time = perf_counter()
            with trace_agent(name, "analyze") as span:
                response = await agent.run(user_prompt)
                result = parse_agent_response(name, response.text or "")
                
                # Record metrics
                duration = perf_counter() - start_time
                eval_tracker.record_agent_result(
                    agent_name=name,
                    phase=1,
                    suggestion_count=len(result.suggestions),
                    findings_length=len(result.findings),
                    cross_reference_count=len(result.cross_references),
                    duration_seconds=duration,
                )
                span.set_attribute("agent.suggestion_count", len(result.suggestions))
                
                return result
        
        tasks = [
            run_single_agent(agent, name)
            for agent, name in zip(agents, self.agent_names)
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _run_team_synthesis(
        self,
        phase1_results: list[AgentResult],
        context: EditingContext,
        eval_tracker: EvaluationTracker,
    ) -> list[AgentResult]:
        """Run team synthesis phase where agents refine their analysis."""
        # Format teammates' findings for context
        all_findings = "\n\n".join(
            f"### {r.agent_name}\n{r.findings}\nSuggestions: {r.suggestions}"
            for r in phase1_results
        )

        # Create refinement agents
        agents: list[Agent] = []
        for name, config in AGENT_CONFIGS.items():
            agent = create_refinement_agent(
                self.client,
                name=name,
                role_description=config["role_description"],
                context=context,
            )
            agents.append(agent)

        # Build concurrent workflow for refinement
        workflow = ConcurrentBuilder(participants=agents).build()

        # Build refinement prompt for each agent
        results: list[AgentResult] = []
        
        async def run_refinement(agent: Agent, name: str, own_result: AgentResult) -> AgentResult:
            start_time = perf_counter()
            with trace_agent(name, "refine") as span:
                teammates = "\n\n".join(
                    f"### {r.agent_name}\n{r.findings}\nSuggestions: {r.suggestions}"
                    for r in phase1_results
                    if r.agent_name != name
                )
                prompt = (
                    f"Original content ({context.label}):\n{context.content}\n\n"
                    f"--- YOUR INITIAL ANALYSIS ---\n{own_result.findings}\n"
                    f"Your suggestions: {own_result.suggestions}\n\n"
                    f"--- TEAMMATES' FINDINGS ---\n{teammates}\n\n"
                    "Refine your analysis considering the above. Produce updated JSON."
                )
                response = await agent.run(prompt)
                result = parse_agent_response(name, response.text or "")
                
                # Record metrics for Phase 2
                duration = perf_counter() - start_time
                eval_tracker.record_agent_result(
                    agent_name=name,
                    phase=2,
                    suggestion_count=len(result.suggestions),
                    findings_length=len(result.findings),
                    cross_reference_count=len(result.cross_references),
                    duration_seconds=duration,
                )
                span.set_attribute("agent.suggestion_count", len(result.suggestions))
                span.set_attribute("agent.initial_suggestions", len(own_result.suggestions))
                
                return result

        # Map phase1 results to agent names
        phase1_by_name = {r.agent_name: r for r in phase1_results}
        
        tasks = [
            run_refinement(agent, name, phase1_by_name.get(name, phase1_results[i]))
            for i, (agent, name) in enumerate(zip(agents, self.agent_names))
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _generate_unified_summary(
        self,
        refined_results: list[AgentResult],
        context: EditingContext,
    ) -> str:
        """Generate unified summary using a lead editor agent."""
        with trace_agent("Lead Editor", "synthesize") as span:
            all_findings = "\n\n".join(
                f"### {r.agent_name}\n{r.findings}\nSuggestions: {r.suggestions}"
                for r in refined_results
            )
            
            lead_editor = self.client.as_agent(
                name="Lead Editor",
                instructions=(
                    "You are the lead editor synthesising your team's feedback into "
                    "a clear, actionable set of recommendations. Organise by priority "
                    "(critical → nice-to-have). Highlight cross-agent agreements and "
                    "note any tensions. Be concise and practical."
                ),
            )

            prompt = (
                f"Content type: {context.label}\n\n"
                f"Original content:\n{context.content}\n\n"
                f"--- TEAM FINDINGS ---\n{all_findings}\n\n"
                "Produce a unified, prioritised list of recommendations."
            )

            response = await lead_editor.run(prompt)
            summary = response.text or ""
            span.set_attribute("summary.length", len(summary))
            span.set_attribute("summary.agent_count", len(refined_results))
            
            return summary

    async def _generate_annotations(
        self,
        refined_results: list[AgentResult],
        context: EditingContext,
    ) -> AnnotatedContent:
        """Generate structured corrections and comments for the editing view."""
        with trace_agent("Annotation Generator", "annotate") as span:
            all_findings = "\n\n".join(
                f"### {r.agent_name}\n{r.findings}\nSuggestions: {r.suggestions}"
                for r in refined_results
            )

            annotator = self.client.as_agent(
                name="Annotation Generator",
                instructions=(
                    "You are an editing assistant. Given the original content and "
                    "team feedback, produce a JSON object with two arrays:\n\n"
                    '1. "corrections": inline text fixes. Each object has:\n'
                    '   - "original": the exact substring from the content to correct\n'
                    '   - "replacement": the corrected text\n'
                    '   - "agent": which agent suggested this (e.g. "Grammar Agent")\n\n'
                    '2. "comments": margin comments / feedback. Each object has:\n'
                    '   - "anchor_text": a short exact quote from the content this comment refers to\n'
                    '   - "comment": the feedback or suggestion\n'
                    '   - "agent": which agent this is from\n'
                    '   - "category": one of "grammar", "fact-check", "style", "audience", "research"\n\n'
                    "IMPORTANT:\n"
                    "- The 'original' and 'anchor_text' values MUST be exact substrings of the content.\n"
                    "- Only include corrections where there is a concrete text change.\n"
                    "- Use comments for general feedback, suggestions, and observations.\n"
                    "- Respond ONLY with the JSON object, no other text."
                ),
            )

            prompt = (
                f"Content type: {context.label}\n\n"
                f"=== ORIGINAL CONTENT ===\n{context.content}\n\n"
                f"=== TEAM FINDINGS ===\n{all_findings}\n\n"
                "Produce the JSON with corrections and comments."
            )

            response = await annotator.run(prompt)
            raw = response.text or "{}"
            span.set_attribute("annotation.raw_length", len(raw))

            # Parse the JSON response
            try:
                # Strip markdown fences if the model wraps its response
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                    cleaned = cleaned.rsplit("```", 1)[0]
                data: dict = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("Failed to parse annotation JSON, returning empty annotations")
                return AnnotatedContent()

            corrections = [
                Correction(
                    original=c.get("original", ""),
                    replacement=c.get("replacement", ""),
                    agent=c.get("agent", ""),
                )
                for c in data.get("corrections", [])
                if c.get("original") and c.get("original") in context.content
            ]
            comments = [
                Comment(
                    anchor_text=c.get("anchor_text", ""),
                    comment=c.get("comment", ""),
                    agent=c.get("agent", ""),
                    category=c.get("category", ""),
                )
                for c in data.get("comments", [])
            ]

            span.set_attribute("annotation.corrections_count", len(corrections))
            span.set_attribute("annotation.comments_count", len(comments))

            return AnnotatedContent(corrections=corrections, comments=comments)
