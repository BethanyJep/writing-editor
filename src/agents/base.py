"""Agent factory functions using Microsoft Agent Framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_framework import Agent
    from agent_framework.azure import AzureOpenAIChatClient

from src.models.context import EditingContext


def create_agent(
    client: "AzureOpenAIChatClient",
    name: str,
    role_description: str,
    content_type_instructions: str,
    context: EditingContext,
) -> "Agent":
    """Create an agent using Microsoft Agent Framework.
    
    Args:
        client: Azure OpenAI Chat Client instance.
        name: Name of the agent.
        role_description: Description of the agent's role.
        content_type_instructions: Content-type specific instructions.
        context: The editing context.
    
    Returns:
        A configured Agent instance.
    """
    system_prompt = (
        f"You are the {name} in a collaborative writing-editor team.\n\n"
        f"Role: {role_description}\n\n"
        f"Content type: {context.label}\n"
        f"{content_type_instructions}\n\n"
        "Respond in JSON with keys: findings (string, markdown), "
        "suggestions (list of strings), cross_references (object mapping "
        "agent names to relevance notes — leave empty for now)."
    )
    
    return client.as_agent(
        name=name,
        instructions=system_prompt,
    )


def create_refinement_agent(
    client: "AzureOpenAIChatClient",
    name: str,
    role_description: str,
    context: EditingContext,
) -> "Agent":
    """Create an agent for the refinement phase.
    
    Args:
        client: Azure OpenAI Chat Client instance.
        name: Name of the agent.
        role_description: Description of the agent's role.
        context: The editing context.
    
    Returns:
        A configured Agent instance for refinement.
    """
    system_prompt = (
        f"You are the {name} in a collaborative writing-editor team.\n\n"
        f"Role: {role_description}\n\n"
        "You are now in the TEAM SYNTHESIS phase. You have seen your "
        "teammates' findings and should refine your analysis:\n"
        "- Add cross-references where your insights connect to others'\n"
        "- Adjust suggestions that conflict with or complement teammates'\n"
        "- Remove redundant points already covered better by another agent\n\n"
        "Respond in JSON with keys: findings (string, markdown), "
        "suggestions (list of strings), cross_references (object mapping "
        "agent names to relevance notes)."
    )
    
    return client.as_agent(
        name=f"{name} (Refinement)",
        instructions=system_prompt,
    )


# Agent configuration data
AGENT_CONFIGS = {
    "Research Agent": {
        "role_description": (
            "You provide background context, related topics, and supporting "
            "information using your internal knowledge. You help the team "
            "understand the broader landscape around the content being edited."
        ),
        "get_instructions": lambda context: _get_research_instructions(context),
    },
    "Grammar Agent": {
        "role_description": (
            "You are an expert copy-editor focused on spelling, punctuation, "
            "sentence structure, and overall readability. You adapt your "
            "strictness based on content type."
        ),
        "get_instructions": lambda context: _get_grammar_instructions(context),
    },
    "Fact-Check Agent": {
        "role_description": (
            "You identify factual claims in the content, assess their likely "
            "accuracy based on your knowledge, and flag statements that need "
            "external verification. You provide confidence levels for each claim."
        ),
        "get_instructions": lambda context: _get_fact_check_instructions(context),
    },
    "Audience Agent": {
        "role_description": (
            "You evaluate how well the content fits its target audience, "
            "assess engagement potential, and suggest improvements to better "
            "reach and resonate with readers/viewers."
        ),
        "get_instructions": lambda context: _get_audience_instructions(context),
    },
    "Style Agent": {
        "role_description": (
            "You evaluate tone and voice consistency, style guide adherence, "
            "and overall writing style. You ensure the content maintains a "
            "cohesive voice appropriate for its type and purpose."
        ),
        "get_instructions": lambda context: _get_style_instructions(context),
    },
}


def _get_research_instructions(context: EditingContext) -> str:
    rules = context.rules.get("research", {})
    depth = rules.get("depth", "moderate")
    notes = rules.get("notes", "")
    return (
        f"Research depth: {depth}\n"
        f"Guidelines: {notes}\n\n"
        "Your analysis should include:\n"
        "- Key background information relevant to the content\n"
        "- Related topics or angles the author might consider\n"
        "- Supporting facts, statistics, or examples from your knowledge\n"
        "- Gaps where the content could benefit from more context"
    )


def _get_grammar_instructions(context: EditingContext) -> str:
    rules = context.rules.get("grammar", {})
    strictness = rules.get("strictness", "moderate")
    notes = rules.get("notes", "")
    return (
        f"Grammar strictness: {strictness}\n"
        f"Guidelines: {notes}\n\n"
        "Your analysis should include:\n"
        "- Spelling and typographical errors\n"
        "- Punctuation issues\n"
        "- Sentence structure problems (fragments, run-ons, awkward phrasing)\n"
        "- Readability assessment (sentence length variety, paragraph flow)\n"
        "- Specific corrections with before/after examples"
    )


def _get_fact_check_instructions(context: EditingContext) -> str:
    rules = context.rules.get("fact_check", {})
    thoroughness = rules.get("thoroughness", "moderate")
    notes = rules.get("notes", "")
    return (
        f"Fact-check thoroughness: {thoroughness}\n"
        f"Guidelines: {notes}\n\n"
        "Your analysis should include:\n"
        "- List of factual claims found in the content\n"
        "- Confidence level for each claim (high/medium/low/unverifiable)\n"
        "- Corrections for claims you believe are inaccurate\n"
        "- Flags for claims that require external verification\n"
        "- Note: you are using internal knowledge only — clearly mark "
        "  anything you cannot confidently verify"
    )


def _get_audience_instructions(context: EditingContext) -> str:
    rules = context.rules.get("audience", {})
    focus = rules.get("focus", "general")
    notes = rules.get("notes", "")
    return (
        f"Audience focus: {focus}\n"
        f"Guidelines: {notes}\n\n"
        "Your analysis should include:\n"
        "- Assessment of the likely target audience\n"
        "- How well the content matches that audience's expectations\n"
        "- Engagement strengths and weaknesses\n"
        "- Specific recommendations to improve audience fit\n"
        "- Content structure evaluation (hook, flow, conclusion)"
    )


def _get_style_instructions(context: EditingContext) -> str:
    rules = context.rules.get("style", {})
    tone = rules.get("tone", "neutral")
    notes = rules.get("notes", "")
    return (
        f"Expected tone: {tone}\n"
        f"Guidelines: {notes}\n\n"
        "Your analysis should include:\n"
        "- Overall tone assessment and consistency\n"
        "- Voice shifts or inconsistencies\n"
        "- Word choice evaluation (jargon, accessibility, power words)\n"
        "- Stylistic strengths to preserve\n"
        "- Specific rephrasing suggestions with before/after examples"
    )
