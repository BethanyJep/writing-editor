#!/usr/bin/env python3
"""Interactive CLI for the Writing Editor multi-agent assistant using Microsoft Agent Framework."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

from src.orchestrator import EditorOrchestrator

# Load environment variables from .env file
load_dotenv()

CONTENT_TYPES = {
    "1": "article",
    "2": "social_media",
    "3": "video_script",
}


def get_content() -> str:
    """Read multi-line content from stdin."""
    print("\n📋 Paste your content below (press Enter twice on an empty line to finish):\n")
    lines: list[str] = []
    empty_count = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append(line)
        else:
            empty_count = 0
            lines.append(line)
    return "\n".join(lines).strip()


def get_content_type() -> str:
    """Prompt user to select a content type."""
    print("\n🔖 Select content type:")
    print("  1. Article")
    print("  2. Social Media Post")
    print("  3. Video Script")
    choice = input("\nEnter choice (1/2/3): ").strip()
    content_type = CONTENT_TYPES.get(choice)
    if not content_type:
        print("Invalid choice — defaulting to 'article'.")
        content_type = "article"
    return content_type


def main() -> None:
    # Get configuration from environment variables
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    
    if not endpoint:
        print("❌ Please set AZURE_OPENAI_ENDPOINT environment variable.")
        print("   You can also create a .env file with the following variables:")
        print("   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini")
        print("   AZURE_OPENAI_API_KEY=your-api-key (optional if using Azure CLI auth)")
        sys.exit(1)

    # Create the Azure OpenAI Chat Client using Microsoft Agent Framework
    # Use API key if provided, otherwise fall back to Azure CLI credential
    if api_key:
        client = AzureOpenAIChatClient(
            endpoint=endpoint,
            deployment_name=deployment,
            api_key=api_key,
        )
    else:
        client = AzureOpenAIChatClient(
            endpoint=endpoint,
            deployment_name=deployment,
            credential=AzureCliCredential(),
        )

    orchestrator = EditorOrchestrator(client, model=deployment)

    print("=" * 60)
    print("  ✍️  Writing Editor — Multi-Agent AI Assistant")
    print("  📦 Powered by Microsoft Agent Framework")
    print("=" * 60)

    while True:
        content = get_content()
        if not content:
            print("No content provided. Exiting.")
            break

        content_type = get_content_type()
        synthesis = orchestrator.run(content, content_type)
        print("\n" + synthesis.format_report())

        again = input("\n🔄 Analyze another piece? (y/n): ").strip().lower()
        if again != "y":
            print("👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
