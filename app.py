#!/usr/bin/env python3
"""Flask web UI for the Writing Editor multi-agent assistant with Microsoft Foundry observability."""

from __future__ import annotations

import logging
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

from src.orchestrator import EditorOrchestrator
from src.observability import (
    auto_configure_observability,
    get_current_trace_id,
    get_tracer,
    ObservabilityConfig,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize observability (Azure Monitor / Foundry / OTLP)
logger.info("🔧 Configuring observability...")
observability_configured = auto_configure_observability()
if observability_configured:
    logger.info("✅ Observability configured successfully")
else:
    logger.warning("⚠️ Observability not fully configured - using local/console output")

# Try to add Flask instrumentation if available
try:
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    FLASK_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    FLASK_INSTRUMENTATION_AVAILABLE = False
    logger.info("Flask instrumentation not available - install opentelemetry-instrumentation-flask")

app = Flask(__name__)

# Instrument Flask app if available
if FLASK_INSTRUMENTATION_AVAILABLE:
    FlaskInstrumentor().instrument_app(app)
    logger.info("✅ Flask instrumentation enabled")

# Content type options
CONTENT_TYPES = {
    "article": "Article",
    "social_media": "Social Media Post",
    "video_script": "Video Script",
}


def get_client() -> AzureOpenAIChatClient:
    """Create Azure OpenAI Chat Client."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    
    if api_key:
        return AzureOpenAIChatClient(
            endpoint=endpoint,
            deployment_name=deployment,
            api_key=api_key,
        )
    else:
        return AzureOpenAIChatClient(
            endpoint=endpoint,
            deployment_name=deployment,
            credential=AzureCliCredential(),
        )


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html", content_types=CONTENT_TYPES)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze content using the multi-agent system."""
    tracer = get_tracer("flask-api")
    
    try:
        data = request.get_json()
        content = data.get("content", "").strip()
        content_type = data.get("content_type", "article")
        
        if not content:
            return jsonify({"error": "No content provided"}), 400
        
        if content_type not in CONTENT_TYPES:
            return jsonify({"error": "Invalid content type"}), 400
        
        # Create client and orchestrator
        client = get_client()
        deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
        orchestrator = EditorOrchestrator(client, model=deployment)
        
        # Run the analysis
        synthesis = orchestrator.run(content, content_type)
        
        # Format results for JSON response
        results = {
            "content_type": synthesis.content_type,
            "trace_id": synthesis.trace_id,
            "original_content": content,
            "individual_results": [
                {
                    "agent_name": r.agent_name,
                    "findings": r.findings,
                    "suggestions": r.suggestions,
                    "cross_references": r.cross_references,
                }
                for r in synthesis.individual_results
            ],
            "refined_results": [
                {
                    "agent_name": r.agent_name,
                    "findings": r.findings,
                    "suggestions": r.suggestions,
                    "cross_references": r.cross_references,
                }
                for r in synthesis.refined_results
            ],
            "unified_summary": synthesis.unified_summary,
            "annotations": synthesis.annotations.to_dict(),
        }
        
        # Add evaluation metrics if available
        if synthesis.evaluation_metrics:
            results["evaluation"] = {
                "total_duration_seconds": synthesis.evaluation_metrics.total_duration_seconds,
                "phase1_duration_seconds": synthesis.evaluation_metrics.phase1_duration_seconds,
                "phase2_duration_seconds": synthesis.evaluation_metrics.phase2_duration_seconds,
                "phase3_duration_seconds": synthesis.evaluation_metrics.phase3_duration_seconds,
                "total_suggestions": synthesis.evaluation_metrics.total_suggestions,
                "refinement_ratio": synthesis.evaluation_metrics.refinement_improvement_ratio,
            }
        
        logger.info(f"Analysis completed - Trace ID: {synthesis.trace_id}")
        return jsonify(results)
    
    except Exception as e:
        logger.exception("Error during analysis")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint with observability status."""
    config = ObservabilityConfig.from_env()
    return jsonify({
        "status": "ok",
        "observability": {
            "configured": observability_configured,
            "foundry_endpoint": bool(config.foundry_project_endpoint),
            "app_insights": bool(config.application_insights_connection_string),
            "otlp_endpoint": bool(config.otlp_endpoint),
        },
        "flask_instrumented": FLASK_INSTRUMENTATION_AVAILABLE,
    })


@app.route("/traces")
def traces():
    """Show information about how to view traces."""
    config = ObservabilityConfig.from_env()
    
    info = {
        "message": "Traces are sent to the configured telemetry backend.",
        "backends": [],
    }
    
    if config.foundry_project_endpoint:
        info["backends"].append({
            "type": "Microsoft Foundry",
            "endpoint": config.foundry_project_endpoint,
            "view_traces": "Azure Portal > Microsoft Foundry > Operate > Tracing",
        })
    
    if config.application_insights_connection_string:
        info["backends"].append({
            "type": "Application Insights",
            "view_traces": "Azure Portal > Application Insights > Transaction search",
        })
    
    if config.otlp_endpoint:
        info["backends"].append({
            "type": "OTLP Endpoint",
            "endpoint": config.otlp_endpoint,
            "view_traces": "Aspire Dashboard or configured collector",
        })
    
    if not info["backends"]:
        info["backends"].append({
            "type": "Console",
            "view_traces": "Check terminal/console output for trace information",
        })
    
    return jsonify(info)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(debug=True, host="0.0.0.0", port=port)
