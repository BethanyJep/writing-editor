"""Observability module for Microsoft Foundry tracing and evaluation.

This module provides:
- OpenTelemetry tracing with Azure Monitor integration
- Custom spans for workflow phases
- Metrics collection for agent performance
- Evaluation tracking for content quality
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Generator, TypeVar

from dotenv import load_dotenv
from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.attributes import service_attributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.span import format_trace_id

load_dotenv()

logger = logging.getLogger(__name__)

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])

# Module-level state
_observability_initialized = False
_tracer: trace.Tracer | None = None
_meter: metrics.Meter | None = None

# Metrics (initialized lazily)
_agent_duration_histogram: metrics.Histogram | None = None
_agent_invocation_counter: metrics.Counter | None = None
_suggestion_counter: metrics.Counter | None = None

import ssl
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


@dataclass
class ObservabilityConfig:
    """Configuration for observability setup."""
    
    service_name: str = "writing-agent-editor"
    service_version: str = "1.0.0"
    enable_sensitive_data: bool = False
    enable_live_metrics: bool = True
    otlp_endpoint: str | None = None
    application_insights_connection_string: str | None = None
    foundry_project_endpoint: str | None = None
    
    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        """Load configuration from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "writing-agent-editor"),
            service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            enable_sensitive_data=os.getenv("OTEL_ENABLE_SENSITIVE_DATA", "false").lower() == "true",
            enable_live_metrics=os.getenv("AZURE_MONITOR_LIVE_METRICS", "true").lower() == "true",
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            application_insights_connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
            foundry_project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        )


def create_resource(config: ObservabilityConfig | None = None) -> Resource:
    """Create an OpenTelemetry resource for the service."""
    if config is None:
        config = ObservabilityConfig.from_env()
    
    return Resource.create({
        service_attributes.SERVICE_NAME: config.service_name,
        service_attributes.SERVICE_VERSION: config.service_version,
        "service.instance.id": os.getenv("HOSTNAME", "local"),
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })


async def configure_foundry_observability(
    config: ObservabilityConfig | None = None,
) -> bool:
    """Configure observability using Microsoft Foundry project connection.
    
    This method connects to Microsoft Foundry and uses the attached Application Insights
    for telemetry collection.
    
    Args:
        config: Optional configuration override. Uses environment variables if not provided.
        
    Returns:
        True if successfully configured, False otherwise.
    """
    global _observability_initialized
    
    if config is None:
        config = ObservabilityConfig.from_env()
    
    if not config.foundry_project_endpoint:
        logger.warning(
            "AZURE_AI_PROJECT_ENDPOINT not set. Foundry observability not configured. "
            "Falling back to local observability."
        )
        return configure_local_observability(config)
    
    try:
        from azure.ai.projects.aio import AIProjectClient
        from azure.identity.aio import AzureCliCredential
        from azure.monitor.opentelemetry import configure_azure_monitor
        
        async with (
            AzureCliCredential() as credential,
            AIProjectClient(
                endpoint=config.foundry_project_endpoint,
                credential=credential,
            ) as project_client,
        ):
            # Get Application Insights connection string from Foundry project
            conn_string = await project_client.telemetry.get_application_insights_connection_string()
            
            configure_azure_monitor(
                connection_string=conn_string,
                enable_live_metrics=config.enable_live_metrics,
                resource=create_resource(config),
                enable_performance_counters=True,
            )
            
            # Enable Agent Framework instrumentation
            from agent_framework.observability import enable_instrumentation
            enable_instrumentation(enable_sensitive_data=config.enable_sensitive_data)
            
            _observability_initialized = True
            logger.info(
                f"✅ Foundry observability configured. "
                f"Project: {config.foundry_project_endpoint}"
            )
            return True
            
    except ImportError as e:
        logger.warning(f"Missing dependencies for Foundry observability: {e}")
        return configure_local_observability(config)
    except Exception as e:
        logger.error(f"Failed to configure Foundry observability: {e}")
        logger.info("Falling back to local observability...")
        return configure_local_observability(config)


def configure_azure_monitor_observability(
    config: ObservabilityConfig | None = None,
) -> bool:
    """Configure observability using Application Insights connection string directly.
    
    Args:
        config: Optional configuration override.
        
    Returns:
        True if successfully configured, False otherwise.
    """
    global _observability_initialized
    
    if config is None:
        config = ObservabilityConfig.from_env()
    
    if not config.application_insights_connection_string:
        logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING not set.")
        return configure_local_observability(config)
    
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        
        configure_azure_monitor(
            connection_string=config.application_insights_connection_string,
            enable_live_metrics=config.enable_live_metrics,
            resource=create_resource(config),
            enable_performance_counters=True,
        )
        
        # Enable Agent Framework instrumentation
        from agent_framework.observability import enable_instrumentation
        enable_instrumentation(enable_sensitive_data=config.enable_sensitive_data)
        
        _observability_initialized = True
        logger.info("✅ Azure Monitor observability configured.")
        return True
        
    except ImportError as e:
        logger.warning(f"Missing azure-monitor-opentelemetry: {e}")
        return configure_local_observability(config)
    except Exception as e:
        logger.error(f"Failed to configure Azure Monitor observability: {e}")
        return configure_local_observability(config)


def configure_otlp_observability(config: ObservabilityConfig | None = None) -> bool:
    """Configure observability using OTLP endpoint (e.g., Aspire Dashboard).
    
    Args:
        config: Optional configuration override.
        
    Returns:
        True if successfully configured, False otherwise.
    """
    global _observability_initialized
    
    if config is None:
        config = ObservabilityConfig.from_env()
    
    if not config.otlp_endpoint:
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT not set.")
        return configure_local_observability(config)
    
    try:
        from agent_framework.observability import configure_otel_providers
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        
        trace_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
        metrics_exporter = OTLPMetricExporter(endpoint=config.otlp_endpoint)
        
        # Use exporters list format
        configure_otel_providers(
            exporters=[trace_exporter, metrics_exporter],
            enable_sensitive_data=config.enable_sensitive_data,
        )
        
        # Enable Agent Framework instrumentation
        from agent_framework.observability import enable_instrumentation
        enable_instrumentation(enable_sensitive_data=config.enable_sensitive_data)
        
        _observability_initialized = True
        logger.info(f"✅ OTLP observability configured. Endpoint: {config.otlp_endpoint}")
        return True
        
    except ImportError as e:
        logger.warning(f"Missing OTLP exporter dependencies: {e}")
        return configure_local_observability(config)
    except Exception as e:
        logger.error(f"Failed to configure OTLP observability: {e}")
        return configure_local_observability(config)


def configure_local_observability(config: ObservabilityConfig | None = None) -> bool:
    """Configure minimal local observability (console output).
    
    Args:
        config: Optional configuration override.
        
    Returns:
        True (always succeeds).
    """
    global _observability_initialized
    
    if config is None:
        config = ObservabilityConfig.from_env()
    
    try:
        from agent_framework.observability import configure_otel_providers, enable_instrumentation
        
        # Configure with no exporters (uses defaults)
        configure_otel_providers(enable_sensitive_data=config.enable_sensitive_data)
        enable_instrumentation(enable_sensitive_data=config.enable_sensitive_data)
        
        _observability_initialized = True
        logger.info("✅ Local observability configured (console output).")
        return True
        
    except Exception as e:
        logger.warning(f"Could not configure local observability: {e}")
        _observability_initialized = False
        return False


def auto_configure_observability() -> bool:
    """Auto-configure observability based on available environment variables.
    
    Priority:
    1. Microsoft Foundry project endpoint (best for Foundry integration)
    2. Application Insights connection string
    3. OTLP endpoint (for Aspire Dashboard or other collectors)
    4. Local/console output (fallback)
    
    Returns:
        True if any configuration succeeded.
    """
    import asyncio
    
    config = ObservabilityConfig.from_env()
    
    # Try Foundry first
    if config.foundry_project_endpoint:
        try:
            return asyncio.run(configure_foundry_observability(config))
        except Exception:
            pass
    
    # Try Application Insights
    if config.application_insights_connection_string:
        if configure_azure_monitor_observability(config):
            return True
    
    # Try OTLP
    if config.otlp_endpoint:
        if configure_otlp_observability(config):
            return True
    
    # Fall back to local
    return configure_local_observability(config)


def get_tracer(name: str = "writing-agent") -> trace.Tracer:
    """Get a tracer for creating spans.
    
    Args:
        name: Name of the instrumentation module.
        
    Returns:
        An OpenTelemetry Tracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(name)
    return _tracer


def get_meter(name: str = "writing-agent") -> metrics.Meter:
    """Get a meter for recording metrics.
    
    Args:
        name: Name of the instrumentation module.
        
    Returns:
        An OpenTelemetry Meter instance.
    """
    global _meter
    if _meter is None:
        _meter = metrics.get_meter(name)
    return _meter


def _ensure_metrics() -> None:
    """Ensure metrics instruments are initialized."""
    global _agent_duration_histogram, _agent_invocation_counter, _suggestion_counter
    
    if _agent_duration_histogram is not None:
        return
        
    meter = get_meter()
    
    _agent_duration_histogram = meter.create_histogram(
        name="agent.duration",
        description="Duration of agent execution in seconds",
        unit="s",
    )
    
    _agent_invocation_counter = meter.create_counter(
        name="agent.invocations",
        description="Number of agent invocations",
        unit="1",
    )
    
    _suggestion_counter = meter.create_counter(
        name="agent.suggestions",
        description="Number of suggestions generated by agents",
        unit="1",
    )


@contextmanager
def trace_phase(
    phase_name: str,
    phase_number: int,
    content_type: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Context manager for tracing workflow phases.
    
    Args:
        phase_name: Name of the phase (e.g., "Independent Analysis")
        phase_number: Phase number (1, 2, or 3)
        content_type: Type of content being analyzed
        attributes: Additional span attributes
        
    Yields:
        The active span for the phase.
    """
    tracer = get_tracer()
    
    span_attributes = {
        "workflow.phase.name": phase_name,
        "workflow.phase.number": phase_number,
        "content.type": content_type,
    }
    if attributes:
        span_attributes.update(attributes)
    
    with tracer.start_as_current_span(
        f"Phase {phase_number}: {phase_name}",
        kind=SpanKind.INTERNAL,
        attributes=span_attributes,
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@contextmanager
def trace_agent(
    agent_name: str,
    operation: str = "analyze",
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Context manager for tracing individual agent operations.
    
    Args:
        agent_name: Name of the agent
        operation: Operation being performed
        attributes: Additional span attributes
        
    Yields:
        The active span for the agent operation.
    """
    tracer = get_tracer()
    _ensure_metrics()
    
    span_attributes = {
        "agent.name": agent_name,
        "agent.operation": operation,
    }
    if attributes:
        span_attributes.update(attributes)
    
    start_time = perf_counter()
    
    with tracer.start_as_current_span(
        f"{agent_name}.{operation}",
        kind=SpanKind.INTERNAL,
        attributes=span_attributes,
    ) as span:
        try:
            if _agent_invocation_counter:
                _agent_invocation_counter.add(1, {"agent.name": agent_name, "operation": operation})
            
            yield span
            
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            duration = perf_counter() - start_time
            span.set_attribute("agent.duration_seconds", duration)
            
            if _agent_duration_histogram:
                _agent_duration_histogram.record(
                    duration,
                    {"agent.name": agent_name, "operation": operation},
                )


def record_suggestions(agent_name: str, suggestion_count: int) -> None:
    """Record the number of suggestions generated by an agent.
    
    Args:
        agent_name: Name of the agent
        suggestion_count: Number of suggestions generated
    """
    _ensure_metrics()
    
    if _suggestion_counter:
        _suggestion_counter.add(
            suggestion_count,
            {"agent.name": agent_name},
        )


def trace_workflow(content_type: str) -> Callable[[F], F]:
    """Decorator for tracing complete workflow execution.
    
    Args:
        content_type: Type of content being processed
        
    Returns:
        Decorated function with tracing.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            
            with tracer.start_as_current_span(
                "WritingAgentWorkflow",
                kind=SpanKind.CLIENT,
                attributes={
                    "workflow.name": "WritingAgentEditor",
                    "content.type": content_type,
                },
            ) as span:
                trace_id = format_trace_id(span.get_span_context().trace_id)
                span.set_attribute("trace.id", trace_id)
                logger.info(f"🔍 Trace ID: {trace_id}")
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper  # type: ignore
    return decorator


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating agent and workflow performance."""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trace_id: str = ""
    content_type: str = ""
    content_length: int = 0
    
    # Timing metrics
    total_duration_seconds: float = 0.0
    phase1_duration_seconds: float = 0.0
    phase2_duration_seconds: float = 0.0
    phase3_duration_seconds: float = 0.0
    
    # Agent metrics
    agent_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Quality metrics
    total_suggestions: int = 0
    total_findings: int = 0
    cross_references_count: int = 0
    
    # Refinement metrics (comparing Phase 1 to Phase 2)
    refinement_improvement_ratio: float = 0.0


class EvaluationTracker:
    """Tracker for collecting evaluation metrics during workflow execution."""
    
    def __init__(self, content: str, content_type: str) -> None:
        self.metrics = EvaluationMetrics(
            content_type=content_type,
            content_length=len(content),
        )
        self._phase_start_times: dict[int, float] = {}
        self._workflow_start_time: float | None = None
        
        # Capture trace ID if available
        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            self.metrics.trace_id = format_trace_id(
                current_span.get_span_context().trace_id
            )
    
    def start_workflow(self) -> None:
        """Mark the start of the workflow."""
        self._workflow_start_time = perf_counter()
    
    def end_workflow(self) -> None:
        """Mark the end of the workflow and calculate total duration."""
        if self._workflow_start_time:
            self.metrics.total_duration_seconds = perf_counter() - self._workflow_start_time
    
    def start_phase(self, phase_number: int) -> None:
        """Mark the start of a workflow phase."""
        self._phase_start_times[phase_number] = perf_counter()
    
    def end_phase(self, phase_number: int) -> None:
        """Mark the end of a workflow phase and record duration."""
        if phase_number in self._phase_start_times:
            duration = perf_counter() - self._phase_start_times[phase_number]
            
            if phase_number == 1:
                self.metrics.phase1_duration_seconds = duration
            elif phase_number == 2:
                self.metrics.phase2_duration_seconds = duration
            elif phase_number == 3:
                self.metrics.phase3_duration_seconds = duration
    
    def record_agent_result(
        self,
        agent_name: str,
        phase: int,
        suggestion_count: int,
        findings_length: int,
        cross_reference_count: int,
        duration_seconds: float | None = None,
    ) -> None:
        """Record metrics for an individual agent result."""
        if agent_name not in self.metrics.agent_metrics:
            self.metrics.agent_metrics[agent_name] = {
                "phase1_suggestions": 0,
                "phase2_suggestions": 0,
                "phase1_duration": 0.0,
                "phase2_duration": 0.0,
            }
        
        agent_data = self.metrics.agent_metrics[agent_name]
        
        if phase == 1:
            agent_data["phase1_suggestions"] = suggestion_count
            agent_data["phase1_findings_length"] = findings_length
            if duration_seconds:
                agent_data["phase1_duration"] = duration_seconds
        elif phase == 2:
            agent_data["phase2_suggestions"] = suggestion_count
            agent_data["phase2_findings_length"] = findings_length
            if duration_seconds:
                agent_data["phase2_duration"] = duration_seconds
        
        # Update totals
        self.metrics.total_suggestions += suggestion_count
        self.metrics.cross_references_count += cross_reference_count
        
        # Record to OpenTelemetry
        record_suggestions(agent_name, suggestion_count)
    
    def calculate_refinement_ratio(self) -> None:
        """Calculate the improvement ratio from Phase 1 to Phase 2."""
        phase1_total = sum(
            m.get("phase1_suggestions", 0)
            for m in self.metrics.agent_metrics.values()
        )
        phase2_total = sum(
            m.get("phase2_suggestions", 0)
            for m in self.metrics.agent_metrics.values()
        )
        
        if phase1_total > 0:
            self.metrics.refinement_improvement_ratio = phase2_total / phase1_total
    
    def to_span_attributes(self) -> dict[str, Any]:
        """Convert metrics to span attributes for tracing."""
        return {
            "eval.content_type": self.metrics.content_type,
            "eval.content_length": self.metrics.content_length,
            "eval.total_duration_seconds": self.metrics.total_duration_seconds,
            "eval.total_suggestions": self.metrics.total_suggestions,
            "eval.cross_references_count": self.metrics.cross_references_count,
            "eval.refinement_ratio": self.metrics.refinement_improvement_ratio,
            "eval.agent_count": len(self.metrics.agent_metrics),
        }
    
    def log_summary(self) -> None:
        """Log a summary of the evaluation metrics."""
        logger.info("=" * 50)
        logger.info("📊 Evaluation Summary")
        logger.info("=" * 50)
        logger.info(f"Trace ID: {self.metrics.trace_id}")
        logger.info(f"Content Type: {self.metrics.content_type}")
        logger.info(f"Content Length: {self.metrics.content_length} chars")
        logger.info(f"Total Duration: {self.metrics.total_duration_seconds:.2f}s")
        logger.info(f"  Phase 1: {self.metrics.phase1_duration_seconds:.2f}s")
        logger.info(f"  Phase 2: {self.metrics.phase2_duration_seconds:.2f}s")
        logger.info(f"  Phase 3: {self.metrics.phase3_duration_seconds:.2f}s")
        logger.info(f"Total Suggestions: {self.metrics.total_suggestions}")
        logger.info(f"Refinement Ratio: {self.metrics.refinement_improvement_ratio:.2f}x")
        logger.info("=" * 50)


def get_current_trace_id() -> str | None:
    """Get the current trace ID if available.
    
    Returns:
        Formatted trace ID string or None if no active trace.
    """
    current_span = trace.get_current_span()
    if current_span and current_span.get_span_context().is_valid:
        return format_trace_id(current_span.get_span_context().trace_id)
    return None
