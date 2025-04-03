"""
Phoenix OpenTelemetry Integration

This module configures Phoenix tracing with OpenTelemetry to automatically
instrument LLM calls and provide observability for the Empress Naturals chatbot.
"""

import os
import logging
from functools import wraps
from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager

# Configure Phoenix logging
logger = logging.getLogger(__name__)

# Initialize tracer provider
try:
    from phoenix.otel import register
    from opentelemetry.trace import get_current_span, set_span_in_context, context
    
    # Check if Phoenix is enabled (default to True)
    PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "true").lower() in ("true", "1", "yes")
    
    # Project name for Phoenix
    PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "empress_naturals_chatbot")
    
    # Automatic instrumentation flag
    AUTO_INSTRUMENT = os.getenv("PHOENIX_AUTO_INSTRUMENT", "true").lower() in ("true", "1", "yes")
    
    # Configure the Phoenix tracer if enabled
    if PHOENIX_ENABLED:
        try:
            # Register Phoenix with OpenTelemetry
            tracer_provider = register(
                project_name=PROJECT_NAME,
                auto_instrument=AUTO_INSTRUMENT
            )
            
            # Get a tracer for this module
            tracer = tracer_provider.get_tracer(__name__)
            logger.info(f"Phoenix OpenTelemetry tracing enabled for project: {PROJECT_NAME}")
            logger.info(f"Automatic instrumentation: {AUTO_INSTRUMENT}")
            
            # Check if collector endpoint is configured
            collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
            if collector_endpoint:
                logger.info(f"Phoenix collector endpoint: {collector_endpoint}")
            else:
                logger.warning("No PHOENIX_COLLECTOR_ENDPOINT specified, using default local file output")
            
            PHOENIX_INITIALIZED = True
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix OpenTelemetry: {str(e)}")
            PHOENIX_INITIALIZED = False
    else:
        logger.info("Phoenix tracing is disabled")
        PHOENIX_INITIALIZED = False
        
except ImportError:
    logger.warning("Phoenix OpenTelemetry packages not installed. Tracing will be disabled.")
    PHOENIX_INITIALIZED = False

def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace a function using Phoenix OpenTelemetry.
    
    Args:
        name: Optional name for the span. Defaults to function name.
        attributes: Optional attributes to add to the span.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PHOENIX_INITIALIZED:
                return func(*args, **kwargs)
            
            span_name = name or func.__name__
            span_attributes = attributes or {}
            
            # Start a new span
            with tracer.start_as_current_span(span_name, attributes=span_attributes) as span:
                # Add function arguments as span attributes if they're serializable
                try:
                    for i, arg in enumerate(args):
                        if isinstance(arg, (str, int, float, bool)):
                            span.set_attribute(f"arg_{i}", str(arg))
                    
                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"kwarg_{key}", str(value))
                except Exception as e:
                    logger.debug(f"Error adding arguments to span: {str(e)}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Add result to span attributes if serializable
                try:
                    if isinstance(result, (str, int, float, bool)):
                        span.set_attribute("result", str(result))
                except Exception as e:
                    logger.debug(f"Error adding result to span: {str(e)}")
                
                return result
        
        return wrapper
    
    return decorator

@contextmanager
def trace_context(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager to create a trace span for a block of code.
    
    Args:
        name: Name for the span
        attributes: Optional attributes to add to the span
    """
    if not PHOENIX_INITIALIZED:
        yield
        return
    
    span_attributes = attributes or {}
    
    with tracer.start_as_current_span(name, attributes=span_attributes) as span:
        yield span

def trace_llm_call(prompt: str, model: str, response: str = None, 
                   latency: float = None, metadata: Dict[str, Any] = None,
                   token_metrics: Dict[str, int] = None):
    """
    Manually add LLM call information to the current span.
    
    Args:
        prompt: The input prompt
        model: The model name
        response: The LLM response (optional)
        latency: The call latency in seconds (optional)
        metadata: Additional metadata (optional)
        token_metrics: Token usage metrics (optional)
    """
    if not PHOENIX_INITIALIZED:
        return
    
    try:
        current_span = get_current_span()
        
        # Add LLM-specific attributes
        current_span.set_attribute("llm.prompt", prompt)
        current_span.set_attribute("llm.model", model)
        
        if response:
            current_span.set_attribute("llm.response", response)
        
        if latency:
            current_span.set_attribute("llm.latency", latency)
        
        # Add token metrics if available
        if token_metrics:
            for key, value in token_metrics.items():
                current_span.set_attribute(f"llm.{key}", value)
        
        # Add any additional metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    current_span.set_attribute(f"metadata.{key}", value)
                else:
                    current_span.set_attribute(f"metadata.{key}", str(value))
    
    except Exception as e:
        logger.error(f"Error adding LLM information to span: {str(e)}") 