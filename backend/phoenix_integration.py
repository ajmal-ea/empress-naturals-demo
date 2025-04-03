from arize.phoenix.trace import trace_context, Trace, Span
from arize.phoenix.config import set_default_outfile
from arize.phoenix.llm_metrics import (
    calculate_response_delay, 
    calculate_token_count, 
    calculate_response_length
)
import os
from datetime import datetime
import json
import logging

# Configure Phoenix logging
logger = logging.getLogger(__name__)

# Set default output file for Phoenix traces
PHOENIX_TRACE_FILE = os.getenv('PHOENIX_TRACE_FILE', 'ea_chatbot_traces.jsonl')
set_default_outfile(PHOENIX_TRACE_FILE)

class PhoenixTracer:
    """
    Handles tracing LLM interactions using Arize Phoenix
    """
    
    def __init__(self):
        self.enabled = True
        logger.info(f"Phoenix tracing initialized with output file: {PHOENIX_TRACE_FILE}")
    
    def start_trace(self, session_id, user_id=None):
        """Start a new trace for a user interaction"""
        trace = Trace(
            name="express_analytics_chatbot",
            metadata={
                "session_id": session_id,
                "user_id": user_id or "anonymous",
                "timestamp": datetime.now().isoformat()
            }
        )
        return trace
    
    def log_user_query(self, trace, query, metadata=None):
        """Log a user query as a span in the trace"""
        with Span(
            name="user_query",
            parent=trace,
            inputs={"query": query},
            metadata=metadata or {}
        ) as span:
            return span
    
    def log_llm_call(self, trace, prompt, response, model, latency=None, metadata=None):
        """Log an LLM call with its prompt, response, and metrics"""
        metrics = {}
        
        # Calculate metrics if not provided
        if latency is not None:
            metrics["latency"] = latency
        
        # Calculate token metrics
        token_metrics = calculate_token_count(prompt, response, model)
        if token_metrics:
            metrics.update(token_metrics)
        
        # Calculate response metrics
        metrics["response_length"] = calculate_response_length(response)
        
        with Span(
            name="llm_call",
            parent=trace,
            inputs={"prompt": prompt},
            outputs={"response": response},
            metadata={
                "model": model,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            },
            metrics=metrics
        ) as span:
            return span
    
    def log_retrieval(self, trace, query, documents, metadata=None):
        """Log retrieval operations from vector database"""
        with Span(
            name="retrieval",
            parent=trace,
            inputs={"query": query},
            outputs={"documents": json.dumps([str(doc) for doc in documents])[:1000]},  # Truncate if too large
            metadata=metadata or {}
        ) as span:
            return span
    
    def end_trace(self, trace):
        """Complete the trace and write to storage"""
        trace.end()
        return trace

# Singleton instance
phoenix_tracer = PhoenixTracer() 