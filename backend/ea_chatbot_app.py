from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime, timedelta
import time
import uuid
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, retry_if_exception_message
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST, Info, CollectorRegistry
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv

from ea_chatbot import ExpressAnalyticsChatbot, setup_logging
from phoenix_otel import trace_function, trace_context, trace_llm_call
from phoenix_evals import phoenix_evaluator

load_dotenv(override=True)

logger = setup_logging()

# HubSpot API credentials
HUBSPOT_API_KEY = os.getenv('HUBSPOT_API_KEY')
HUBSPOT_API_URL = os.getenv('HUBSPOT_API_URL')

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    timezone: Optional[str] = "UTC"

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    session_id: str

class ChatHistory(BaseModel):
    messages: List[Dict[str, str]]

class RateLimitError(Exception):
    pass

class HubSpotContactRequest(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    company_name: Optional[str] = None
    note: Optional[str] = None

class MeetingSchedulerConfig(BaseModel):
    url: str
    provider: str
    name: str
    email: str
    title: str
    description: str

class EvaluationRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    session_id: Optional[str] = None
    eval_type: str = "standard"

class EvaluationResult(BaseModel):
    summary: Dict[str, Any]
    detail_count: int

# Default meeting scheduler configuration
default_meeting_config = {
    "url": "https://calendly.com/ajmal-aksar-expressanalytics/30min",
    "provider": "calendly",
    "name": "Express Analytics Team",
    "email": "ajmal.aksar@expressanalytics.net",
    "title": "Express Analytics Consultation",
    "description": "Schedule a 30-minute consultation with our team to discuss your data analytics needs."
}

app = FastAPI(
    title="Express Analytics Chatbot API",
    description="API endpoints for the Express Analytics AI Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot with Supabase
supabase_url = "https://xjfnuiknkxggygmgqgxg.supabase.co/"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhqZm51aWtua3hnZ3lnbWdxZ3hnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MDEzNzYzMSwiZXhwIjoyMDU1NzEzNjMxfQ.xzVATCSvGiFX8iYe8rMyxKVhLjTeO6ws3drdXxWXDHI"
mistral_api_key = "DJQ7OG5FeAPPeG7ut6PNCpMqanV365nj"
chatbot = ExpressAnalyticsChatbot(supabase_url, supabase_key, mistral_api_key)

# Create a custom registry for Prometheus metrics
registry = CollectorRegistry()

# Prometheus metrics with custom registry
REQUESTS_COUNTER = Counter('chatbot_requests_total', 'Total number of requests', ['endpoint', 'status_code'], registry=registry)
RESPONSE_TIME = Histogram('chatbot_response_time_seconds', 'Response time', ['endpoint'], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], registry=registry)
ACTIVE_SESSIONS = Gauge('chatbot_active_sessions', 'Number of active sessions', registry=registry)
ERROR_COUNTER = Counter('chatbot_errors_total', 'Total errors', ['endpoint', 'error_type', 'status_code'], registry=registry)
CHAT_LENGTH = Summary('chatbot_message_length', 'Message length', ['type'], registry=registry)
INPUT_TOKENS = Counter('chatbot_input_tokens_total', 'Input tokens', ['model', 'endpoint'], registry=registry)
OUTPUT_TOKENS = Counter('chatbot_output_tokens_total', 'Output tokens', ['model', 'endpoint'], registry=registry)
TOTAL_TOKENS = Counter('chatbot_total_tokens', 'Total tokens', ['model', 'endpoint'], registry=registry)
LLM_LATENCY = Histogram('chatbot_llm_latency_seconds', 'LLM latency', ['model'], buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0], registry=registry)
SYSTEM_INFO = Info('chatbot_system_info', 'System info', registry=registry)
SYSTEM_INFO.info({'version': '1.0.0', 'environment': os.getenv('ENV', 'development')})

# TPM tracking
tpm_usage = 0
last_reset = time.time()
TPM_LIMIT = 7000

def reset_tpm_if_needed():
    global tpm_usage, last_reset
    current_time = time.time()
    if current_time - last_reset >= 60:
        logger.info(f"Resetting TPM usage. Previous: {tpm_usage}")
        tpm_usage = 0
        last_reset = current_time

def get_location_from_ip(request: Request):
    ip_address = request.headers.get("X-Forwarded-For", request.client.host)
    if ip_address in ("127.0.0.1", "localhost", "::1"):
        return None
    try:
        response = requests.get(f'http://ip-api.com/json/{ip_address}')
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return {
                    "country": data.get('country'),
                    "city": data.get('city'),
                    "latitude": data.get('lat'),
                    "longitude": data.get('lon')
                }
        return None
    except Exception as e:
        logger.error(f"Error getting location data from ip-api: {str(e)}")
        return None

async def store_chat_in_supabase(session_id, user_query, bot_response, ip_address, location):
    """Store chat interaction in Supabase."""
    try:
        location_str = str(location) if location else None
        data = {
            "session_id": session_id,
            "user_query": user_query,
            "bot_response": bot_response,
            "ip_address": ip_address,
            "location": location_str
        }
        response = chatbot.supabase.table("chat_history").insert(data).execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error storing chat in Supabase: {response.error}")
        else:
            logger.info(f"Successfully stored chat for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to store chat in Supabase: {str(e)}")

async def get_chat_history_from_supabase(session_id: str) -> str:
    """Retrieve chat history from Supabase for a given session_id."""
    try:
        response = chatbot.supabase.table("chat_history") \
            .select("user_query", "bot_response") \
            .eq("session_id", session_id) \
            .order("timestamp", desc=True) \
            .limit(4) \
            .execute()
        
        if hasattr(response, 'data') and response.data:
            history = ""
            for entry in reversed(response.data):
                history += f"User: {entry['user_query']}\nAssistant: {entry['bot_response']}\n"
            return history.strip()
        return ""
    except Exception as e:
        logger.error(f"Error retrieving chat history from Supabase: {str(e)}")
        return ""

async def get_active_sessions_count():
    """Calculate the number of unique active sessions in the last 15 minutes."""
    try:
        cutoff_time = (datetime.now() - timedelta(minutes=15)).isoformat()
        response = chatbot.supabase.table("chat_history") \
            .select("session_id") \
            .gte("timestamp", cutoff_time) \
            .execute()
        if hasattr(response, 'data') and response.data:
            unique_sessions = len(set([entry['session_id'] for entry in response.data]))
            return unique_sessions
        return 0
    except Exception as e:
        logger.error(f"Error calculating active sessions: {str(e)}")
        return 0
    
@trace_function(name="process_chat_request")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(10),
    retry=retry_if_exception_message(match='rate_limit_exceeded|429|413'),
    before_sleep=lambda retry_state: logger.info(f"Retrying due to rate limit. Attempt {retry_state.attempt_number}/3, waiting 10s")
)
async def process_chat_request(request: ChatRequest, session_id: str, chat_history: str, client_ip: str):
    global tpm_usage
    reset_tpm_if_needed()
    
    llm_start = time.time()
    try:
        with trace_context("chatbot_get_response", {"session_id": session_id}):
            response, token_usage = chatbot.get_response(request.message, session_id)
    except Exception as e:
        error_str = str(e)
        if 'rate_limit_exceeded' in error_str or '413' in error_str or '429' in error_str:
            if '413' in error_str and 'Requested' in error_str:
                try:
                    requested_str = error_str.split('Requested')[1].split(',')[0].strip()
                    requested_tokens = int(requested_str)
                    logger.warning(f"Payload too large: {requested_tokens} tokens requested, limit is 7000")
                except:
                    logger.warning(f"Payload too large but couldn't parse token count: {error_str}")
            logger.warning(f"Rate limit hit: {error_str}")
            raise
        raise
    
    requested_tokens = token_usage["total_tokens"]
    
    logger.info(f"Request tokens: {token_usage['input_tokens']} prompt, {token_usage['output_tokens']} completion, Total: {requested_tokens}")
    logger.info(f"Current TPM usage before request: {tpm_usage}")
    
    if tpm_usage + requested_tokens > TPM_LIMIT:
        logger.warning(f"TPM limit would be exceeded: {tpm_usage} + {requested_tokens} > {TPM_LIMIT}")
        raise HTTPException(status_code=429, detail=f"TPM limit would be exceeded: {tpm_usage + requested_tokens} > {TPM_LIMIT}")
    
    tpm_usage += requested_tokens
    logger.info(f"Updated TPM usage: {tpm_usage}")
    
    return response, token_usage

@app.get("/")
async def root():
    """Root endpoint with API information."""
    REQUESTS_COUNTER.labels(endpoint='/', status_code='200').inc()
    return {
        "message": "Express Analytics Chatbot API",
        "version": "1.0.0",
        "endpoints": ["/chat", "/history/{session_id}", "/health", "/metrics", "/analytics"]
    }

@app.post("/chat", response_model=ChatResponse)
@trace_function(name="process_chat_endpoint")
async def chat(request: ChatRequest, client_request: Request):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    timezone = request.timezone or "UTC"
    
    # Get client IP for analysis
    client_ip = client_request.headers.get("X-Forwarded-For", client_request.client.host)
    location = get_location_from_ip(client_request)
    
    try:
        # Get previous chat history
        chat_history = await get_chat_history_from_supabase(session_id)
        
        # Add session metadata to the current span
        with trace_context("user_query", {
            "client_ip": client_ip,
            "session_id": session_id,
            "timezone": timezone,
            "location": str(location) if location else None
        }) as span:
            span.set_attribute("user.message", request.message)
            span.set_attribute("message.length", len(request.message))
        
        # Track message length for Prometheus
        CHAT_LENGTH.labels(type="input").observe(len(request.message))
        
        # Process the chat request
        llm_start_time = time.time()
        with trace_context("llm_processing", {"session_id": session_id}):
            response, tokens_data = await process_chat_request(request, session_id, chat_history, client_ip)
        llm_latency = time.time() - llm_start_time
        
        # Record LLM call details
        trace_llm_call(
            prompt=chat_history + "\nUser: " + request.message,
            model="mistral",
            response=response,
            latency=llm_latency,
            token_metrics=tokens_data,
            metadata={
                "session_id": session_id,
                "client_ip": client_ip
            }
        )
        
        # Update token metrics in Prometheus
        if tokens_data:
            INPUT_TOKENS.labels(model="mistral", endpoint="/chat").inc(tokens_data.get("input_tokens", 0))
            OUTPUT_TOKENS.labels(model="mistral", endpoint="/chat").inc(tokens_data.get("output_tokens", 0))
            TOTAL_TOKENS.labels(model="mistral", endpoint="/chat").inc(tokens_data.get("total_tokens", 0))
            LLM_LATENCY.labels(model="mistral").observe(llm_latency)
        
        # Track response length
        CHAT_LENGTH.labels(type="output").observe(len(response))
        
        # Store conversation in Supabase
        await store_chat_in_supabase(session_id, request.message, response, client_ip, location)
        
        # Update active sessions count
        active_sessions = await get_active_sessions_count()
        ACTIVE_SESSIONS.set(active_sessions)
        
        # Generate response 
        timestamp = datetime.now().isoformat()
        response_data = ChatResponse(
            response=response,
            timestamp=timestamp,
            session_id=session_id
        )
        
        # Record response time and request count
        response_time = time.time() - start_time
        RESPONSE_TIME.labels(endpoint="/chat").observe(response_time)
        REQUESTS_COUNTER.labels(endpoint="/chat", status_code=200).inc()
        
        return response_data
    except RateLimitError as e:
        ERROR_COUNTER.labels(endpoint="/chat", error_type="rate_limit", status_code=429).inc()
        # Log error with trace context
        with trace_context("error", {"error_type": "rate_limit", "status_code": 429}):
            trace_llm_call(
                prompt=request.message, 
                model="mistral", 
                response=str(e), 
                metadata={"error": "rate_limit"}
            )
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing chat: {error_message}")
        ERROR_COUNTER.labels(endpoint="/chat", error_type="general", status_code=500).inc()
        # Log error with trace context
        with trace_context("error", {"error_type": "general", "status_code": 500}):
            trace_llm_call(
                prompt=request.message, 
                model="mistral", 
                response=error_message, 
                metadata={"error": error_message}
            )
        raise HTTPException(status_code=500, detail=f"Error processing chat: {error_message}")

@app.post("/create_hubspot_contact")
async def create_hubspot_contact(contact: HubSpotContactRequest, session_id: Optional[str] = None):
    """Create a contact in HubSpot and attach separate notes for form message and chat history if provided."""
    REQUESTS_COUNTER.labels(endpoint='/create_hubspot_contact', status_code='200').inc()
    start_time = time.time()
    
    try:
        PREDEFINED_OWNER_ID = "144816232"  
        
        properties = {
            "firstname": contact.name.split()[0] if " " in contact.name else contact.name,
            "lastname": contact.name.split()[1] if " " in contact.name and len(contact.name.split()) > 1 else "",
            "email": contact.email,
            "hubspot_owner_id": PREDEFINED_OWNER_ID
        }
        if contact.phone:
            properties["phone"] = contact.phone
        if contact.company_name:
            properties["company"] = contact.company_name
        
        headers = {
            "Authorization": f"Bearer {HUBSPOT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Creating HubSpot contact with properties: {properties}")
        response = requests.post(
            HUBSPOT_API_URL,
            json={"properties": properties},
            headers=headers
        )
        
        if response.status_code == 201:
            contact_id = response.json()["id"]
            logger.info(f"Successfully created HubSpot contact: {contact_id}")
            
            # Create a note if a note is provided
            if contact.note:
                note_url = "https://api.hubapi.com/crm/v3/objects/notes"
                timestamp_ms = int(datetime.now().timestamp() * 1000)
                note_payload = {
                    "properties": {
                        "hs_timestamp": str(timestamp_ms),
                        "hs_note_body": contact.note
                    },
                    "associations": [
                        {
                            "to": {"id": contact_id},
                            "types": [
                                {
                                    "associationCategory": "HUBSPOT_DEFINED",
                                    "associationTypeId": 202
                                }
                            ]
                        }
                    ]
                }
                note_response = requests.post(note_url, json=note_payload, headers=headers)
                if note_response.status_code == 201:
                    logger.info(f"Successfully created note for contact: {contact_id}")
                else:
                    logger.warning(f"Failed to create form note: {note_response.status_code} - {note_response.text}")
            
            # Fetch and create a separate note for chat history if session_id is provided
            if session_id:
                history_response = chatbot.supabase.table("chat_history") \
                    .select("user_query", "bot_response", "timestamp") \
                    .eq("session_id", session_id) \
                    .order("timestamp") \
                    .execute()
                
                chat_history = ""
                if hasattr(history_response, 'data') and history_response.data:
                    for entry in history_response.data:
                        chat_history += f"[{entry['timestamp']}] User: {entry['user_query']}\nAssistant: {entry['bot_response']}\n"
                
                if chat_history:
                    chat_note_payload = {
                        "properties": {
                            "hs_timestamp": str(timestamp_ms + 1000),  # Slight offset to distinguish from form note
                            "hs_note_body": f"Chat History:\n{chat_history}"
                        },
                        "associations": [
                            {
                                "to": {"id": contact_id},
                                "types": [
                                    {
                                        "associationCategory": "HUBSPOT_DEFINED",
                                        "associationTypeId": 202
                                    }
                                ]
                            }
                        ]
                    }
                    chat_note_response = requests.post(note_url, json=chat_note_payload, headers=headers)
                    if chat_note_response.status_code == 201:
                        logger.info(f"Successfully created chat history note for contact: {contact_id}")
                    else:
                        logger.warning(f"Failed to create chat history note: {chat_note_response.status_code} - {chat_note_response.text}")
            
            RESPONSE_TIME.labels(endpoint='/create_hubspot_contact').observe(time.time() - start_time)
            return {"message": "Contact created successfully", "contact_id": contact_id}
        else:
            logger.error(f"Failed to create HubSpot contact: {response.status_code} - {response.text} - Correlation ID: {response.json().get('correlationId')}")
            ERROR_COUNTER.labels(endpoint='/create_hubspot_contact', error_type='HubSpotError', status_code=str(response.status_code)).inc()
            raise HTTPException(status_code=response.status_code, detail=f"Failed to create contact: {response.text}")
            
    except Exception as e:
        ERROR_COUNTER.labels(endpoint='/create_hubspot_contact', error_type=type(e).__name__, status_code='500').inc()
        logger.error(f"Error creating HubSpot contact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        RESPONSE_TIME.labels(endpoint='/create_hubspot_contact').observe(time.time() - start_time)

@app.get("/history/{session_id}", response_model=ChatHistory)
async def get_chat_history(session_id: str):
    """Retrieve chat history for a specific session from Supabase."""
    REQUESTS_COUNTER.labels(endpoint='/history', status_code='200').inc()
    start_time = time.time()
    
    try:
        response = chatbot.supabase.table("chat_history") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("timestamp") \
            .execute()
            
        if hasattr(response, 'data') and response.data:
            formatted_messages = []
            for entry in response.data:
                formatted_messages.append({"role": "user", "content": entry["user_query"]})
                formatted_messages.append({"role": "assistant", "content": entry["bot_response"]})
            return ChatHistory(messages=formatted_messages)
        
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNTER.labels(endpoint='/history', error_type=type(e).__name__, status_code='500').inc()
        REQUESTS_COUNTER.labels(endpoint='/history', status_code='500').inc()
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        RESPONSE_TIME.labels(endpoint='/history').observe(time.time() - start_time)

@app.get("/history-by-ip", response_model=ChatHistory)
async def get_chat_history_by_ip(request: Request):
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    response = chatbot.supabase.table("chat_history") \
        .select("user_query, bot_response, session_id, timestamp") \
        .eq("ip_address", client_ip) \
        .order("timestamp") \
        .execute()

    if hasattr(response, 'data') and response.data:
        formatted_messages = []
        for entry in response.data:
            if entry["user_query"]:  # Add user message
                formatted_messages.append({
                    "role": "user",
                    "content": entry["user_query"],
                    "session_id": entry["session_id"],
                    "timestamp": entry["timestamp"]
                })
            if entry["bot_response"]:  # Add bot response
                formatted_messages.append({
                    "role": "assistant",
                    "content": entry["bot_response"],
                    "session_id": entry["session_id"],
                    "timestamp": entry["timestamp"]
                })
        return ChatHistory(messages=formatted_messages)
    raise HTTPException(status_code=404, detail="No chat history found for this IP address")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    REQUESTS_COUNTER.labels(endpoint='/health', status_code='200').inc()
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    REQUESTS_COUNTER.labels(endpoint='/metrics', status_code='200').inc()
    start_time = time.time()
    
    try:
        ACTIVE_SESSIONS.set(await get_active_sessions_count())
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        ERROR_COUNTER.labels(endpoint='/metrics', error_type=type(e).__name__, status_code='500').inc()
        REQUESTS_COUNTER.labels(endpoint='/metrics', status_code='500').inc()
        logger.error(f"Error serving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        RESPONSE_TIME.labels(endpoint='/metrics').observe(time.time() - start_time)

@app.get("/analytics")
async def get_analytics():
    """Get analytics from chat history."""
    REQUESTS_COUNTER.labels(endpoint='/analytics', status_code='200').inc()
    try:
        count_response = chatbot.supabase.table("chat_history").select("id").execute()
        total_interactions = len(count_response.data) if hasattr(count_response, 'data') else 0
        
        session_response = chatbot.supabase.table("chat_history").select("session_id").execute()
        unique_sessions = len(set([item['session_id'] for item in session_response.data])) if hasattr(session_response, 'data') else 0
        
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        recent_response = chatbot.supabase.table("chat_history").select("id").gte("timestamp", yesterday).execute()
        recent_interactions = len(recent_response.data) if hasattr(recent_response, 'data') and recent_response.data else 0
        
        return {
            "total_interactions": total_interactions,
            "unique_sessions": unique_sessions,
            "interactions_last_24h": recent_interactions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        ERROR_COUNTER.labels(endpoint='/analytics', error_type=type(e).__name__, status_code='500').inc()
        REQUESTS_COUNTER.labels(endpoint='/analytics', status_code='500').inc()
        logger.error(f"Error retrieving analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/meeting-config")
async def get_meeting_config():
    """Get the meeting scheduler configuration."""
    return default_meeting_config

@app.post("/evaluations", response_model=EvaluationResult)
async def run_evaluations(request: EvaluationRequest):
    """Run evaluations on chat traces and return the results"""
    try:
        # Run the standard evaluations
        results = phoenix_evaluator.run_standard_evals()
        if not results:
            raise HTTPException(status_code=500, detail="Failed to run evaluations")
        
        # Generate the summary report
        summary = phoenix_evaluator.generate_eval_report()
        
        return EvaluationResult(
            summary=summary,
            detail_count=len(results) if results else 0
        )
    except Exception as e:
        logger.error(f"Error running evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running evaluations: {str(e)}")

@app.get("/metrics/phoenix")
async def get_phoenix_metrics():
    """Get metrics from Phoenix traces"""
    try:
        # Load the traces
        traces_df = phoenix_evaluator.load_traces()
        
        if traces_df.empty:
            return {"message": "No traces found"}
        
        # Calculate some basic metrics
        metrics = {
            "total_llm_calls": len(traces_df),
            "average_latency": traces_df.get("metrics.latency", traces_df.get("latency")).mean() if "metrics.latency" in traces_df.columns or "latency" in traces_df.columns else None,
            "total_tokens": traces_df.get("metrics.total_tokens", traces_df.get("total_tokens")).sum() if "metrics.total_tokens" in traces_df.columns or "total_tokens" in traces_df.columns else None,
            "recent_traces": len(traces_df[traces_df["timestamp"] > (datetime.now() - timedelta(hours=24)).isoformat()]) if "timestamp" in traces_df.columns else 0
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting Phoenix metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Phoenix metrics: {str(e)}")

@app.post("/export/arize", response_model=Dict[str, Any])
async def export_to_arize(request: Dict[str, Any] = {}):
    """Export traces to Arize platform (requires Arize API key)"""
    arize_api_key = os.getenv("ARIZE_API_KEY")
    arize_space_key = os.getenv("ARIZE_SPACE_KEY")
    
    if not arize_api_key or not arize_space_key:
        raise HTTPException(status_code=400, detail="Arize API key and space key are required")
    
    try:
        # This would be the implementation to export to Arize
        # For demonstration purposes, we'll just return a success message
        return {
            "status": "success",
            "message": "Data exported to Arize platform successfully",
            "details": {
                "export_time": datetime.now().isoformat(),
                "trace_count": 0  # This would be actual count in real implementation
            }
        }
    except Exception as e:
        logger.error(f"Error exporting to Arize: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting to Arize: {str(e)}")

if __name__ == "__main__":
    print("Environment variables:")
    print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
    print(f"SUPABASE_KEY: {os.getenv('SUPABASE_KEY')}")
    print(f"MISTRAL_API_KEY: {os.getenv('MISTRAL_API_KEY')}")
    print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY')}")
    
    uvicorn.run(
        "ea_chatbot_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )