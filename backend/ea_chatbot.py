import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from supabase.client import Client, create_client
from langchain_groq import ChatGroq
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pytz
import uuid
import requests
import time

# Phoenix OTEL integration
try:
    from phoenix_otel import trace_function, trace_context, trace_llm_call, PHOENIX_INITIALIZED
except ImportError:
    PHOENIX_INITIALIZED = False
    logging.warning("Phoenix OTEL not available, tracing will be disabled")
    # Create dummy decorators if Phoenix is unavailable
    def trace_function(name=None, attributes=None):
        def decorator(func):
            return func
        return decorator
    
    @contextmanager
    def trace_context(name, attributes=None):
        yield None
    
    def trace_llm_call(**kwargs):
        pass

# Set up logging
def setup_logging():
    """Configure logging for cloud deployment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def get_chat_prompt() -> ChatPromptTemplate:
    """Create a ChatPromptTemplate that includes chat history as messages."""
    system_template = """You are an AI assistant for Empress Naturals founded by Elizabeth Koshy, a company that crafts natural skincare and wellness products. Empress Naturals — a brand dedicated to clean, natural skincare and wellness. Our products address the unique changes women experience during perimenopause and menopause, whether combating dryness, restoring elasticity, or soothing irritation. Your role is to provide helpful, concise, and engaging responses about Empress Naturals' services and skincare topics ONLY FOR RELEVANT QUESTIONS.

Key Guidelines:
- Keep responses brief and conversational (2-3 short paragraphs maximum)
- Use a professional yet friendly tone
- Break down complex concepts into simple explanations
- Focus on skincare and wellness products.
- Return the response in proper HTML Format.

If you're unsure about specific information/response or if you determine that the question is irrelevant to Empress Naturals, follow these steps:
1. Acknowledge the limitation politely if it is a relevant question but you don't know the answer; if it is irrelevant, politely inform the user that you are unable to provide an answer.
2. Direct users to:
   - Website: https://empressnaturals.co/
   - Email: support@empressnaturals.co

Context from knowledge base:
{context}
"""

# Removed Info
# Current Time Information:
# - Current Date: {current_date}
# - Current Time: {current_time}
# - Timezone: {timezone}

    # Use ChatPromptTemplate to include system prompt and chat history
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        # Chat history will be dynamically added here
        # Placeholder for the human message (current question)
        ("human", "{question}")
    ])

class ExpressAnalyticsChatbot:
    def __init__(self, supabase_url: str, supabase_key: str, mistral_api_key: str):
        """Initialize the chatbot with Supabase vector store."""
        logger.info("Initializing Express Analytics Chatbot")
        try:
            # Initialize embeddings
            if mistral_api_key:
                self.embeddings = MistralAIEmbeddings(api_key=mistral_api_key)
            else:
                self.embeddings = MistralAIEmbeddings(api_key=os.getenv("MISTRALAI_API_KEY"))
            logger.info("Successfully initialized embeddings")
            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Embedding dimension: {len(test_embedding)}")
            
            # Initialize Supabase client
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("Successfully initialized Supabase client")
            
            # Initialize Supabase vector store
            self.vector_store = SupabaseVectorStore(
                embedding=self.embeddings,
                client=self.supabase,
                table_name="empress_naturals_documents",
                query_name="match_empress_naturals_documents"
            )
            logger.info("Successfully initialized Supabase vector store")

            # Initialize conversation memory (though we'll manage history manually)
            self.memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='response',
                input_key='question'
            )
            
            # Initialize LLM Chain
            self.chain = self.setup_llm_chain()
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def setup_llm_chain(self) -> LLMChain:
        """Set up the LLM chain with memory."""
        logger.info("Setting up LLM chain")
        try:
            llm = ChatGroq(
                temperature=0.2,
                groq_api_key="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",
                model_name="llama-3.2-3b-preview",
                callbacks=[StreamingStdOutCallbackHandler()]
            )

            chain = LLMChain(
                llm=llm,
                prompt=get_chat_prompt(),
                memory=self.memory,
                output_key='response'
            )
            
            logger.info("Successfully created LLM chain")
            return chain
            
        except Exception as e:
            logger.error(f"Error setting up LLM chain: {str(e)}")
            raise

    def _prepare_chat_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Retrieve and format chat history from Supabase as a list of (role, message) tuples."""
        try:
            response = self.supabase.table("chat_history") \
                .select("user_query", "bot_response") \
                .eq("session_id", session_id) \
                .order("timestamp", desc=True) \
                .limit(2) \
                .execute()
            
            chat_history = []
            if hasattr(response, 'data') and response.data:
                for entry in reversed(response.data):
                    # Add user message
                    chat_history.append(("human", entry['user_query']))
                    # Add bot response
                    chat_history.append(("ai", entry['bot_response']))
            return chat_history
        except Exception as e:
            logger.error(f"Error retrieving chat history for session {session_id}: {str(e)}")
            return []

    @trace_function(name="get_response")
    def get_response(self, question: str, session_id: str) -> Tuple[str, Dict[str, int]]:
        """Generate response using the LLM chain with chat history and track with Phoenix."""
        start_time = time.time()
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        try:
            # Use Supabase for similarity search
            with trace_context("similarity_search", {"session_id": session_id, "query": question}):
                docs = self.vector_store.similarity_search(question, k=3)
                logger.info(f"Found {len(docs)} relevant documents")
                context = "\n".join([doc.page_content for doc in docs])
                query_embedding = self.embeddings.embed_query(question)
            
            # Retrieve and prepare chat history
            with trace_context("prepare_history", {"session_id": session_id}):
                chat_history = self._prepare_chat_history(session_id)
            
            current_time = datetime.now(pytz.UTC)
            user_timezone = pytz.timezone('UTC')
            local_time = current_time.astimezone(user_timezone)

            # Create the full prompt with chat history
            with trace_context("prepare_prompt", {"session_id": session_id}):
                prompt = get_chat_prompt()
                messages = prompt.messages
                # Insert chat history messages before the final human message
                for role, message in chat_history:
                    messages.insert(-1, (role, message))
                
                # Update the chain's prompt with the new messages
                self.chain.prompt = ChatPromptTemplate.from_messages(messages)
                
                # Record the final prompt for tracing
                final_prompt = f"System: {messages[0][1]}\n"
                for i in range(1, len(messages)):
                    role, message = messages[i]
                    final_prompt += f"{role.capitalize()}: {message}\n"

            # Track token usage
            with trace_context("llm_call", {"session_id": session_id, "model": "llama-3.2-3b-preview"}):
                with get_openai_callback() as cb:
                    response = self.chain.predict(
                        question=question,
                        context=context
                    )
                    
                    # Record token usage for both Prometheus and Phoenix
                    token_usage = {
                        "input_tokens": cb.prompt_tokens,
                        "output_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens
                    }
                    
                    logger.info(f"Token usage for session {session_id}: {cb.total_tokens} total "
                               f"({cb.prompt_tokens} prompt, {cb.completion_tokens} completion)")
                    
                    # Record detailed trace information
                    trace_llm_call(
                        prompt=final_prompt, 
                        model="llama-3.2-3b-preview",
                        response=response,
                        latency=time.time() - start_time,
                        token_metrics=token_usage,
                        metadata={"session_id": session_id}
                    )
                    
                    return response, token_usage
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_message = "I'm having trouble processing your request right now. Please try again in a moment."
            
            # Record error in trace
            if PHOENIX_INITIALIZED:
                with trace_context("error", {"error_type": "llm_call_error", "session_id": session_id}):
                    trace_llm_call(
                        prompt=question,
                        model="llama-3.2-3b-preview",
                        response=error_message,
                        metadata={"error": str(e), "session_id": session_id}
                    )
            
            return error_message, token_usage

    def store_chat(self, session_id: str, user_query: str, bot_response: str):
        """Store chat interaction in Supabase."""
        try:
            # Get location using ip-api.com
            ip_address = get_client_ip()
            location = get_location_from_ip(ip_address)
            
            # Insert into Supabase table
            data = {
                "session_id": session_id,
                "user_query": user_query,
                "bot_response": bot_response,
                "ip_address": ip_address,
                "location": str(location) if location else None
            }
            
            response = self.supabase.table("chat_history").insert(data).execute()
            logger.info(f"Stored chat in Supabase for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store chat in Supabase: {str(e)}")

def get_client_ip():
    """Get the client IP address using ipify."""
    try:
        response = requests.get('https://api.ipify.org')
        return response.text
    except:
        return "127.0.0.1"

def get_location_from_ip(ip_address):
    """Get location information from IP address using ip-api.com."""
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

def main():
    logger.info("Starting application")
    st.set_page_config(page_title="Express Analytics Chatbot", page_icon="🤖")
    st.title("Express Analytics AI Assistant")

    # Load environment variables
    supabase_url = os.getenv("SUPABASE_URL", "https://xjfnuiknkxggygmgqgxg.supabase.co/")
    supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhqZm51aWtua3hnZ3lnbWdxZ3hnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MDEzNzYzMSwiZXhwIjoyMDU1NzEzNjMxfQ.xzVATCSvGiFX8iYe8rMyxKVhLjTeO6ws3drdXxWXDHI")
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "DJQ7OG5FeAPPeG7ut6PNCpMqanV365nj")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = ExpressAnalyticsChatbot(
                supabase_url, 
                supabase_key, 
                mistral_api_key
            )
            logger.info("Chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            st.error(f"Error initializing chatbot: {str(e)}")
            return
    
    # Generate or retrieve session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {st.session_state.session_id}")

    # Initialize chat history if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display welcome message for first-time users
    if not st.session_state.messages:
        welcome_msg = """
        👋 Welcome to Empress Naturals! I'm your AI assistant, ready to help you with:
        - Skincare and wellness product inquiries
        - Information about our natural products for perimenopause and menopause
        
        How can I assist you today?
        """
        st.markdown(welcome_msg)
        
        # Store welcome message in Supabase
        try:
            st.session_state.chatbot.store_chat(
                session_id=st.session_state.session_id,
                user_query="[SESSION_START]",
                bot_response=welcome_msg
            )
        except Exception as e:
            logger.error(f"Failed to store welcome message: {str(e)}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you with your skincare needs?"):
        logger.info(f"Received user prompt: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                try:
                    response, token_usage = st.session_state.chatbot.get_response(
                        prompt,
                        st.session_state.session_id
                    )
                    response_text = response
                    st.write(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.session_state.chatbot.store_chat(
                        session_id=st.session_state.session_id,
                        user_query=prompt,
                        bot_response=response_text
                    )
                except Exception as e:
                    logger.error(f"Error in chat: {str(e)}")
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()