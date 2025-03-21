# Standard Library
import os
import re
import asyncio
import sys
import argparse
import logging
import time
import json
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any

# Third-party Libraries
import aiohttp
from dotenv import load_dotenv
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from supabase import create_client
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

# Custom Modules
from utils import parseDate

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("website_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def abbreviate_company_name(name: str) -> str:
    """
    Create an abbreviation from a company name by taking the first letter of each word.
    
    Args:
        name: Company name
        
    Returns:
        Abbreviation of the company name
    """
    # Remove common legal suffixes
    name = re.sub(r'\b(Inc|LLC|Ltd|Limited|Corp|Corporation|Private|Public|Co)\b', '', name, flags=re.IGNORECASE)
    
    # Split by spaces, hyphens, underscores, and handle camelCase
    words = re.findall(r'[A-Z][a-z]*|\b[a-z]+\b', name)
    
    # Take first letter of each word and convert to uppercase
    abbreviation = ''.join(word[0].upper() for word in words if word)
    
    # If abbreviation is too short (less than 2 chars), use more characters
    if len(abbreviation) < 2 and name:
        abbreviation = name[:3].upper()
        
    return abbreviation

class ProgressTracker:
    """Track and display progress for website scraping"""
    
    def __init__(self):
        """Initialize the progress tracker"""
        self.console = Console()
        self.progress = None
        self.task_id = None
        self.total = 0
        self.scraped = 0
        self.failed = 0
        self.skipped = 0
        self.http_errors = 0  # Track HTTP errors specifically
        self.start_time = None
        self.last_update_time = 0  # Track when we last updated
        
    def initialize(self, total_urls: int):
        """Initialize the progress bar with the total number of URLs to process
        
        Args:
            total_urls: Total number of URLs to process
        """
        self.total = total_urls
        self.scraped = 0
        self.failed = 0
        self.skipped = 0
        self.http_errors = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        self.progress = Progress(
            TextColumn("Scraping websites"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            "•",
            TextColumn("{task.fields[scraped]} scraped"),
            "•", 
            TextColumn("{task.fields[failed]} failed"),
            "•",
            TextColumn("{task.fields[skipped]} skipped"),
            "•",
            TextColumn("{task.fields[http_errors]} HTTP errors"),
            TimeRemainingColumn(),
            refresh_per_second=4  # Increase refresh rate
        )
        
        self.progress.start()
        self.task_id = self.progress.add_task(
            "Scraping", 
            total=total_urls,
            scraped=self.scraped,
            failed=self.failed,
            skipped=self.skipped,
            http_errors=self.http_errors
        )
    
    def update(self, status: str, http_code: int = None):
        """Update the progress bar
        
        Args:
            status: Status of the URL processing ('scraped', 'failed', or 'skipped')
            http_code: HTTP status code (if applicable)
        """
        current_time = time.time()
        # Ensure updates don't happen too frequently (throttle to prevent display issues)
        if current_time - self.last_update_time < 0.1:  # 100ms minimum between updates
            time.sleep(0.1)
        
        if status == "scraped":
            self.scraped += 1
        elif status == "failed":
            self.failed += 1
            # If we have an HTTP error code in the 4xx or 5xx range
            if http_code and (400 <= http_code < 600):
                self.http_errors += 1
        elif status == "skipped":
            self.skipped += 1
            
        # Update task with forced refresh
        try:
            self.progress.update(
                self.task_id, 
                advance=1, 
                scraped=self.scraped,
                failed=self.failed,
                skipped=self.skipped,
                http_errors=self.http_errors,
                refresh=True  # Force refresh to update display immediately
            )
            
            # Explicitly refresh the display
            self.progress.refresh()
        except Exception as e:
            # Handle any issues with progress updates
            print(f"Error updating progress: {e}")
        
        self.last_update_time = time.time()
    
    def finish(self):
        """Complete the progress tracking and show summary statistics"""
        if self.progress:
            try:
                self.progress.stop()
            except Exception as e:
                print(f"Error stopping progress: {e}")
        
        # Calculate statistics
        processed = self.scraped + self.failed + self.skipped
        duration = time.time() - self.start_time
        
        # Create summary table
        table = Table(title="Website Scraping Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total URLs", str(self.total))
        table.add_row("Processed", str(processed))
        table.add_row("Scraped", str(self.scraped))
        table.add_row("Failed", str(self.failed))
        table.add_row("Skipped", str(self.skipped))
        table.add_row("HTTP Errors", str(self.http_errors))
        table.add_row("Duration", f"{duration:.2f} seconds")
        
        if duration > 0:
            rate = processed / duration
            table.add_row("Processing Rate", f"{rate:.2f} URLs/second")
        
        self.console.print(table)

class WebsiteContentProcessor:
    def __init__(
        self, 
        company_name: str = "Express Analytics",
        website_url: str = "https://www.expressanalytics.com",
        sitemap_url: str = "https://www.expressanalytics.com/sitemap.xml",
        batch_size: int = 5,
        setup_database: bool = False
    ):
        """Initialize the website content processor
        
        Args:
            company_name: Name of the company
            website_url: Base URL of the website
            sitemap_url: URL of the sitemap
            batch_size: Number of URLs to process in parallel
            setup_database: Whether to generate and return SQL setup commands when tables don't exist
        """
        # Get company abbreviation for table names
        self.company_abbr = abbreviate_company_name(company_name)
        self.company_name = company_name
        self.website_url = website_url
        self.sitemap_url = sitemap_url
        self.batch_size = batch_size
        self.setup_database = setup_database
        
        # Set table names based on company abbreviation
        self.documents_table = f"{self.company_abbr}_documents"
        self.website_data_table = f"{self.company_abbr}_website_data"
        
        # Initialize the progress tracker
        self.progress = ProgressTracker()
        
        # Initialize supabase client
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
            
        self.supabase_client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize langchain clients
        self.mistral_api_key = os.environ.get("MISTRALAI_API_KEY")
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        
        if not self.mistral_api_key:
            raise ValueError("MISTRALAI_API_KEY environment variable must be set")
            
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable must be set")
        
        # Initialize embeddings model
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=self.mistral_api_key
        )
        
        # Initialize summarizer
        self.summarizer = load_summarize_chain(
            llm=ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192",
                groq_api_key=self.groq_api_key
            ),
            chain_type="refine"
        )
        
        # Retry configuration for API calls
        self.api_retry_config = {
            "stop": stop_after_attempt(5),
            "wait": wait_exponential(multiplier=1, min=4, max=60),
            "retry": retry_if_exception_type(Exception),
            "reraise": True,
        }
        
        # Try to initialize vector store
        try:
            vector_available = self._check_vector_extension()
            logger.info(f"Vector extension available: {vector_available}")
            
            self._initialize_supabase_tables()
            
            # Initialize vector store
            self.vector_store = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name=self.documents_table,
                query_name=f"match_{self.documents_table}"
            )
            logger.info("Successfully initialized Supabase vector store")
        except Exception as e:
            logger.warning(f"Error initializing Supabase vector store: {e}")
            logger.warning("Will continue without vector store functionality")
            # Create a placeholder vector_store attribute to avoid attribute errors
            self.vector_store = None

        # Rate limits with more conservative values
        self.MISTRAL_CALLS = 15  # Reduced from 20 to 15 calls per minute
        self.GROQ_CALLS = 20     # Reduced from 25 to 20 calls per minute
        self.last_mistral_call = 0
        self.last_groq_call = 0
        
        logger.info(f"Initialized WebsiteContentProcessor for {company_name}")
        logger.info(f"Using table names: {self.documents_table} and {self.website_data_table}")

    def _check_vector_extension(self):
        """Check if the vector extension is enabled in Supabase"""
        # NOTE: We can't directly check extensions from the Supabase REST API
        # We'll assume the vector extension isn't available by default
        # and handle errors gracefully when they occur
        logger.warning("Vector extension check not available through REST API - assuming not available")
        return False

    def _initialize_supabase_tables(self):
        """Initialize required tables and functions in Supabase using REST API instead of direct SQL"""
        missing_tables = []
        setup_script = ""
        
        try:
            logger.info(f"Creating/verifying tables using Supabase REST API methods")
            
            # Check if the website_data table exists
            try:
                self.supabase_client.table(self.website_data_table).select("*").limit(1).execute()
                logger.info(f"Table {self.website_data_table} exists")
            except Exception as e:
                if "does not exist" in str(e).lower():
                    logger.warning(f"Table {self.website_data_table} does not exist.")
                    missing_tables.append(self.website_data_table)
                    
                    # Generate SQL script for website_data table
                    website_data_sql = f"""
                    CREATE TABLE IF NOT EXISTS "{self.website_data_table}" (
                        id BIGINT GENERATED BY DEFAULT AS IDENTITY NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT now(),
                        url TEXT NOT NULL,
                        service_name TEXT NULL,
                        sitemap_last_update TIMESTAMP WITH TIME ZONE NULL,
                        last_information_update TIMESTAMP WITH TIME ZONE NULL,
                        type TEXT NOT NULL,
                        vector_store_row_ids BIGINT[] NOT NULL DEFAULT '{{}}'::BIGINT[],
                        http_status INTEGER NULL,
                        last_extracted TIMESTAMP WITH TIME ZONE NULL DEFAULT now(),
                        CONSTRAINT {self.website_data_table}_pkey PRIMARY KEY (url),
                        CONSTRAINT {self.website_data_table}_type_check CHECK (
                            (
                                type = ANY (
                                    ARRAY[
                                        'Other'::TEXT,
                                        'Solution'::TEXT,
                                        'Product'::TEXT,
                                        'Blog'::TEXT,
                                        'Service'::TEXT,
                                        'About'::TEXT,
                                        'Contact'::TEXT
                                    ]
                                )
                            )
                        )
                    ) TABLESPACE pg_default;
                    """
                    setup_script += website_data_sql
            
            # Check if the documents table exists
            try:
                self.supabase_client.table(self.documents_table).select("*").limit(1).execute()
                logger.info(f"Table {self.documents_table} exists")
            except Exception as e:
                if "does not exist" in str(e).lower():
                    logger.warning(f"Table {self.documents_table} does not exist.")
                    missing_tables.append(self.documents_table)
                    
                    # Generate SQL script for documents table
                    documents_sql = f"""
                    CREATE TABLE IF NOT EXISTS "{self.documents_table}" (
                        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                        content TEXT,
                        metadata JSONB,
                        embedding VECTOR(1024),
                        url TEXT
                    );

                    -- Create vector index if available
                    CREATE INDEX IF NOT EXISTS {self.documents_table}_embedding_idx
                    ON "{self.documents_table}"
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                    """
                    setup_script += documents_sql
                    
                    # Add match function for vector search
                    match_function_sql = f"""
                    -- Create match function for semantic search
                    CREATE OR REPLACE FUNCTION match_{self.documents_table}(
                        query_embedding VECTOR(1024),
                        match_count INT DEFAULT 5,
                        filter JSONB DEFAULT '{{}}'::jsonb,
                        match_threshold REAL DEFAULT 0.5
                    ) RETURNS TABLE (
                        id BIGINT,
                        content TEXT,
                        metadata JSONB,
                        url TEXT,
                        similarity REAL
                    ) 
                    LANGUAGE plpgsql 
                    AS $$
                    BEGIN
                        RETURN QUERY
                        SELECT
                            id,
                            content,
                            metadata,
                            url,
                            1 - (embedding <=> query_embedding) AS similarity
                        FROM
                            "{self.documents_table}"
                        WHERE
                            1 - (embedding <=> query_embedding) > match_threshold
                        ORDER BY
                            embedding <=> query_embedding
                        LIMIT match_count;
                    END;
                    $$;
                    """
                    setup_script += match_function_sql
                    
                    # Add function and trigger for updating vector_store_row_ids
                    update_function_sql = f"""
                    -- Create function for updating vector_store_row_ids
                    CREATE OR REPLACE FUNCTION update_{self.company_abbr}_vector_store_row_ids()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                      -- On INSERT: Add new document ID, ensuring uniqueness
                      IF TG_OP = 'INSERT' THEN
                        UPDATE "{self.website_data_table}"
                        SET vector_store_row_ids = ARRAY(
                          SELECT DISTINCT unnest(array_append(vector_store_row_ids, NEW.id))
                        )
                        WHERE url = NEW.url;
                      END IF;

                      -- On DELETE: Remove document ID from array
                      IF TG_OP = 'DELETE' THEN
                        UPDATE "{self.website_data_table}"
                        SET vector_store_row_ids = array_remove(vector_store_row_ids, OLD.id)
                        WHERE url = OLD.url;
                      END IF;

                      -- On UPDATE: If URL changes, move ID from old URL to new URL
                      IF TG_OP = 'UPDATE' AND OLD.url IS DISTINCT FROM NEW.url THEN
                        -- Remove from old URL
                        UPDATE "{self.website_data_table}"
                        SET vector_store_row_ids = array_remove(vector_store_row_ids, OLD.id)
                        WHERE url = OLD.url;

                        -- Add to new URL
                        UPDATE "{self.website_data_table}"
                        SET vector_store_row_ids = ARRAY(
                          SELECT DISTINCT unnest(array_append(vector_store_row_ids, NEW.id))
                        )
                        WHERE url = NEW.url;
                      END IF;

                      RETURN NULL;
                    END;
                    $function$;

                    -- Create trigger for automatically updating vector_store_row_ids
                    DROP TRIGGER IF EXISTS trigger_update_{self.company_abbr}_vector_store_row_ids ON "{self.documents_table}";

                    CREATE TRIGGER trigger_update_{self.company_abbr}_vector_store_row_ids
                    AFTER DELETE OR INSERT OR UPDATE
                    ON "{self.documents_table}"
                    FOR EACH ROW
                    EXECUTE FUNCTION update_{self.company_abbr}_vector_store_row_ids();
                    """
                    setup_script += update_function_sql
            
            # Check if vector extension needs to be enabled
            if missing_tables and "VECTOR" in setup_script:
                vector_extension_sql = """
                -- Enable vector extension if not already enabled
                CREATE EXTENSION IF NOT EXISTS vector;
                """
                setup_script = vector_extension_sql + setup_script
            
            # If any tables are missing, generate SQL and prompt the user
            if missing_tables:
                # Save SQL to a file for convenience
                sql_file_path = f"{self.company_abbr.lower()}_db_setup.sql"
                with open(sql_file_path, "w") as f:
                    f.write(setup_script)
                
                # Print clear setup instructions
                print("\n" + "="*80)
                print(f"DATABASE SETUP REQUIRED FOR {self.company_name}")
                print("="*80)
                print("The required tables do not exist in your Supabase project.")
                print("\nTo create the required tables and functions, run the following SQL in your Supabase SQL Editor:")
                print("\n" + setup_script)
                print("\n" + "="*80)
                print(f"SQL commands have been saved to {os.path.abspath(sql_file_path)} for your convenience.")
                print("="*80 + "\n")
                
                # Prompt the user to run the SQL and wait for confirmation
                self._wait_for_sql_execution_confirmation(sql_file_path)
                
                # After confirmation, verify that tables now exist
                try:
                    self.supabase_client.table(self.website_data_table).select("*").limit(1).execute()
                    self.supabase_client.table(self.documents_table).select("*").limit(1).execute()
                    logger.info("Table verification successful after SQL execution - all required tables exist")
                except Exception as e:
                    logger.error(f"Tables still do not exist after SQL execution: {str(e)}")
                    raise ValueError("Database tables still missing after SQL execution. Please check your Supabase SQL Editor for errors.")
            else:
                logger.info("Supabase initialization complete - all required tables exist")
                
        except Exception as e:
            logger.error(f"Error during Supabase initialization: {str(e)}")
            raise
    
    def _wait_for_sql_execution_confirmation(self, sql_file_path: str):
        """Wait for the user to confirm they have executed the SQL script
        
        Args:
            sql_file_path: Path to the SQL file
        """
        console = Console()  # Create a Rich console for better formatting
        
        console.print("\n[bold yellow]IMPORTANT DATABASE SETUP REQUIRED[/bold yellow]")
        console.print("[bold yellow]--------------------------------[/bold yellow]")
        console.print(f"Supabase does not allow table creation via the API. You need to manually run the SQL script.")
        console.print("\n[bold]Steps to follow:[/bold]")
        console.print("1. Open your [link=https://app.supabase.com]Supabase dashboard[/link]")
        console.print("2. Navigate to the SQL Editor section")
        console.print(f"3. Copy the SQL from the file: [bold cyan]{os.path.abspath(sql_file_path)}[/bold cyan]")
        console.print("4. Paste and execute the SQL in your Supabase SQL Editor")
        console.print("5. Return to this terminal and confirm completion")
        
        with open(sql_file_path, 'r') as f:
            sql_content = f.read()
        
        console.print("\n[bold]SQL to execute:[/bold]")
        console.print(f"[dim]{sql_content}[/dim]")
        
        console.print("\n[bold yellow]After executing the SQL in Supabase:[/bold yellow]")
        
        confirmation = ""
        while confirmation.lower() not in ["yes", "y"]:
            confirmation = input("\nHave you executed the SQL in your Supabase SQL Editor? (yes/no): ")
            if confirmation.lower() in ["no", "n"]:
                console.print("[bold red]Please execute the SQL before continuing.[/bold red]")
                console.print(f"The SQL script is located at: [cyan]{os.path.abspath(sql_file_path)}[/cyan]")
            elif confirmation.lower() not in ["yes", "y"]:
                console.print("[bold red]Please answer 'yes' or 'no'.[/bold red]")
        
        console.print("\n[bold green]Thank you! Continuing with the process...[/bold green]\n")
        logger.info("User confirmed SQL execution in Supabase SQL Editor")

    async def generate_embedding(self, text: str):
        """Generate embedding with rate limiting and retry logic"""
        @retry(**self.api_retry_config)
        async def _generate():
            # Rate limiting
            current_time = time.time()
            elapsed = current_time - self.last_mistral_call
            required_wait = (60 / self.MISTRAL_CALLS)
            
            if elapsed < required_wait:
                wait_time = required_wait - elapsed
                logger.info(f"Rate limiting: Waiting {wait_time:.2f}s for Mistral API")
                await asyncio.sleep(wait_time)
            
            try:
                result = self.embeddings.embed_query(text)
                self.last_mistral_call = time.time()
                return result
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                if "rate limit" in str(e).lower():
                    logger.info("Rate limit hit, waiting 61 seconds before retry")
                    await asyncio.sleep(61)  # Wait a full minute plus 1 second
                raise
        return await _generate()

    async def generate_summary(self, docs: List[Document]) -> str:
        """Generate summary with rate limiting and retry logic"""
        @retry(**self.api_retry_config)
        async def _generate():
            # Rate limiting
            current_time = time.time()
            elapsed = current_time - self.last_groq_call
            required_wait = (60 / self.GROQ_CALLS)
            
            if elapsed < required_wait:
                wait_time = required_wait - elapsed
                logger.info(f"Rate limiting: Waiting {wait_time:.2f}s for Groq API")
                await asyncio.sleep(wait_time)
            
            try:
                result = self.summarizer.run(docs)
                self.last_groq_call = time.time()
                return result
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                if "rate limit" in str(e).lower():
                    logger.info("Rate limit hit, waiting 61 seconds before retry")
                    await asyncio.sleep(61)  # Wait a full minute plus 1 second
                raise
        return await _generate()

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, int]:
        """Fetch URL content with retry logic
        
        Returns:
            Tuple of (content, status_code)
        """
        @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30))
        async def _fetch():
            try:
                async with session.get(url, timeout=30) as response:
                    status_code = response.status
                    content = await response.text()
                    response.raise_for_status()  # Still raise for error codes
                    return content, status_code
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error {e.status} fetching {url}: {str(e)}")
                # Return empty content but the error code
                return "", e.status
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching {url}")
                raise
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                raise
        try:
            return await _fetch()
        except Exception as e:
            logger.error(f"Failed all retries fetching {url}: {str(e)}")
            # Return empty string and 0 status for non-HTTP errors
            return "", 0

    def check_content_update_needed(self, url: str, last_sitemap_update: str) -> bool:
        """Check if the content needs to be updated based on the last sitemap update
        
        Args:
            url: The URL to check
            last_sitemap_update: Last update from sitemap
            
        Returns:
            True if update is needed, False otherwise
        """
        try:
            # Query the database to see if we already have this URL
            result = self.supabase_client.table(self.website_data_table).select("*").eq("url", url).execute()
            
            # If we have results, check if we need to update
            if result.data and len(result.data) > 0:
                # Get the last update time from the database
                db_last_update = result.data[0].get("sitemap_last_update")
                
                # If we have a last update time, compare with the sitemap
                if db_last_update and last_sitemap_update:
                    # Convert strings to datetime objects for comparison
                    db_last_update_dt = datetime.fromisoformat(db_last_update.replace('Z', '+00:00'))
                    sitemap_update_dt = datetime.fromisoformat(last_sitemap_update.replace('Z', '+00:00'))
                    
                    # If the sitemap update is newer, we need to update
                    if sitemap_update_dt > db_last_update_dt:
                        logger.info(f"Content update needed for {url}: Sitemap lastmod {sitemap_update_dt} is newer than DB lastmod {db_last_update_dt}")
                        return True
                    else:
                        logger.info(f"Content up to date for {url}: DB lastmod {db_last_update_dt} is newer than or equal to Sitemap lastmod {sitemap_update_dt}")
                        return False
                
                # If we don't have a last update time in either the DB or sitemap, we need to update
                logger.info(f"Content update needed for {url}: Missing lastmod information")
                return True
            
            # If we don't have any results, we need to update
            logger.info(f"Content update needed for {url}: Not found in database")
            return True
        except Exception as e:
            # If the table doesn't exist, we need to proceed with processing
            if "does not exist" in str(e).lower():
                logger.warning(f"Error checking update status: {e}")
                logger.warning("Table doesn't exist - will process URL regardless")
                return True
            
            # For other errors, log and proceed with processing
            logger.error(f"Error checking update status: {e}")
            return True

    async def process_content(self, url: str, site_type: str, content_html: str) -> Dict:
        """Process webpage content"""
        soup = BeautifulSoup(content_html, features="html.parser")
        
        # Select content based on type and URL
        if url.startswith(f"{self.website_url}/blog/"):
            content_div = soup.find("div", class_="blog-wrap")
            if not content_div:
                # Fallback selectors for blog
                content_div = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
        else:
            content_div = soup.find("div", class_="wpb-content-wrapper")
            if not content_div:
                # Fallback selectors for other pages
                content_div = soup.find("main") or soup.find("div", class_="content") or soup.find("div", id="content")
            
        if not content_div:
            logger.warning(f"Content div not found for {url}, using body instead")
            content_div = soup.find("body")
            
        if not content_div:
            raise ValueError(f"Could not extract any content from {url}")
            
        service_info = " ".join(line.strip() for line in content_div.text.split("\n") if line.strip())
        
        # Extract title from h1, meta title, or URL
        title_tag = soup.find("h1")
        if title_tag:
            service_name = title_tag.text.strip()
        else:
            meta_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "title"})
            if meta_title and meta_title.get("content"):
                service_name = meta_title["content"].strip()
            else:
                # Extract name from URL path
                path = url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ").title()
                service_name = path
        
        # Create a Document object for summarization
        doc = Document(
            page_content=service_info,
            metadata={"source": url, "type": site_type}
        )
        
        # Generate summary using Document object with retry
        summary = await self.generate_summary([doc])
        
        return {
            "service_name": service_name,
            "content": service_info,
            "summary": summary
        }

    async def update_vector_store(self, url: str, site_type: str, 
                                content: str, summary: str) -> List[str]:
        """Update the vector store with the content
        
        Args:
            url: The URL of the page
            site_type: The type of the page
            content: The content of the page
            summary: The summary of the page
            
        Returns:
            List of IDs of the created documents
        """
        try:
            # First check if we have an entry in the website_data table
            try:
                website_result = self.supabase_client.table(self.website_data_table).select("*").eq("url", url).execute()
                website_exists = website_result.data and len(website_result.data) > 0
                if website_exists:
                    logger.info(f"Found existing entry in website_data for {url}")
                    # Extract existing document IDs to ensure they're properly removed
                    existing_doc_ids = website_result.data[0].get("vector_store_row_ids", [])
                    if existing_doc_ids:
                        logger.info(f"Found {len(existing_doc_ids)} existing document IDs in website_data: {existing_doc_ids}")
            except Exception as e:
                if "does not exist" in str(e).lower():
                    logger.warning(f"Website data table does not exist: {e}")
                    website_exists = False
                else:
                    logger.error(f"Error checking website data table: {e}")
                    website_exists = False

            # First identify any existing documents for this URL that we might need to delete
            existing_ids = []
            try:
                result = self.supabase_client.table(self.documents_table).select("id").eq("url", url).execute()
                existing_ids = [item["id"] for item in result.data] if result.data else []
                if existing_ids:
                    logger.info(f"Found {len(existing_ids)} existing documents in documents table for {url}")
            except Exception as e:
                if "does not exist" in str(e).lower():
                    logger.warning(f"Error identifying existing documents for {url}: {e}")
                    logger.warning("Will proceed with creating new documents anyway")
                else:
                    logger.error(f"Error identifying existing documents for {url}: {e}")
                    return []
                
            # Delete existing documents if any
            if existing_ids:
                logger.info(f"Deleting {len(existing_ids)} existing documents for {url}")
                try:
                    for doc_id in existing_ids:
                        delete_result = self.supabase_client.table(self.documents_table).delete().eq("id", doc_id).execute()
                        logger.debug(f"Deleted document {doc_id} with result: {delete_result}")
                except Exception as e:
                    logger.error(f"Error deleting existing documents for {url}: {e}")
                    # Continue with adding new documents
            
            # Split the content into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Create document objects for each chunk
            docs = []
            for i, chunk in enumerate(text_splitter.split_text(content)):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "url": url,
                        "type": site_type,
                        "chunk": i,
                        "summary": summary
                    }
                )
                docs.append(doc)
            
            if not docs:
                logger.warning(f"No content chunks generated for {url}")
                return []
            
            logger.info(f"Generated {len(docs)} content chunks for {url}")
            
            # If vector_store is available, add documents to it
            document_ids = []
            
            # Insert documents into database
            for doc in docs:
                try:
                    embedding = await self.generate_embedding(doc.page_content)
                    
                    # Create document record
                    document_data = {
                        "content": doc.page_content,
                        "metadata": json.dumps(doc.metadata),
                        "url": url
                    }
                    
                    # Check if we have vector extension
                    if self.vector_store:
                        # Add embedding directly as vector type
                        document_data["embedding"] = embedding
                    else:
                        # Store as string for future conversion
                        document_data["embedding"] = json.dumps(embedding)
                    
                    # Insert document
                    @retry(**self.api_retry_config)
                    async def insert_doc():
                        # No need to check has_vector - always store as string
                        # Convert embedding list to string representation
                        result = self.supabase_client.table(self.documents_table).insert(document_data).execute()
                        return result
                    
                    try:
                        result = await insert_doc()
                        if result.data and len(result.data) > 0:
                            document_ids.append(result.data[0]["id"])
                    except Exception as e:
                        if "does not exist" in str(e).lower():
                            logger.error(f"Error inserting document for {url}: {e}")
                            logger.error("Database table does not exist - please run setup first")
                            return []
                        else:
                            logger.error(f"Error inserting document for {url}: {e}")
                
                except Exception as e:
                    logger.error(f"Error processing document chunk for {url}: {e}")
            
            # Update website_data table with the document IDs and other metadata
            if document_ids:
                logger.info(f"Generated {len(document_ids)} new document IDs for {url}")
                website_data = {
                    "url": url,
                    "service_name": url.split("/")[-1].replace("-", " ").title() if url.split("/")[-1] else url,
                    "sitemap_last_update": datetime.now(timezone.utc).isoformat(),
                    "last_information_update": datetime.now(timezone.utc).isoformat(),
                    "type": site_type,
                    "vector_store_row_ids": document_ids,
                    "last_extracted": datetime.now(timezone.utc).isoformat()
                }
                
                try:
                    # Use upsert to create or update the entry
                    logger.info(f"Upserting website_data for {url} with document IDs: {document_ids}")
                    upsert_result = self.supabase_client.table(self.website_data_table).upsert(website_data).execute()
                    
                    # Verify the upsert was successful
                    if upsert_result.data and len(upsert_result.data) > 0:
                        logger.info(f"Successfully upserted website_data for {url}")
                        
                        # Double-check the update by retrieving the row again
                        check_result = self.supabase_client.table(self.website_data_table).select("*").eq("url", url).execute()
                        if check_result.data and len(check_result.data) > 0:
                            stored_ids = check_result.data[0].get("vector_store_row_ids", [])
                            logger.info(f"Verified website_data for {url} has {len(stored_ids)} document IDs")
                        else:
                            logger.warning(f"Could not verify website_data update for {url} - entry not found after upsert")
                    else:
                        logger.warning(f"No data returned from upsert operation for {url}")
                except Exception as e:
                    if "does not exist" in str(e).lower():
                        logger.error(f"Error upserting website_data for {url}: {e}")
                        logger.error("Database table does not exist - please run setup first")
                    else:
                        logger.error(f"Error upserting website_data for {url}: {e}")
            else:
                logger.warning(f"No document IDs generated for {url}, website_data table not updated")
            
            return document_ids
            
        except Exception as e:
            logger.error(f"Error updating vector store for {url}: {e}")
            return []

    async def _safe_update_progress(self, status: str, http_code: int = None):
        """Safely update the progress tracker from async code
        
        Args:
            status: Status of the URL processing ('scraped', 'failed', or 'skipped')
            http_code: HTTP status code (if applicable)
        """
        try:
            # Run the progress update in a separate thread to avoid blocking the event loop
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.progress.update(status, http_code)
            )
        except Exception as e:
            # Catch and log any issues with progress updates to prevent them from breaking processing
            logger.error(f"Error updating progress ({status}, code {http_code}): {e}")

    async def process_website(self, url: str, last_sitemap_update: str) -> bool:
        """Process a single website, returns True if successful, False otherwise"""
        # Normalize website and URL for consistent matching
        normalized_website_url = self.website_url.replace("https://www.", "https://")
        normalized_url = url.replace("https://www.", "https://")
        
        # Determine site type based on URL pattern
        if not normalized_url.startswith(normalized_website_url):
            logger.warning(f"URL {url} does not belong to {self.website_url}, skipping")
            # Update progress with skipped status (no HTTP code)
            await self._safe_update_progress("skipped")
            return False
            
        if "/product/" in url:
            site_type = "Product"
        elif "/solutions/" in url:
            site_type = "Solution"
        elif "/blog/" in url:
            site_type = "Blog"
        elif "/service/" in url:
            site_type = "Service"
        elif "/about/" in url:
            site_type = "About"
        elif "/contact/" in url:
            site_type = "Contact"
        else:
            site_type = "Other"

        # Check if content update is needed based on sitemap lastmod
        if not self.check_content_update_needed(url, last_sitemap_update):
            logger.info(f"Content up to date for {url}, skipping processing")
            # Update progress with skipped status (no HTTP code)
            await self._safe_update_progress("skipped")
            return True

        # At this point, we know we need to update the content
        logger.info(f"Processing content for {url}")
        http_status_code = 0  # Default status code
        
        try:
            # First check if we have entry in website_data but need to update it
            try:
                website_result = self.supabase_client.table(self.website_data_table).select("*").eq("url", url).execute()
                if website_result.data and len(website_result.data) > 0:
                    # We have an existing entry, we'll update it later in update_vector_store
                    logger.info(f"Found existing entry in website_data for {url}, will update")
            except Exception as e:
                if "does not exist" in str(e).lower():
                    logger.warning(f"website_data table does not exist: {e}")
                else:
                    logger.error(f"Error checking website data table: {e}")

            async with aiohttp.ClientSession() as session:
                # Fetch content with status code
                content_html, http_status_code = await self.fetch_url(session, url)
                
                # If the response was an error (empty content), return failure
                if not content_html and http_status_code >= 400:
                    logger.error(f"Failed to fetch content for {url} with HTTP status {http_status_code}")
                    # Update progress with failed status and HTTP code
                    await self._safe_update_progress("failed", http_status_code)
                    return False
                
                processed_content = await self.process_content(url, site_type, content_html)
                
                # Update vector store
                vector_ids = await self.update_vector_store(
                    url, 
                    site_type,
                    processed_content["content"],
                    processed_content["summary"]
                )
                
                # Update website data
                current_datetime = datetime.now(timezone.utc)
                website_data = {
                    "url": url,
                    "service_name": processed_content["service_name"],
                    "sitemap_last_update": last_sitemap_update or current_datetime.isoformat(),
                    "last_information_update": current_datetime.isoformat(),
                    "type": site_type,
                    "last_extracted": current_datetime.isoformat(),
                    "vector_store_row_ids": vector_ids,
                    "http_status": http_status_code  # Store the HTTP status code
                }
                
                try:
                    # Use upsert to create or update the entry
                    logger.info(f"Upserting website_data from process_website for {url}")
                    upsert_result = self.supabase_client.table(self.website_data_table).upsert(website_data).execute()
                    
                    if upsert_result.data and len(upsert_result.data) > 0:
                        logger.info(f"Successfully updated website_data for {url}")
                    else:
                        logger.warning(f"No data returned from website_data upsert for {url}")
                except Exception as e:
                    logger.error(f"Error updating website_data for {url}: {e}")
                
                logger.info(f"Successfully processed {url} with HTTP status {http_status_code}")
                # Update progress with scraped status and HTTP code
                await self._safe_update_progress("scraped", http_status_code)
                return True
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            # Update progress with failed status and HTTP code
            await self._safe_update_progress("failed", http_status_code)
            return False

    async def process_batch(self, batch: List[Tuple[str, str]]) -> List[bool]:
        """Process a batch of websites concurrently"""
        tasks = [self.process_website(url, last_mod) for url, last_mod in batch]
        return await asyncio.gather(*tasks)

    async def run(self):
        """Run the website content processor"""
        logger.info(f"Starting website scraping for {self.company_name}")
        logger.info(f"Using database tables: {self.documents_table} and {self.website_data_table}")
        
        try:
            logger.info(f"Fetching main sitemap from {self.sitemap_url}")
            
            # Normalize the website URL by removing www if present for matching
            normalized_website_url = self.website_url.replace("https://www.", "https://")
            logger.info(f"Normalized website URL for matching: {normalized_website_url}")
            
            # Fetch sitemap
            async with aiohttp.ClientSession() as session:
                all_urls = []
                sitemap_content, sitemap_status = await self.fetch_url(session, self.sitemap_url)
                
                # Check if we got a successful response
                if not sitemap_content or sitemap_status >= 400:
                    logger.error(f"Failed to fetch sitemap from {self.sitemap_url} with status code {sitemap_status}")
                    print(f"\nERROR: Failed to fetch sitemap from {self.sitemap_url} with status code {sitemap_status}")
                    return False
                
                # Log a snippet of the sitemap content for debugging
                logger.info(f"Sitemap content snippet: {sitemap_content[:500]}...")
                
                # Parse sitemap using BeautifulSoup with XML parser instead of ElementTree
                main_soup = BeautifulSoup(sitemap_content, features="xml")
                
                # Check if this is a sitemap index (has <sitemap> tags)
                sitemap_tags = main_soup.find_all("sitemap")
                
                logger.info(f"Found {len(sitemap_tags)} sitemap tags in the main sitemap")
                
                if sitemap_tags:
                    # This is a sitemap index
                    sub_sitemaps = [tag.find("loc").text for tag in sitemap_tags if tag.find("loc")]
                    
                    logger.info(f"Found {len(sub_sitemaps)} sub-sitemaps")
                    
                    # Fetch each sub-sitemap
                    for sub_sitemap_url in sub_sitemaps:
                        logger.info(f"Fetching sub-sitemap: {sub_sitemap_url}")
                        sub_content, sub_status = await self.fetch_url(session, sub_sitemap_url)
                        
                        # Check if we got a successful response
                        if not sub_content or sub_status >= 400:
                            logger.error(f"Failed to fetch sub-sitemap from {sub_sitemap_url} with status code {sub_status}")
                            print(f"\nWARNING: Failed to fetch sub-sitemap from {sub_sitemap_url} with status code {sub_status}")
                            continue  # Try next sub-sitemap instead of failing completely
                        
                        # Log a snippet of the sub-sitemap content for debugging
                        logger.info(f"Sub-sitemap content snippet: {sub_content[:500]}...")
                        
                        # Parse sub-sitemap
                        sub_soup = BeautifulSoup(sub_content, features="xml")
                        url_tags = sub_soup.find_all("url")
                        
                        logger.info(f"Found {len(url_tags)} URL tags in sub-sitemap: {sub_sitemap_url}")
                        
                        for url_tag in url_tags:
                            loc_tag = url_tag.find("loc")
                            lastmod_tag = url_tag.find("lastmod")
                            
                            if loc_tag and loc_tag.text:
                                url = loc_tag.text
                                lastmod = lastmod_tag.text if lastmod_tag else None
                                
                                if lastmod:
                                    logger.info(f"URL {url} has lastmod: {lastmod}")
                                else:
                                    logger.info(f"URL {url} has no lastmod")
                                
                                # Normalize the URL by removing www if present
                                normalized_url = url.replace("https://www.", "https://")
                                
                                if normalized_url.startswith(normalized_website_url):
                                    url_data = (url, lastmod)
                                    all_urls.append(url_data)
                                    logger.info(f"Added URL to processing list: {url}")
                                else:
                                    logger.info(f"Skipping URL that doesn't match website URL: {url}")
                                    logger.info(f"  Normalized URL: {normalized_url}")
                                    logger.info(f"  Expected to start with: {normalized_website_url}")
                else:
                    # This is a regular sitemap (has <url> tags)
                    url_tags = main_soup.find_all("url")
                    
                    logger.info(f"Found {len(url_tags)} URL tags in main sitemap")
                    
                    for url_tag in url_tags:
                        loc_tag = url_tag.find("loc")
                        lastmod_tag = url_tag.find("lastmod")
                        
                        if loc_tag and loc_tag.text:
                            url = loc_tag.text
                            lastmod = lastmod_tag.text if lastmod_tag else None
                            
                            if lastmod:
                                logger.info(f"URL {url} has lastmod: {lastmod}")
                            else:
                                logger.info(f"URL {url} has no lastmod")
                            
                            # Normalize the URL by removing www if present
                            normalized_url = url.replace("https://www.", "https://")
                            
                            # Compare normalized URLs to handle www vs non-www variations
                            if normalized_url.startswith(normalized_website_url):
                                url_data = (url, lastmod)
                                all_urls.append(url_data)
                            else:
                                logger.info(f"Skipping URL that doesn't match website URL: {url}")
            
            logger.info(f"Extracted {len(all_urls)} URLs from sitemaps")
            
            # Make sure we have URLs to process
            if not all_urls:
                logger.warning(f"No URLs found in sitemap. Please check the sitemap URL: {self.sitemap_url}")
                print(f"\nWARNING: No URLs found in sitemap. Please check the sitemap URL: {self.sitemap_url}")
                return False
            
            # Create a new progress tracker to ensure it's in a clean state
            self.progress = ProgressTracker()
            self.progress.initialize(len(all_urls))
            
            # Process URLs in batches
            batch_num = 0
            while batch_num * self.batch_size < len(all_urls):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, len(all_urls))
                
                batch = all_urls[start_idx:end_idx]
                logger.info(f"Processing batch {batch_num+1}/{(len(all_urls) + self.batch_size - 1) // self.batch_size} ({len(batch)} URLs)")
                
                # Process the batch concurrently
                results = await self.process_batch(batch)
                logger.info(f"Completed batch {batch_num+1} with {sum(results)} successes / {len(results)} total")
                
                # Add a small delay to ensure progress updates are visible
                await asyncio.sleep(0.2)
                
                # Increment batch counter
                batch_num += 1
            
            # Complete progress tracking
            self.progress.finish()
            
            logger.info("Website scraping completed successfully")
            return True
            
        except ValueError as e:
            if "Database tables missing" in str(e):
                logger.warning(f"Database setup required: {e}")
                if self.setup_database:
                    # This is expected when setup_database is True and tables are missing
                    logger.info("Please run the generated SQL script and try again.")
                else:
                    logger.warning("Run this script with --setup-database to generate SQL setup script")
                return False
            else:
                logger.error(f"Error running website content processor: {e}")
                return False
        except Exception as e:
            logger.error(f"Error running website content processor: {e}")
            return False

def main():
    """Main function to run the script"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process website content and store it in a vector database")
    parser.add_argument("--company", type=str, default="Empress Naturals", 
                        help="Company name (used for database naming)")
    parser.add_argument("--website", type=str, default="https://www.empressnaturals.co", 
                        help="Website base URL")
    parser.add_argument("--sitemap", type=str, default="https://empressnaturals.co/sitemap.xml", 
                        help="Sitemap URL")
    parser.add_argument("--batch-size", type=int, default=5, 
                        help="Number of URLs to process concurrently")
    parser.add_argument("--setup-database", action="store_true", 
                        help="Generate SQL setup script if tables don't exist")
    parser.add_argument("--verify-tables", action="store_true",
                        help="Verify database tables exist before processing")
    
    args = parser.parse_args()
    
    print(f"\n=== Website Content Processor Configuration ===")
    print(f"Company: {args.company}")
    print(f"Website: {args.website}")
    print(f"Sitemap: {args.sitemap}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Setup Database: {args.setup_database}")
    print(f"Verify Tables: {args.verify_tables}")
    print(f"==============================================\n")
    
    try:
        # Initialize the processor with the given parameters
        processor = WebsiteContentProcessor(
            company_name=args.company,
            website_url=args.website,
            sitemap_url=args.sitemap,
            batch_size=args.batch_size,
            setup_database=args.setup_database
        )
        
        # If verify tables is set, check if database tables exist
        if args.verify_tables:
            print("Verifying database tables...")
            # Get table names
            documents_table = f"{processor.company_abbr}_documents"
            website_data_table = f"{processor.company_abbr}_website_data"
            
            # Check if tables exist
            supabase_client = create_client(
                os.getenv("SUPABASE_URL"), 
                os.getenv("SUPABASE_KEY")
            )
            
            # Query for tables
            try:
                documents_result = supabase_client.table(documents_table).select("count(*)", count="exact").execute()
                website_data_result = supabase_client.table(website_data_table).select("count(*)", count="exact").execute()
                
                # Log counts
                documents_count = documents_result.count if hasattr(documents_result, 'count') else 0
                website_data_count = website_data_result.count if hasattr(website_data_result, 'count') else 0
                
                print(f"Documents table exists with {documents_count} rows")
                print(f"Website data table exists with {website_data_count} rows")
                
                # If tables exist but website_data is empty, ensure setup is enabled
                if website_data_count == 0 and args.setup_database == False:
                    print("WARNING: Website data table is empty. Consider running with --setup-database")
            except Exception as e:
                if "does not exist" in str(e).lower():
                    print(f"ERROR: Database tables do not exist: {e}")
                    if not args.setup_database:
                        print("Please run with --setup-database to create tables")
                        return False
                else:
                    print(f"ERROR: Failed to verify tables: {e}")
        
        # Run the processor
        asyncio.run(processor.run())
        return True
    except Exception as e:
        logger.error(f"Error running website content processor: {e}")
        return False

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)

