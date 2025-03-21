"""
Database setup and management for RAG tools.
"""
import os
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from supabase import create_client, Client
import json

from rag_tools.utils import get_sql_file_path, prompt_for_sql_execution

logger = logging.getLogger("rag_tools")

class DatabaseManager:
    """
    Manages database operations for vector storage.
    """
    
    def __init__(
        self,
        company_abbr: str,
        setup_mode: bool = False,
        custom_types: Optional[List[str]] = None
    ):
        """
        Initialize the database manager.
        
        Args:
            company_abbr: Abbreviation for the company name
            setup_mode: Whether to set up database tables if they don't exist
            custom_types: Custom content types to add to the type check constraint
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Save basic parameters
        self.company_abbr = company_abbr
        self.setup_mode = setup_mode
        
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
            
        self.supabase_client = create_client(supabase_url, supabase_key)
        
        # Set up table names
        self.documents_table = f"{company_abbr}_documents"
        self.website_data_table = f"{company_abbr}_website_data"
        
        # Set up content types for the type check constraint
        self.content_types = [
            "home",
            "article",
            "blog",
            "product",
            "service",
            "about",
            "contact",
            "faq",
            "testimonial",
            "terms",
            "privacy",
            "support",
            "documentation",
            "case_study",
            "other"
        ]
        
        # Add custom types if provided
        if custom_types:
            self.content_types.extend(custom_types)
            
        # Remove duplicates
        self.content_types = list(set(self.content_types))
        
        # Flag for schema compatibility mode
        self._needs_schema_compat = False
    
    def verify_tables(self) -> bool:
        """
        Verify that required database tables exist.
        
        Returns:
            True if tables exist, False otherwise
        """
        tables_exist = True
        
        try:
            # Check if documents table exists
            self.supabase_client.table(self.documents_table).select("*").limit(1).execute()
            logger.info(f"Table {self.documents_table} exists")
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.warning(f"Table {self.documents_table} does not exist")
                tables_exist = False
            else:
                logger.error(f"Error checking {self.documents_table} table: {e}")
                raise
        
        try:
            # Check if website_data table exists
            self.supabase_client.table(self.website_data_table).select("*").limit(1).execute()
            logger.info(f"Table {self.website_data_table} exists")
            
            # Check if http_status column exists and add it if it doesn't
            self.check_and_update_schema()
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.warning(f"Table {self.website_data_table} does not exist")
                tables_exist = False
            else:
                logger.error(f"Error checking {self.website_data_table} table: {e}")
                raise
        
        return tables_exist
    
    def check_and_update_schema(self) -> None:
        """
        Check if required columns exist in the website_data table and add them if they don't.
        """
        try:
            # Try to select the http_status column specifically to see if it exists
            self.supabase_client.table(self.website_data_table).select("http_status").limit(1).execute()
            logger.info(f"http_status column exists in {self.website_data_table}")
        except Exception as e:
            error_message = str(e).lower()
            # Check for specific error about the column not existing
            if "http_status" in error_message and ("column" in error_message or "does not exist" in error_message):
                logger.warning(f"http_status column missing in {self.website_data_table}")
                
                # As a workaround, since we can't directly alter the table through the API,
                # let's modify our upsert_website_data method to handle both schema versions
                logger.warning(f"Will use schema compatibility mode for {self.website_data_table}")
                self._needs_schema_compat = True
            else:
                logger.error(f"Error checking schema for {self.website_data_table}: {str(e)}")
    
    def generate_setup_sql(self) -> str:
        """
        Generate SQL for setting up the database tables.
        
        Returns:
            SQL script string
        """
        # Create the content type array string for the constraint
        type_array_items = [f"'{type_name}'::text" for type_name in self.content_types]
        type_array_str = ",\n                        ".join(type_array_items)
        
        # Start with vector extension
        sql = f"""-- Enable vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create website_data table
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
                    {type_array_str}
                ]
            )
        )
    )
) TABLESPACE pg_default;

-- Create documents table for vector storage
CREATE TABLE IF NOT EXISTS "{self.documents_table}" (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1024),
    url TEXT,
    CONSTRAINT {self.documents_table}_url_fkey FOREIGN KEY (url)
        REFERENCES "{self.website_data_table}" (url)
        ON DELETE CASCADE
);

-- Create vector index
CREATE INDEX IF NOT EXISTS {self.documents_table}_embedding_idx
ON "{self.documents_table}"
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

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
        return sql
    
    def setup_database(self) -> bool:
        """
        Set up the database tables if they don't exist.
        
        Returns:
            True if setup succeeded or tables already exist, False otherwise
        """
        # First check if tables already exist
        if self.verify_tables():
            logger.info("Database tables already exist, skipping setup")
            return True
        
        if not self.setup_mode:
            logger.error("Database tables do not exist and setup_mode is disabled")
            logger.error("Run with --setup-database to create tables")
            return False
        
        # Generate setup SQL
        sql = self.generate_setup_sql()
        sql_file_path = get_sql_file_path(self.company_abbr)
        
        # Write SQL to file
        with open(sql_file_path, "w") as f:
            f.write(sql)
        
        logger.info(f"SQL script written to {sql_file_path}")
        
        # Prompt user to execute SQL
        if not prompt_for_sql_execution(sql_file_path):
            logger.error("User did not confirm SQL execution")
            return False
        
        # Wait a bit for the tables to be created
        time.sleep(2)
        
        # Verify tables now exist
        if not self.verify_tables():
            logger.error("Tables still do not exist after SQL execution")
            logger.error("Please check for errors in the Supabase SQL Editor")
            return False
        
        logger.info("Database setup completed successfully")
        return True
    
    async def delete_documents_for_url(self, url: str) -> bool:
        """
        Delete all documents for a specific URL.
        
        Args:
            url: URL to delete documents for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get document IDs for the URL
            result = self.supabase_client.table(self.documents_table).select("id").eq("url", url).execute()
            
            if not result.data:
                logger.info(f"No documents found for URL {url}")
                return True
            
            document_ids = [item["id"] for item in result.data]
            logger.info(f"Found {len(document_ids)} documents to delete for URL {url}")
            
            # Delete documents
            for doc_id in document_ids:
                self.supabase_client.table(self.documents_table).delete().eq("id", doc_id).execute()
            
            logger.info(f"Successfully deleted {len(document_ids)} documents for URL {url}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents for URL {url}: {e}")
            return False
    
    def get_website_data(self, url: str) -> Dict[str, Any]:
        """
        Get website data for a specific URL.
        
        Args:
            url: URL to get data for
            
        Returns:
            Website data or empty dict if not found
        """
        try:
            result = self.supabase_client.table(self.website_data_table).select("*").eq("url", url).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            
            return {}
        except Exception as e:
            logger.error(f"Error getting website data for URL {url}: {e}")
            return {}
    
    def upsert_website_data(self, website_data: Dict[str, Any]) -> bool:
        """
        Create or update website data.
        
        Args:
            website_data: Website data to upsert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we need schema compatibility mode (working with older schema without http_status)
            if getattr(self, '_needs_schema_compat', False) and 'http_status' in website_data:
                # Create a copy of the data without the http_status field
                compat_data = {k: v for k, v in website_data.items() if k != 'http_status'}
                logger.info(f"Using schema compatibility mode for {website_data.get('url')} (removing http_status)")
                result = self.supabase_client.table(self.website_data_table).upsert(compat_data).execute()
            else:
                # Normal operation with full schema
                result = self.supabase_client.table(self.website_data_table).upsert(website_data).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully upserted website data for URL {website_data.get('url')}")
                return True
            
            logger.warning(f"No data returned from upsert operation for URL {website_data.get('url')}")
            return False
        except Exception as e:
            logger.error(f"Error upserting website data for URL {website_data.get('url')}: {e}")
            return False
    
    async def insert_document(self, content: str, metadata: Dict[str, Any], embedding: List[float], url: str) -> Optional[int]:
        """
        Insert a document into the vector store.
        
        Args:
            content: Document content
            metadata: Document metadata
            embedding: Document embedding
            url: URL associated with the document
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            # Create document record
            document_data = {
                "content": content,
                "metadata": json.dumps(metadata),
                "embedding": embedding,
                "url": url
            }
            
            # Insert document
            result = self.supabase_client.table(self.documents_table).insert(document_data).execute()
            
            if result.data and len(result.data) > 0:
                doc_id = result.data[0]["id"]
                logger.debug(f"Successfully inserted document {doc_id} for URL {url}")
                return doc_id
            
            logger.warning(f"No data returned from insert operation for URL {url}")
            return None
        except Exception as e:
            logger.error(f"Error inserting document for URL {url}: {e}")
            return None 