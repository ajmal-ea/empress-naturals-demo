"""
Vector storage for RAG tools - main module.

This module provides functionality for processing website content and storing it in a vector database.
"""
import os
import asyncio
import argparse
import logging
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime, timezone

# Import our modules
from rag_tools.utils import setup_logging, abbreviate_company_name, check_environment_variables
from rag_tools.database import DatabaseManager
from rag_tools.sitemap import SitemapParser
from rag_tools.content import ContentExtractor
from rag_tools.progress import ProgressTracker

class WebsiteVectorStorage:
    """
    Main class for processing website content and storing it in a vector database.
    """
    
    def __init__(
        self,
        company_name: str,
        website_url: str,
        sitemap_url: str,
        batch_size: int = 5,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_urls: Optional[int] = None,
        setup_database: bool = False,
        custom_types: Optional[List[str]] = None,
        log_level: int = logging.INFO,
        request_timeout: int = 30,
        mistral_rate_limit: int = 15,
        groq_rate_limit: int = 20,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the website vector storage processor.
        
        Args:
            company_name: Name of the company (used for table naming)
            website_url: Base URL of the website
            sitemap_url: URL of the sitemap
            batch_size: Number of URLs to process in parallel
            include_patterns: URL patterns to include (regex strings)
            exclude_patterns: URL patterns to exclude (regex strings)
            max_urls: Maximum number of URLs to process
            setup_database: Whether to set up database tables if they don't exist
            custom_types: Custom content types to add to the type check constraint
            log_level: Logging level
            request_timeout: Timeout for HTTP requests in seconds
            mistral_rate_limit: Rate limit for MistralAI API calls per minute
            groq_rate_limit: Rate limit for Groq API calls per minute
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between text chunks
        """
        # Set up logging
        self.logger = setup_logging(log_file=f"{company_name.lower().replace(' ', '_')}_vector_storage.log", level=log_level)
        
        # Check required environment variables
        if not check_environment_variables(["SUPABASE_URL", "SUPABASE_KEY", "MISTRALAI_API_KEY", "GROQ_API_KEY"]):
            raise ValueError("Missing required environment variables")
        
        # Save basic parameters
        self.company_name = company_name
        self.website_url = website_url
        self.sitemap_url = sitemap_url
        self.batch_size = batch_size
        self.max_urls = max_urls
        
        # Get company abbreviation for table names
        self.company_abbr = abbreviate_company_name(company_name)
        
        # Log configuration details
        self.logger.info(f"Initializing WebsiteVectorStorage for {company_name}")
        self.logger.info(f"Website URL: {website_url}")
        self.logger.info(f"Sitemap URL: {sitemap_url}")
        self.logger.info(f"Batch size: {batch_size}")
        
        # Initialize components
        self.db_manager = DatabaseManager(
            company_abbr=self.company_abbr,
            setup_mode=setup_database,
            custom_types=custom_types
        )
        
        self.sitemap_parser = SitemapParser(
            base_url=website_url,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            max_urls=max_urls,
            request_timeout=request_timeout
        )
        
        self.content_extractor = ContentExtractor(
            mistral_api_key=os.environ.get("MISTRALAI_API_KEY"),
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            request_timeout=request_timeout,
            mistral_rate_limit=mistral_rate_limit,
            groq_rate_limit=groq_rate_limit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize progress tracker
        self.progress = ProgressTracker()
        
        self.logger.info("WebsiteVectorStorage initialization complete")
    
    async def update_needed(self, url: str, lastmod: Optional[str]) -> bool:
        """
        Check if a URL needs to be updated based on the lastmod date.
        
        Args:
            url: URL to check
            lastmod: Last modification date from sitemap (ISO format)
            
        Returns:
            True if update is needed, False otherwise
        """
        # If no lastmod provided, assume update is needed
        if not lastmod:
            self.logger.info(f"No lastmod date for {url}, update needed")
            return True
        
        try:
            # Get website data from database
            website_data = self.db_manager.get_website_data(url)
            
            if not website_data:
                self.logger.info(f"No existing data for {url}, update needed")
                return True
            
            # Get the last update time from the database
            db_lastmod = website_data.get("sitemap_last_update")
            
            if not db_lastmod:
                self.logger.info(f"No lastmod date in database for {url}, update needed")
                return True
            
            # Parse ISO format dates and compare
            db_date = datetime.fromisoformat(db_lastmod.replace('Z', '+00:00'))
            sitemap_date = datetime.fromisoformat(lastmod.replace('Z', '+00:00'))
            
            if sitemap_date > db_date:
                self.logger.info(f"Update needed for {url}: sitemap lastmod {sitemap_date} is newer than database lastmod {db_date}")
                return True
            else:
                self.logger.info(f"Update not needed for {url}: database lastmod {db_date} is newer than or equal to sitemap lastmod {sitemap_date}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error checking update status for {url}: {e}")
            # On error, assume update is needed
            return True
    
    async def process_url(self, url_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a single URL.
        
        Args:
            url_data: Dictionary with URL data from sitemap
                - url: URL to process
                - lastmod: Last modification date (ISO format)
                - sitemap: URL of the sitemap that contained this URL
                
        Returns:
            Processing result with status information
        """
        url = url_data["url"]
        lastmod = url_data["lastmod"]
        
        self.logger.info(f"Processing URL: {url}")
        self.progress.url_started(url)
        
        try:
            # Check if update is needed
            if not await self.update_needed(url, lastmod):
                self.logger.info(f"Skipping {url} - already up to date")
                self.progress.url_completed(url, "skipped")
                return {"url": url, "status": "skipped", "message": "Already up to date"}
            
            # Get content type based on URL
            content_type = self.sitemap_parser.categorize_url(url)
            
            # Process URL content
            url_result = await self.content_extractor.process_url(url, content_type)
            
            # Check for errors in processing
            if "error" in url_result:
                self.logger.error(f"Failed to process {url}: {url_result['error']}")
                self.progress.url_completed(
                    url, 
                    "failed", 
                    http_code=url_result.get("http_status"),
                    error_type="http" if url_result.get("http_status", 0) >= 400 else "other"
                )
                return {
                    "url": url, 
                    "status": "failed", 
                    "message": url_result["error"],
                    "http_status": url_result.get("http_status")
                }
            
            # No chunks indicates a problem with content extraction
            if not url_result["chunks"]:
                self.logger.error(f"No content chunks generated for {url}")
                self.progress.url_completed(url, "failed", error_type="other")
                return {"url": url, "status": "failed", "message": "No content chunks generated"}
            
            # Delete any existing documents for this URL
            await self.db_manager.delete_documents_for_url(url)
            
            # First create/update the website data record to satisfy the foreign key constraint
            website_data = {
                "url": url,
                "service_name": url_result["title"],
                "sitemap_last_update": lastmod or datetime.now(timezone.utc).isoformat(),
                "last_information_update": datetime.now(timezone.utc).isoformat(),
                "type": content_type,
                "vector_store_row_ids": [],  # Initially empty, will be updated after document insertion
                "http_status": url_result["http_status"],
                "last_extracted": datetime.now(timezone.utc).isoformat()
            }
            
            # First insert/update the website data record
            if not self.db_manager.upsert_website_data(website_data):
                self.logger.error(f"Failed to create initial website data for {url}")
                self.progress.url_completed(url, "failed", error_type="database")
                return {"url": url, "status": "failed", "message": "Failed to create website data record"}
            
            # Now that the website_data record exists, we can insert the documents
            document_ids = []
            for chunk in url_result["chunks"]:
                doc_id = await self.db_manager.insert_document(
                    content=chunk["content"],
                    metadata=chunk["metadata"],
                    embedding=chunk["embedding"],
                    url=url
                )
                if doc_id:
                    document_ids.append(doc_id)
            
            # Update website data with document IDs if any were created
            if document_ids:
                website_data["vector_store_row_ids"] = document_ids
                
                if self.db_manager.upsert_website_data(website_data):
                    self.logger.info(f"Successfully processed {url} with {len(document_ids)} chunks")
                    self.progress.url_completed(url, "success", http_code=url_result["http_status"])
                    return {
                        "url": url, 
                        "status": "success", 
                        "chunks": len(document_ids),
                        "http_status": url_result["http_status"]
                    }
                else:
                    self.logger.error(f"Failed to update website data with document IDs for {url}")
                    self.progress.url_completed(url, "partial_success", error_type="database")
                    return {"url": url, "status": "partial_success", "message": "Documents created but failed to update website data"}
            else:
                self.logger.error(f"No document IDs generated for {url}")
                self.progress.url_completed(url, "failed", error_type="database")
                return {"url": url, "status": "failed", "message": "No documents created"}
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            self.progress.url_completed(url, "failed", error_type="other")
            return {"url": url, "status": "failed", "message": str(e)}
    
    async def process_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process a batch of URLs concurrently.
        
        Args:
            batch: List of URL data dictionaries
                
        Returns:
            List of processing results
        """
        tasks = [self.process_url(url_data) for url_data in batch]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the website vector storage processor.
        
        Returns:
            Processing statistics
        """
        self.logger.info(f"Starting website processing for {self.company_name}")
        
        # Set up database if needed
        if not self.db_manager.setup_database():
            self.logger.error("Database setup failed")
            return {"status": "failed", "message": "Database setup failed"}
        
        try:
            # Extract URLs from sitemap
            urls = await self.sitemap_parser.extract_urls(self.sitemap_url)
            
            if not urls:
                self.logger.error(f"No URLs found in sitemap: {self.sitemap_url}")
                return {"status": "failed", "message": "No URLs found in sitemap"}
            
            self.logger.info(f"Found {len(urls)} URLs in sitemap")
            
            # Initialize progress tracking
            self.progress.initialize(len(urls))
            
            # Process URLs in batches
            results = []
            batch_num = 0
            
            while batch_num * self.batch_size < len(urls):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, len(urls))
                
                batch = urls[start_idx:end_idx]
                self.logger.info(f"Processing batch {batch_num+1}/{(len(urls) + self.batch_size - 1) // self.batch_size} ({len(batch)} URLs)")
                
                batch_results = await self.process_batch(batch)
                results.extend(batch_results)
                
                # Small delay to prevent UI issues
                await asyncio.sleep(0.1)
                
                batch_num += 1
            
            # Finish progress tracking and get statistics
            stats = self.progress.finish()
            
            self.logger.info(f"Website processing completed for {self.company_name}")
            return {
                "status": "success",
                "urls_processed": len(results),
                "statistics": stats
            }
            
        except Exception as e:
            self.logger.error(f"Error in main execution: {str(e)}")
            if hasattr(self, 'progress') and self.progress:
                self.progress.finish()
            return {"status": "failed", "message": str(e)}

async def _run_cli():
    """CLI entry point for async execution"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process website content and store it in a vector database")
    
    # Company and website parameters
    parser.add_argument("--company", type=str, required=True, 
                        help="Company name (used for table naming)")
    parser.add_argument("--website", type=str, required=True, 
                        help="Website base URL (e.g., https://example.com)")
    parser.add_argument("--sitemap", type=str, required=True, 
                        help="Sitemap URL (e.g., https://example.com/sitemap.xml)")
    
    # Processing parameters
    parser.add_argument("--batch-size", type=int, default=5, 
                        help="Number of URLs to process concurrently")
    parser.add_argument("--max-urls", type=int, default=None, 
                        help="Maximum number of URLs to process")
    parser.add_argument("--include", type=str, nargs="*", default=None, 
                        help="URL patterns to include (regex)")
    parser.add_argument("--exclude", type=str, nargs="*", default=None, 
                        help="URL patterns to exclude (regex)")
    parser.add_argument("--types", type=str, nargs="*", default=None, 
                        help="Custom content types to add")
    
    # Database parameters
    parser.add_argument("--setup-database", action="store_true", 
                        help="Generate SQL setup script if tables don't exist")
    
    # Rate limiting parameters
    parser.add_argument("--mistral-rate-limit", type=int, default=15, 
                        help="Rate limit for MistralAI API calls per minute")
    parser.add_argument("--groq-rate-limit", type=int, default=20, 
                        help="Rate limit for Groq API calls per minute")
    
    # Content parameters
    parser.add_argument("--chunk-size", type=int, default=512, 
                        help="Size of text chunks for embedding")
    parser.add_argument("--chunk-overlap", type=int, default=50, 
                        help="Overlap between text chunks")
    
    # Other parameters
    parser.add_argument("--timeout", type=int, default=30, 
                        help="Timeout for HTTP requests in seconds")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Initialize and run the processor
    processor = WebsiteVectorStorage(
        company_name=args.company,
        website_url=args.website,
        sitemap_url=args.sitemap,
        batch_size=args.batch_size,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_urls=args.max_urls,
        setup_database=args.setup_database,
        custom_types=args.types,
        log_level=log_level,
        request_timeout=args.timeout,
        mistral_rate_limit=args.mistral_rate_limit,
        groq_rate_limit=args.groq_rate_limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Run the processor
    result = await processor.run()
    
    # Return success based on status
    return result["status"] == "success"

def main():
    """CLI entry point"""
    # Load environment variables
    load_dotenv()
    
    # Run the async main function
    success = asyncio.run(_run_cli())
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 