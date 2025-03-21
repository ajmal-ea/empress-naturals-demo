# Import Libraries
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
import pandas as pd
from langchain_groq.chat_models import ChatGroq
from dateutil.parser import isoparse
from datetime import datetime, timezone
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from typing import Optional, List, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from ratelimit import limits, sleep_and_retry
import time

# Custom Modules
from utils import parseDate

load_dotenv(override=True)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebsiteContentProcessor:
    def __init__(self):
        load_dotenv()
        
        # Configure retry settings
        self.api_retry_config = {
            "stop": stop_after_attempt(3),
            "wait": wait_exponential(multiplier=1, min=4, max=10),
            "before_sleep": before_sleep_log(logger, logging.INFO),
            "retry": retry_if_exception_type((Exception))
        }
        
        # Initialize clients and models
        self.supabase_client = create_client(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_KEY")
        )
        
        self.summarizer = load_summarize_chain(
            ChatGroq(model_name="llama3-8b-8192"),
            chain_type="refine",
            verbose=True
        )
        
        self.mistral_embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=os.getenv("MISTRALAI_API_KEY")
        )
        
        self.vector_store = SupabaseVectorStore(
            client=self.supabase_client,
            embedding=self.mistral_embeddings,
            table_name="documents",
            query_name="match_documents"
        )

        # Rate limits
        self.MISTRAL_CALLS = 20  # Calls per minute (conservative)
        self.GROQ_CALLS = 25     # Calls per minute (conservative)
        self.last_mistral_call = 0
        self.last_groq_call = 0

    async def generate_embedding(self, text: str):
        """Generate embedding with rate limiting and retry logic"""
        @retry(**self.api_retry_config)
        async def _generate():
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_mistral_call < (60 / self.MISTRAL_CALLS):
                await asyncio.sleep((60 / self.MISTRAL_CALLS) - (current_time - self.last_mistral_call))
            
            result = self.mistral_embeddings.embed_query(text)
            self.last_mistral_call = time.time()
            return result
        return await _generate()

    async def generate_summary(self, docs: List[Document]) -> str:
        """Generate summary with rate limiting and retry logic"""
        @retry(**self.api_retry_config)
        async def _generate():
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_groq_call < (60 / self.GROQ_CALLS):
                await asyncio.sleep((60 / self.GROQ_CALLS) - (current_time - self.last_groq_call))
            
            result = self.summarizer.run(docs)
            self.last_groq_call = time.time()
            return result
        return await _generate()

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch URL content with retry logic"""
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        async def _fetch():
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
        try:
            return await _fetch()
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            raise

    def check_content_update_needed(self, url: str, last_sitemap_update: str) -> bool:
        """Check if content needs updating based on sitemap date"""
        try:
            result = self.supabase_client.table("website_data").select("*").eq("url", url).execute()
            if not result.data:
                return True
            
            existing_record = result.data[0]
            existing_update = isoparse(existing_record["sitemap_last_update"])
            new_update = isoparse(last_sitemap_update)
            
            return new_update > existing_update
        except Exception as e:
            logger.error(f"Error checking update status: {str(e)}")
            return True

    async def process_content(self, url: str, site_type: str, content_html: str) -> Dict:
        """Process webpage content"""
        soup = BeautifulSoup(content_html, features="html.parser")
        
        # Select content based on type
        if site_type == "Blog":
            content_div = soup.find("div", class_="blog-wrap")
        else:
            content_div = soup.find("div", class_="wpb-content-wrapper")
            
        if not content_div:
            raise ValueError(f"Content div not found for {url}")
            
        service_info = " ".join(line.strip() for line in content_div.text.split("\n\n"))
        service_name = soup.find("h1").text if soup.find("h1") else url
        
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
        """Update vector store and return document IDs"""
        try:
            # Delete existing vectors if any
            existing_docs = self.vector_store.similarity_search(url, k=100)
            existing_ids = [doc.metadata.get("id") for doc in existing_docs if doc.metadata.get("id") is not None]
            
            if existing_ids:
                # Delete existing documents in parallel
                delete_tasks = [
                    self.supabase_client.table("documents")
                    .delete()
                    .eq("id", doc_id)
                    .execute() 
                    for doc_id in existing_ids
                ]
                await asyncio.gather(*delete_tasks)
            
            # First ensure the website_data record exists
            try:
                self.supabase_client.table("website_data").upsert({
                    "url": url,
                    "type": site_type,
                    "vector_store_row_ids": []  # Initialize empty array
                }).execute()
            except Exception as e:
                logger.error(f"Error upserting website_data for {url}: {str(e)}")
                raise
            
            # Create new document
            doc = Document(
                page_content=f"""Source: {url}
Type: {site_type}

Website Information: {content}

{summary}
""",
                metadata={
                    "source": url,
                    "type": site_type,
                    "url": url
                }
            )
            
            # Split and store new documents
            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents([doc])
            
            logger.info(f"Length of Split Docs: {len(split_docs)}")
            
            # Add documents directly to Supabase with retry
            doc_ids = []
            for split_doc in split_docs:
                try:
                    # Generate embedding with retry
                    embedding = await self.generate_embedding(split_doc.page_content)
                    
                    # Insert document with retry
                    @retry(**self.api_retry_config)
                    async def insert_doc():
                        result = self.supabase_client.table("documents").insert({
                            "content": split_doc.page_content,
                            "metadata": split_doc.metadata,
                            "embedding": embedding,
                            "url": url
                        }).execute()
                        return result
                    
                    result = await insert_doc()
                    if result.data:
                        doc_ids.append(result.data[0]["id"])
                except Exception as e:
                    logger.error(f"Error processing document chunk for {url}: {str(e)}")
                    raise
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error updating vector store for {url}: {str(e)}")
            raise

    async def process_website(self, url: str, last_sitemap_update: str) -> None:
        """Process a single website"""
        # Determine site type
        if url.startswith("https://www.expressanalytics.com/product/"):
            site_type = "Product"
        elif url.startswith("https://www.expressanalytics.com/solutions/"):
            site_type = "Solution"
        elif url.startswith("https://www.expressanalytics.com/blog/"):
            site_type = "Blog"
        else:
            logger.info(f"Skipping unsupported URL: {url}")
            return

        if not self.check_content_update_needed(url, last_sitemap_update):
            logger.info(f"Content up to date for {url}")
            return

        async with aiohttp.ClientSession() as session:
            try:
                content_html = await self.fetch_url(session, url)
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
                self.supabase_client.table("website_data").upsert({
                    "url": url,
                    "service_name": processed_content["service_name"],
                    "sitemap_last_update": last_sitemap_update,
                    "last_information_update": current_datetime.isoformat(),
                    "type": site_type,
                    "last_extracted": current_datetime.isoformat(),
                    "vector_store_row_ids": vector_ids
                }).execute()
                
                logger.info(f"Successfully processed {url}")
                
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                raise

    async def main(self):
        """Main execution function"""
        try:
            print("MISTRALAI_API_KEY: ", os.getenv("MISTRALAI_API_KEY"))
            print("SUPABASE_KEY: ", os.getenv("SUPABASE_KEY"))
            # Fetch main sitemap
            async with aiohttp.ClientSession() as session:
                sitemap_content = await self.fetch_url(
                    session, 
                    "https://www.expressanalytics.com/sitemap.xml"
                )
                
            # Parse sub-sitemaps
            soup = BeautifulSoup(sitemap_content, features="xml")
            sub_sitemaps = [
                sitemap.loc.text 
                for sitemap in soup.find_all("sitemap")
            ]
            
            # Process each sub-sitemap
            for sub_sitemap in sub_sitemaps:
                async with aiohttp.ClientSession() as session:
                    sub_content = await self.fetch_url(session, sub_sitemap)
                    
                sub_soup = BeautifulSoup(sub_content, features="xml")
                pages = sub_soup.find_all("url")
                
                # Process pages concurrently
                tasks = []
                for page in pages:
                    url = page.loc.text
                    last_mod = page.lastmod.text
                    tasks.append(self.process_website(url, last_mod))
                
                await asyncio.gather(*tasks)
                
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
            raise

if __name__ == "__main__":
    processor = WebsiteContentProcessor()
    asyncio.run(processor.main())

