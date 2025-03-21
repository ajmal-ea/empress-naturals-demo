"""
Content extraction and processing for RAG tools.
"""
import logging
import asyncio
import aiohttp
import time
from typing import Dict, Any, Optional, List, Tuple
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from datetime import datetime, timezone
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain

logger = logging.getLogger("rag_tools")

class ContentExtractor:
    """
    Extract and process content from web pages.
    """
    
    def __init__(
        self,
        mistral_api_key: str,
        groq_api_key: str,
        request_timeout: int = 30,
        mistral_rate_limit: int = 15,  # Calls per minute
        groq_rate_limit: int = 20,     # Calls per minute
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the content extractor.
        
        Args:
            mistral_api_key: MistralAI API key
            groq_api_key: Groq API key
            request_timeout: Timeout for HTTP requests in seconds
            mistral_rate_limit: Rate limit for MistralAI API calls per minute
            groq_rate_limit: Rate limit for Groq API calls per minute
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between text chunks
        """
        self.request_timeout = request_timeout
        
        # Initialize embedding model
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=mistral_api_key
        )
        
        # Initialize summarizer
        self.summarizer = load_summarize_chain(
            llm=ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192",
                groq_api_key=groq_api_key
            ),
            chain_type="refine"
        )
        
        # Rate limiting
        self.mistral_rate_limit = mistral_rate_limit
        self.groq_rate_limit = groq_rate_limit
        self.last_mistral_call = 0
        self.last_groq_call = 0
        
        # Text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, int]:
        """
        Fetch URL content with retry logic.
        
        Args:
            session: aiohttp client session
            url: URL to fetch
            
        Returns:
            Tuple of (content, status_code)
        """
        try:
            async with session.get(url, timeout=self.request_timeout) as response:
                content = await response.text()
                return content, response.status
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return "", 408  # Request Timeout
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return "", 500  # Internal Server Error
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a text with rate limiting.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_mistral_call
        required_wait = (60 / self.mistral_rate_limit)
        
        if elapsed < required_wait:
            wait_time = required_wait - elapsed
            logger.debug(f"Rate limiting: Waiting {wait_time:.2f}s for Mistral API")
            await asyncio.sleep(wait_time)
        
        try:
            embedding = self.embeddings.embed_query(text)
            self.last_mistral_call = time.time()
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Handle rate limit errors specifically
            if "rate limit" in str(e).lower():
                logger.warning("Mistral API rate limit hit, waiting 60 seconds")
                await asyncio.sleep(60)
                # Retry once after waiting
                embedding = self.embeddings.embed_query(text)
                self.last_mistral_call = time.time()
                return embedding
            raise
    
    async def generate_summary(self, text: str) -> str:
        """
        Generate a summary for a text with rate limiting.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_groq_call
        required_wait = (60 / self.groq_rate_limit)
        
        if elapsed < required_wait:
            wait_time = required_wait - elapsed
            logger.debug(f"Rate limiting: Waiting {wait_time:.2f}s for Groq API")
            await asyncio.sleep(wait_time)
        
        try:
            # Create a document for the summarizer
            doc = Document(page_content=text)
            
            # Generate summary
            summary = self.summarizer.run([doc])
            self.last_groq_call = time.time()
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Handle rate limit errors specifically
            if "rate limit" in str(e).lower():
                logger.warning("Groq API rate limit hit, waiting 60 seconds")
                await asyncio.sleep(60)
                # Retry once after waiting
                doc = Document(page_content=text)
                summary = self.summarizer.run([doc])
                self.last_groq_call = time.time()
                return summary
            raise
    
    def extract_content(self, html: str, content_type: str) -> Dict[str, Any]:
        """
        Extract content from HTML based on content type.
        
        Args:
            html: HTML content
            content_type: Type of content (Blog, Product, etc.)
            
        Returns:
            Dictionary with extracted content:
            - title: Page title
            - content: Main content text
            - metadata: Additional metadata
        """
        try:
            soup = BeautifulSoup(html, features="html.parser")
            
            # Extract title
            title = ""
            h1_tag = soup.find("h1")
            if h1_tag:
                title = h1_tag.text.strip()
            else:
                meta_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "title"})
                if meta_title and meta_title.get("content"):
                    title = meta_title["content"].strip()
                else:
                    title_tag = soup.find("title")
                    if title_tag:
                        title = title_tag.text.strip()
            
            # Extract main content based on content type
            main_content = ""
            content_div = None
            
            if content_type == "Blog":
                # Try common blog content selectors
                selectors = [
                    "article", 
                    "div.blog-content", 
                    "div.post-content", 
                    "div.blog-wrap", 
                    "div.entry-content",
                    "main"
                ]
                
                for selector in selectors:
                    content_div = soup.select_one(selector)
                    if content_div and len(content_div.text.strip()) > 200:
                        break
            else:
                # Try common content selectors for other types
                selectors = [
                    "div.content", 
                    "div.main-content", 
                    "div.product-content", 
                    "div.wpb-content-wrapper", 
                    "main",
                    "div#content"
                ]
                
                for selector in selectors:
                    content_div = soup.select_one(selector)
                    if content_div and len(content_div.text.strip()) > 200:
                        break
            
            # If we couldn't find content with specific selectors, try more generic approach
            if not content_div or len(content_div.text.strip()) < 200:
                # Try to find the element with the most paragraph tags
                paragraphs_count = {}
                for div in soup.find_all(["div", "article", "section", "main"]):
                    p_count = len(div.find_all("p", recursive=False))
                    if p_count > 0:
                        paragraphs_count[div] = p_count
                
                if paragraphs_count:
                    content_div = max(paragraphs_count.items(), key=lambda x: x[1])[0]
                else:
                    # Fall back to body
                    content_div = soup.find("body")
            
            # Extract text from content div
            if content_div:
                # Remove navigation, headers, footers, etc.
                for element in content_div.find_all(["nav", "header", "footer", "aside", "script", "style"]):
                    element.decompose()
                
                # Get text from paragraphs and headings
                main_content = " ".join([
                    p.text.strip() for p in content_div.find_all(["p", "h2", "h3", "h4", "h5", "h6", "li"]) 
                    if p.text.strip()
                ])
                
                # If we don't have much content from paragraphs, fall back to all text
                if len(main_content) < 200:
                    main_content = content_div.text.strip()
                    
                    # Clean up whitespace
                    main_content = " ".join([line.strip() for line in main_content.splitlines() if line.strip()])
            
            # Extract metadata
            metadata = {}
            
            # Get meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                metadata["description"] = meta_desc["content"].strip()
            
            # Get meta keywords
            meta_keywords = soup.find("meta", attrs={"name": "keywords"})
            if meta_keywords and meta_keywords.get("content"):
                metadata["keywords"] = meta_keywords["content"].strip()
            
            # Get publish date for blogs
            if content_type == "Blog":
                # Look for time tags
                time_tag = soup.find("time")
                if time_tag and time_tag.get("datetime"):
                    metadata["publish_date"] = time_tag["datetime"]
                else:
                    # Look for common date containers
                    date_selectors = [
                        "span.date", 
                        "div.date", 
                        "p.date", 
                        "span.post-date", 
                        "div.post-date"
                    ]
                    
                    for selector in date_selectors:
                        date_el = soup.select_one(selector)
                        if date_el:
                            metadata["publish_date"] = date_el.text.strip()
                            break
            
            # Normalize content
            if main_content:
                # Limit content length and clean up whitespace
                main_content = " ".join(main_content.split())
                
                # Truncate if extremely long
                if len(main_content) > 10000:
                    main_content = main_content[:10000] + "..."
            
            return {
                "title": title,
                "content": main_content,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return {"title": "", "content": "", "metadata": {}}
    
    async def process_url(
        self, 
        url: str, 
        content_type: str
    ) -> Dict[str, Any]:
        """
        Process a URL by fetching content, extracting information, and generating embeddings.
        
        Args:
            url: URL to process
            content_type: Type of content (Blog, Product, etc.)
            
        Returns:
            Dictionary with processed data:
            - url: Original URL
            - title: Page title
            - content: Extracted content
            - summary: Generated summary
            - http_status: HTTP status code
            - chunks: List of content chunks with embeddings
        """
        logger.info(f"Processing URL: {url}")
        
        async with aiohttp.ClientSession() as session:
            # Fetch URL content
            html, status_code = await self.fetch_url(session, url)
            
            if not html or status_code >= 400:
                logger.error(f"Failed to fetch content for {url} with status code {status_code}")
                return {
                    "url": url,
                    "title": "",
                    "content": "",
                    "summary": "",
                    "http_status": status_code,
                    "chunks": [],
                    "error": f"Failed to fetch content with status code {status_code}"
                }
            
            # Extract content from HTML
            extracted = self.extract_content(html, content_type)
            
            # If we didn't get any content, return error
            if not extracted["content"]:
                logger.error(f"Failed to extract content from {url}")
                return {
                    "url": url,
                    "title": extracted["title"],
                    "content": "",
                    "summary": "",
                    "http_status": status_code,
                    "chunks": [],
                    "error": "Failed to extract meaningful content"
                }
            
            # Generate summary
            try:
                summary = await self.generate_summary(extracted["content"])
            except Exception as e:
                logger.error(f"Error generating summary for {url}: {str(e)}")
                summary = ""
            
            # Split content into chunks
            chunks = []
            texts = self.text_splitter.split_text(extracted["content"])
            
            # Generate embeddings for each chunk
            for i, chunk_text in enumerate(texts):
                try:
                    # Create chunk metadata
                    chunk_metadata = {
                        "chunk_index": i,
                        "source": url,
                        "title": extracted["title"],
                        "type": content_type
                    }
                    
                    # Add extracted metadata
                    chunk_metadata.update(extracted["metadata"])
                    
                    # Generate embedding
                    embedding = await self.generate_embedding(chunk_text)
                    
                    # Add chunk to results
                    chunks.append({
                        "content": chunk_text,
                        "embedding": embedding,
                        "metadata": chunk_metadata
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i} for {url}: {str(e)}")
            
            logger.info(f"Successfully processed {url} with {len(chunks)} chunks")
            
            return {
                "url": url,
                "title": extracted["title"],
                "content": extracted["content"],
                "summary": summary,
                "http_status": status_code,
                "chunks": chunks,
                "processed_at": datetime.now(timezone.utc).isoformat()
            } 