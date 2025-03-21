"""
Sitemap parsing and URL extraction for RAG tools.
"""
import logging
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Optional, Set
from bs4 import BeautifulSoup
import re
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from rag_tools.utils import parse_date

logger = logging.getLogger("rag_tools")

class SitemapParser:
    """
    Parse sitemap XML files to extract URLs and metadata.
    """
    
    def __init__(
        self,
        base_url: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_urls: int = None,
        request_timeout: int = 30
    ):
        """
        Initialize the sitemap parser.
        
        Args:
            base_url: Base URL of the website
            include_patterns: URL patterns to include (regex strings)
            exclude_patterns: URL patterns to exclude (regex strings)
            max_urls: Maximum number of URLs to return (None for unlimited)
            request_timeout: Timeout for HTTP requests in seconds
        """
        self.base_url = base_url.rstrip("/")
        
        # Normalize base URL for matching
        self.normalized_base_url = self.base_url.replace("https://www.", "https://")
        if self.normalized_base_url.startswith("http://www."):
            self.normalized_base_url = "https://" + self.normalized_base_url[11:]
        elif self.normalized_base_url.startswith("http://"):
            self.normalized_base_url = "https://" + self.normalized_base_url[7:]
            
        # Compile include/exclude patterns
        self.include_patterns = [re.compile(pattern) for pattern in include_patterns] if include_patterns else []
        self.exclude_patterns = [re.compile(pattern) for pattern in exclude_patterns] if exclude_patterns else []
        
        self.max_urls = max_urls
        self.request_timeout = request_timeout
    
    def _should_include_url(self, url: str) -> bool:
        """
        Check if a URL should be included based on patterns.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be included, False otherwise
        """
        # Always exclude non-matching base URL
        normalized_url = url.replace("https://www.", "https://")
        if not normalized_url.startswith(self.normalized_base_url):
            return False
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.search(url):
                return False
        
        # If include patterns are specified, at least one must match
        if self.include_patterns:
            for pattern in self.include_patterns:
                if pattern.search(url):
                    return True
            return False
        
        # If we get here, include the URL
        return True
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, int]:
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
    
    async def _parse_sitemap_content(
        self,
        session: aiohttp.ClientSession,
        content: str,
        url: str,
        processed_urls: Set[str]
    ) -> List[Tuple[str, Optional[str], str]]:
        """
        Parse sitemap content and extract URLs.
        
        Args:
            session: aiohttp client session
            content: Sitemap XML content
            url: URL of the sitemap
            processed_urls: Set of already processed sitemap URLs
            
        Returns:
            List of tuples (url, lastmod, sitemap_url)
        """
        results = []
        
        try:
            # Parse XML content
            soup = BeautifulSoup(content, features="xml")
            
            # Check if this is a sitemap index (contains other sitemaps)
            sitemap_tags = soup.find_all("sitemap")
            
            if sitemap_tags:
                # This is a sitemap index
                subsitemap_urls = []
                
                for sitemap_tag in sitemap_tags:
                    loc_tag = sitemap_tag.find("loc")
                    
                    if loc_tag and loc_tag.text:
                        subsitemap_url = loc_tag.text.strip()
                        
                        if subsitemap_url not in processed_urls:
                            processed_urls.add(subsitemap_url)
                            subsitemap_urls.append(subsitemap_url)
                
                # Process sub-sitemaps concurrently
                tasks = []
                for subsitemap_url in subsitemap_urls:
                    tasks.append(self._process_sitemap(session, subsitemap_url, processed_urls))
                
                # Wait for all sub-sitemaps to be processed
                subsitemap_results = await asyncio.gather(*tasks)
                
                # Combine results
                for subresult in subsitemap_results:
                    results.extend(subresult)
            else:
                # This is a regular sitemap with URLs
                url_tags = soup.find_all("url")
                
                for url_tag in url_tags:
                    loc_tag = url_tag.find("loc")
                    lastmod_tag = url_tag.find("lastmod")
                    
                    if loc_tag and loc_tag.text:
                        page_url = loc_tag.text.strip()
                        lastmod = lastmod_tag.text.strip() if lastmod_tag else None
                        
                        if self._should_include_url(page_url):
                            results.append((page_url, lastmod, url))
        except Exception as e:
            logger.error(f"Error parsing sitemap {url}: {str(e)}")
        
        return results
    
    async def _process_sitemap(
        self,
        session: aiohttp.ClientSession,
        sitemap_url: str,
        processed_urls: Set[str]
    ) -> List[Tuple[str, Optional[str], str]]:
        """
        Process a sitemap and extract URLs.
        
        Args:
            session: aiohttp client session
            sitemap_url: URL of the sitemap
            processed_urls: Set of already processed sitemap URLs
            
        Returns:
            List of tuples (url, lastmod, sitemap_url)
        """
        logger.info(f"Processing sitemap: {sitemap_url}")
        
        # Fetch sitemap content
        content, status_code = await self._fetch_url(session, sitemap_url)
        
        if not content or status_code >= 400:
            logger.error(f"Failed to fetch sitemap {sitemap_url} with status code {status_code}")
            return []
        
        # Parse sitemap content
        return await self._parse_sitemap_content(session, content, sitemap_url, processed_urls)
    
    async def extract_urls(self, sitemap_url: str) -> List[Dict[str, str]]:
        """
        Extract URLs from a sitemap.
        
        Args:
            sitemap_url: URL of the sitemap
            
        Returns:
            List of URL dictionaries with keys:
            - url: URL of the page
            - lastmod: Last modification date (ISO format)
            - sitemap: URL of the sitemap that contained this URL
        """
        logger.info(f"Extracting URLs from sitemap: {sitemap_url}")
        
        async with aiohttp.ClientSession() as session:
            results = await self._process_sitemap(session, sitemap_url, set([sitemap_url]))
            
            # Convert results to dictionaries
            urls = []
            
            for url, lastmod, sitemap in results:
                url_data = {
                    "url": url,
                    "lastmod": lastmod,
                    "sitemap": sitemap
                }
                
                urls.append(url_data)
                
                # Apply max_urls limit if specified
                if self.max_urls and len(urls) >= self.max_urls:
                    logger.info(f"Reached maximum URL limit of {self.max_urls}")
                    break
            
            logger.info(f"Extracted {len(urls)} URLs from sitemap")
            return urls
    
    def categorize_url(self, url: str) -> str:
        """
        Categorize a URL based on its path.
        
        Args:
            url: URL to categorize
            
        Returns:
            Category string
        """
        # Remove base URL and query parameters to get the path
        path = url.replace(self.base_url, "").split("?")[0].strip("/")
        
        # Check for common patterns
        if '/product/' in url or '/products/' in url:
            return 'product'
        elif '/solution/' in url or '/solutions/' in url:
            return 'other'  # Convert "Solution" to "other" as it's not in allowed types
        elif '/blog/' in url or '/blogs/' in url or '/news/' in url:
            return 'blog'
        elif '/service/' in url or '/services/' in url:
            return 'service'
        elif '/about/' in url or '/about-us/' in url:
            return 'about'
        elif '/contact/' in url or '/contact-us/' in url:
            return 'contact'
        elif '/collections/' in url:
            return 'other'  # Collections pages categorized as "other"
        elif '/pages/' in url:
            # Try to categorize pages based on common names
            if 'faq' in path.lower():
                return 'faq'
            elif 'terms' in path.lower() or 'tos' in path.lower():
                return 'terms'
            elif 'privacy' in path.lower():
                return 'privacy'
            elif 'about' in path.lower():
                return 'about'
            elif 'contact' in path.lower():
                return 'contact'
            else:
                return 'article'  # Default for other pages
        elif not path:  # Empty path means it's the homepage
            return 'home'
        
        # Default category
        return "other" 