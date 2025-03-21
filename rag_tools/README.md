# RAG Tools - Website Vector Storage

A robust tool for processing website content and storing it in a vector database for RAG (Retrieval-Augmented Generation) applications.

## Features

- Extracts content from websites using sitemap.xml
- Processes content with intelligent extraction based on page type
- Generates embeddings using Mistral AI
- Summarizes content using Groq LLM
- Stores content and embeddings in Supabase vector database
- Tracks processing progress with a rich UI
- Handles errors gracefully with retry logic
- Supports rate limiting for API calls
- Provides detailed logging and statistics

## Installation

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
MISTRALAI_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Basic Usage

```bash
python -m rag_tools.vector_storage --company "Company Name" --website "https://example.com" --sitemap "https://example.com/sitemap.xml" --setup-database
```

### Command Line Arguments

- `--company`: Company name (used for table naming)
- `--website`: Website base URL (e.g., https://example.com)
- `--sitemap`: Sitemap URL (e.g., https://example.com/sitemap.xml)
- `--batch-size`: Number of URLs to process concurrently (default: 5)
- `--max-urls`: Maximum number of URLs to process (default: unlimited)
- `--include`: URL patterns to include (regex)
- `--exclude`: URL patterns to exclude (regex)
- `--types`: Custom content types to add
- `--setup-database`: Generate SQL setup script if tables don't exist
- `--mistral-rate-limit`: Rate limit for MistralAI API calls per minute (default: 15)
- `--groq-rate-limit`: Rate limit for Groq API calls per minute (default: 20)
- `--chunk-size`: Size of text chunks for embedding (default: 512)
- `--chunk-overlap`: Overlap between text chunks (default: 50)
- `--timeout`: Timeout for HTTP requests in seconds (default: 30)
- `--verbose`: Enable verbose logging

### Examples

Process only blog posts:
```bash
python -m rag_tools.vector_storage --company "Company Name" --website "https://example.com" --sitemap "https://example.com/sitemap.xml" --include ".*blog.*"
```

Process with custom rate limits:
```bash
python -m rag_tools.vector_storage --company "Company Name" --website "https://example.com" --sitemap "https://example.com/sitemap.xml" --mistral-rate-limit 10 --groq-rate-limit 15
```

Process with custom chunk size:
```bash
python -m rag_tools.vector_storage --company "Company Name" --website "https://example.com" --sitemap "https://example.com/sitemap.xml" --chunk-size 1024 --chunk-overlap 100
```

## Database Setup

Since Supabase does not allow table creation via the API, the tool will generate a SQL script that you need to execute manually in the Supabase SQL Editor. When you run the tool with the `--setup-database` flag, it will:

1. Check if the required tables exist
2. If not, generate a SQL script and save it to a file
3. Prompt you to execute the SQL in the Supabase SQL Editor
4. Wait for your confirmation before continuing

## Using as a Library

You can also use the tool as a library in your Python code:

```python
from rag_tools import WebsiteVectorStorage
import asyncio

async def process_website():
    processor = WebsiteVectorStorage(
        company_name="Company Name",
        website_url="https://example.com",
        sitemap_url="https://example.com/sitemap.xml",
        setup_database=True
    )
    
    result = await processor.run()
    print(f"Processing completed with status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Processed {result['urls_processed']} URLs")
        print(f"Success rate: {result['statistics']['successful_urls'] / result['statistics']['processed_urls'] * 100:.2f}%")

# Run the async function
asyncio.run(process_website())
```

## License

MIT 