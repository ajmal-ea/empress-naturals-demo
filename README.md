# Express Analytics Chatbot

A versatile AI-powered chatbot for Express Analytics that can be embedded on any website or WordPress site.

## Project Structure Note

This project is organized into three main components:
1. **Backend**: A Python FastAPI backend (`backend/ea_chatbot_app.py`) that serves as the chatbot API
2. **Frontend**: A Node.js development server (`frontend/dev-server.js`) for testing the frontend interface
3. **WordPress**: A WordPress plugin (`wordpress/ea-chatbot.php`) for WordPress integration

These components are designed to work together but can be deployed separately as needed.

## Features

- AI-powered responses to questions about data analytics, marketing analytics, AI solutions, and business intelligence
- Multiple deployment options:
  - Standalone webpage
  - Embedded via iframe
  - WordPress plugin with floating widget
  - WordPress plugin (shortcode) for inline embedding
- Session management for conversation history
- Chat reset functionality to start a new conversation
- "Contact Us" button for scheduling meetings with the Express Analytics team
- Responsive design that works on desktop and mobile
- Markdown-like formatting support for rich text responses
- Customizable appearance and behavior

## Directory Structure

```
ea-website-rag-chatbot/
├── backend/                 # Python API server
│   ├── ea_chatbot.py        # Chatbot implementation
│   ├── ea_chatbot_app.py    # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── prometheus.yml       # Prometheus configuration
│   └── chatbot-dashboard.json # Grafana dashboard
├── frontend/                # Frontend interface
│   ├── static/              # Static assets
│   │   ├── index.html       # Main HTML file
│   │   ├── styles.css       # CSS styles
│   │   ├── script.js        # Core chatbot functionality
│   │   ├── config.js        # Configuration file
│   │   └── wordpress-embed.js # Script for WordPress embedding
│   ├── dev-server.js        # Development server
│   ├── test-embed.html      # Example of iframe embedding
│   └── package.json         # Node.js dependencies
├── wordpress/               # WordPress integration
│   └── ea-chatbot.php       # WordPress plugin file
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Backend Docker image
├── Dockerfile.frontend      # Frontend Docker image
├── entrypoint.sh            # Docker entrypoint script
└── package.json             # Root package.json for project management
```

## Development Setup

### Prerequisites

- Python 3.8+ (for the API server)
- Node.js v14+ (for the development server)
- npm v6+ (for the development server)
- Docker and Docker Compose (optional, for containerized deployment)

### Running the API Server

1. Install Python dependencies:
   ```
   npm run install:backend
   ```
   or directly:
   ```
   cd backend && pip install -r requirements.txt
   ```

2. Set up environment variables (or use the .env file in the backend directory):
   ```
   export SUPABASE_URL=your_supabase_url
   export SUPABASE_KEY=your_supabase_key
   export MISTRAL_API_KEY=your_mistral_api_key
   export GROQ_API_KEY=your_groq_api_key
   ```

3. Run the API server:
   ```
   npm run start:backend
   ```
   or directly:
   ```
   cd backend && python ea_chatbot_app.py
   ```
   
   The API server will be available at http://localhost:8000

### Running the Frontend Development Server

1. Install Node.js dependencies:
   ```
   npm run install:frontend
   ```
   or directly:
   ```
   cd frontend && npm install
   ```

2. Configure the API endpoint:
   - Open `frontend/static/config.js` and update the `apiBaseUrl` to point to your API server
   - By default, it points to `http://localhost:8000`

3. Start the development server:
   ```
   npm run start:frontend
   ```
   or directly:
   ```
   cd frontend && npm run start
   ```

   For production mode (listens on all interfaces):
   ```
   npm run start:frontend:prod
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:4000
   ```

### Windows-Specific Notes

If you're running on Windows, make sure to use the appropriate commands:

1. For the backend, you may need to set environment variables using:
   ```
   set SUPABASE_URL=your_supabase_url
   set SUPABASE_KEY=your_supabase_key
   set MISTRAL_API_KEY=your_mistral_api_key
   set GROQ_API_KEY=your_groq_api_key
   ```

2. The frontend scripts have been updated to work on both Windows and Unix-based systems.

### Using Docker Compose (Optional)

To run the entire stack (API, Frontend, Prometheus, and Grafana) using Docker:

```
npm run docker:up
```
or directly:
```
docker-compose up -d
```

This will start:
- The chatbot API at http://localhost:8000
- The frontend interface at http://localhost:4000
- Prometheus at http://localhost:9090
- Grafana at http://localhost:3000 (admin/admin)

To stop all services:
```
npm run docker:down
```

#### Docker Services

1. **chatbot-api**: The Python FastAPI backend
   - Built from `Dockerfile`
   - Exposes port 8000
   - Contains the chatbot logic and API endpoints

2. **chatbot-frontend**: The Node.js frontend server
   - Built from `Dockerfile.frontend`
   - Exposes port 4000
   - Serves the chatbot interface and test page

3. **prometheus**: Metrics collection
   - Uses the official Prometheus image
   - Exposes port 9090
   - Collects metrics from the API

4. **grafana**: Metrics visualization
   - Uses the official Grafana image
   - Exposes port 3000
   - Displays dashboards for monitoring the chatbot

#### Docker Networking

All services are connected through the `chatbot-network` bridge network, allowing them to communicate with each other using their service names as hostnames.

## Usage Options

### Option 1: Standalone Webpage

Simply host the files in the `frontend/static` directory on any web server. Users can access the chatbot by navigating to the `index.html` file.

### Option 2: Embed via iframe

You can embed the chatbot on any webpage using an iframe:

```html
<iframe src="path/to/static/index.html" width="100%" height="600px" frameborder="0"></iframe>
```

See `frontend/test-embed.html` for a complete example.

### Option 3: WordPress Plugin (Floating Widget)

1. Upload the `wordpress` directory to your WordPress plugins directory
2. Activate the "Express Analytics Chatbot" plugin in WordPress
3. Go to Settings > EA Chatbot to configure the chatbot
4. Set "Display Mode" to "Floating Widget"
5. Configure the API URL and other settings
6. Save changes

### Option 4: WordPress Plugin (Shortcode)

1. Follow steps 1-3 from Option 3
2. Set "Display Mode" to "Inline (via Shortcode)"
3. Use the `[ea_chatbot]` shortcode on any page or post
4. Optionally, you can specify dimensions: `[ea_chatbot height="400px" width="100%"]`

## Configuration

### API Endpoint

The chatbot needs to connect to your Express Analytics API. You can configure this in:

- For standalone/iframe: Edit the `apiBaseUrl` variable in `frontend/static/config.js`
- For WordPress: Use the plugin settings page

### Meeting Scheduler

The chatbot includes a "Contact Us" button that redirects users to a meeting scheduling service. You can configure this in:

- For standalone/iframe: Edit the `meetingSchedulerUrl` variable in `frontend/static/config.js`
- The backend also provides a `/meeting-config` endpoint that can override the frontend configuration

### WordPress Plugin Settings

The WordPress plugin provides the following settings:

- **API URL**: The URL of your chatbot API
- **Display Mode**: Choose between floating widget, inline shortcode, or disabled
- **Widget Position**: Left or right side of the screen (for floating widget)
- **Primary Color**: Main color for the chatbot interface
- **Chatbot Title**: Title displayed in the header
- **Welcome Message**: Initial message shown to users
- **Meeting Scheduler URL**: URL for the "Contact Us" button

## API Requirements

The chatbot expects the API to provide the following endpoints:

- `POST /chat`: Process a chat message
  - Request body: `{ "message": "user message", "session_id": "optional-session-id", "timezone": "UTC" }`
  - Response: `{ "response": "bot response", "timestamp": "ISO timestamp", "session_id": "session-id" }`

- `GET /history/{session_id}`: Retrieve chat history
  - Response: `{ "messages": [{ "role": "user|assistant", "content": "message content" }, ...] }`

- `GET /meeting-config`: Get meeting scheduler configuration
  - Response: `{ "url": "meeting-url", "provider": "service-provider", "name": "contact-name", "email": "contact-email", "title": "meeting-title", "description": "meeting-description" }`

## Troubleshooting

### CORS Issues

If you encounter CORS (Cross-Origin Resource Sharing) issues when testing locally, make sure your API server has CORS enabled. For development purposes, you can add the following headers to your API responses:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

### Rate Limit Handling

The chatbot API implements two approaches to handle rate limit errors when the conversation becomes too large:

1. **Approach 1: Dynamic Conversation Summarization**
   - When a rate limit error occurs, the system uses LangChain's summarization capabilities to create a concise summary of the conversation
   - The summarization uses a separate LLM call to generate a contextually relevant summary of the chat history
   - This preserves the important context and topics from the conversation while significantly reducing token usage
   - If summarization fails for any reason, the system falls back to a generic summary
   - The system logs when this approach is used and includes the generated summary in the logs

2. **Approach 2: History Truncation**
   - If summarization still results in rate limit errors, the system will limit the chat history to the last 2 exchanges
   - This ensures the most recent context is preserved while reducing token usage
   - The system logs when this approach is used

The frontend provides user-friendly error messages and visual cues when these limits are reached, suggesting the user reset their chat if needed.

### API Connection Issues

If the chatbot cannot connect to your API:

1. Check that the API server is running
2. Verify the API URL in `frontend/static/config.js` is correct
3. Ensure your API endpoints match the expected format
4. Check browser console for any error messages

## Browser Compatibility

The chatbot is compatible with:
- Chrome 60+
- Firefox 60+
- Safari 12+
- Edge 79+
- Opera 47+

## License

Copyright © 2025 Express Analytics. All rights reserved.

## Monitoring

The Express Analytics Chatbot includes a monitoring setup with Prometheus and Grafana.

### Prometheus

Prometheus is used to collect metrics from the chatbot API. It's configured to scrape metrics from the `/metrics` endpoint of the chatbot API.

- Prometheus UI: http://localhost:9090
- Metrics endpoint: http://localhost:8000/metrics

### Grafana

Grafana is used to visualize the metrics collected by Prometheus. It's pre-configured with a dashboard for the chatbot metrics.

- Grafana UI: http://localhost:3000
- Default credentials: admin/admin

The Express Analytics Chatbot Dashboard provides insights into:
- Chat request rates
- Active sessions
- Response times
- Error rates
- Message lengths
- Overall traffic

# Website Content Vector Store Generator

This tool allows you to scrape website content and store it in a Supabase vector database for use with AI applications. It automatically processes content from a sitemap, generates embeddings and summaries, and stores them in company-specific tables.

## Features

- **Company-specific Database Naming**: Automatically generates abbreviations for company names to create distinct database tables
- **Parameterized Execution**: Run the script with different company names, websites, and configuration
- **Progress Tracking**: Visual progress bar and summary statistics for the scraping process
- **Robust Error Handling**: Gracefully handles rate limits, timeouts, and API errors
- **Resumable Processing**: Skip already processed URLs that haven't changed
- **Automatic Database Setup**: Creates necessary Supabase tables, functions, and triggers

## Prerequisites

1. Supabase account with valid API key
2. Mistral AI API key for embeddings
3. Groq API key for summarization
4. Python 3.8+

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/website-vector-store.git
cd website-vector-store

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your API keys
touch .env
```

Add the following to your `.env` file:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
MISTRALAI_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
```

## Database Setup

**IMPORTANT**: Before running the script, you need to create the necessary tables in Supabase:

1. Go to your Supabase project dashboard
2. Navigate to the SQL Editor
3. Open the `db_setup.sql` file from this repository
4. Replace all instances of `COMPANY_ABBR` with your company abbreviation (e.g., "EA" for Express Analytics)
5. Execute the SQL script

This will create:
- The document table for storing content chunks with embeddings
- The website data table for storing metadata
- Functions and triggers for maintaining relationships between tables
- Indexes for efficient searching

You only need to do this setup once per company. The script will automatically use the correct tables based on the company name you provide.

## Usage

Run the script with default parameters (Express Analytics website):

```bash
python scripts/vector_storage.py
```

### Command Line Arguments

- `--company`: Company name (used for database naming)
- `--website`: Website base URL
- `--sitemap`: Sitemap URL
- `--batch-size`: Number of URLs to process concurrently

Example with custom parameters:

```bash
python scripts/vector_storage.py --company "Acme Corporation" --website "https://www.acme.com" --sitemap "https://www.acme.com/sitemap.xml" --batch-size 3
```

## How It Works

1. The script first abbreviates the company name to create database table names
2. It checks if the necessary database tables exist
3. It fetches the sitemap and extracts all URLs to process
4. For each URL:
   - It checks if the content needs updating based on sitemap lastmod date
   - It extracts content from the webpage
   - It generates a summary using Groq's LLM
   - It splits the content into chunks and generates embeddings using Mistral
   - It stores everything in the company-specific tables in Supabase
5. Progress is tracked and displayed throughout the process

## Database Structure

The script creates two main tables in Supabase:

1. `{COMPANY_ABBR}_documents`: Stores content chunks with embeddings
   - `id`: Primary key
   - `content`: Text content
   - `metadata`: JSON metadata
   - `embedding`: Vector embedding
   - `url`: Source URL

2. `{COMPANY_ABBR}_website_data`: Stores website metadata
   - `id`: Primary key
   - `url`: Website URL (primary key)
   - `service_name`: Title of the page
   - `sitemap_last_update`: Last modified date from sitemap
   - `last_information_update`: When the information was updated
   - `type`: Content type (Blog, Product, etc.)
   - `last_extracted`: When content was last extracted
   - `vector_store_row_ids`: Array of related document IDs

It also creates functions and triggers to keep these tables in sync.

## Troubleshooting

### Table creation errors

If you encounter errors about missing tables, make sure you've run the `db_setup.sql` script in the Supabase SQL Editor first. 

The Python Supabase client doesn't have functionality to directly execute SQL commands to create tables, so this step must be done manually before running the script.

### Vector extension not available

If your Supabase plan doesn't include the pgvector extension, the script will still work but will store embeddings as text strings instead of vectors. This means semantic search won't be available, but you can still store and retrieve documents.

To enable full vector functionality:
1. Upgrade to a Supabase plan that includes the pgvector extension
2. In the SQL Editor, run: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Run the `db_setup.sql` script again

## Rate Limiting and Error Handling

The script implements conservative rate limiting to avoid hitting API quotas:
- 15 calls per minute for Mistral embeddings
- 20 calls per minute for Groq summaries

If a rate limit is hit, the script will wait and retry automatically.

## Logging

Detailed logs are written to:
- Console output
- `website_scraper.log` file

## Contributing

Contributions are welcome! See the [todo.md](todo.md) file for planned improvements.

## License

MIT
