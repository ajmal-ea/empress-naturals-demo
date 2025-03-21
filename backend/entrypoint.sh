#!/bin/bash
set -e

# Make sure uvicorn is installed
pip install uvicorn

# Run the API server
exec python -m uvicorn ea_chatbot_app:app --host 0.0.0.0 --port 8000