#!/bin/bash
set -e

# Make sure uvicorn is installed
pip install uvicorn

# Set default Phoenix environment variables if not already set
if [ -z "$PHOENIX_COLLECTOR_ENDPOINT" ]; then
  echo "Setting default Phoenix collector endpoint"
  export PHOENIX_COLLECTOR_ENDPOINT="http://phoenix:6006/v1/traces"
fi

if [ -z "$PHOENIX_PROJECT_NAME" ]; then
  echo "Setting default Phoenix project name"
  export PHOENIX_PROJECT_NAME="empress_naturals_chatbot"
fi

if [ -z "$PHOENIX_ENABLED" ]; then
  echo "Enabling Phoenix tracing by default"
  export PHOENIX_ENABLED="true"
fi

echo "Phoenix configuration:"
echo "PHOENIX_COLLECTOR_ENDPOINT: $PHOENIX_COLLECTOR_ENDPOINT"
echo "PHOENIX_PROJECT_NAME: $PHOENIX_PROJECT_NAME"
echo "PHOENIX_ENABLED: $PHOENIX_ENABLED"

# Run the API server
exec python -m uvicorn ea_chatbot_app:app --host 0.0.0.0 --port 8000 --reload-exclude="./traces/*"