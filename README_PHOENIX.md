# Arize Phoenix Integration

This document describes how Empress Naturals Chatbot integrates with Arize Phoenix for AI observability and LLM evaluation.

## Overview

Arize Phoenix provides comprehensive observability and evaluation for LLM applications. In our application, Phoenix is used for:

1. **Tracing LLM interactions** - Track every user query, model response, and retrieval operation
2. **Monitoring performance metrics** - Token usage, latency, response quality
3. **Evaluating model outputs** - Analyze hallucinations, bias, and other quality issues
4. **Debugging issues** - Identify problematic patterns in user-model interactions
5. **Export for retraining** - Export data for model improvements

## Architecture

The integration follows this architecture:

```
User Query → FastAPI → OpenTelemetry → Phoenix → LLM → Response
```

Key components:
- `phoenix_otel.py` - OpenTelemetry-based tracing (decorators and context managers)
- `phoenix_evals.py` - Evaluation framework for assessing LLM responses
- Phoenix Server - Collector and UI for traces and evaluations

## Docker Deployment

We use the official Phoenix Docker image in our composition:

```yaml
# Phoenix service using official Docker image
phoenix:
  image: arizephoenix/phoenix:latest
  ports:
    - "6006:6006"  # UI and OTLP HTTP collector
    - "4317:4317"  # OTLP gRPC collector
  environment:
    - PHOENIX_WORKING_DIR=/mnt/data
    - PHOENIX_SQL_DATABASE_URL=sqlite:////mnt/data/phoenix.db
  volumes:
    - phoenix_data:/mnt/data
```

Our backend service connects to Phoenix via OpenTelemetry:

```yaml
chatbot-api:
  # ...
  environment:
    # Phoenix configuration
    - PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces
    - PHOENIX_ENABLED=true
  depends_on:
    - phoenix
```

## Usage in Development

When running locally:

1. Start the application using docker-compose:
   ```bash
   docker-compose up -d
   ```

2. Access the Phoenix UI at [http://localhost:6006](http://localhost:6006)

## Usage in Code

Our application uses OpenTelemetry-based tracing through decorators and context managers:

```python
# Decorate functions for automatic tracing
@trace_function(name="function_name")
def my_function():
    # Function will be traced automatically
    
# Use context managers for blocks of code
with trace_context("operation_name", {"attribute": "value"}):
    # Code here will be traced
    
# Manually record LLM interactions
trace_llm_call(
    prompt="Your prompt",
    model="model_name",
    response="LLM response",
    latency=1.25,
    token_metrics={
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150
    }
)
```

## Evaluations

We use Phoenix's evaluation framework to assess the quality of LLM responses:

```python
# Run standard evaluations on traces
results = phoenix_evaluator.run_standard_evals()

# Generate a summary report
summary = phoenix_evaluator.generate_eval_report()
```

Available evaluations include:
- **Factuality** - Accuracy of information
- **Relevance** - Relevance to user query
- **Completeness** - Thoroughness of response
- **Toxicity** - Detection of harmful content
- **Helpfulness** - Overall usefulness to the user

## Production Integration with Arize

For production, you can connect to the Arize platform:

1. Sign up at [https://app.arize.com/](https://app.arize.com/)
2. Create an API key
3. Configure environment variables:
   ```yaml
   environment:
     - ARIZE_API_KEY=your_api_key
     - ARIZE_SPACE_KEY=your_space_key
     - PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com
     - PHOENIX_CLIENT_HEADERS=api_key=${ARIZE_API_KEY}
   ```

## Available Metrics

The Phoenix integration captures:

- **Token usage** - Input, output, and total tokens
- **Latency** - Response time for LLM calls
- **Response length** - Size of model responses
- **Error rates** - Tracking of failures and issues
- **Session information** - User session context

## Data Export for Retraining

To export data for retraining:

1. Use the Phoenix UI to select relevant data points
2. Export from the UI to a local file
3. For production data from Arize:
   ```python
   from arize.exporter import ArizeExportClient
   from arize.utils.types import Environments

   client = ArizeExportClient()
   df = client.export_model_to_df(
       space_id='your_space_id',
       model_name='empress_naturals_llm',
       environment=Environments.PRODUCTION,
       start_time=datetime.fromisoformat('2023-02-11T07:00:00.000+00:00'),
       end_time=datetime.fromisoformat('2023-03-14T00:59:59.999+00:00'),
   )
   ```

## Resources

- [Arize Phoenix Documentation](https://docs.arize.com/phoenix)
- [Phoenix Docker Deployment](https://docs.arize.com/phoenix/deployment/docker)
- [Phoenix Tracing Guide](https://docs.arize.com/phoenix/tracing/llm-traces-1/quickstart-tracing-python)
- [Production Observability with Arize](https://docs.arize.com/arize)
- [Phoenix Slack Community](https://join.slack.com/t/arize-ai/shared_invite/zt-1ntwku49s-JwU5_TdNQPujT_oPyB_Gfg) 