# Stock Agent

A conversational AI agent for stock data analysis, built with LangGraph and deployed on Databricks Apps. The agent uses Google Gemini (via Nexus) as the LLM and connects to Databricks MCP servers for stock data tools.

## Architecture

- **LLM**: Google Gemini 3.1 Pro Preview (via Nexus API gateway)
- **Framework**: LangGraph + MLflow Responses API
- **Tools**:
  - **Genie Space** (`stock-agent-genie-space`) — natural language queries over stock data
  - **Unity Catalog Functions** (`workspace.stock`) — stock-related UC functions
- **Deployment**: Databricks Apps with built-in chat UI
- **Tracing**: MLflow autologging

## Project Structure

```
agent_server/
  agent.py           # Agent logic, model config, MCP tool setup
  start_server.py    # FastAPI server + MLflow setup
  utils.py           # Stream processing, auth helpers
  evaluate_agent.py  # Agent evaluation with MLflow scorers
scripts/
  quickstart.py      # One-command setup
  start_app.py       # Local dev launcher
  discover_tools.py  # Discover workspace resources
app.yaml             # Databricks Apps config (for UI deploy)
databricks.yml       # Databricks Asset Bundles config (for CLI deploy)
```

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [nvm](https://github.com/nvm-sh/nvm) (Node version manager, for chat UI)
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)

## Setup

1. **Clone and configure environment**

   ```bash
   cp .env.example .env
   ```

   Fill in your `.env`:
   ```
   DATABRICKS_CONFIG_PROFILE=DEFAULT
   MLFLOW_EXPERIMENT_ID=<your-experiment-id>
   NEXUS_BASE_URL=https://genai-nexus.int.api.corpinter.net
   NEXUS_API_KEY=<your-nexus-api-key>
   ```

2. **Authenticate with Databricks**

   ```bash
   databricks auth login
   ```

3. **Run quickstart** (handles experiment setup + starts the app)

   ```bash
   uv run quickstart
   ```

## Running Locally

```bash
uv run start-app
```

The agent server and chat UI start at http://localhost:8000.

**Server options:**

```bash
uv run start-server --reload    # Hot-reload on code changes
uv run start-server --port 8001 # Custom port
uv run start-server --workers 4 # Multiple workers
```

**Test via API:**

```bash
# Streaming
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "What is the current price of AAPL?" }], "stream": true }'

# Non-streaming
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "What is the current price of AAPL?" }] }'
```

## Deploying to Databricks Apps

### Via Databricks UI

1. Create an app in the Databricks UI
2. Add resources:
   - **MLflow Experiment** — for tracing
   - **Genie Space** (`01f11baeb79e12de8ce1e5bbdbe2b6a1`) — `CAN_RUN`
   - **Secret** (`nexus`) — scope: `llm-test`, key: `nexus-api-key`
3. Sync source code and deploy

### Via CLI

```bash
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/agent-langgraph"
databricks apps deploy agent-langgraph \
  --source-code-path /Workspace/Users/$DATABRICKS_USERNAME/agent-langgraph
```

### Querying the Deployed Agent

Databricks Apps require OAuth (PATs are not supported):

```bash
databricks auth token
curl -X POST <app-url>.databricksapps.com/invocations \
  -H "Authorization: Bearer <oauth-token>" \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "Show me TSLA stock performance" }], "stream": true }'
```

## Evaluation

Update test cases in `agent_server/evaluate_agent.py`, then run:

```bash
uv run agent-evaluate
```

Results are logged to your MLflow experiment.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABRICKS_CONFIG_PROFILE` | Databricks CLI auth profile | Yes |
| `MLFLOW_EXPERIMENT_ID` | MLflow experiment for tracing | Yes |
| `NEXUS_BASE_URL` | Nexus API gateway URL | Yes |
| `NEXUS_API_KEY` | Nexus API key (Gemini access) | Yes |
| `CHAT_APP_PORT` | Chat UI port (default: 3000) | No |
| `CHAT_PROXY_TIMEOUT_SECONDS` | Request timeout (default: 300) | No |
