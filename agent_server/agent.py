import logging, os
from datetime import datetime
from typing import AsyncGenerator, Optional

import litellm
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from agent_server.utils import (
    get_databricks_host_from_env,
    get_session_id,
    get_user_workspace_client,
    process_agent_astream_events,
)

litellm.suppress_debug_info = True
sp_workspace_client = WorkspaceClient()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "chat_history.db")

AGENT_INSTRUCTIONS = """You are a helpful stock market analyst assistant.

You have access to:
- Stock trading data via Genie Space (30min, 5min, 60min, and daily intervals)
- Unity Catalog functions for stock analysis
- Web search and web fetch tools for researching market news

When answering questions:
- Use the available tools to look up real data before responding
- All stock data should be sourced from the Genie Space tool, not from memory, not from internet search results
- Provide specific numbers, dates, and sources when possible
- If asked about stock trends, query the relevant time interval data
- Present data clearly with tables or summaries when appropriate
- Always clarify the time period and data source you're referencing
"""


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


def init_mcp_client(workspace_client: WorkspaceClient) -> DatabricksMultiServerMCPClient:
    host_name = get_databricks_host_from_env()
    return DatabricksMultiServerMCPClient(
        [
            DatabricksMCPServer(
                name="stock-agent-genie-space",
                url=f"{host_name}/api/2.0/mcp/genie/01f11baeb79e12de8ce1e5bbdbe2b6a1",
                workspace_client=workspace_client,
            ),
             DatabricksMCPServer(
                name="workspace-stock-function",
                url=f"{host_name}/api/2.0/mcp/functions/workspace/stock",
                workspace_client=workspace_client,
            ),
        ]
    )


GRADIO_MCP_URL = "https://victor-web.hf.space/gradio_api/mcp/sse"


def _sanitize_input_schema(schema: dict) -> dict:
    """Remove properties with None definitions from an MCP tool input schema."""
    schema = dict(schema)  # shallow copy
    if "properties" in schema and isinstance(schema["properties"], dict):
        schema["properties"] = {
            k: v for k, v in schema["properties"].items() if isinstance(v, dict)
        }
        if "required" in schema:
            schema["required"] = [
                r for r in schema["required"] if r in schema["properties"]
            ]
    return schema


def _schema_dict_to_model(name: str, schema: dict):
    """Convert a JSON Schema dict to a Pydantic model class."""
    from pydantic import create_model, Field

    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
    fields = {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for prop_name, prop_def in properties.items():
        if not isinstance(prop_def, dict):
            continue
        py_type = type_map.get(prop_def.get("type", "string"), str)
        description = prop_def.get("description", "")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            default = prop_def.get("default")
            fields[prop_name] = (Optional[py_type], Field(default=default, description=description))

    return create_model(name, **fields)




async def init_agent(workspace_client: Optional[WorkspaceClient] = None, checkpointer=None):
    tools = []
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",  # see table above to set the desired model id
        client_options={"api_endpoint": os.getenv("NEXUS_BASE_URL")},
        # Use your Nexus API key here
        google_api_key=os.getenv("NEXUS_API_KEY"),
        temperature=0.1,
        max_output_tokens=512,  # Unlocks the full 64K output capacity
    )
    # To use MCP server tools instead, replace the line above with:
    mcp_client = init_mcp_client(workspace_client or sp_workspace_client)
    mcp_tools = await mcp_client.get_tools()
    tools.extend(mcp_tools)
    return create_agent(tools=tools, model=model, checkpointer=checkpointer)


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:


    agent = await init_agent()
    thread_id = "default"
        # With a checkpointer, LangGraph manages history internally.
        # Only send the last user message to avoid duplicating history.
    all_messages = to_chat_completions_input(
            [i.model_dump() for i in request.input])
    last_user_messages = []
    for msg in reversed(all_messages):
            last_user_messages.insert(0, msg)
            if msg.get("role") == "user":
                break
    messages = {"messages": [{"role": "system", "content": AGENT_INSTRUCTIONS}] + last_user_messages}
    config = {"configurable": {"thread_id": thread_id}}

    async for event in process_agent_astream_events(
            agent.astream(input=messages, stream_mode=["updates", "messages"], config=config)
        ):
            yield event
