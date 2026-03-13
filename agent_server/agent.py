import logging, os
from datetime import datetime
from typing import AsyncGenerator, Optional

import litellm
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import tool
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

mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True
sp_workspace_client = WorkspaceClient()


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
    if "properties" in schema and isinstance(schema["properties"], dict):
        schema["properties"] = {
            k: v for k, v in schema["properties"].items() if isinstance(v, dict)
        }
        # Also remove sanitized keys from 'required' if present
        if "required" in schema:
            schema["required"] = [
                r for r in schema["required"] if r in schema["properties"]
            ]
    return schema


async def load_gradio_tools() -> list:
    """Load tools from the Gradio MCP server via SSE, sanitizing invalid schemas."""
    from langchain_core.tools import StructuredTool
    from langchain_mcp_adapters.sessions import create_session

    connection = {"transport": "sse", "url": GRADIO_MCP_URL}
    tools = []
    async with create_session(connection) as session:
        await session.initialize()
        result = await session.list_tools()
        for mcp_tool in result.tools:
            input_schema = _sanitize_input_schema(mcp_tool.inputSchema)

            async def _make_call(session_conn=connection, tool_name=mcp_tool.name, **kwargs):
                async with create_session(session_conn) as s:
                    await s.initialize()
                    result = await s.call_tool(tool_name, kwargs)
                    return result.content

            tools.append(StructuredTool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                args_schema=input_schema,
                coroutine=_make_call,
            ))
    return tools


async def init_agent(workspace_client: Optional[WorkspaceClient] = None):
    tools = []
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",  # see table above to set the desired model id
        client_options={"api_endpoint": os.getenv("NEXUS_BASE_URL")},
        # Use your Nexus API key here
        google_api_key=os.getenv("NEXUS_API_KEY"),
     #   transport="rest",       # otherwise will use gRPC which is not supported by Nexus
    )
    # To use MCP server tools instead, replace the line above with:
    mcp_client = init_mcp_client(workspace_client or sp_workspace_client)
    tools.extend(await mcp_client.get_tools())
    try:
        gradio_tools = await load_gradio_tools()
        tools.extend(gradio_tools)
    except Exception as e:
        logging.warning(f"Failed to load Gradio MCP tools: {e}")
    return create_agent(tools=tools, model=model)


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
    if session_id := get_session_id(request):
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id})

    # By default, uses service principal credentials.
    # For on-behalf-of user authentication, use get_user_workspace_client() instead:
    #   agent = await init_agent(workspace_client=get_user_workspace_client())
    agent = await init_agent()
    messages = {"messages": to_chat_completions_input(
        [i.model_dump() for i in request.input])}

    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
