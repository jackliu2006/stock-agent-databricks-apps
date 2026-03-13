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

mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True
sp_workspace_client = WorkspaceClient()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "chat_history.db")


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
            sanitized = _sanitize_input_schema(mcp_tool.inputSchema)
            args_model = _schema_dict_to_model(mcp_tool.name + "_args", sanitized)

            async def _make_call(session_conn=connection, tool_name=mcp_tool.name, **kwargs):
                async with create_session(session_conn) as s:
                    await s.initialize()
                    call_result = await s.call_tool(tool_name, kwargs)
                    return call_result.content

            tools.append(StructuredTool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                args_schema=args_model,
                coroutine=_make_call,
            ))
    return tools


async def init_agent(workspace_client: Optional[WorkspaceClient] = None, checkpointer=None):
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
    session_id = get_session_id(request)
    if session_id:
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id})

    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH) as checkpointer:
        await checkpointer.setup()
        agent = await init_agent(checkpointer=checkpointer)
        thread_id = session_id or "default"
        # With a checkpointer, LangGraph manages history internally.
        # Only send the last user message to avoid duplicating history.
        all_messages = to_chat_completions_input(
            [i.model_dump() for i in request.input])
        last_user_messages = []
        for msg in reversed(all_messages):
            last_user_messages.insert(0, msg)
            if msg.get("role") == "user":
                break
        messages = {"messages": last_user_messages}
        config = {"configurable": {"thread_id": thread_id}}

        async for event in process_agent_astream_events(
            agent.astream(input=messages, stream_mode=["updates", "messages"], config=config)
        ):
            yield event
