"""Model Context Protocol (MCP) server integration and tool interfaces."""

from __future__ import annotations

from production_rag_evaluation.mcp.models import (
    JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPConfig,
    MCPPrompt,
    MCPPromptArgument,
    MCPPromptMessage,
    MCPProtocolVersion,
    MCPResource,
    MCPTool,
    MCPTransport,
    ToolCallResult,
    ToolParameter,
)
from production_rag_evaluation.mcp.prompts import (
    MCPPromptRegistry,
    get_default_prompt_registry,
)
from production_rag_evaluation.mcp.resources import (
    MCPResourceRegistry,
    get_default_resource_registry,
)
from production_rag_evaluation.mcp.server import (
    MCPServer,
    create_mcp_server,
)
from production_rag_evaluation.mcp.tools import (
    MCPToolRegistry,
    get_default_tool_registry,
)

__all__ = [
    # Constants & Enums
    "JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    "MCPProtocolVersion",
    "MCPTransport",
    # Models
    "MCPConfig",
    "ToolParameter",
    "MCPTool",
    "ToolCallResult",
    "MCPResource",
    "MCPPromptArgument",
    "MCPPromptMessage",
    "MCPPrompt",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    # Registries
    "MCPToolRegistry",
    "get_default_tool_registry",
    "MCPResourceRegistry",
    "get_default_resource_registry",
    "MCPPromptRegistry",
    "get_default_prompt_registry",
    # Server
    "MCPServer",
    "create_mcp_server",
]
