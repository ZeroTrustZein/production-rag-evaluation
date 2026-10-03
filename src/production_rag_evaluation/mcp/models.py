"""Data models, protocols, and schemas for the MCP (Model Context Protocol) server."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

JSONRPC_VERSION: str = "2.0"
MCP_PROTOCOL_VERSION: str = "2024-11-05"


class MCPProtocolVersion(str, Enum):
    """Supported MCP protocol specification versions."""

    V2024_11_05 = "2024-11-05"


class MCPTransport(str, Enum):
    """Transport layer modes for MCP server communication."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPConfig(BaseModel):
    """Runtime configuration settings for the MCP server instance."""

    model_config = ConfigDict(frozen=True)

    server_name: str = Field(
        default="production-rag-evaluation-mcp",
        description="Logical name exposed in MCP handshake",
    )
    server_version: str = Field(
        default="0.1.0",
        description="SemVer string identifying server build",
    )
    transport: MCPTransport = Field(
        default=MCPTransport.STDIO,
        description="Transport mechanism (stdio, sse, or http)",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host interface for network-bound transports",
    )
    port: int = Field(
        default=8000,
        description="Port for SSE or HTTP listeners",
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Optional shared bearer token for request authentication",
    )
    log_level: str = Field(
        default="INFO",
        description="Server logging verbosity",
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose JSON-RPC trace output",
    )


class ToolParameter(BaseModel):
    """Schema descriptor for a single tool input argument."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Parameter key identifier")
    type: str = Field(default="string", description="JSON schema data type")
    description: str = Field(default="", description="Explanatory text for client LLMs")
    required: bool = Field(default=True, description="Whether client must supply this field")
    default: Optional[Any] = Field(default=None, description="Default value if parameter is omitted")


class MCPTool(BaseModel):
    """Exposed executable tool definition compliant with MCP specifications."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="Behavioral synopsis provided to client LLMs")
    inputSchema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="JSON Schema specifying acceptable arguments",
    )
    handler: Optional[Callable[..., Any]] = Field(
        default=None,
        description="Optional underlying callable handling execution",
    )


class ToolCallResult(BaseModel):
    """Structured response returned by tool invocation."""

    model_config = ConfigDict(frozen=True)

    content: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of MCP content objects (e.g. text blocks)",
    )
    isError: bool = Field(
        default=False,
        description="Flag indicating tool runtime error or validation failure",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary supplementary metrics or tracing metadata",
    )

    @classmethod
    def text(cls, text: str, is_error: bool = False, metadata: Optional[Dict[str, Any]] = None) -> ToolCallResult:
        """Helper to create a plain text tool response block."""
        return cls(
            content=[{"type": "text", "text": text}],
            isError=is_error,
            metadata=metadata or {},
        )


class MCPResource(BaseModel):
    """Representational resource readable by clients via MCP URIs."""

    model_config = ConfigDict(frozen=True)

    uri: str = Field(description="Uniform Resource Identifier identifying this entity")
    name: str = Field(description="Human-readable resource label")
    description: Optional[str] = Field(
        default=None,
        description="Functional description of resource content",
    )
    mimeType: str = Field(
        default="application/json",
        description="MIME type classification of resource payload",
    )
    text: Optional[str] = Field(
        default=None,
        description="Raw UTF-8 text representation if applicable",
    )
    blob: Optional[str] = Field(
        default=None,
        description="Base64 encoded binary payload if applicable",
    )


class MCPPromptArgument(BaseModel):
    """Variable substitution argument for an MCP prompt template."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Argument placeholder identifier")
    description: Optional[str] = Field(default=None, description="Argument purpose")
    required: bool = Field(default=True, description="Whether argument is mandatory")


class MCPPromptMessage(BaseModel):
    """Chat message node inside an MCP prompt template response."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(default="user", description="Participant role: user, assistant, system")
    content: Union[Dict[str, Any], str] = Field(
        description="Message payload or formatted content dictionary",
    )


class MCPPrompt(BaseModel):
    """Prompt template exposed to clients for guided interactive workflows."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique prompt key")
    description: str = Field(description="Operational summary of prompt intent")
    arguments: List[MCPPromptArgument] = Field(
        default_factory=list,
        description="Supported parameters for prompt rendering",
    )
    template: Optional[str] = Field(
        default=None,
        description="Raw text template string with format specifiers",
    )


class JSONRPCError(BaseModel):
    """Standard JSON-RPC 2.0 error specification payload."""

    model_config = ConfigDict(frozen=True)

    code: int = Field(description="Numeric JSON-RPC error code")
    message: str = Field(description="Short textual summary of error condition")
    data: Optional[Any] = Field(default=None, description="Structured diagnostics context")


class JSONRPCRequest(BaseModel):
    """Inbound JSON-RPC 2.0 request envelope."""

    model_config = ConfigDict(frozen=True)

    jsonrpc: str = Field(default=JSONRPC_VERSION)
    id: Optional[Union[str, int]] = Field(default=None)
    method: str = Field(description="Target RPC method string")
    params: Optional[Dict[str, Any]] = Field(default=None)


class JSONRPCResponse(BaseModel):
    """Outbound JSON-RPC 2.0 response envelope."""

    model_config = ConfigDict(frozen=True)

    jsonrpc: str = Field(default=JSONRPC_VERSION)
    id: Optional[Union[str, int]] = Field(default=None)
    result: Optional[Any] = Field(default=None)
    error: Optional[JSONRPCError] = Field(default=None)
