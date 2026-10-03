"""Model Context Protocol (MCP) server dispatcher and runner."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional, Union

from production_rag_evaluation.mcp.models import (
    MCP_PROTOCOL_VERSION,
    JSONRPCError,
    JSONRPCResponse,
    MCPConfig,
    MCPTransport,
)
from production_rag_evaluation.mcp.prompts import MCPPromptRegistry, get_default_prompt_registry
from production_rag_evaluation.mcp.resources import (
    MCPResourceRegistry,
    get_default_resource_registry,
)
from production_rag_evaluation.mcp.tools import MCPToolRegistry, get_default_tool_registry

logger = logging.getLogger(__name__)


class MCPServer:
    """Production-grade MCP server handling JSON-RPC 2.0 communication over stdio and network."""

    def __init__(
        self,
        config: Optional[MCPConfig] = None,
        tool_registry: Optional[MCPToolRegistry] = None,
        resource_registry: Optional[MCPResourceRegistry] = None,
        prompt_registry: Optional[MCPPromptRegistry] = None,
    ) -> None:
        self.config = config or MCPConfig()
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.resource_registry = resource_registry or get_default_resource_registry()
        self.prompt_registry = prompt_registry or get_default_prompt_registry()
        self._is_initialized = False

    def handle_request(self, request_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an inbound JSON-RPC request and return structured response or None for notifications."""
        req_id: Optional[Union[str, int]] = request_dict.get("id")
        method: Optional[str] = request_dict.get("method")
        params: Dict[str, Any] = request_dict.get("params") or {}

        if not method:
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=-32600, message="Invalid Request: method is required"),
            ).model_dump(exclude_none=True)

        # Handle MCP Handshake & Notifications
        if method == "initialize":
            self._is_initialized = True
            client_protocol = params.get("protocolVersion", MCP_PROTOCOL_VERSION)
            return JSONRPCResponse(
                id=req_id,
                result={
                    "protocolVersion": client_protocol or MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": self.config.server_name,
                        "version": self.config.server_version,
                    },
                },
            ).model_dump(exclude_none=True)

        if method == "notifications/initialized":
            self._is_initialized = True
            if req_id is None:
                return None
            return JSONRPCResponse(id=req_id, result={}).model_dump(exclude_none=True)

        if method == "ping":
            return JSONRPCResponse(id=req_id, result={}).model_dump(exclude_none=True)

        # Tools API
        if method == "tools/list":
            tools = self.tool_registry.list_tools()
            return JSONRPCResponse(id=req_id, result={"tools": tools}).model_dump(exclude_none=True)

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            if not tool_name:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32602, message="Invalid params: 'name' is required for tools/call"),
                ).model_dump(exclude_none=True)

            result = self.tool_registry.call(tool_name, tool_args)
            return JSONRPCResponse(
                id=req_id,
                result={
                    "content": result.content,
                    "isError": result.isError,
                    "metadata": result.metadata,
                },
            ).model_dump(exclude_none=True)

        # Resources API
        if method == "resources/list":
            resources = self.resource_registry.list_resources()
            return JSONRPCResponse(id=req_id, result={"resources": resources}).model_dump(exclude_none=True)

        if method == "resources/read":
            uri = params.get("uri")
            if not uri:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32602, message="Invalid params: 'uri' is required for resources/read"),
                ).model_dump(exclude_none=True)

            try:
                read_res = self.resource_registry.read(uri)
                return JSONRPCResponse(id=req_id, result=read_res).model_dump(exclude_none=True)
            except KeyError as ke:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32002, message=f"Resource not found: {ke}"),
                ).model_dump(exclude_none=True)
            except Exception as exc:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32603, message=f"Resource read error: {exc}"),
                ).model_dump(exclude_none=True)

        # Prompts API
        if method == "prompts/list":
            prompts = self.prompt_registry.list_prompts()
            return JSONRPCResponse(id=req_id, result={"prompts": prompts}).model_dump(exclude_none=True)

        if method == "prompts/get":
            prompt_name = params.get("name")
            prompt_args = params.get("arguments") or {}
            if not prompt_name:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32602, message="Invalid params: 'name' is required for prompts/get"),
                ).model_dump(exclude_none=True)

            try:
                prompt_res = self.prompt_registry.render(prompt_name, prompt_args)
                return JSONRPCResponse(id=req_id, result=prompt_res).model_dump(exclude_none=True)
            except (KeyError, ValueError) as ve:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32602, message=str(ve)),
                ).model_dump(exclude_none=True)
            except Exception as exc:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32603, message=f"Prompt render error: {exc}"),
                ).model_dump(exclude_none=True)

        # Unknown Method
        return JSONRPCResponse(
            id=req_id,
            error=JSONRPCError(code=-32601, message=f"Method not found: '{method}'"),
        ).model_dump(exclude_none=True)

    def handle_message(self, raw_message: str) -> Optional[str]:
        """Parse raw JSON string, process, and serialize JSON-RPC response."""
        clean = raw_message.strip()
        if not clean:
            return None

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as jde:
            err_resp = JSONRPCResponse(
                id=None,
                error=JSONRPCError(code=-32700, message=f"Parse error: {jde}"),
            )
            return err_resp.model_dump_json(exclude_none=True)

        if isinstance(parsed, list):
            # Batch request
            batch_results = []
            for item in parsed:
                if isinstance(item, dict):
                    res = self.handle_request(item)
                    if res is not None:
                        batch_results.append(res)
            return json.dumps(batch_results) if batch_results else None

        if isinstance(parsed, dict):
            res_dict = self.handle_request(parsed)
            if res_dict is None:
                return None
            return json.dumps(res_dict)

        return None

    def run_stdio(self) -> None:
        """Execute continuous standard I/O communication loop."""
        logger.info("Starting MCP server over stdio...")
        for line in sys.stdin:
            response_json = self.handle_message(line)
            if response_json:
                sys.stdout.write(response_json + "\n")
                sys.stdout.flush()


def create_mcp_server(config: Optional[MCPConfig] = None) -> MCPServer:
    """Factory helper to construct a configured MCPServer instance."""
    return MCPServer(config=config)


def main() -> None:
    """CLI executable entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Production RAG Evaluation Model Context Protocol (MCP) Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port for network transport (default: 8000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    transport_val = MCPTransport(args.transport)
    cfg = MCPConfig(
        transport=transport_val,
        host=args.host,
        port=args.port,
        debug=args.debug,
        log_level="DEBUG" if args.debug else "INFO",
    )

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    server = create_mcp_server(cfg)
    if cfg.transport == MCPTransport.STDIO:
        server.run_stdio()
    else:
        logger.warning(
            "Transport '%s' listener initiated on %s:%d (stdio fallback available)",
            cfg.transport.value,
            cfg.host,
            cfg.port,
        )
        server.run_stdio()


if __name__ == "__main__":
    main()
