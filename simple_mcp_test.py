#!/usr/bin/env python3
"""
Minimal MCP Server Test
"""

import asyncio
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

class SimpleMCPServer(Server):
    def __init__(self):
        super().__init__("simple-test")
        self.tools = {
            "hello": Tool(
                name="hello",
                description="Say hello",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name to greet"
                        }
                    },
                    "required": []
                }
            )
        }

    async def list_tools(self):
        from mcp.types import ListToolsResult
        return ListToolsResult(tools=list(self.tools.values()))

    async def call_tool(self, name: str, arguments: dict):
        from mcp.types import CallToolResult
        if name == "hello":
            name_to_greet = arguments.get("name", "World")
            result = {"message": f"Hello, {name_to_greet}!"}
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2))]
            )

async def main():
    server = SimpleMCPServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="simple-test",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 