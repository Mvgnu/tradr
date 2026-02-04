#!/usr/bin/env python3
"""
Test script for the AURA MCP Server
"""

import asyncio
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    ListToolsRequest,
    TextContent
)

# Import the actual server components
from mcp_server import server as mcp_server_instance
from mcp_server import toolbox as mcp_toolbox_instance
from mcp_server import tools as mcp_tool_definitions

async def test_mcp_server():
    """Test the MCP server functionality by directly accessing its components."""
    print("🧪 Testing AURA MCP Server components directly...")
    
    # Test listing tools
    print("\n📋 Testing tool listing...")
    
    # Access the globally defined tools dictionary directly
    registered_tools = list(mcp_tool_definitions.values())
    
    print(f"Found {len(registered_tools)} tools:")
    for tool in registered_tools:
        print(f"  - {tool.name}: {tool.description}")
        
    if not registered_tools:
        print("❌ No tools found!")
        return

    # Test a simple tool call (e.g., get_market_news)
    print("\n🔧 Testing tool call (get_market_news)...")
    test_request_news = CallToolRequest(
        name="get_market_news",
        arguments={}
    )
    
    try:
        # Directly call the handler function with the toolbox instance
        # This bypasses the stdio communication for direct testing
        result_content = await mcp_server_instance.call_tool_handler(test_request_news.name, test_request_news.arguments)
        
        # result_content is a list of TextContent objects
        if result_content and isinstance(result_content[0], TextContent):
            result_json = json.loads(result_content[0].text)
            print("✅ Tool call successful!")
            print(f"Result (first 200 chars): {result_content[0].text[:200]}...")
        else:
            print(f"❌ Tool call failed: Unexpected result format: {result_content}")

    except Exception as e:
        print(f"❌ Tool call failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Test analyze_technicals
    print("\n🔧 Testing tool call (analyze_technicals for AAPL)...")
    test_request_tech = CallToolRequest(
        name="analyze_technicals",
        arguments={"symbol": "AAPL"}
    )
    try:
        result_content = await mcp_server_instance.call_tool_handler(test_request_tech.name, test_request_tech.arguments)
        if result_content and isinstance(result_content[0], TextContent):
            result_json = json.loads(result_content[0].text)
            print("✅ analyze_technicals call successful!")
            print(f"Result (first 200 chars): {result_content[0].text[:200]}...")
        else:
            print(f"❌ analyze_technicals call failed: Unexpected result format: {result_content}")
    except Exception as e:
        print(f"❌ analyze_technicals call failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ MCP Server test completed!") 