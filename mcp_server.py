#!/usr/bin/env python3
"""
MCP Server for AURA Trading System

This server exposes trading tools to the Gemini CLI via the Model-Context Protocol.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Load environment variables from .env file at project root
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

from tradr.core.config import CONFIG, setup_clients
from tradr.tools.toolbox import AsyncToolbox
from tradr.memory.agent_memory import AgentMemory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the server
server = Server("aura-trading")

# Initialize trading components
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=20)
clients = setup_clients()
memory = AgentMemory(file_path="/tmp/agent_memory.json")
toolbox = AsyncToolbox(clients, memory=memory, executor=executor)

# Define tools
tools = {
    # Research Tools
    "get_market_news": Tool(
        name="get_market_news",
        description="Get latest market news and sentiment analysis",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "assess_market_regime": Tool(
        name="assess_market_regime",
        description="Analyze current market regime (bull/bear, volatility, risk-on/off)",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "scan_top_movers": Tool(
        name="scan_top_movers",
        description="Scan for top moving stocks with unusual activity",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of stocks to return",
                    "default": 50
                }
            },
            "required": []
        }
    ),
    "scan_unusual_options_flow": Tool(
        name="scan_unusual_options_flow",
        description="Scan for unusual options activity (volume vs open interest)",
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbols to scan"
                },
                "min_volume": {
                    "type": "integer",
                    "description": "Minimum options volume filter",
                    "default": 500
                },
                "min_volume_oi_ratio": {
                    "type": "number",
                    "description": "Minimum volume/open interest ratio",
                    "default": 2.0
                },
                "max_expirations": {
                    "type": "integer",
                    "description": "Max expirations to scan per symbol",
                    "default": 3
                },
                "max_days_out": {
                    "type": "integer",
                    "description": "Max days to expiration",
                    "default": 45
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 50
                }
            },
            "required": ["symbols"]
        }
    ),
    "scan_insider_activity": Tool(
        name="scan_insider_activity",
        description="Scan insider transactions (requires EARNINGSFEED_API_KEY)",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Optional symbol filter"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 25
                },
                "only_buys": {
                    "type": "boolean",
                    "description": "Filter to insider buys only",
                    "default": True
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "Bypass cache and refetch",
                    "default": False
                }
            },
            "required": []
        }
    ),
    "scan_13f_changes": Tool(
        name="scan_13f_changes",
        description="Scan institutional 13F holdings (requires EARNINGSFEED_API_KEY)",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Optional symbol filter"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 25
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "Bypass cache and refetch",
                    "default": False
                }
            },
            "required": []
        }
    ),
    "analyze_sector_exposure": Tool(
        name="analyze_sector_exposure",
        description="Analyze current portfolio sector exposure and concentration",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "analyze_intermarket_correlation": Tool(
        name="analyze_intermarket_correlation",
        description="Analyze correlations between equities, bonds, commodities, and currencies",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    
    # Technical Analysis Tools
    "analyze_technicals": Tool(
        name="analyze_technicals",
        description="Perform comprehensive technical analysis on a stock",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to analyze"
                }
            },
            "required": ["symbol"]
        }
    ),
    "analyze_volume_trend": Tool(
        name="analyze_volume_trend",
        description="Analyze volume trends and quality for a stock",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to analyze"
                }
            },
            "required": ["symbol"]
        }
    ),
    "add_to_watchlist": Tool(
        name="add_to_watchlist",
        description="Add a symbol to the watchlist",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to add"
                },
                "reason": {
                    "type": "string",
                    "description": "Optional reason for adding"
                },
                "technicals": {
                    "type": "object",
                    "description": "Optional technicals snapshot (stored in reason if provided)"
                }
            },
            "required": ["symbol"]
        }
    ),
    "get_watchlist": Tool(
        name="get_watchlist",
        description="Return watchlist with fresh technicals for each symbol",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "remove_from_watchlist": Tool(
        name="remove_from_watchlist",
        description="Remove a symbol from the watchlist",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to remove"
                }
            },
            "required": ["symbol"]
        }
    ),
    
    # Portfolio Management Tools
    "get_portfolio_state": Tool(
        name="get_portfolio_state",
        description="Get current portfolio positions and performance",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "calculate_risk_metrics": Tool(
        name="calculate_risk_metrics",
        description="Calculate portfolio risk metrics (Sharpe ratio, drawdown, VaR)",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "calculate_portfolio_beta": Tool(
        name="calculate_portfolio_beta",
        description="Calculate portfolio beta relative to the market",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    
    # Trading Execution Tools
    "calculate_position_size": Tool(
        name="calculate_position_size",
        description="Calculate optimal position size based on risk parameters",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol"
                },
                "confidence": {
                    "type": "string",
                    "description": "Confidence level (low/medium/high)",
                    "enum": ["low", "medium", "high"]
                },
                "entry_price": {
                    "type": "number",
                    "description": "Entry price for the position"
                },
                "force_minimum": {
                    "type": "boolean",
                    "description": "Force minimum position size",
                    "default": True
                }
            },
            "required": ["symbol", "confidence", "entry_price"]
        }
    ),
    "smart_order_entry": Tool(
        name="smart_order_entry",
        description="Execute a smart order with stop-loss and take-profit",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to trade"
                },
                "shares": {
                    "type": "number",
                    "description": "Number of shares to trade"
                },
                "order_type": {
                    "type": "string",
                    "description": "Order type (market/limit/smart)",
                    "default": "smart"
                },
                "take_profit_pct": {
                    "type": "number",
                    "description": "Take profit percentage"
                },
                "stop_loss_pct": {
                    "type": "number",
                    "description": "Stop loss percentage"
                }
            },
            "required": ["symbol", "shares"]
        }
    ),
    "modify_trade_parameters": Tool(
        name="modify_trade_parameters",
        description="Modify an existing order's stop/limit parameters (currently supports stop price updates)",
        inputSchema={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID to modify"
                },
                "new_stop_price": {
                    "type": "number",
                    "description": "New stop price (optional)"
                },
                "new_limit_price": {
                    "type": "number",
                    "description": "New limit price (optional)"
                },
                "trail_percent": {
                    "type": "number",
                    "description": "Trailing stop percentage (optional)"
                }
            },
            "required": ["order_id"]
        }
    ),
    "close_position": Tool(
        name="close_position",
        description="Close an existing position",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to close"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for closing position"
                },
                "shares": {
                    "type": "number",
                    "description": "Number of shares to close (optional, closes all if not specified)"
                }
            },
            "required": ["symbol", "reason"]
        }
    ),
    "buy_option": Tool(
        name="buy_option",
        description="Buy an option contract.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The underlying stock symbol"
                },
                "strike": {
                    "type": ["number", "string"],
                    "description": "The strike price of the option"
                },
                "expiration_date": {
                    "type": "string",
                    "description": "The expiration date of the option in YYYY-MM-DD format"
                },
                "quantity": {
                    "type": "integer",
                    "description": "The number of option contracts to buy"
                },
                "contract_type": {
                    "type": "string",
                    "description": "The type of option to buy, either 'put' or 'call'",
                    "enum": ["put", "call"]
                }
            },
            "required": ["symbol", "strike", "expiration_date", "quantity", "contract_type"]
        }
    ),
    
    # Utility Tools
    "save_notes": Tool(
        name="save_notes",
        description="Save notes to memory system",
        inputSchema={
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Notes to save"
                }
            },
            "required": ["notes"]
        }
    ),
    "read_notes": Tool(
        name="read_notes",
        description="Read notes from memory system",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "get_company_news": Tool(
        name="get_company_news",
        description="Get recent company-specific news for a given stock symbol (North American equities only) in a date range.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., AAPL, MSFT, TSLA)"
                },
                "from_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD)"
                },
                "to_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD)"
                }
            },
            "required": ["symbol", "from_date", "to_date"]
        }
    ),
}

def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    import enum
    from types import MappingProxyType

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, MappingProxyType):
        return {k: make_json_serializable(v) for k, v in dict(obj).items()}
    if isinstance(obj, type):
        return obj.__name__
    if hasattr(obj, "__dict__"):
        return make_json_serializable(obj.__dict__)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if asyncio.iscoroutine(obj):
        return f"[Coroutine: {obj.__name__}]"
    return obj

class NotificationOptions:
    def __init__(self):
        self.tools_changed = None

@server.list_tools()
async def handle_list_tools():
    """List all available tools."""
    return list(tools.values())

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    """Execute a tool call."""
    logger.info(f"Calling tool: {name} with args: {arguments}")
    
    try:
        if hasattr(toolbox, name):
            tool_method = getattr(toolbox, name)
            
            if asyncio.iscoroutinefunction(tool_method):
                result = await tool_method(**arguments)
            else:
                result = tool_method(**arguments)
            
            result = make_json_serializable(result)
            logger.info(f"Tool {name} completed successfully")
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        else:
            error_msg = f"Tool '{name}' not found in toolbox"
            logger.error(error_msg)
            return [TextContent(type="text", text=json.dumps({"error": error_msg}, indent=2))]
            
    except Exception as e:
        error_msg = f"Error executing tool {name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": error_msg}, indent=2))]

async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="aura-trading",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 
