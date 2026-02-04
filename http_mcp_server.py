#!/usr/bin/env python3
"""
HTTP/WebSocket MCP Server for AURA Trading System

This server exposes the MCP trading tools over HTTP/WebSocket for remote access.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

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

# Initialize trading components
clients = setup_clients()
memory = AgentMemory()
toolbox = AsyncToolbox(clients, memory=memory)

# Create FastAPI app
app = FastAPI(title="AURA Trading MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for requests/responses
class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}

class ToolCallResponse(BaseModel):
    result: List[Dict[str, str]]
    error: Optional[str] = None

class ToolInfo(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]

# Define tools (same as in mcp_server.py)
tools = {
    # Research Tools
    "get_market_news": ToolInfo(
        name="get_market_news",
        description="Get latest market news and sentiment analysis",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "assess_market_regime": ToolInfo(
        name="assess_market_regime",
        description="Analyze current market regime (bull/bear, volatility, risk-on/off)",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "scan_top_movers": ToolInfo(
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
    "scan_unusual_options_flow": ToolInfo(
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
    "scan_insider_activity": ToolInfo(
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
    "scan_13f_changes": ToolInfo(
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
    "analyze_sector_exposure": ToolInfo(
        name="analyze_sector_exposure",
        description="Analyze current portfolio sector exposure and concentration",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "analyze_intermarket_correlation": ToolInfo(
        name="analyze_intermarket_correlation",
        description="Analyze correlations between equities, bonds, commodities, and currencies",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    
    # Technical Analysis Tools
    "analyze_technicals": ToolInfo(
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
    "analyze_volume_trend": ToolInfo(
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
    "add_to_watchlist": ToolInfo(
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
    "get_watchlist": ToolInfo(
        name="get_watchlist",
        description="Return watchlist with fresh technicals for each symbol",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "remove_from_watchlist": ToolInfo(
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
    "get_portfolio_state": ToolInfo(
        name="get_portfolio_state",
        description="Get current portfolio positions and performance",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "calculate_risk_metrics": ToolInfo(
        name="calculate_risk_metrics",
        description="Calculate portfolio risk metrics (Sharpe ratio, drawdown, VaR)",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    "calculate_portfolio_beta": ToolInfo(
        name="calculate_portfolio_beta",
        description="Calculate portfolio beta relative to the market",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    
    # Trading Execution Tools
    "calculate_position_size": ToolInfo(
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
    "smart_order_entry": ToolInfo(
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
                },
                "limit_price": {
                    "type": "number",
                    "description": "Limit price for limit orders (optional)"
                },
                "reason": {
                    "type": "string",
                    "description": "Optional trade rationale"
                }
            },
            "required": ["symbol", "shares"]
        }
    ),
    "modify_trade_parameters": ToolInfo(
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
    "close_position": ToolInfo(
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

    "buy_option": ToolInfo(
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
                },
                "reason": {
                    "type": "string",
                    "description": "Optional trade rationale"
                }
            },
            "required": ["symbol", "strike", "expiration_date", "quantity", "contract_type"]
        }
    ),
    
    # Utility Tools
    "save_notes": ToolInfo(
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
    "read_notes": ToolInfo(
        name="read_notes",
        description="Read notes from memory system",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
}

def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif asyncio.iscoroutine(obj):
        return f"[Coroutine: {obj.__name__}]"
    else:
        return obj

@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "name": "AURA Trading MCP Server",
        "version": "1.0.0",
        "status": "running",
        "tools_available": len(tools)
    }

@app.post("/")
async def mcp_initialize():
    """MCP initialization endpoint."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "aura-trading",
                "version": "1.0.0"
            }
        }
    }

@app.post("/tools/list")
async def mcp_list_tools():
    """MCP tools listing endpoint."""
    tools_list = []
    for tool_name, tool_info in tools.items():
        tools_list.append({
            "name": tool_name,
            "description": tool_info.description,
            "inputSchema": tool_info.inputSchema
        })
    
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": tools_list
        }
    }

@app.post("/tools/call")
async def mcp_call_tool(request: ToolCallRequest):
    """MCP tool call endpoint."""
    logger.info(f"Calling tool: {request.name} with args: {request.arguments}")
    
    try:
        if hasattr(toolbox, request.name):
            tool_method = getattr(toolbox, request.name)
            
            if asyncio.iscoroutinefunction(tool_method):
                result = await tool_method(**request.arguments)
            else:
                result = tool_method(**request.arguments)
            
            result = make_json_serializable(result)
            logger.info(f"Tool {request.name} completed successfully")
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
        else:
            error_msg = f"Tool '{request.name}' not found in toolbox"
            logger.error(error_msg)
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {
                    "code": -32601,
                    "message": error_msg
                }
            }
            
    except Exception as e:
        error_msg = f"Error executing tool {request.name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32603,
                "message": error_msg
            }
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time tool calls."""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "list_tools":
                response = {"type": "tools_list", "tools": list(tools.values())}
            elif message.get("type") == "call_tool":
                tool_name = message.get("name")
                arguments = message.get("arguments", {})
                
                try:
                    if hasattr(toolbox, tool_name):
                        tool_method = getattr(toolbox, tool_name)
                        
                        if asyncio.iscoroutinefunction(tool_method):
                            result = await tool_method(**arguments)
                        else:
                            result = tool_method(**arguments)
                        
                        result = make_json_serializable(result)
                        response = {
                            "type": "tool_result",
                            "name": tool_name,
                            "result": [{"type": "text", "text": json.dumps(result, indent=2)}]
                        }
                    else:
                        response = {
                            "type": "tool_error",
                            "name": tool_name,
                            "error": f"Tool '{tool_name}' not found"
                        }
                except Exception as e:
                    response = {
                        "type": "tool_error",
                        "name": tool_name,
                        "error": str(e)
                    }
            else:
                response = {"type": "error", "message": "Unknown message type"}
            
            # Send response
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/.well-known/oauth-authorization-server")
async def oauth_discovery():
    return {
        "issuer": "http://localhost:8000",
        "authorization_endpoint": "http://localhost:8000/authorize",
        "token_endpoint": "http://localhost:8000/token",
        "jwks_uri": "http://localhost:8000/jwks"
    }

@app.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource():
    return {"message": "Dummy protected resource"}

@app.get("/authorize")
async def authorize():
    return {"message": "Dummy authorize endpoint"}

@app.post("/token")
async def token():
    return {"access_token": "dummy-token", "token_type": "bearer"}

@app.get("/jwks")
async def jwks():
    return {"keys": []}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA Trading MCP HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting AURA Trading MCP Server on {args.host}:{args.port}")
    print(f"Available endpoints:")
    print(f"  - HTTP: http://{args.host}:{args.port}")
    print(f"  - WebSocket: ws://{args.host}:{args.port}/ws")
    print(f"  - Tools list: http://{args.host}:{args.port}/tools")
    
    uvicorn.run(
        "http_mcp_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    ) 
