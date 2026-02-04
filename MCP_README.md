# AURA Trading System - MCP Server

This directory contains the Model-Context Protocol (MCP) server implementation for the AURA trading system, allowing AI assistants (Claude, Gemini, etc.) to directly discover and use trading tools.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements-mcp.txt
```

### 2. Configure Your Client

**For Claude Desktop:**
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "aura-trading": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["/absolute/path/to/tradr/mcp_server.py"],
      "env": {
        "ALPACA_KEY": "your_key",
        "ALPACA_SECRET": "your_secret",
        "GOOGLE_API_KEY": "your_key",
        "FINNHUB_KEY": "your_key"
      }
    }
  }
}
```

**For Gemini Code Assist:**
Add to `.gemini/settings.json`:
```json
{
  "mcpServers": {
    "tradr": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["/absolute/path/to/tradr/mcp_server.py"],
      "cwd": "/absolute/path/to/tradr",
      "timeout": 30000,
      "trust": true
    }
  }
}
```

## Available Tools

### Research
- `get_market_news` - Get latest market news and sentiment
- `assess_market_regime` - Analyze market regime (bull/bear, volatility)
- `scan_top_movers` - Find stocks with unusual activity
- `scan_unusual_options_flow` - Scan for whale activity and unusual volume/OI ratios
- `scan_insider_activity` - Monitor insider buys/sells
- `scan_13f_changes` - Track institutional holdings

### Technical Analysis
- `analyze_technicals` - Comprehensive technical analysis (RSI, MACD, etc.)
- `analyze_volume_trend` - Volume trend and quality analysis

### Portfolio Management
- `get_portfolio_state` - Current portfolio positions
- `calculate_risk_metrics` - Portfolio risk metrics (Sharpe, drawdown, VaR)
- `calculate_portfolio_beta` - Portfolio beta calculation
- `analyze_sector_exposure` - Portfolio sector concentration

### Trading Execution
- `calculate_position_size` - Optimal position sizing based on risk
- `smart_order_entry` - Execute trades with stop-loss/take-profit brackets
- `close_position` - Close existing positions
- `buy_option` - Purchase call/put option contracts

### Utility
- `save_notes` - Save insights to memory
- `read_notes` - Retrieve past insights
- `add_to_watchlist` / `remove_from_watchlist` - Manage symbol tracking

## Architecture

The MCP Server (`mcp_server.py`) exposes the `AsyncToolbox` via standard stdio, allowing any MCP-compliant client to orchestrate the trading tools without custom integration code.