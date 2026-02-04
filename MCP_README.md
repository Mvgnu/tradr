# AURA Trading System - MCP Server

This directory contains the Model-Context Protocol (MCP) server implementation for the AURA trading system, allowing Gemini CLI to directly discover and use trading tools.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements-mcp.txt
```

### 2. Configure Gemini CLI

Copy the `gemini-mcp-config.json` to your Gemini CLI configuration directory:

```bash
# On macOS/Linux
cp gemini-mcp-config.json ~/.config/gemini/config.json

# Or merge with existing config
```

### 3. Test the MCP Server

```bash
cd tradr
python test_mcp_server.py
```

## 📋 Available Tools

The MCP server exposes the following trading tools to Gemini CLI:

### Research Tools
- `get_market_news` - Get latest market news and sentiment
- `assess_market_regime` - Analyze market regime (bull/bear, volatility)
- `scan_top_movers` - Find stocks with unusual activity
- `analyze_sector_exposure` - Analyze portfolio sector exposure
- `analyze_intermarket_correlation` - Analyze correlations between asset classes

### Technical Analysis Tools
- `analyze_technicals` - Comprehensive technical analysis
- `analyze_volume_trend` - Volume trend and quality analysis

### Portfolio Management Tools
- `get_portfolio_state` - Current portfolio positions
- `calculate_risk_metrics` - Portfolio risk metrics (Sharpe, drawdown, VaR)
- `calculate_portfolio_beta` - Portfolio beta calculation

### Trading Execution Tools
- `calculate_position_size` - Optimal position sizing
- `smart_order_entry` - Execute trades with stop-loss/take-profit
- `close_position` - Close existing positions

### Utility Tools
- `save_notes` - Save notes to memory system

## 🔧 Configuration

### Gemini CLI Config (`gemini-mcp-config.json`)

```json
{
  "mcpServers": {
    "auraTrading": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "./tradr",
      "timeout": 30000,
      "trust": true
    }
  }
}
```

### Environment Variables

Make sure your `.env` file contains the necessary API keys:

```env
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
FINNHUB_KEY=your_finnhub_key
```

## 🧪 Testing

### Test the MCP Server

```bash
python test_mcp_server.py
```

### Test with Gemini CLI

Once configured, you can use the tools directly in Gemini CLI:

```
# Example conversation
You: "Get the latest market news and analyze the current market regime"

Gemini CLI will automatically discover and use:
- get_market_news
- assess_market_regime
```

## 🏗️ Architecture

### MCP Server (`mcp_server.py`)
- Exposes all trading tools via MCP protocol
- Handles tool discovery and execution
- Manages async operations and error handling

### Tool Integration
- All tools from `AsyncToolbox` are automatically exposed
- JSON schema validation for tool inputs
- Proper error handling and logging

### Benefits of MCP Approach
1. **Standard Protocol**: Uses industry-standard MCP
2. **Tool Discovery**: Gemini CLI automatically discovers available tools
3. **No Custom Orchestration**: Eliminates need for custom prompt engineering
4. **Better Integration**: Native integration with Gemini CLI
5. **Extensible**: Easy to add new tools

## 🔍 Troubleshooting

### Common Issues

1. **MCP Server Not Starting**
   - Check Python path and dependencies
   - Verify `.env` file exists with API keys

2. **Tools Not Discovered**
   - Check Gemini CLI configuration
   - Verify MCP server is running
   - Check logs for errors

3. **Tool Execution Errors**
   - Check API keys and permissions
   - Verify market hours (if applicable)
   - Check network connectivity

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Usage Examples

### Market Analysis
```
"Analyze the current market regime and get the latest news"
```

### Portfolio Review
```
"Get my current portfolio state and calculate risk metrics"
```

### Trading Setup
```
"Scan for top movers, analyze AAPL technically, and calculate position size"
```

### Trade Execution
```
"Enter a smart order for AAPL with 10 shares, 5% take profit, 2% stop loss"
```

## 🔄 Migration from Custom Orchestration

The MCP approach eliminates the need for:
- Custom prompt engineering
- Two-agent pipeline orchestration
- Complex JSON extraction logic
- Manual tool calling enforcement

Instead, Gemini CLI handles all of this natively through the MCP protocol. 