# Tradr

Async trading toolbox + MCP server for AURA. Includes market scans, portfolio/risk analytics, trade execution, and lightweight memory (watchlist + notes).

## Disclaimer

This software can place live orders if configured with your broker API key.

It is for research/experimentation. Running it against real money is entirely at your own risk. This is not financial advice or a managed service.

## Quick start

Install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-mcp.txt
```

Set environment:
```bash
cp .env.example .env
```

Run MCP server (stdio):
```bash
python mcp_server.py
```

Run HTTP MCP server:
```bash
python http_mcp_server.py
```

## Environment variables

See `.env.example` for the full list. Required:
- `ALPACA_KEY`
- `ALPACA_SECRET`
- `GOOGLE_API_KEY`
- `FINNHUB_KEY`

Optional:
- `SEC_API_KEY`
- `EARNINGSFEED_API_KEY`

## MCP tools (high level)

Research / scanning:
- `get_market_news`
- `get_latest_sec_filings`
- `assess_market_regime`
- `scan_top_movers`
- `scan_unusual_options_flow`
- `scan_insider_activity` (EarningsFeed)
- `scan_13f_changes` (EarningsFeed)

Technicals:
- `analyze_technicals`
- `analyze_volume_trend`

Portfolio / risk:
- `get_portfolio_state`
- `analyze_sector_exposure`
- `calculate_risk_metrics`
- `calculate_portfolio_beta`

Execution:
- `smart_order_entry` (bracket orders)
- `modify_trade_parameters` (stop updates)
- `close_position`
- `buy_option`

Watchlist + notes:
- `add_to_watchlist`
- `get_watchlist` (returns fresh technicals per symbol)
- `remove_from_watchlist`
- `save_notes` / `read_notes`

## Storage locations

- Notes: `notes.md` (sliding window of last 10 entries).
- MCP memory (watchlist, trades, etc.): `/tmp/agent_memory.json`.
- CLI memory: `agent_memory.json` in repo root.

## Tests

Tool suite:
```bash
python run_tests.py --type tools
```

### 1. Claude Desktop
Add this to your `claude_desktop_config.json` (usually `~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

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
*Note: We recommend using absolute paths to your venv python executable. Also, ensure the Python environment has `mcp` installed.*

### 2. Gemini Code Assist / Editor
Add this to your project's `.gemini/settings.json` (or global settings):

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


## Notes

I personally suggest using this with Gemini 3 Pro. In my personal experience Gemini 3 outperformed Opus 4.5 -
this is strictly my personal experience and should not be taken as financial advice or a recommendation.

I have not yet tested GPT 5.2 or Grok with this toolset.

## How to Vibe trade

Wire up the MCP server to your AI agent of choice, set the environment variables, and start vibing.

Example prompt:

"Hmm google earnings today, stock down a lil
I know:
- anthropic trained opus 4.5 on tpus
- cost reduction likely due to tpus
- inference on tpus
- msft cloud disappointment
- openai previously also eyed tpus for their models

its basically the if transformers are a thing, google is the sauce, and i am extremely bullish on them, obviously doesnt mean the market agrees - but earnings are after hours today, opinion?"

Agent will look up options flows, insider activity, and 13f changes, company news, and technicals to make a informed suggestion or YOLO your money its an LLM you never know - beware risks THERE ARE NO GUARDRAILS IN PLACE.

## LICENSE

MIT