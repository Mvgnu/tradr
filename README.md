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
