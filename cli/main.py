#!/usr/bin/env python3
"""
LLM-Driven Paper-Trading Bot (educational sample) - v6 Refactored

Modular architecture with async tooling, caching, and proper separation of concerns.
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file at project root
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

# Debug: Check if FINNHUB_KEY is loaded (masked for security)
finnhub_key = os.getenv('FINNHUB_KEY')
if finnhub_key:
    print(f"FINNHUB_KEY loaded: {finnhub_key[:6]}...{finnhub_key[-4:] if len(finnhub_key) > 10 else '***'}")
else:
    print("FINNHUB_KEY not found in environment variables")

from tradr.core.config import CONFIG, setup_clients
from tradr.core.models import safe_serialize
from tradr.tools.toolbox import AsyncToolbox
from tradr.tools.cli_executor import GeminiCliRunner
from tradr.tools.json_extractor import JsonExtractor
from tradr.memory.agent_memory import AgentMemory

# Setup comprehensive logging for the 3-cycle test
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agentic_3cycle_test.log"),
        logging.StreamHandler()
    ]
)


class TradingAction(BaseModel):
    action: str
    symbol: str
    quantity: int
    reason: str


def extract_tool_examples(toolbox_instance: AsyncToolbox) -> str:
    """
    Extract tool usage examples from toolbox methods programmatically.
    Only extracts docstrings, does not call the actual methods.
    
    Args:
        toolbox_instance: Instance of AsyncToolbox
        
    Returns:
        Formatted string with all tool examples
    """
    examples = []
    
    # Define the tools we want examples for
    tool_methods = [
        'get_market_news',
        'get_latest_sec_filings', 
        'analyze_technicals',
        'analyze_volume_trend',
        'assess_market_regime',
        'calculate_risk_metrics',
        'analyze_sector_exposure',
        'get_portfolio_state',
        'analyze_intermarket_correlation',
        'calculate_portfolio_beta',
        'calculate_position_size',
        'smart_order_entry',
        'modify_trade_parameters',
        'close_position',
        'add_to_watchlist',
        'remove_from_watchlist',
        'save_notes',  # Add the new save_notes method
        'scan_top_movers',
    ]
    
    for method_name in tool_methods:
        if hasattr(toolbox_instance, method_name):
            method = getattr(toolbox_instance, method_name)
            if hasattr(method, '__doc__') and method.__doc__:
                doc = method.__doc__
                
                # Extract example tool input and output from docstring
                input_match = re.search(r'Example Tool Input:(.*?)(?=Example Output:|$)', doc, re.DOTALL)
                output_match = re.search(r'Example Output:(.*?)(?=\n\s*\n|$)', doc, re.DOTALL)
                
                if input_match and output_match:
                    tool_input = input_match.group(1).strip()
                    output = output_match.group(1).strip()
                    
                    # Clean up the examples
                    tool_input = re.sub(r'^\s*', '', tool_input, flags=re.MULTILINE)
                    output = re.sub(r'^\s*', '', output, flags=re.MULTILINE)
                    
                    example_text = f"""
**{method_name}**
**Tool Input:**
```json
{tool_input}
```
"""
                    examples.append(example_text)
    
    if examples:
        return "\n".join(examples)
    else:
        return "No tool examples found."


class AdvancedAutonomousTradingAgent:
    """AURA (Autonomous Unified Reasoning Agent) - Tier-1 autonomous trading system."""
    
    def __init__(self):
        """Initialize the agent."""
        self.clients = setup_clients()
        self.memory = AgentMemory()
        self.toolbox = AsyncToolbox(self.clients, memory=self.memory)
        self.json_extractor = JsonExtractor()
        self.running = False
        self.consecutive_errors = 0
        
        # Initialize ReAct components
        print("🔧 Initializing ReAct components...")
        self.cli_runner = GeminiCliRunner()
        print("✅ ReAct components initialized")
        print("🎯 AURA agent initialization complete! Using Researcher-Trader Pipeline.")

    # --------------------------------------------------------------------
    # SECTION 1: PROMPT GENERATION
    # --------------------------------------------------------------------

    def _generate_researcher_prompt(self, world_state: dict) -> str:
        """Generates the specialized prompt for AURA-R, injecting the current world state."""
        return f"""You are AURA-R, a Tier-1 macro and quantitative research system.
Your SOLE PURPOSE is to analyze the market and generate a watchlist of 5-10 high-potential tickers.

**Current World State:**
- Time: {world_state.get('time')}
- Market Regime: {world_state.get('market_regime', {}).get('trend', 'unknown')} ({world_state.get('market_regime', {}).get('volatility', 'normal')} volatility)
- Existing Positions: {len(world_state.get('portfolio_state', []))}

**Your Workflow:**
1.  **Establish a Controlling Narrative:** Use `get_market_news` and `assess_market_regime` to form a market thesis.
2.  **Identify Strong/Weak Sectors:** Use `scan_top_movers` to find active areas that align with your narrative.
3.  **Finalize and Output:** Once you have a high-conviction list, your FINAL action MUST be to call the `return_research_results` tool.

**CRITICAL OUTPUT REQUIREMENT:**
Your final action must be a call to `return_research_results` with a `controlling_narrative` and a list of `candidates`. Each candidate needs a `symbol` and `reason`.

**EXAMPLE FINAL OUTPUT:**
```json
{{
  "tool_name": "return_research_results",
  "args": {{
    "controlling_narrative": "AI sector shows strength post-earnings, with spillover into semiconductors.",
    "candidates": [
      {{ "symbol": "MRVL", "reason": "Strong momentum in semiconductors, sympathy play." }},
      {{ "symbol": "SMCI", "reason": "High volume gainer, leader in AI servers." }}
    ]
  }}
}}
```

You are FORBIDDEN from using `analyze_technicals` or `smart_order_entry`.
Your available tools are: `get_market_news`, `assess_market_regime`, `scan_top_movers`, and `return_research_results`.

Begin your research process now.
"""

    def _generate_trader_prompt(self, watchlist_data: dict, world_state: dict) -> str:
        """Generates the specialized prompt for AURA-T, injecting the watchlist and world state."""
        candidates_str = "\n".join([f"- {c['symbol']}: {c['reason']}" for c in watchlist_data.get('candidates', [])])
        
        return f"""You are AURA-T, a Tier-1 technical trading and execution system.
The research division has provided you with a watchlist. Your SOLE PURPOSE is to perform due diligence and execute a single, high-conviction trade.

**Current World State:**
- Time: {world_state.get('time')}
- Portfolio Value: ${world_state.get('portfolio_state', [{}])[0].get('market_value', 'N/A')}
- Open Positions: {len(world_state.get('portfolio_state', []))}

**Your Input from Research Division:**
- Controlling Narrative: {watchlist_data.get('controlling_narrative', 'Not provided')}
- Candidates to Analyze:
{candidates_str}

**Your Workflow:**
1.  **Analyze Each Candidate:** For each ticker, perform a technical and volume check using `analyze_technicals` and `analyze_volume_trend`.
2.  **Synthesize and Decide:** Select the single BEST setup. A high-conviction setup requires confluence between the narrative, technicals (price structure), and volume (confirmation).
3.  **Execute or Pass:**
    a. **If a high-conviction setup exists (Conviction Score > 7):** Your FINAL action must be to call `smart_order_entry` after calculating the position size.
    b. **If NO candidate meets your strict criteria:** Your FINAL action must be to call `pass_on_trades` with a clear `reason`.

Your response MUST be a tool call. You are FORBIDDEN from using `get_market_news`.
Your available tools are: `analyze_technicals`, `analyze_volume_trend`, `calculate_position_size`, `get_portfolio_state`, `smart_order_entry`, and `pass_on_trades`.

Begin your technical analysis now.
"""

    # --------------------------------------------------------------------
    # SECTION 2: ORCHESTRATION & EXECUTION
    # --------------------------------------------------------------------

    async def run_pipeline_cycle(self):
        """Executes the full, stateless Researcher-Trader pipeline."""
        print("🚀 STARTING RESEARCH-TRADER PIPELINE CYCLE 🚀")
        
        world_state = await self._get_world_state()
        if not world_state:
            print("🚨 Could not build world state. Aborting cycle.")
            return

        # === PHASE 1: RESEARCH (AURA-R) ===
        print("\n--- PHASE 1: AURA-R (Researcher) ---")
        research_prompt = self._generate_researcher_prompt(world_state)
        research_tools = {
            "get_market_news": self.toolbox.get_market_news,
            "assess_market_regime": self.toolbox.assess_market_regime,
            "scan_top_movers": self.toolbox.scan_top_movers,
            "return_research_results": self.toolbox.return_research_results,
        }
        # The "terminal tool" for this phase is return_research_results
        watchlist_data = await self._run_agent_phase(
            research_prompt, research_tools, terminal_tool="return_research_results"
        )
        
        if not watchlist_data:
            print("🚨 RESEARCH PHASE FAILED: Agent did not return a watchlist. Ending cycle.")
            return
        print(f"✅ RESEARCH PHASE COMPLETE. Watchlist received.")

        # === PHASE 2: TRADING (AURA-T) ===
        print("\n--- PHASE 2: AURA-T (Trader) ---")
        trading_prompt = self._generate_trader_prompt(watchlist_data, world_state)
        trading_tools = {
            "analyze_technicals": self.toolbox.analyze_technicals,
            "analyze_volume_trend": self.toolbox.analyze_volume_trend,
            "calculate_position_size": self.toolbox.calculate_position_size,
            "get_portfolio_state": self.toolbox.get_portfolio_state,
            "smart_order_entry": self.toolbox.smart_order_entry,
            "pass_on_trades": self.toolbox.pass_on_trades,
        }
        final_action = await self._run_agent_phase(
            trading_prompt, trading_tools, terminal_tool=["smart_order_entry", "pass_on_trades"]
        )
        
        if not final_action:
            print("🚨 TRADING PHASE FAILED: Agent did not produce a final action. Ending cycle.")
            return
            
        if final_action.get("tool_name") == "smart_order_entry":
            print(f"✅ TRADE EXECUTED: {final_action.get('args')}")
        elif final_action.get("tool_name") == "pass_on_trades":
            print(f"🟡 PASSED: No trade executed. Reason: {final_action.get('args', {}).get('reason')}")

        print("\n🏁 PIPELINE CYCLE COMPLETE 🏁")

    async def _run_agent_phase(self, prompt: str, available_tools: dict, terminal_tool: list or str):
        """
        Runs a single agent phase until a terminal tool is called.

        This is a pure ReAct loop: Reason -> Act (call tool) -> Observe (get result) -> Repeat.
        """
        prompt_history = prompt
        if isinstance(terminal_tool, str):
            terminal_tool = [terminal_tool]

        for turn in range(10): # Max 10 turns per phase
            print(f"\n🔄 Phase Turn {turn + 1}/10")
            
            # 1. REASON
            llm_output = await self._get_llm_response(prompt_history)
            prompt_history += f"\n\n--- Agent Turn {turn + 1} ---\n{llm_output}"
            
            # 2. ACT
            extracted_json = self.json_extractor.extract(llm_output)
            if not extracted_json:
                print("⚠️ Agent did not provide a valid tool call. Continuing...")
                prompt_history += "\n\n--- System ---\nError: No valid JSON tool call found. You must respond with a tool call."
                continue

            plan = extracted_json[0]
            tool_name = plan.get("tool_name")
            tool_args = plan.get("args", {})

            print(f"🤖 Agent wants to call: {tool_name} with args: {tool_args}")

            # Check if it's the terminal tool for this phase
            if tool_name in terminal_tool:
                # If the tool is a real one (like smart_order_entry), execute it.
                if tool_name in available_tools:
                    await available_tools[tool_name](**tool_args)
                # Return the arguments from the terminal call as the result of the phase.
                print(f"✅ Terminal tool '{tool_name}' called. Phase complete.")
                return plan

            # If not a terminal tool, it must be an available research/analysis tool
            if tool_name in available_tools:
                # 3. OBSERVE
                tool_result = await available_tools[tool_name](**tool_args)
                tool_result_str = json.dumps(self._make_json_serializable(tool_result), indent=2)
                
                print(f"🛠️ Tool Result:\n{tool_result_str}")
                prompt_history += f"\n\n--- Tool Result ---\n{tool_result_str}"
            else:
                print(f"❌ ERROR: Agent called an unavailable or unknown tool: {tool_name}")
                prompt_history += f"\n\n--- System ---\nError: Tool '{tool_name}' is not available in this phase. Available tools are: {list(available_tools.keys())}"

        print("🚨 Phase timed out after 10 turns.")
        return None

    async def _get_world_state(self) -> dict:
        """Gathers essential, high-level context for the agents."""
        try:
            portfolio_state = await self.toolbox.get_portfolio_state()
            market_regime = await self.toolbox.assess_market_regime()
            
            return {
                "time": datetime.now(timezone.utc).isoformat(),
                "portfolio_state": portfolio_state,
                "market_regime": market_regime,
            }
        except Exception as e:
            print(f"Error building world state: {e}")
            return None
            
    # --------------------------------------------------------------------
    # SECTION 3: UTILITIES & MAIN LOOP
    # --------------------------------------------------------------------
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif asyncio.iscoroutine(obj):
            return f"[Coroutine: {obj.__name__}]"
        else:
            return obj

    async def _get_llm_response(self, prompt_history: str) -> str:
        """Get LLM response using the CLI runner."""
        print(f"📝 Sending prompt to agent (length: {len(prompt_history)} chars)")
        return await self.cli_runner.execute(prompt_history)

    async def run(self, test_mode: bool = False):
        """Main run loop that now uses the pipeline."""
        self.running = True
        print("🚀 AURA (Advanced Autonomous Trading Agent) starting...")
        
        cycle_count = 0
        max_cycles = 3 if test_mode else float('inf')
        
        while self.running and cycle_count < max_cycles:
            cycle_count += 1
            print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"🧪 CYCLE {cycle_count}/{max_cycles} STARTING")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # Call the new pipeline cycle instead of the old run_cycle
            await self.run_pipeline_cycle()

            # === ADD DAILY SNAPSHOT LOGGING ===
            try:
                portfolio_state = await self.toolbox.get_portfolio_state()
                account = await asyncio.get_running_loop().run_in_executor(None, self.clients["alpaca"].get_account)
                total_equity = float(account.equity)
                self.memory.log_daily_snapshot(total_equity)
                logging.info(f"Logged daily snapshot. Equity: ${total_equity:.2f}")
            except Exception as e:
                logging.error(f"Could not log daily portfolio snapshot: {e}")
            
            print(f"✅ CYCLE {cycle_count}/{max_cycles} COMPLETED")
            if test_mode and cycle_count >= max_cycles:
                break
            
            await asyncio.sleep(CONFIG.get("main_loop_sleep_seconds", 300))
        
        print("🏁 AURA stopped")

def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AURA Trading Agent')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode (3 cycles with comprehensive logging)')
    args = parser.parse_args()
    
    agent = AdvancedAutonomousTradingAgent()
    try:
        asyncio.run(agent.run(test_mode=args.test))
    except KeyboardInterrupt:
        print("🛑 Agent terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()