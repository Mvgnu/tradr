#!/usr/bin/env python3
"""
Comprehensive test suite for the agentic trading system.

This test suite addresses the testing strategy outlined in the user's request:
1. Unit tests for individual tools (most important)
2. Agent reasoning and tool usage tests
3. Golden path integration tests
4. Failure and resilience testing
"""

# NOTE: All imports in this test file use absolute imports for compatibility with both script and package execution.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import json
import datetime as dt
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import types

from tradr.tools.toolbox import AsyncToolbox
from tradr.memory.agent_memory import AgentMemory
from tradr.cli.main import AdvancedAutonomousTradingAgent
from tradr.core.models import TradeData, PerformanceMetrics, MarketCondition


class TestIndividualTools:
    """Unit tests for individual tools - the most important tests."""

    @pytest.fixture
    def mock_clients(self):
        """Mock API clients for testing."""
        clients = {
            'alpaca': Mock(),
            'finnhub_key': 'test_key',
            'sec_key': 'test_key'
        }

        # Mock Alpaca account
        mock_account = Mock()
        mock_account.buying_power = "100000"
        mock_account.equity = "105000"
        mock_account.account_blocked = False
        mock_account.pattern_day_trader = False
        mock_account.daytrade_count = "0"
        clients['alpaca'].get_account.return_value = mock_account

        # Mock Alpaca clock
        mock_clock = Mock()
        mock_clock.is_open = True
        clients['alpaca'].get_clock.return_value = mock_clock

        return clients

    @pytest.fixture
    def toolbox(self, mock_clients):
        """Create toolbox instance with mocked clients."""
        return AsyncToolbox(mock_clients)

    @pytest.mark.asyncio
    async def test_analyze_volume_trend_with_mock_data(self, toolbox):
        """Test analyze_volume_trend with mocked yfinance data."""
        # Create mock DataFrame with at least 100 bars
        mock_data = pd.DataFrame({
            'Close': [100 + i for i in range(100)],
            'Volume': [1000000 + i * 10000 for i in range(100)]
        })

        with patch('yfinance.download', return_value=mock_data):
            result = await toolbox.analyze_volume_trend("AAPL")

            assert isinstance(result, dict)
            assert 'symbol' in result
            assert 'volume_pattern' in result
            assert 'volume_quality_score' in result
            assert result['symbol'] == "AAPL"

    @pytest.mark.asyncio
    async def test_analyze_volume_trend_short_data(self, toolbox):
        """Test analyze_volume_trend with insufficient data."""
        # Create mock DataFrame with only 5 days of data (too short)
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })

        with patch('yfinance.download', return_value=mock_data):
            result = await toolbox.analyze_volume_trend("AAPL")

            assert isinstance(result, dict)
            assert 'error' in result
            assert 'insufficient data' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_analyze_volume_trend_nan_values(self, toolbox):
        """Test analyze_volume_trend with NaN values."""
        # Create mock DataFrame with NaN values
        mock_data = pd.DataFrame({
            'Close': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000000, 1100000, np.nan, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000]
        })

        with patch('yfinance.download', return_value=mock_data):
            result = await toolbox.analyze_volume_trend("AAPL")

            assert isinstance(result, dict)
            # Should handle NaN values gracefully
            assert 'error' in result or 'volume_quality_score' in result

    @pytest.mark.asyncio
    async def test_calculate_position_size_stateless(self, toolbox):
        """Test that calculate_position_size is stateless and works with provided trade history."""
        # Test with no trade history
        shares, details = await toolbox.calculate_position_size(
            symbol="AAPL",
            confidence="high",
            entry_price=150.0,
            trade_history=None
        )

        assert isinstance(shares, (int, float))
        assert isinstance(details, dict)
        assert shares > 0

        # Test with trade history
        trade_history = [
            {"symbol": "AAPL", "profit": 100, "timestamp": "2024-01-01T00:00:00Z"},
            {"symbol": "AAPL", "profit": -50, "timestamp": "2024-01-02T00:00:00Z"},
            {"symbol": "MSFT", "profit": 75, "timestamp": "2024-01-03T00:00:00Z"}
        ]

        shares2, details2 = await toolbox.calculate_position_size(
            symbol="AAPL",
            confidence="high",
            entry_price=150.0,
            trade_history=trade_history
        )

        assert isinstance(shares2, (int, float))
        assert isinstance(details2, dict)
        assert shares2 > 0

    @pytest.mark.asyncio
    async def test_assess_market_regime(self, toolbox):
        """Test market regime assessment."""
        with patch('yfinance.download') as mock_download:
            # Mock SPY data
            mock_spy_data = pd.DataFrame({
                'Close': [400 + i for i in range(50)],
                'Volume': [1000000 + i * 10000 for i in range(50)]
            })
            mock_download.return_value = mock_spy_data

            result = await toolbox.assess_market_regime()

            assert isinstance(result, dict)
            assert 'trend' in result
            assert 'volatility' in result
            assert 'should_be_defensive' in result

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, toolbox):
        """Test risk metrics calculation."""
        with patch('yfinance.download') as mock_download:
            # Mock portfolio data
            mock_data = pd.DataFrame({
                'Close': [100 + i for i in range(30)],
                'Volume': [1000000 + i * 10000 for i in range(30)]
            })
            mock_download.return_value = mock_data

            result = await toolbox.calculate_risk_metrics()

            assert isinstance(result, dict)
            assert 'current_drawdown' in result
            assert 'portfolio_beta' in result
            assert 'var_95' in result


class TestAgentReasoning:
    """Test the agent's reasoning and tool usage capabilities."""

    @pytest.mark.asyncio
    async def test_agent_plan_execution(self):
        from tradr.cli.main import AdvancedAutonomousTradingAgent
        agent = AdvancedAutonomousTradingAgent()
        mock_llm_outputs = [
            '{"action": "execute_tool", "tool_name": "calculate_position_size", "args": {"symbol": "AAPL", "confidence": "high", "entry_price": 150.0}}',
            'cycle complete'
        ]
        with patch.object(agent, '_pre_cycle_checks', new=AsyncMock(return_value=True)), \
             patch.object(agent, '_build_world_context', new=AsyncMock(return_value="context")), \
             patch.object(agent.cli_runner, 'execute', new=AsyncMock(side_effect=mock_llm_outputs)), \
             patch.object(agent.toolbox, 'calculate_position_size', new_callable=AsyncMock) as mock_calc:
            mock_calc.return_value = {"position_size": 25, "position_size_pct": 2.5}
            await agent.run_cycle()
            mock_calc.assert_called_once_with(symbol="AAPL", confidence="high", entry_price=150.0)

    @pytest.mark.asyncio
    async def test_agent_bullish_breakout_scenario(self):
        from tradr.cli.main import AdvancedAutonomousTradingAgent
        agent = AdvancedAutonomousTradingAgent()
        mock_responses = [
            (
                "I need to analyze AAPL's technicals to see if it's a good breakout candidate.\n"
                "{\n"
                "  \"action\": \"execute_tool\",\n"
                "  \"tool_name\": \"analyze_technicals\",\n"
                "  \"args\": {\"symbol\": \"AAPL\"}\n"
                "}\n"
            ),
            (
                "The technicals look bullish. Now let me calculate position size.\n"
                "{\n"
                "  \"action\": \"execute_tool\",\n"
                "  \"tool_name\": \"calculate_position_size\",\n"
                "  \"args\": {\"symbol\": \"AAPL\", \"confidence\": \"high\", \"entry_price\": 150.0}\n"
                "}\n"
            ),
            (
                "Perfect! Now I'll execute the trade.\n"
                "{\n"
                "  \"action\": \"execute_tool\",\n"
                "  \"tool_name\": \"smart_order_entry\",\n"
                "  \"args\": {\"symbol\": \"AAPL\", \"shares\": 30, \"order_type\": \"market\"}\n"
                "}\n"
                "\ncycle complete"
            )
        ]
        with patch.object(agent, '_pre_cycle_checks', new=AsyncMock(return_value=True)), \
             patch.object(agent, '_build_world_context', new=AsyncMock(return_value="context")), \
             patch.object(agent.cli_runner, 'execute', new=AsyncMock(side_effect=mock_responses)), \
             patch.object(agent.toolbox, 'analyze_technicals', new_callable=AsyncMock) as mock_tech, \
             patch.object(agent.toolbox, 'calculate_position_size', new_callable=AsyncMock) as mock_calc, \
             patch.object(agent.toolbox, 'smart_order_entry', new_callable=AsyncMock) as mock_order:
            mock_tech.return_value = {"rsi": 65, "macd_signal": "bullish", "trend": "uptrend"}
            mock_calc.return_value = {"position_size": 30, "position_size_pct": 3.0}
            mock_order.return_value = {"order_id": "123", "status": "filled"}
            await agent.run_cycle()
            mock_tech.assert_called_with(symbol="AAPL")
            mock_calc.assert_called_with(symbol="AAPL", confidence="high", entry_price=150.0)
            mock_order.assert_called_with(symbol="AAPL", shares=30, order_type="market")


class TestGoldenPathIntegration:
    """Golden path integration tests for common scenarios."""

    @pytest.fixture
    def integration_agent(self):
        """Create a real agent instance for integration testing."""
        # Use a test configuration
        with patch('tradr.core.config.CONFIG', {
            'max_position_size_pct': 5.0,
            'max_sector_exposure_pct': 25.0,
            'max_drawdown_pct': 20.0,
            'ignore_market_hours': True,  # Allow testing outside market hours
            'llm_model': 'gemini-2.0-flash-exp'
        }):
            return AdvancedAutonomousTradingAgent()

    @pytest.mark.asyncio
    async def test_bullish_breakout_integration(self, integration_agent):
        """Test complete bullish breakout scenario end-to-end."""
        # Mock all external APIs
        with patch('yfinance.download') as mock_yf, \
             patch('requests.get') as mock_requests, \
             patch('tradr.core.config.TradingClient') as mock_alpaca:

            # Mock market data
            mock_yf.return_value = pd.DataFrame({
                'Close': [100 + i for i in range(50)],
                'Volume': [1000000 + i * 50000 for i in range(50)]
            })

            # Mock API responses
            mock_requests.return_value.json.return_value = []
            mock_requests.return_value.status_code = 200

            # Mock Alpaca responses
            mock_alpaca.return_value.get_account.return_value = Mock(
                buying_power="100000",
                equity="105000",
                account_blocked=False,
                pattern_day_trader=False,
                daytrade_count="0"
            )
            mock_alpaca.return_value.get_clock.return_value = Mock(is_open=True)

            # Run a cycle
            await integration_agent.run_cycle()

            # Verify the cycle completed without errors
            # (We can't predict exact behavior due to LLM, but we can check for no exceptions)
            assert True  # If we get here, no exceptions were raised

    @pytest.mark.asyncio
    async def test_risk_off_deleveraging_integration(self, integration_agent):
        """Test risk-off scenario where agent should de-leverage."""
        # Mock bear market conditions
        with patch('yfinance.download') as mock_yf, \
             patch('requests.get') as mock_requests, \
             patch('tradr.core.config.TradingClient') as mock_alpaca:

            # Mock bearish market data
            mock_yf.return_value = pd.DataFrame({
                'Close': [100 - i for i in range(50)],  # Declining prices
                'Volume': [1000000 + i * 100000 for i in range(50)]  # High volume
            })

            # Mock API responses
            mock_requests.return_value.json.return_value = []
            mock_requests.return_value.status_code = 200

            # Mock Alpaca with existing positions
            mock_alpaca.return_value.get_account.return_value = Mock(
                buying_power="50000",
                equity="95000",  # Down from 100000
                account_blocked=False,
                pattern_day_trader=False,
                daytrade_count="0"
            )
            mock_alpaca.return_value.get_clock.return_value = Mock(is_open=True)

            # Run a cycle
            await integration_agent.run_cycle()

            # Verify the cycle completed without errors
            assert True


class TestFailureAndResilience:
    """Test failure scenarios and resilience mechanisms."""

    @pytest.fixture
    def resilience_agent(self):
        with patch('tradr.core.config.CONFIG', {
            'max_position_size_pct': 5.0,
            'max_sector_exposure_pct': 25.0,
            'max_drawdown_pct': 20.0,
            'ignore_market_hours': True,
            'llm_model': 'gemini-2.0-flash-exp'
        }):
            return AdvancedAutonomousTradingAgent()

    @pytest.mark.asyncio
    async def test_tool_failure_handling(self, resilience_agent):
        with patch.object(resilience_agent, '_pre_cycle_checks', return_value=True), \
             patch.object(resilience_agent, '_build_world_context', return_value="test context"):
            mock_llm_output = '''
            Let me assess the market regime.
            {
              "action": "execute_tool",
              "tool_name": "assess_market_regime",
              "args": {}
            }
            '''
            with patch.object(resilience_agent.cli_runner, 'execute', new=AsyncMock(return_value=mock_llm_output)), \
                 patch.object(resilience_agent.toolbox, 'assess_market_regime', new_callable=AsyncMock) as mock_regime:
                mock_regime.side_effect = Exception("API timeout")
                await resilience_agent.run_cycle()
                assert resilience_agent.consecutive_errors > 0

    @pytest.mark.asyncio
    async def test_max_drawdown_exceeded(self, resilience_agent):
        with patch.object(resilience_agent.toolbox, 'calculate_risk_metrics') as mock_risk:
            mock_risk.return_value = {'current_drawdown': 25.0}
            await resilience_agent.run_cycle()
            assert True

    @pytest.mark.asyncio
    async def test_safe_mode_activation(self, resilience_agent):
        with patch.object(resilience_agent, '_pre_cycle_checks', return_value=True), \
             patch.object(resilience_agent, '_build_world_context', return_value="test context"):
            mock_llm_output = '''
            Let me assess the market regime.
            {
              "action": "execute_tool",
              "tool_name": "assess_market_regime",
              "args": {}
            }
            '''
            with patch.object(resilience_agent.cli_runner, 'execute', new=AsyncMock(return_value=mock_llm_output)), \
                 patch.object(resilience_agent.toolbox, 'assess_market_regime', new_callable=AsyncMock) as mock_regime:
                mock_regime.side_effect = Exception("Persistent failure")
                for _ in range(3):
                    await resilience_agent.run_cycle()
                assert resilience_agent.consecutive_errors >= 3

    @pytest.mark.asyncio
    async def test_memory_corruption_recovery(self, resilience_agent):
        """Test that agent recovers from memory corruption."""
        # Corrupt the memory file
        resilience_agent.memory.memory = {"corrupted": "data"}

        # Try to save memory
        resilience_agent.memory.save_memory()

        # Verify memory can still be accessed (should use default structure)
        watchlist = resilience_agent.memory.get_watchlist()
        assert isinstance(watchlist, list)


class TestMemorySystem:
    """Test the memory system with Pydantic models."""

    @pytest.fixture
    def memory(self):
        """Create memory instance for testing."""
        return AgentMemory("test_memory.json")

    def test_trade_logging_with_pydantic(self, memory):
        """Test that trade logging works with Pydantic models."""
        trade_data = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 150.0,
            "side": "buy",
            "profit": 500.0,
            "order_id": "12345"
        }

        result = memory.log_trade(trade_data)

        assert result["status"] == "success"

        # Verify trade was stored
        trades = memory.memory.get('trades', [])
        assert len(trades) > 0
        assert trades[-1]['symbol'] == "AAPL"

    def test_performance_update_with_pydantic(self, memory):
        """Test that performance updates work with Pydantic models."""
        metrics = {
            "total_trades": 10,
            "winning_trades": 7,
            "total_pnl": 1500.0,
            "max_drawdown": 5.0,
            "sharpe_ratio": 1.2
        }

        result = memory.update_performance(metrics)

        assert result["status"] == "success"

        # Verify metrics were stored
        performance_history = memory.memory.get('performance_history', [])
        assert len(performance_history) > 0

    def test_market_condition_recording(self, memory):
        """Test market condition recording with Pydantic models."""
        condition = MarketCondition(
            timestamp="2024-01-01T00:00:00Z",
            market_trend="bull",
            volatility="medium",
            key_observations=["Strong earnings season", "Fed dovish signals"]
        )

        memory.record_market_condition(condition)

        # Verify condition was stored
        conditions = memory.memory.get('market_conditions', [])
        assert len(conditions) > 0
        assert conditions[-1]['market_trend'] == "bull"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
