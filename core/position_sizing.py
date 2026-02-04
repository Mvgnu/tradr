import logging
from typing import Tuple, Optional


def calculate_position_size(
    symbol: str,
    entry_price: float,
    position_size: float,
    total_equity: float,
    buying_power: float,
    force_minimum: bool = False,
) -> Tuple[int, float]:
    """
    Calculate position size with fixes for high-priced tickers.

    Args:
        symbol: Stock symbol
        entry_price: Entry price per share
        position_size: Position size as percentage of total equity (0.0 to 1.0)
        total_equity: Total account equity
        buying_power: Available buying power
        force_minimum: Whether to force at least 1 share if possible

    Returns:
        Tuple of (shares, actual_position_value)
    """
    if entry_price <= 0:
        logging.warning(f"Invalid entry price for {symbol}: {entry_price}")
        return 0, 0.0

    if position_size <= 0 or position_size > 1:
        logging.warning(f"Invalid position size for {symbol}: {position_size}")
        return 0, 0.0

    # Calculate target position value
    target_position_value = total_equity * position_size

    # Calculate shares based on target position value
    shares = int(target_position_value / entry_price)

    # Force minimum logic - if we want at least 1 share and have buying power
    if force_minimum and shares < 1 and buying_power >= entry_price:
        shares = 1
        actual_position_value = entry_price
        logging.info(f"Force minimum: 1 share of {symbol} at ${entry_price:.2f}")
    elif shares >= 1:
        actual_position_value = shares * entry_price

        # Check if we have enough buying power
        if actual_position_value > buying_power:
            # Reduce shares to fit buying power
            shares = int(buying_power / entry_price)
            actual_position_value = shares * entry_price

            if shares < 1:
                logging.warning(
                    f"Insufficient buying power for {symbol}: need ${entry_price:.2f}, have ${buying_power:.2f}"
                )
                return 0, 0.0
    else:
        # Can't afford even 1 share
        logging.warning(f"Cannot afford {symbol}: 1 share = ${entry_price:.2f}, buying power = ${buying_power:.2f}")
        return 0, 0.0

    logging.info(f"Position size for {symbol}: {shares} shares at ${entry_price:.2f} = ${actual_position_value:.2f}")
    return shares, actual_position_value


def calculate_portfolio_beta(positions: list, market_data: dict, lookback_days: int = 60) -> float:
    """
    Calculate portfolio beta with proper handling of thin histories.

    Args:
        positions: List of position dictionaries
        market_data: Market data dictionary
        lookback_days: Number of days to look back for correlation

    Returns:
        Portfolio beta (defaults to 1.0 for thin histories with warning)
    """
    if not positions:
        return 1.0

    # Check if we have enough data points
    if lookback_days < 30:
        logging.warning(f"Lookback period too short ({lookback_days} days), defaulting to beta 1.0")
        return 1.0

    try:
        # Calculate weighted average beta
        total_value = sum(pos.get("market_value", 0) for pos in positions)

        if total_value == 0:
            logging.warning("Total portfolio value is 0, defaulting to beta 1.0")
            return 1.0

        weighted_beta = 0.0

        for position in positions:
            symbol = position.get("symbol")
            market_value = position.get("market_value", 0)
            weight = market_value / total_value

            # Get individual stock beta (simplified - in practice would calculate from price data)
            stock_beta = market_data.get(symbol, {}).get("beta", 1.0)

            weighted_beta += weight * stock_beta

        # Warn if we're using default betas for many positions
        default_beta_count = sum(
            1 for pos in positions if market_data.get(pos.get("symbol"), {}).get("beta", 1.0) == 1.0
        )

        if default_beta_count > len(positions) * 0.5:
            logging.warning(f"Using default beta 1.0 for {default_beta_count}/{len(positions)} positions")

        return max(0.1, min(3.0, weighted_beta))  # Clamp between 0.1 and 3.0

    except Exception as e:
        logging.error(f"Error calculating portfolio beta: {e}")
        return 1.0


def calculate_kelly_fraction(
    win_rate: float, avg_win_pct: float, avg_loss_pct: float, max_kelly: float = 0.25
) -> float:
    """
    Calculate Kelly fraction for position sizing using percentage returns.

    Args:
        win_rate: Win rate (0.0 to 1.0)
        avg_win_pct: Average win percentage (e.g., 0.05 for 5%)
        avg_loss_pct: Average loss percentage (e.g., 0.03 for 3%)
        max_kelly: Maximum Kelly fraction to prevent over-leverage

    Returns:
        Kelly fraction (0.0 to max_kelly)
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0

    if avg_win_pct <= 0 or avg_loss_pct <= 0:
        return 0.0

    # Kelly formula: f = (bp - q) / b
    # where b = avg_win_pct / avg_loss_pct, p = win_rate, q = 1 - win_rate
    b = avg_win_pct / avg_loss_pct
    p = win_rate
    q = 1 - win_rate

    kelly_fraction = (b * p - q) / b

    # Clamp to reasonable range
    kelly_fraction = max(0.0, min(max_kelly, kelly_fraction))

    return kelly_fraction
