import pandas as pd
from typing import List, Dict

def calculate_twrr(daily_snapshots: List[Dict]) -> float:
    """
    Calculates the Time-Weighted Rate of Return (TWRR).

    Args:
        daily_snapshots: A list of dictionaries, each containing 'timestamp'
                         and 'market_value'. Must be sorted by timestamp.
                         It should also include snapshots on days of cash flows.

    Returns:
        The TWRR as a percentage (e.g., 15.5 for 15.5%).
    """
    if len(daily_snapshots) < 2:
        return 0.0

    df = pd.DataFrame(daily_snapshots)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # Calculate cash flows as the change in market value minus the investment return
    df['end_of_day_value'] = df['market_value']
    df['start_of_day_value'] = df['end_of_day_value'].shift(1)
    # On the first day, the starting value is the first market value
    df.loc[df.index[0], 'start_of_day_value'] = df.loc[df.index[0], 'end_of_day_value']

    # Use provided cash flow data when available; otherwise assume 0
    if 'cash_flow' not in df.columns:
        df['cash_flow'] = 0.0

    # Calculate the return for each sub-period
    # HPR = (End Value - Cash Flow) / Start Value
    df = df[df['start_of_day_value'] > 0]
    df['hpr'] = (df['end_of_day_value'] - df['cash_flow']) / df['start_of_day_value']

    # Geometric linking of the holding period returns
    # TWRR = (Product of all (1 + R_subperiod)) - 1
    # We use df['hpr'] directly because it's already (1+R)
    twrr = df['hpr'].prod() - 1

    return twrr * 100 # Return as a percentage 
