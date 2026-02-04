from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from enum import Enum
import json

class MarketTrend(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"

class VolatilityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class TradeData(BaseModel):
    """Trade data model for logging and analysis."""
    symbol: str
    qty: float
    price: float
    side: OrderSide
    profit: Optional[float] = None
    timestamp: str
    order_id: Optional[str] = None
    reason: Optional[str] = None
    estimated_slippage: Optional[float] = None
    actual_slippage: Optional[float] = None

class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None

class MarketCondition(BaseModel):
    """Market condition model."""
    timestamp: str
    market_trend: MarketTrend
    volatility: VolatilityLevel
    key_observations: List[str]

class WatchlistItem(BaseModel):
    """Watchlist item model."""
    symbol: str
    added_at: str
    reason: str = ""

class RiskMetrics(BaseModel):
    """Risk metrics model."""
    current_drawdown: float
    portfolio_beta: float
    sector_concentration: Dict[str, float]
    position_concentration: Dict[str, float]
    volatility_30d: float
    var_95: float
    cvar_95: float # New field for Conditional Value at Risk
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None # New field for the matrix
    max_position_size: float
    total_exposure: float

class MarketRegime(BaseModel):
    """Market regime analysis model."""
    trend: MarketTrend
    volatility: VolatilityLevel
    confidence: float
    key_indicators: Dict[str, Any]
    regime_duration_days: int
    regime_strength: float

class SectorExposure(BaseModel):
    """Sector exposure model."""
    sectors: Dict[str, float]
    top_sectors: List[str]
    concentration_risk: float
    diversification_score: float

class TechnicalAnalysis(BaseModel):
    """Technical analysis model."""
    symbol: str
    rsi: float
    macd_signal: str
    trend: str
    support_levels: List[float]
    resistance_levels: List[float]
    volume_trend: str
    momentum_score: float

class VolumeAnalysis(BaseModel):
    """Volume analysis model."""
    symbol: str
    current_volume: int
    volume_pattern: str
    volume_ratio_20d: float
    volume_ratio_50d: float
    volume_quality_score: float
    vpt_trend: str
    obv_trend: str

class PortfolioPosition(BaseModel):
    """Portfolio position model."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    sector: str

class EconomicEvent(BaseModel):
    """Economic calendar event model."""
    date: str
    event: str
    impact: str
    country: str

class NewsItem(BaseModel):
    """Market news item model."""
    category: str
    datetime: int
    headline: str
    id: int
    image: Optional[str] = None
    related: Optional[str] = None
    source: str
    summary: str
    url: str

class SECFiling(BaseModel):
    """SEC filing model."""
    symbol: str
    filing_type: str
    filing_date: str
    description: str
    url: str
    impact_score: Optional[float] = None

class OrderData(BaseModel):
    """Order data model."""
    symbol: str
    qty: float
    side: OrderSide
    type: OrderType
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None

class OrderResult(BaseModel):
    """Order result model."""
    order_id: str
    status: str
    filled_qty: float
    filled_avg_price: Optional[float] = None
    commission: Optional[float] = None
    error_message: Optional[str] = None

class WorldContext(BaseModel):
    """World context model for agent reasoning."""
    market_regime: MarketRegime
    risk_metrics: RiskMetrics
    sector_exposure: SectorExposure
    strategy_parameters: Dict[str, Any]
    current_time: str
    market_status: str
    recent_news: List[NewsItem] = []
    upcoming_events: List[EconomicEvent] = []

class AgentPlan(BaseModel):
    """Agent plan model."""
    reasoning: str
    actions: List[Dict[str, Any]]
    confidence: ConfidenceLevel
    risk_assessment: str
    expected_outcome: str

def safe_serialize(obj: Any) -> Any:
    """Safely serialize any object using Pydantic models when possible."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode='json')
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        # Try to convert to dict and serialize
        try:
            return safe_serialize(obj.__dict__)
        except Exception:
            return str(obj)
    else:
        return str(obj)
