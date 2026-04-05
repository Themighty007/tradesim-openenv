from pydantic import BaseModel, field_validator
from typing import Literal, List, Dict, Optional

class Bar(BaseModel):
    timestep: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class Observation(BaseModel):
    current_price: float
    last_5_bars: List[Bar]
    indicators: Dict[str, float]  # rsi, ma20, macd, bb_upper, bb_lower
    portfolio: Dict[str, float]   # cash, position, unrealized_pnl
    timestep: int
    task_description: str
    market_regime_hint: str

class Action(BaseModel):
    decision: Literal["BUY", "SELL", "HOLD"]
    position_size: float = 0.5
    stop_loss_pct: float = 0.05
    reasoning: str = ""

    @field_validator("position_size")
    @classmethod
    def clamp_size(cls, v):
        return max(0.0, min(1.0, v))

class Trade(BaseModel):
    timestep: int
    decision: str
    price: float
    quantity: float
    cash_after: float

class Reward(BaseModel):
    score: float
    feedback: str
    partial_breakdown: Dict[str, float]

class State(BaseModel):
    task_id: int
    current_timestep: int
    portfolio_value: float
    peak_value: float
    trade_history: List[Trade]
    pnl_curve: List[float]