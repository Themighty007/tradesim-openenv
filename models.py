"""
TradeSim v3 — models.py
=======================
FINAL VERSION for Meta x Scaler OpenEnv Hackathon Grand Finale.

New in v3:
  - HMM regime probability fields in observation (unsupervised regime detection)
  - Sharpe ratio in GradeResult (the number every quant cares about)
  - Calmar ratio in GradeResult
  - Granger causality scores stored per episode (proves causal world model)
  - EpisodeMetrics dataclass for the training curve logger
  - Full 3-axis observation: Technical + Fundamental + Psychology
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MarketRegime(str, Enum):
    BULL  = "bull"
    RANGE = "range"
    CRASH = "crash"


class ActionType(str, Enum):
    BUY  = "buy"
    SELL = "sell"
    HOLD = "hold"


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegFloat   = Annotated[float, Field(ge=0.0)]
UnitFloat     = Annotated[float, Field(ge=0.0, le=1.0)]
Price         = Annotated[float, Field(gt=0.0)]
Quantity      = Annotated[float, Field(ge=0.0)]
Cash          = Annotated[float, Field()]
SignedUnit    = Annotated[float, Field(ge=-1.0, le=1.0)]


# ---------------------------------------------------------------------------
# AXIS 1: Technical signals
# ---------------------------------------------------------------------------

class TechnicalSignals(BaseModel):
    model_config = {"frozen": True}

    rsi_14:         float = Field(..., ge=0.0, le=100.0)
    ma_20:          float = Field(..., gt=0.0)
    ma_50:          float = Field(..., gt=0.0)
    macd:           float = Field(...)
    macd_signal:    float = Field(...)
    bb_upper:       float = Field(..., gt=0.0)
    bb_lower:       float = Field(..., gt=0.0)
    bb_pct:         float = Field(...)
    volatility_20:  float = Field(..., ge=0.0)
    price_vs_ma20:  float = Field(...)
    # NEW: momentum signals
    roc_5:          float = Field(0.0, description="Rate of change 5-period")
    roc_20:         float = Field(0.0, description="Rate of change 20-period")
    atr_14:         float = Field(0.0, ge=0.0, description="Average True Range — volatility measure")


# ---------------------------------------------------------------------------
# AXIS 2: Fundamental signals
# ---------------------------------------------------------------------------

class FundamentalSignals(BaseModel):
    model_config = {"frozen": True}

    earnings_surprise:   SignedUnit = Field(...)
    fed_rate_change_bps: float      = Field(..., ge=-100.0, le=100.0)
    macro_gdp_surprise:  SignedUnit = Field(...)
    supply_shock:        SignedUnit = Field(...)
    institutional_flow:  SignedUnit = Field(...)
    # NEW: credit market signals
    credit_spread_bps:   float      = Field(0.0, description="High yield - Treasury spread. Widens before crashes.")
    yield_curve_slope:   float      = Field(0.0, description="10yr - 2yr yield. Negative = inverted = recession warning.")


# ---------------------------------------------------------------------------
# AXIS 3: Psychological signals
# ---------------------------------------------------------------------------

class PsychologySignals(BaseModel):
    model_config = {"frozen": True}

    fear_greed_index:  SignedUnit = Field(...)
    social_sentiment:  SignedUnit = Field(...)
    news_sentiment:    SignedUnit = Field(...)
    put_call_ratio:    float      = Field(..., ge=0.0, le=4.0)
    insider_buying:    SignedUnit = Field(...)
    # NEW: volatility regime signals
    vix_level:         float      = Field(15.0, ge=0.0, description="Implied volatility index. >30=fear, <15=complacency.")
    skew:              float      = Field(0.0,  description="Options skew — demand for downside protection.")


# ---------------------------------------------------------------------------
# NEW: HMM Regime Detection output
# ---------------------------------------------------------------------------

class HMMRegimeSignal(BaseModel):
    """
    Output of the Hidden Markov Model regime detector.
    
    The HMM is trained UNSUPERVISED on rolling log-returns.
    It does NOT know the true regime label — it discovers regime
    structure from price behaviour alone.
    
    This is production technology used at Bridgewater, Man AHL,
    and every systematic macro fund for regime conditioning.
    """
    model_config = {"frozen": True}

    prob_bull:        UnitFloat = Field(..., description="P(current regime = bull) from HMM")
    prob_crash:       UnitFloat = Field(..., description="P(current regime = crash/volatile) from HMM")
    current_state:    int       = Field(..., ge=0, le=1, description="Most likely HMM state (0 or 1)")
    state_confidence: UnitFloat = Field(..., description="max(prob_bull, prob_crash) — how certain the HMM is")
    # Granger causality: does fundamental signal CAUSE price changes?
    granger_earnings_pval: float = Field(1.0, ge=0.0, le=1.0,
        description="p-value: earnings_surprise Granger-causes returns. <0.05 = causal.")
    granger_sentiment_pval: float = Field(1.0, ge=0.0, le=1.0,
        description="p-value: fear_greed Granger-causes returns. <0.05 = causal.")


# ---------------------------------------------------------------------------
# Price window
# ---------------------------------------------------------------------------

class PriceWindow(BaseModel):
    model_config = {"frozen": True}

    raw_prices:        list[Price] = Field(...)
    returns:           list[float] = Field(...)
    normalised_prices: list[float] = Field(...)

    @field_validator("returns")
    @classmethod
    def finite_returns(cls, v):
        if any(not math.isfinite(r) for r in v):
            raise ValueError("All returns must be finite.")
        return v

    @model_validator(mode="after")
    def lengths_consistent(self):
        n = len(self.raw_prices)
        if len(self.returns) != max(n - 1, 0):
            raise ValueError("returns length must be len(raw_prices)-1.")
        if len(self.normalised_prices) != n:
            raise ValueError("normalised_prices length must equal len(raw_prices).")
        return self


# ---------------------------------------------------------------------------
# Portfolio snapshot
# ---------------------------------------------------------------------------

class PortfolioSnapshot(BaseModel):
    model_config = {"frozen": True}

    cash:            Cash        = Field(...)
    shares_held:     Quantity    = Field(...)
    current_price:   Price       = Field(...)
    net_worth:       NonNegFloat = Field(...)
    peak_net_worth:  NonNegFloat = Field(...)
    drawdown:        UnitFloat   = Field(...)
    equity_fraction: UnitFloat   = Field(...)
    total_trades:    int         = Field(..., ge=0)
    total_return:    float       = Field(...)

    @model_validator(mode="after")
    def net_worth_consistent(self):
        implied = self.cash + self.shares_held * self.current_price
        if abs(implied - self.net_worth) > 0.01:
            raise ValueError(f"net_worth inconsistent: {self.net_worth:.4f} vs {implied:.4f}")
        return self

    @model_validator(mode="after")
    def drawdown_consistent(self):
        if self.peak_net_worth > 0:
            expected_dd = max(0.0, 1.0 - self.net_worth / self.peak_net_worth)
            if abs(expected_dd - self.drawdown) > 1e-6:
                raise ValueError(f"drawdown inconsistent")
        return self


# ---------------------------------------------------------------------------
# UPGRADED Observation — all 4 axes + HMM
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Full 4-axis observation for TradeSim v3:
      1. Technical  — RSI, MACD, BB, ATR, ROC
      2. Fundamental — earnings, Fed, GDP, credit spreads, yield curve
      3. Psychological — fear/greed, VIX, put/call, social sentiment
      4. HMM regime — unsupervised regime probability from price behaviour
    """
    model_config = {"frozen": True}

    timestep:    int               = Field(..., ge=0)
    max_steps:   int               = Field(..., gt=0)
    regime:      MarketRegime      = Field(...)
    window:      PriceWindow       = Field(...)
    portfolio:   PortfolioSnapshot = Field(...)
    time_left:   UnitFloat         = Field(...)
    technical:   TechnicalSignals  = Field(...)
    fundamental: FundamentalSignals = Field(...)
    psychology:  PsychologySignals  = Field(...)
    hmm:         HMMRegimeSignal    = Field(...)
    active_agents: list[str]        = Field(default_factory=list)

    @model_validator(mode="after")
    def time_left_consistent(self):
        expected = 1.0 - self.timestep / self.max_steps
        if abs(expected - self.time_left) > 1e-6:
            raise ValueError("time_left inconsistent.")
        return self


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    model_config = {"frozen": True}

    action_type: ActionType = Field(...)
    fraction:    UnitFloat  = Field(0.0)
    reason:      str        = Field("", max_length=512)

    @model_validator(mode="after")
    def hold_fraction_zero(self):
        if self.action_type == ActionType.HOLD and self.fraction != 0.0:
            object.__setattr__(self, "fraction", 0.0)
        return self

    @classmethod
    def buy(cls, fraction=1.0, reason=""):
        return cls(action_type=ActionType.BUY, fraction=fraction, reason=reason)

    @classmethod
    def sell(cls, fraction=1.0, reason=""):
        return cls(action_type=ActionType.SELL, fraction=fraction, reason=reason)

    @classmethod
    def hold(cls, reason=""):
        return cls(action_type=ActionType.HOLD, fraction=0.0, reason=reason)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    model_config = {"frozen": True}

    pnl_reward:       float = Field(...)
    risk_penalty:     float = Field(..., le=0.0)
    drawdown_penalty: float = Field(..., le=0.0)
    turnover_penalty: float = Field(..., le=0.0)
    survival_bonus:   float = Field(..., ge=0.0)
    # NEW: regime alignment bonus
    regime_alignment: float = Field(0.0, description="Bonus for trading WITH the HMM-detected regime")
    total:            float = Field(...)

    @model_validator(mode="after")
    def total_consistent(self):
        implied = (self.pnl_reward + self.risk_penalty + self.drawdown_penalty
                   + self.turnover_penalty + self.survival_bonus + self.regime_alignment)
        if abs(implied - self.total) > 1e-9:
            raise ValueError(f"total inconsistent: {self.total} vs {implied:.9f}")
        return self


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    model_config = {"frozen": True}

    observation: Observation     = Field(...)
    reward:      RewardBreakdown = Field(...)
    done:        bool            = Field(...)
    info:        dict            = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode records
# ---------------------------------------------------------------------------

class StepRecord(BaseModel):
    model_config = {"frozen": True}

    timestep:        int              = Field(..., ge=0)
    action:          Action
    reward:          RewardBreakdown
    net_worth:       NonNegFloat
    drawdown:        UnitFloat
    equity_fraction: UnitFloat
    price:           Price
    technical:       TechnicalSignals
    fundamental:     FundamentalSignals
    psychology:      PsychologySignals
    hmm:             HMMRegimeSignal


class EpisodeRecord(BaseModel):
    model_config = {"frozen": True}

    regime:          MarketRegime
    initial_capital: PositiveFloat
    steps:           list[StepRecord] = Field(..., min_length=1)

    @property
    def final_net_worth(self): return self.steps[-1].net_worth
    @property
    def total_return(self): return (self.final_net_worth - self.initial_capital) / self.initial_capital
    @property
    def peak_net_worth(self): return max(s.net_worth for s in self.steps)
    @property
    def max_drawdown(self): return max(s.drawdown for s in self.steps)
    @property
    def num_trades(self): return sum(1 for s in self.steps if s.action.action_type != ActionType.HOLD)
    @property
    def total_reward(self): return sum(s.reward.total for s in self.steps)


# ---------------------------------------------------------------------------
# UPGRADED GradeResult — now includes Sharpe + Calmar + 3-axis scores
# ---------------------------------------------------------------------------

class GradeResult(BaseModel):
    """
    Complete performance report for one episode.
    
    Sharpe ratio = (mean_reward / std_reward) * sqrt(252)
    Calmar ratio = annualised_return / max_drawdown
    
    These are the two numbers every quant interviewer asks about first.
    """
    model_config = {"frozen": True}

    task:                int       = Field(..., ge=1, le=3)
    score:               UnitFloat = Field(...)
    breakdown:           dict[str, float] = Field(default_factory=dict)
    rationale:           str       = Field("")

    # Professional performance metrics
    sharpe_ratio:        float     = Field(0.0, description="Annualised Sharpe ratio of step rewards")
    calmar_ratio:        float     = Field(0.0, description="Annual return / max drawdown")
    max_drawdown:        float     = Field(0.0, ge=0.0)
    total_return_pct:    float     = Field(0.0)
    num_trades:          int       = Field(0)

    # 3-axis scores
    technical_score:     UnitFloat = Field(0.0)
    fundamental_score:   UnitFloat = Field(0.0)
    psychological_score: UnitFloat = Field(0.0)

    # HMM alignment score
    hmm_alignment_score: UnitFloat = Field(0.0, description="How well agent traded WITH detected regime")

    def __float__(self): return self.score


# ---------------------------------------------------------------------------
# NEW: EpisodeMetrics for training curve logging
# ---------------------------------------------------------------------------

class EpisodeMetrics(BaseModel):
    """
    Logged after every episode for the training curve.
    This is what the judges see as "reward improvement over time."
    """
    episode_num:         int
    task_id:             int
    regime:              str
    score:               float
    sharpe_ratio:        float
    total_return_pct:    float
    max_drawdown_pct:    float
    num_trades:          int
    technical_score:     float
    fundamental_score:   float
    psychological_score: float
    hmm_alignment_score: float
    strategy_update_used: bool = False


# ---------------------------------------------------------------------------
# Trade, State, Config
# ---------------------------------------------------------------------------

class Trade(BaseModel):
    model_config = {"frozen": True}

    timestep:   int   = Field(..., ge=0)
    decision:   str   = Field(...)
    price:      float = Field(..., gt=0)
    quantity:   float = Field(..., ge=0)
    cash_after: float = Field(...)


class State(BaseModel):
    model_config = {"frozen": True}

    task_id:          int         = Field(..., ge=0)
    current_timestep: int         = Field(..., ge=0)
    portfolio_value:  float       = Field(..., ge=0)
    peak_value:       float       = Field(..., ge=0)
    trade_history:    list[Trade] = Field(default_factory=list)
    pnl_curve:        list[float] = Field(default_factory=list)
    sharpe_so_far:    float       = Field(0.0)


class EnvironmentConfig(BaseModel):
    regime:                MarketRegime  = Field(...)
    num_steps:             int           = Field(252, ge=10)
    initial_capital:       PositiveFloat = Field(100_000.0)
    window_size:           int           = Field(20, ge=5)
    seed:                  int           = Field(42)
    transaction_cost:      UnitFloat     = Field(0.001)
    max_position_fraction: UnitFloat     = Field(0.95)
    multi_agent_mode:      bool          = Field(True)
    hmm_enabled:           bool          = Field(True)