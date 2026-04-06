"""
TradeSim — models.py
====================
The canonical data contracts for every object that flows through TradeSim.
Built with Pydantic v2 for runtime validation and rich JSON schema generation.

Design philosophy:
  - Every field has a semantic meaning and a tight validator.
  - Enum values are never bare strings — use the provided enums.
  - All monetary values are in USD with at most 6 decimal places.
  - Ratios / fractions are always in [0, 1] unless explicitly annotated otherwise.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MarketRegime(str, Enum):
    """The three canonical market scenarios."""
    BULL   = "bull"        # Steady upward trend — Task 1
    RANGE  = "range"       # Choppy, mean-reverting — Task 2
    CRASH  = "crash"       # Flash crash + slow recovery — Task 3


class ActionType(str, Enum):
    """Discrete action space for the agent."""
    BUY  = "buy"
    SELL = "sell"
    HOLD = "hold"


class RewardComponent(str, Enum):
    """Named components of the composite reward signal."""
    PNL          = "pnl"           # Mark-to-market P&L change
    RISK         = "risk"          # Penalty for concentration / leverage
    DRAWDOWN     = "drawdown"      # Penalty for sitting in a drawdown
    TURNOVER     = "turnover"      # Penalty for excessive trading (friction)
    SURVIVAL     = "survival"      # Bonus for capital preservation in crashes


# ---------------------------------------------------------------------------
# Positive-constrained scalar aliases
# ---------------------------------------------------------------------------

PositiveFloat  = Annotated[float, Field(gt=0.0)]
NonNegFloat    = Annotated[float, Field(ge=0.0)]
UnitFloat      = Annotated[float, Field(ge=0.0, le=1.0)]   # fraction in [0,1]
Price          = Annotated[float, Field(gt=0.0, description="Asset price in USD")]
Quantity       = Annotated[float, Field(ge=0.0, description="Number of shares (fractional OK)")]
Cash           = Annotated[float, Field(description="USD cash balance (may be 0)")]


# ---------------------------------------------------------------------------
# 1. Observation — what the agent sees at each timestep
# ---------------------------------------------------------------------------

class PriceWindow(BaseModel):
    """A rolling window of normalised price features."""

    model_config = {"frozen": True}

    raw_prices:        list[Price]  = Field(..., description="Absolute prices, oldest→newest")
    returns:           list[float]  = Field(..., description="Log-returns, length = len(raw_prices)-1")
    normalised_prices: list[float]  = Field(..., description="z-score normalised over the window")

    @field_validator("returns")
    @classmethod
    def finite_returns(cls, v: list[float]) -> list[float]:
        if any(not math.isfinite(r) for r in v):
            raise ValueError("All returns must be finite (no inf / NaN).")
        return v

    @field_validator("normalised_prices")
    @classmethod
    def finite_normalised(cls, v: list[float]) -> list[float]:
        if any(not math.isfinite(p) for p in v):
            raise ValueError("Normalised prices must be finite.")
        return v

    @model_validator(mode="after")
    def lengths_consistent(self) -> "PriceWindow":
        n = len(self.raw_prices)
        if len(self.returns) != max(n - 1, 0):
            raise ValueError(
                f"returns length ({len(self.returns)}) must be len(raw_prices)-1 ({n-1})."
            )
        if len(self.normalised_prices) != n:
            raise ValueError(
                f"normalised_prices length ({len(self.normalised_prices)}) must equal "
                f"len(raw_prices) ({n})."
            )
        return self


class PortfolioSnapshot(BaseModel):
    """The agent's current financial state."""

    model_config = {"frozen": True}

    cash:              Cash         = Field(...,  description="Liquid USD cash")
    shares_held:       Quantity     = Field(...,  description="Shares currently held")
    current_price:     Price        = Field(...,  description="Latest market price")
    net_worth:         NonNegFloat  = Field(...,  description="cash + shares * price")
    peak_net_worth:    NonNegFloat  = Field(...,  description="Highest net_worth seen so far")
    drawdown:          UnitFloat    = Field(...,  description="Current drawdown fraction from peak")
    equity_fraction:   UnitFloat    = Field(...,  description="Fraction of net_worth in equities")
    total_trades:      int          = Field(...,  ge=0, description="Cumulative trade count")
    total_return:      float        = Field(...,  description="Cumulative return vs. initial capital")

    @model_validator(mode="after")
    def net_worth_consistent(self) -> "PortfolioSnapshot":
        implied = self.cash + self.shares_held * self.current_price
        if abs(implied - self.net_worth) > 0.01:
            raise ValueError(
                f"net_worth ({self.net_worth:.4f}) is inconsistent with "
                f"cash + shares*price ({implied:.4f})."
            )
        return self

    @model_validator(mode="after")
    def drawdown_consistent(self) -> "PortfolioSnapshot":
        if self.peak_net_worth > 0:
            expected_dd = max(0.0, 1.0 - self.net_worth / self.peak_net_worth)
            if abs(expected_dd - self.drawdown) > 1e-6:
                raise ValueError(
                    f"drawdown ({self.drawdown:.6f}) inconsistent with "
                    f"peak/net_worth ({expected_dd:.6f})."
                )
        return self


class Observation(BaseModel):
    """Full observation delivered to the agent at each timestep."""

    model_config = {"frozen": True}

    timestep:   int             = Field(..., ge=0,  description="Current step index (0-based)")
    max_steps:  int             = Field(..., gt=0,  description="Episode length")
    regime:     MarketRegime    = Field(...,         description="Which scenario is running")
    window:     PriceWindow     = Field(...,         description="Price history window")
    portfolio:  PortfolioSnapshot = Field(...,       description="Current portfolio state")
    time_left:  UnitFloat       = Field(...,         description="Fraction of episode remaining")

    @model_validator(mode="after")
    def time_left_consistent(self) -> "Observation":
        expected = 1.0 - self.timestep / self.max_steps
        if abs(expected - self.time_left) > 1e-6:
            raise ValueError(
                f"time_left ({self.time_left}) inconsistent with "
                f"1 - timestep/max_steps ({expected:.6f})."
            )
        return self


# ---------------------------------------------------------------------------
# 2. Action — what the agent sends back
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    The agent's decision.

    fraction is interpreted differently per action_type:
      BUY  → fraction of *available cash* to deploy (0.0–1.0)
      SELL → fraction of *current shares* to liquidate (0.0–1.0)
      HOLD → ignored (set to 0.0 by convention)
    """

    model_config = {"frozen": True}

    action_type: ActionType = Field(...,            description="BUY / SELL / HOLD")
    fraction:    UnitFloat  = Field(0.0,            description="Fraction of resource to commit")
    reason:      str        = Field("",  max_length=512, description="Optional natural-language rationale")

    @model_validator(mode="after")
    def hold_fraction_zero(self) -> "Action":
        if self.action_type == ActionType.HOLD and self.fraction != 0.0:
            # Silently coerce — the agent might forget but we don't crash.
            object.__setattr__(self, "fraction", 0.0)
        return self

    # Convenience constructors
    @classmethod
    def buy(cls, fraction: float = 1.0, reason: str = "") -> "Action":
        return cls(action_type=ActionType.BUY, fraction=fraction, reason=reason)

    @classmethod
    def sell(cls, fraction: float = 1.0, reason: str = "") -> "Action":
        return cls(action_type=ActionType.SELL, fraction=fraction, reason=reason)

    @classmethod
    def hold(cls, reason: str = "") -> "Action":
        return cls(action_type=ActionType.HOLD, fraction=0.0, reason=reason)


# ---------------------------------------------------------------------------
# 3. Reward — granular breakdown at every timestep
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """
    The full reward signal for one timestep, decomposed into named components.
    All components are additive; total = sum of all.
    """

    model_config = {"frozen": True}

    pnl_reward:      float = Field(..., description="Reward for mark-to-market P&L change")
    risk_penalty:    float = Field(..., le=0.0, description="Penalty for concentration risk (≤ 0)")
    drawdown_penalty:float = Field(..., le=0.0, description="Penalty for being in drawdown (≤ 0)")
    turnover_penalty:float = Field(..., le=0.0, description="Penalty for excessive trading (≤ 0)")
    survival_bonus:  float = Field(..., ge=0.0, description="Bonus for capital survival in crash (≥ 0)")
    total:           float = Field(..., description="Sum of all components")

    @model_validator(mode="after")
    def total_consistent(self) -> "RewardBreakdown":
        implied = (
            self.pnl_reward
            + self.risk_penalty
            + self.drawdown_penalty
            + self.turnover_penalty
            + self.survival_bonus
        )
        if abs(implied - self.total) > 1e-9:
            raise ValueError(
                f"total ({self.total}) != sum of components ({implied:.9f})."
            )
        return self


# ---------------------------------------------------------------------------
# 4. StepResult — what environment.step() returns
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """The full result of a single environment step."""

    model_config = {"frozen": True}

    observation:  Observation      = Field(..., description="New observation after action")
    reward:       RewardBreakdown  = Field(..., description="Reward breakdown for this step")
    done:         bool             = Field(..., description="True if episode has ended")
    info:         dict             = Field(default_factory=dict, description="Diagnostic metadata")


# ---------------------------------------------------------------------------
# 5. EpisodeRecord — full history for grading
# ---------------------------------------------------------------------------

class StepRecord(BaseModel):
    """One row in the episode history — immutable."""

    model_config = {"frozen": True}

    timestep:       int             = Field(..., ge=0)
    action:         Action
    reward:         RewardBreakdown
    net_worth:      NonNegFloat
    drawdown:       UnitFloat
    equity_fraction:UnitFloat
    price:          Price


class EpisodeRecord(BaseModel):
    """Complete episode trajectory — input to graders."""

    model_config = {"frozen": True}

    regime:          MarketRegime
    initial_capital: PositiveFloat
    steps:           list[StepRecord]  = Field(..., min_length=1)

    # Derived convenience properties
    @property
    def final_net_worth(self) -> float:
        return self.steps[-1].net_worth

    @property
    def total_return(self) -> float:
        return (self.final_net_worth - self.initial_capital) / self.initial_capital

    @property
    def peak_net_worth(self) -> float:
        return max(s.net_worth for s in self.steps)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown observed across the episode."""
        return max(s.drawdown for s in self.steps)

    @property
    def num_trades(self) -> int:
        return sum(1 for s in self.steps if s.action.action_type != ActionType.HOLD)

    @property
    def total_reward(self) -> float:
        return sum(s.reward.total for s in self.steps)


# ---------------------------------------------------------------------------
# 6. GradeResult — output of each grader function
# ---------------------------------------------------------------------------

class GradeResult(BaseModel):
    """Structured output from a grader, [0.0, 1.0]."""

    model_config = {"frozen": True}

    task:        int        = Field(..., ge=1, le=3, description="Task number (1, 2, or 3)")
    score:       UnitFloat  = Field(..., description="Final normalised score in [0, 1]")
    breakdown:   dict[str, float] = Field(default_factory=dict, description="Sub-scores")
    rationale:   str        = Field("", description="Human-readable explanation")

    def __float__(self) -> float:
        return self.score


# ---------------------------------------------------------------------------
# 7. EnvironmentConfig — top-level runtime configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 8. Trade — a single executed trade (for State.trade_history)
# ---------------------------------------------------------------------------

class Trade(BaseModel):
    """Record of a single executed trade."""

    model_config = {"frozen": True}

    timestep:   int   = Field(..., ge=0)
    decision:   str   = Field(..., description="BUY / SELL / HOLD")
    price:      float = Field(..., gt=0)
    quantity:   float = Field(..., ge=0)
    cash_after: float = Field(..., description="Cash balance after trade")


# ---------------------------------------------------------------------------
# 9. State — readable episode summary (for env.state())
# ---------------------------------------------------------------------------

class State(BaseModel):
    """High-level episode state summary."""

    model_config = {"frozen": True}

    task_id:           int          = Field(..., ge=0)
    current_timestep:  int          = Field(..., ge=0)
    portfolio_value:   float        = Field(..., ge=0)
    peak_value:        float        = Field(..., ge=0)
    trade_history:     list[Trade]  = Field(default_factory=list)
    pnl_curve:         list[float]  = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 10. EnvironmentConfig — top-level runtime configuration
# ---------------------------------------------------------------------------

class EnvironmentConfig(BaseModel):
    """Configuration object for instantiating a TradeSim environment."""

    regime:          MarketRegime  = Field(...,            description="Which market scenario")
    num_steps:       int           = Field(252,  ge=10,    description="Episode length (trading days)")
    initial_capital: PositiveFloat = Field(100_000.0,      description="Starting cash in USD")
    window_size:     int           = Field(20,   ge=5,     description="Lookback window for features")
    seed:            int           = Field(42,             description="RNG seed for reproducibility")
    transaction_cost:UnitFloat     = Field(0.001,          description="Cost per trade as fraction of trade value")
    max_position_fraction: UnitFloat = Field(
        0.95,
        description="Max fraction of net_worth that can be in equities"
    )

    @field_validator("num_steps")
    @classmethod
    def steps_exceeds_window(cls, v: int, info) -> int:
        # We don't have access to window_size here easily, so just a floor check.
        if v < 10:
            raise ValueError("num_steps must be at least 10.")
        return v
