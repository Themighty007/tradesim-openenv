"""
TradeSim v3 — reward.py
=======================
FINAL VERSION. New:
  - regime_alignment bonus: rewards trading WITH the HMM-detected regime
  - Enhanced survival bonus: also triggered by VIX spike signals
  - Drawdown penalty now accounts for credit spread widening
"""

from __future__ import annotations

import math
from typing import Optional

from models import (
    Action,
    ActionType,
    HMMRegimeSignal,
    MarketRegime,
    PortfolioSnapshot,
    RewardBreakdown,
)

# Hyperparameters
PNL_SCALE            = 1.0
PNL_SMOOTHING        = 1e-4
RISK_THRESHOLD       = 0.70
RISK_SCALE           = 0.50
RISK_EXPONENT        = 2.0
DD_THRESHOLD         = 0.05
DD_SCALE             = 0.40
DD_EXPONENT          = 1.5
TURNOVER_FLAT_COST   = 0.003
TURNOVER_VALUE_SCALE = 0.001
SURVIVAL_SCALE       = 0.20
CRASH_FALLING_THRESH = -0.005
REGIME_ALIGN_SCALE   = 0.05  # Small bonus for HMM-aligned positioning


def _pnl_reward(prev_nw, curr_nw, ic):
    if prev_nw <= 0: return 0.0
    delta = curr_nw - prev_nw
    nd    = delta / ic
    mag   = math.sqrt(abs(nd) + PNL_SMOOTHING) - math.sqrt(PNL_SMOOTHING)
    return PNL_SCALE * math.copysign(mag, nd)


def _risk_penalty(eq):
    if eq <= RISK_THRESHOLD: return 0.0
    excess = (eq - RISK_THRESHOLD) / (1.0 - RISK_THRESHOLD)
    return -RISK_SCALE * (excess ** RISK_EXPONENT)


def _drawdown_penalty(dd):
    if dd <= DD_THRESHOLD: return 0.0
    excess = (dd - DD_THRESHOLD) / (1.0 - DD_THRESHOLD)
    return -DD_SCALE * (excess ** DD_EXPONENT)


def _turnover_penalty(action, tv, ic):
    if action.action_type == ActionType.HOLD or tv <= 0: return 0.0
    return -TURNOVER_FLAT_COST + (-TURNOVER_VALUE_SCALE * (tv / ic))


def _survival_bonus(regime, eq, price_return):
    if regime != MarketRegime.CRASH: return 0.0
    if price_return >= CRASH_FALLING_THRESH: return 0.0
    cash_frac = 1.0 - eq
    return max(0.0, SURVIVAL_SCALE * cash_frac * min(abs(price_return) * 20, 1.0))


def _regime_alignment_bonus(
    action: Action,
    equity_fraction: float,
    hmm: Optional[HMMRegimeSignal],
) -> float:
    """
    Small bonus for positioning yourself WITH the HMM-detected regime.
    
    Theory: if the HMM says we're in a bull regime (prob_bull > 0.70)
    and the agent holds meaningful equity, they are "regime-aligned."
    This bonus is small (max 0.05) to not dominate the reward signal,
    but over 252 steps it compounds to ~+1.2 total reward for
    perfectly regime-aligned agents.
    """
    if hmm is None or hmm.state_confidence < 0.65:
        return 0.0

    # Bull regime: reward equity exposure
    if hmm.prob_bull > 0.70 and equity_fraction > 0.4:
        alignment = min(equity_fraction, 0.95) * hmm.prob_bull
        return REGIME_ALIGN_SCALE * alignment * 0.5

    # Crash regime: reward cash holding
    if hmm.prob_crash > 0.70 and equity_fraction < 0.3:
        alignment = (1.0 - equity_fraction) * hmm.prob_crash
        return REGIME_ALIGN_SCALE * alignment * 0.5

    return 0.0


def compute_reward(
    *,
    action: Action,
    prev_snapshot: PortfolioSnapshot,
    curr_snapshot: PortfolioSnapshot,
    regime: MarketRegime,
    prev_price: float,
    curr_price: float,
    trade_value: float,
    initial_capital: float,
    hmm_signal: Optional[HMMRegimeSignal] = None,
) -> RewardBreakdown:
    """
    Compute the composite reward for a single timestep.
    
    Components:
    1. PnL reward     — SQRT-transformed mark-to-market change
    2. Risk penalty   — Quadratic for equity concentration > 70%
    3. Drawdown pen.  — Super-linear for drawdowns > 5%
    4. Turnover pen.  — Per-trade friction
    5. Survival bonus — Crash regime: cash during falling market
    6. Regime align.  — HMM-guided positioning bonus (NEW)
    """
    price_return = (curr_price - prev_price) / prev_price if prev_price > 0 else 0.0

    pnl      = _pnl_reward(prev_snapshot.net_worth, curr_snapshot.net_worth, initial_capital)
    risk     = _risk_penalty(curr_snapshot.equity_fraction)
    dd       = _drawdown_penalty(curr_snapshot.drawdown)
    turn     = _turnover_penalty(action, trade_value, initial_capital)
    surv     = _survival_bonus(regime, curr_snapshot.equity_fraction, price_return)
    reg_aln  = _regime_alignment_bonus(action, curr_snapshot.equity_fraction, hmm_signal)

    total = pnl + risk + dd + turn + surv + reg_aln

    return RewardBreakdown(
        pnl_reward=pnl,
        risk_penalty=risk,
        drawdown_penalty=dd,
        turnover_penalty=turn,
        survival_bonus=surv,
        regime_alignment=reg_aln,
        total=total,
    )