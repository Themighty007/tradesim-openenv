"""
TradeSim v3 — graders.py
========================
FINAL VERSION. New features:

1. SHARPE RATIO computed from step-level rewards
   sharpe = (mean(rewards) / std(rewards)) * sqrt(252)
   This is THE professional metric. Every quant interviewer asks this first.

2. CALMAR RATIO = annualised return / max drawdown
   Used by trend-following CTAs (Man AHL, Winton).
   Measures return per unit of worst-case drawdown.

3. HMM ALIGNMENT SCORE
   Did the agent trade WITH the HMM-detected regime?
   Bull HMM probability + bullish position = aligned = good score.
   Crash HMM probability + short/cash position = aligned = good score.

4. FULL 3-AXIS GRADING (technical + fundamental + psychological)
   Each axis independently scored, then weighted into final score.

The combined score formula is designed to:
- Reward agents that understand ALL THREE axes simultaneously
- Penalise agents that ONLY trade technically (ignoring fundamentals)
- Reward psychological resilience (not panicking with the crowd)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Callable

from models import (
    ActionType,
    EpisodeRecord,
    GradeResult,
    MarketRegime,
    StepRecord,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp(v, lo=0.0, hi=1.0): return max(lo, min(hi, v))

def _linear_score(value, worst, best):
    if abs(best - worst) < 1e-12: return 0.5
    return _clamp((value - worst) / (best - worst))

def _return_score(record, worst, best):
    return _linear_score(record.total_return, worst, best)

def _drawdown_score(record, worst_dd):
    if record.max_drawdown >= worst_dd: return 0.0
    return 1.0 - record.max_drawdown / worst_dd

def _churn_score(record, max_trades):
    return _linear_score(record.num_trades, max_trades, 0)


# ---------------------------------------------------------------------------
# SHARPE RATIO COMPUTATION
# ---------------------------------------------------------------------------

def compute_sharpe(record: EpisodeRecord, annualise: bool = True) -> float:
    """
    Compute Sharpe ratio from step-level portfolio returns.
    
    Uses actual net-worth percentage changes (not raw rewards).
    This is the correct financial Sharpe calculation.
    
    Sharpe = (mean_daily_return / std_daily_return) * sqrt(252)
    
    A Sharpe > 1.0 is "good". Renaissance Medallion's is ~6.0.
    """
    if len(record.steps) < 3:
        return 0.0
    
    # Compute step-to-step portfolio returns
    nw = np.array([s.net_worth for s in record.steps])
    step_returns = np.diff(nw) / (nw[:-1] + 1e-10)
    
    mean_r = np.mean(step_returns)
    std_r  = np.std(step_returns)
    
    if std_r < 1e-10:
        return 0.0
    
    sharpe = mean_r / std_r
    if annualise:
        sharpe *= math.sqrt(252)
    
    return float(np.clip(sharpe, -10, 20))


def compute_calmar(record: EpisodeRecord) -> float:
    """
    Calmar ratio = annualised return / max drawdown.
    
    Preferred by CTAs (trend-following funds) because it focuses
    on the worst-case scenario rather than volatility.
    
    A Calmar > 0.5 is considered acceptable.
    A Calmar > 2.0 is considered excellent.
    """
    if record.max_drawdown < 1e-6:
        return 0.0 if record.total_return <= 0 else 10.0
    
    # Approximate annualisation: assume 252 steps per year
    episodes_per_year = 252.0 / max(len(record.steps), 1)
    annualised_return = record.total_return * episodes_per_year
    
    calmar = annualised_return / record.max_drawdown
    return float(np.clip(calmar, -10, 10))


# ---------------------------------------------------------------------------
# AXIS 1: Technical competency
# ---------------------------------------------------------------------------

def grade_technical(record: EpisodeRecord) -> float:
    """
    Score: did the agent correctly read and ACT on technical signals?
    
    Correct behaviours:
    - RSI > 70 (overbought) → SELL or HOLD (not BUY)
    - RSI < 30 (oversold)   → BUY or HOLD (not SELL)
    - MACD bullish cross    → BUY
    - MACD bearish cross    → SELL
    - BB upper breach       → SELL (stretched)
    - BB lower breach       → BUY (oversold)
    - High ATR (volatility) → HOLD or small position (risk management)
    """
    if not record.steps:
        return 0.0

    correct = 0
    total   = 0

    for step in record.steps:
        t = step.technical
        action = step.action.action_type

        if t.rsi_14 > 70:
            total += 1
            if action in (ActionType.SELL, ActionType.HOLD): correct += 1

        elif t.rsi_14 < 30:
            total += 1
            if action in (ActionType.BUY, ActionType.HOLD): correct += 1

        if t.macd > t.macd_signal and t.macd > 0:
            if action == ActionType.BUY:
                total += 1; correct += 1

        if t.macd < t.macd_signal and t.macd < 0:
            if action == ActionType.SELL:
                total += 1; correct += 1

        if t.bb_pct > 1.0 and action == ActionType.SELL:
            total += 1; correct += 1

        if t.bb_pct < 0.0 and action == ActionType.BUY:
            total += 1; correct += 1

        # ATR risk management: high volatility → avoid large positions
        if t.atr_14 > 2.0 and action == ActionType.BUY and step.action.fraction > 0.7:
            total += 1  # Penalise: large buy in high vol without caution
        elif t.atr_14 > 2.0 and action == ActionType.BUY and step.action.fraction <= 0.5:
            total += 1; correct += 1  # Reward: cautious sizing in high vol

    if total == 0:
        return 0.3
    return _clamp(correct / total)


# ---------------------------------------------------------------------------
# AXIS 2: Fundamental competency
# ---------------------------------------------------------------------------

def grade_fundamental(record: EpisodeRecord) -> float:
    """
    Score: did the agent correctly respond to economic fundamentals?
    
    Correct behaviours:
    - Fed hike > 25bps        → SELL (bonds more attractive, equities fall)
    - Fed cut < -15bps        → BUY (rate cut = equity bullish)
    - Earnings beat > 0.4     → BUY (PEAD: post-earnings announcement drift)
    - Earnings miss < -0.4    → SELL (negative PEAD)
    - Supply shock < -0.5     → SELL (supply glut = bearish)
    - Credit spread > 600bps  → SELL (credit stress = systemic risk)
    - Yield curve inverted     → REDUCE exposure (recession warning)
    """
    if not record.steps:
        return 0.0

    correct = 0
    total   = 0

    for step in record.steps:
        f = step.fundamental
        action = step.action.action_type

        if f.fed_rate_change_bps > 25:
            total += 1
            if action in (ActionType.SELL, ActionType.HOLD): correct += 1

        elif f.fed_rate_change_bps < -15:
            total += 1
            if action in (ActionType.BUY, ActionType.HOLD): correct += 1

        if f.earnings_surprise > 0.4:
            total += 1
            if action == ActionType.BUY: correct += 1

        elif f.earnings_surprise < -0.4:
            total += 1
            if action in (ActionType.SELL, ActionType.HOLD): correct += 1

        if f.supply_shock < -0.5:
            total += 1
            if action in (ActionType.SELL, ActionType.HOLD): correct += 1

        # Credit spread signal: widening spreads = systemic stress
        if f.credit_spread_bps > 600:
            total += 1
            if action in (ActionType.SELL, ActionType.HOLD): correct += 1

        # Inverted yield curve = recession warning → reduce exposure
        if f.yield_curve_slope < -0.3:
            total += 1
            if step.equity_fraction < 0.5: correct += 1  # Reduced exposure = correct

        # Strong institutional buying = hold/buy
        if f.institutional_flow > 0.5 and step.equity_fraction > 0.3:
            total += 1; correct += 1

    if total == 0:
        return 0.4
    return _clamp(correct / total)


# ---------------------------------------------------------------------------
# AXIS 3: Psychological competency
# ---------------------------------------------------------------------------

def grade_psychological(record: EpisodeRecord) -> float:
    """
    Score: did the agent FADE crowd psychology extremes?
    
    Contrarian rules:
    - Fear/greed > 0.80 (extreme greed) → SELL (top signal)
    - Fear/greed < -0.70 (extreme fear) → BUY (bottom signal)  
    - VIX > 35 (fear spike) + BUY = contrarian correct
    - Put/call > 2.0 + BUY = correct (everyone hedged = often bottom)
    - Social euphoria > 0.75 + SELL = contrarian correct
    - Insider buying > 0.5 + BUY = following smart money = correct
    """
    if not record.steps:
        return 0.0

    smart  = 0
    traps  = 0
    total  = 0

    for step in record.steps:
        ps = step.psychology
        action = step.action.action_type

        if ps.fear_greed_index > 0.80:
            total += 1
            if action in (ActionType.SELL, ActionType.HOLD): smart += 1
            elif action == ActionType.BUY: traps += 1

        elif ps.fear_greed_index < -0.70:
            total += 1
            if action in (ActionType.BUY, ActionType.HOLD): smart += 1
            elif action == ActionType.SELL: traps += 1

        # VIX spike: fear is highest at bottoms
        if ps.vix_level > 35:
            total += 1
            if action == ActionType.BUY: smart += 1  # Buying during fear spike
            elif action == ActionType.SELL: traps += 1

        # High put/call = bottoming signal
        if ps.put_call_ratio > 2.0 and action == ActionType.BUY:
            total += 1; smart += 1

        # Social euphoria: retail frenzy = sell signal
        if ps.social_sentiment > 0.75 and action == ActionType.SELL:
            total += 1; smart += 1

        # Insider buying: smart money
        if ps.insider_buying > 0.5 and action == ActionType.BUY:
            total += 1; smart += 1

        # Extreme negative skew = massive downside hedging = crash imminent
        if ps.skew < -0.30 and action in (ActionType.SELL, ActionType.HOLD):
            total += 1; smart += 1

    if total == 0:
        return _churn_score(record, max_trades=15)

    base = _clamp(smart / total)
    trap_penalty = _clamp(traps / max(total, 1)) * 0.3
    return _clamp(base - trap_penalty)


# ---------------------------------------------------------------------------
# HMM ALIGNMENT SCORE
# ---------------------------------------------------------------------------

def grade_hmm_alignment(record: EpisodeRecord) -> float:
    """
    Score: did the agent position itself correctly based on
    the HMM regime detector's output?
    
    Bull HMM probability > 0.7 + equity fraction > 0.5 = aligned
    Crash HMM probability > 0.7 + equity fraction < 0.3 = aligned
    
    This rewards agents that learn to use the unsupervised regime
    signal, not just the labelled regime hint.
    """
    if not record.steps:
        return 0.5

    aligned = 0
    total   = 0

    for step in record.steps:
        hmm = step.hmm
        eq  = step.equity_fraction

        if hmm.state_confidence < 0.6:
            continue  # HMM is uncertain, don't score

        total += 1

        if hmm.prob_bull > 0.7:
            # HMM says bull regime — being invested is correct
            if eq > 0.4: aligned += 1

        elif hmm.prob_crash > 0.7:
            # HMM says crash regime — being in cash is correct
            if eq < 0.4: aligned += 1

    if total == 0:
        return 0.5
    return _clamp(aligned / total)


# ---------------------------------------------------------------------------
# Shared financial metric extractors
# ---------------------------------------------------------------------------

def _buy_timing_score(record):
    n = len(record.steps)
    for i, step in enumerate(record.steps):
        if step.action.action_type == ActionType.BUY and step.action.fraction >= 0.1:
            frac_through = i / n
            return _linear_score(frac_through, worst=0.8, best=0.0)
    return 0.0

def _equity_peak_capture(record):
    if not record.steps: return 0.0
    initial = record.initial_capital
    passive_peak = max(initial * (s.price / record.steps[0].price) for s in record.steps)
    return _linear_score(record.peak_net_worth, worst=initial, best=passive_peak)

def _crash_survival_score(record):
    steps = record.steps
    n = len(steps)
    if n < 20: return 0.0
    prices = [s.price for s in steps]
    nadir_idx = prices.index(min(prices))
    pre_crash_end = min(nadir_idx, n // 3)
    if pre_crash_end < 5: pre_crash_end = min(nadir_idx, n // 4)
    pre_crash_steps = steps[:max(pre_crash_end, 1)]
    nadir_step = steps[nadir_idx]
    avg_pre_equity = sum(s.equity_fraction for s in pre_crash_steps) / max(len(pre_crash_steps), 1)
    participation_score = _clamp(avg_pre_equity / 0.30)
    cash_at_nadir = 1.0 - nadir_step.equity_fraction
    return _clamp(participation_score * cash_at_nadir)

def _crash_exit_timing_score(record):
    steps = record.steps
    if len(steps) < 10: return 0.0
    prices = [s.price for s in steps]
    peak_price = max(prices)
    for step in steps:
        if step.action.action_type == ActionType.SELL and step.action.fraction >= 0.25:
            drop = (peak_price - step.price) / peak_price
            if drop <= 0.0: return 1.0
            return _clamp(math.exp(-drop * 8.0))
    return 0.0

def _crash_recovery_score(record):
    steps = record.steps
    n = len(steps)
    if n < 10: return 0.0
    prices = [s.price for s in steps]
    nadir_idx = prices.index(min(prices))
    recovery_steps = steps[nadir_idx:]
    if not recovery_steps: return 0.0
    avg_eq = sum(s.equity_fraction for s in recovery_steps) / len(recovery_steps)
    if len(recovery_steps) >= 2:
        rec_ret = (recovery_steps[-1].net_worth - recovery_steps[0].net_worth) / max(recovery_steps[0].net_worth, 1.0)
    else:
        rec_ret = 0.0
    return _clamp(0.6 * avg_eq + 0.4 * _linear_score(rec_ret, -0.05, 0.10))


# ---------------------------------------------------------------------------
# Task graders
# ---------------------------------------------------------------------------

def _make_grade_result(
    task: int,
    fin_score: float,
    tech_score: float,
    fund_score: float,
    psych_score: float,
    hmm_score: float,
    record: EpisodeRecord,
    sub: dict,
    weights: dict,
    axis_weights: dict,
) -> GradeResult:
    """Helper: compute final score and build GradeResult with all metrics."""
    score = _clamp(
        sum(weights[k] * sub[k] for k in weights)
        + axis_weights["technical"]    * tech_score
        + axis_weights["fundamental"]  * fund_score
        + axis_weights["psychological"] * psych_score
        + axis_weights["hmm"]          * hmm_score
    )

    sharpe  = compute_sharpe(record)
    calmar  = compute_calmar(record)
    n_steps = len(record.steps)

    return GradeResult(
        task=task,
        score=score,
        breakdown={
            **{k: round(v, 4) for k, v in sub.items()},
            "technical":     round(tech_score, 4),
            "fundamental":   round(fund_score, 4),
            "psychological": round(psych_score, 4),
            "hmm_alignment": round(hmm_score, 4),
        },
        rationale=(
            f"Task {task}: score={score:.4f} | "
            f"Sharpe={sharpe:.2f} Calmar={calmar:.2f} | "
            f"tech={tech_score:.3f} fund={fund_score:.3f} "
            f"psych={psych_score:.3f} hmm={hmm_score:.3f}"
        ),
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        max_drawdown=record.max_drawdown,
        total_return_pct=record.total_return * 100,
        num_trades=record.num_trades,
        technical_score=tech_score,
        fundamental_score=fund_score,
        psychological_score=psych_score,
        hmm_alignment_score=hmm_score,
    )


def grade_task1(record: EpisodeRecord) -> GradeResult:
    assert record.regime == MarketRegime.BULL
    weights      = {"return": 0.20, "peak_capture": 0.15, "buy_timing": 0.12, "drawdown": 0.08, "churn": 0.08}
    axis_weights = {"technical": 0.10, "fundamental": 0.10, "psychological": 0.07, "hmm": 0.10}
    sub = {
        "return":       _return_score(record, -0.02, 0.12),
        "peak_capture": _equity_peak_capture(record),
        "buy_timing":   _buy_timing_score(record),
        "drawdown":     _drawdown_score(record, 0.15),
        "churn":        _churn_score(record, 12),
    }
    return _make_grade_result(
        1, sum(weights[k]*sub[k] for k in weights),
        grade_technical(record), grade_fundamental(record),
        grade_psychological(record), grade_hmm_alignment(record),
        record, sub, weights, axis_weights
    )


def grade_task2(record: EpisodeRecord) -> GradeResult:
    assert record.regime == MarketRegime.RANGE
    num_trades = record.num_trades
    if num_trades == 0: engagement = 0.0
    elif num_trades <= 6: engagement = min(1.0, num_trades / 5.0)
    elif num_trades <= 12: engagement = 1.0 - (num_trades - 6) / 12.0
    else: engagement = _clamp(1.0 - num_trades / 20.0)

    weights      = {"preservation": 0.12, "drawdown": 0.08, "engagement": 0.25, "churn": 0.05, "return": 0.05}
    axis_weights = {"technical": 0.18, "fundamental": 0.10, "psychological": 0.10, "hmm": 0.07}
    sub = {
        "preservation": _return_score(record, -0.08, 0.04),
        "drawdown":     _drawdown_score(record, 0.10),
        "engagement":   _clamp(engagement),
        "churn":        _churn_score(record, 8),
        "return":       _return_score(record, -0.06, 0.06),
    }
    return _make_grade_result(
        2, sum(weights[k]*sub[k] for k in weights),
        grade_technical(record), grade_fundamental(record),
        grade_psychological(record), grade_hmm_alignment(record),
        record, sub, weights, axis_weights
    )


def grade_task3(record: EpisodeRecord) -> GradeResult:
    assert record.regime == MarketRegime.CRASH
    weights      = {"survival": 0.20, "exit_timing": 0.20, "drawdown": 0.08, "recovery": 0.08}
    axis_weights = {"technical": 0.10, "fundamental": 0.14, "psychological": 0.12, "hmm": 0.08}
    sub = {
        "survival":    _crash_survival_score(record),
        "exit_timing": _crash_exit_timing_score(record),
        "drawdown":    _drawdown_score(record, 0.35),
        "recovery":    _crash_recovery_score(record),
    }
    return _make_grade_result(
        3, sum(weights[k]*sub[k] for k in weights),
        grade_technical(record), grade_fundamental(record),
        grade_psychological(record), grade_hmm_alignment(record),
        record, sub, weights, axis_weights
    )


_GRADERS = {1: grade_task1, 2: grade_task2, 3: grade_task3}
_REGIME_TO_TASK = {MarketRegime.BULL: 1, MarketRegime.RANGE: 2, MarketRegime.CRASH: 3}


def grade_episode(record: EpisodeRecord) -> GradeResult:
    task = _REGIME_TO_TASK[record.regime]
    return _GRADERS[task](record)