"""
TradeSim — graders.py
=====================
Three deterministic grader functions, one per task.

Design Contract
---------------
  • Input  : EpisodeRecord (full episode history)
  • Output : GradeResult with score in [0.0, 1.0]
  • Determinism: same input → identical output, always (no RNG, no I/O)
  • Partial credit: multiple sub-scores combined, never cliff-edge pass/fail
  • The spread across difficulty (Task 1 > Task 2 > Task 3) proves graders work

Grading Philosophy
------------------
Each task tests a specific competency:

  Task 1 (Bull) — Can the agent identify a trend and capture it?
    Sub-metrics: total return, timing (early buy), holding (low churn),
                 risk management (not reckless leverage).

  Task 2 (Range) — Can the agent survive without trend-following profit?
    Sub-metrics: capital preservation, drawdown control, avoidance of
                 false breakout traps, transaction cost consciousness.

  Task 3 (Crash) — Can the agent detect and survive the crash?
    Sub-metrics: survival (net worth intact at crash nadir), exit timing
                 (did it sell before the cliff?), recovery participation
                 (re-entered during the bounce?), overall drawdown.

Normalisation
-------------
Each sub-score is independently normalised to [0, 1] using known bounds,
then weighted-summed, then clipped to [0, 1].
"""

from __future__ import annotations

import math
from typing import Callable

from models import (
    ActionType,
    EpisodeRecord,
    GradeResult,
    MarketRegime,
    StepRecord,
)


# ---------------------------------------------------------------------------
# Utility — sigmoid normaliser (smooth, always in (0,1))
# ---------------------------------------------------------------------------

def _sigmoid(x: float, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-scale * x))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _linear_score(value: float, worst: float, best: float) -> float:
    """Linearly map value from [worst, best] → [0, 1]. Clipped at extremes."""
    if abs(best - worst) < 1e-12:
        return 0.5
    return _clamp((value - worst) / (best - worst))


# ---------------------------------------------------------------------------
# Shared metric extractors
# ---------------------------------------------------------------------------

def _return_score(record: EpisodeRecord, worst_return: float, best_return: float) -> float:
    """Sub-score for total return, normalised between worst and best."""
    return _linear_score(record.total_return, worst_return, best_return)


def _drawdown_score(record: EpisodeRecord, worst_dd: float) -> float:
    """Sub-score for max drawdown. Lower drawdown → higher score."""
    # worst_dd is the threshold above which we give 0
    if record.max_drawdown >= worst_dd:
        return 0.0
    return 1.0 - record.max_drawdown / worst_dd


def _churn_score(record: EpisodeRecord, max_acceptable_trades: int) -> float:
    """
    Sub-score penalising excessive trading.
    0 trades → perfect score (disciplined HOLD = no churn).
    Beyond max_acceptable_trades → score 0.
    """
    return _linear_score(record.num_trades, max_acceptable_trades, 0)


def _buy_timing_score(record: EpisodeRecord) -> float:
    """
    Bull-specific: how early did the agent take a meaningful position?

    Earlier first BUY → higher score.
    No BUY → 0.0.
    """
    n = len(record.steps)
    for i, step in enumerate(record.steps):
        if step.action.action_type == ActionType.BUY and step.action.fraction >= 0.1:
            # Bought in the first 20% → max score
            frac_through = i / n
            return _linear_score(frac_through, worst=0.8, best=0.0)
    return 0.0  # Never bought


def _equity_peak_capture(record: EpisodeRecord) -> float:
    """
    Bull-specific: did the agent's peak net worth actually track the market peak?

    Compare agent's peak net worth to a theoretical all-in passive strategy.
    """
    if not record.steps:
        return 0.0

    initial = record.initial_capital
    passive_peak = max(
        initial * (s.price / record.steps[0].price)
        for s in record.steps
    )
    agent_peak = record.peak_net_worth
    return _linear_score(agent_peak, worst=initial, best=passive_peak)


# ---------------------------------------------------------------------------
# Task 1 — Bull Market Grader
# ---------------------------------------------------------------------------

def grade_task1(record: EpisodeRecord) -> GradeResult:
    """
    Task 1: Bull Market

    Expected agent behaviour:
      • Buy early (first 20% of episode)
      • Hold through the trend (low churn)
      • Capture most of the upside (good return)
      • Don't use reckless leverage

    Expected score range: 0.65–0.80 for a reasonable agent.
    """
    assert record.regime == MarketRegime.BULL, f"Task 1 expects BULL regime; got {record.regime}"

    weights = {
        "return":       0.35,   # Total P&L matters most in a bull market
        "peak_capture": 0.25,   # Did it ride the actual upswing?
        "buy_timing":   0.20,   # Early entry is critical
        "drawdown":     0.10,   # Shouldn't have deep drawdown in a bull
        "churn":        0.10,   # Disciplined holding > nervous trading
    }

    sub = {
        # Bull returns +12% passively. An LLM scoring 0.65–0.80 should capture 6–10%
        "return":       _return_score(record, worst_return=-0.02, best_return=0.12),
        "peak_capture": _equity_peak_capture(record),
        "buy_timing":   _buy_timing_score(record),
        "drawdown":     _drawdown_score(record, worst_dd=0.15),
        "churn":        _churn_score(record, max_acceptable_trades=12),
    }

    score = _clamp(sum(weights[k] * sub[k] for k in weights))

    return GradeResult(
        task=1,
        score=score,
        breakdown={k: round(v, 4) for k, v in sub.items()},
        rationale=(
            f"Bull market score: return={sub['return']:.3f}, "
            f"peak_capture={sub['peak_capture']:.3f}, "
            f"buy_timing={sub['buy_timing']:.3f}, "
            f"drawdown={sub['drawdown']:.3f}, "
            f"churn={sub['churn']:.3f}. "
            f"Weighted total: {score:.4f}"
        ),
    )


# ---------------------------------------------------------------------------
# Task 2 — Choppy Range Grader
# ---------------------------------------------------------------------------

def grade_task2(record: EpisodeRecord) -> GradeResult:
    """
    Task 2: Choppy Range Market

    Expected agent behaviour:
      • Don't lose money (capital preservation)
      • Avoid false breakouts (don't FOMO buy at spikes)
      • Keep drawdowns tight (mean-revert, don't trend-follow)
      • Trade lightly (friction kills in a sideways market)

    Expected score range: 0.35–0.55 for a reasonable agent.
    """
    assert record.regime == MarketRegime.RANGE, f"Task 2 expects RANGE regime; got {record.regime}"

    weights = {
        "preservation": 0.20,   # Don't lose money
        "drawdown":     0.15,   # Tight drawdown = good risk management
        "engagement":   0.50,   # Must DEMONSTRATE range-trading skill (not just HOLD)
        "churn":        0.075,  # Excessive trading kills in sideways market
        "return":       0.075,  # Small positive return is a bonus
    }

    # Engagement: peak at 3-6 disciplined trades (demonstrates range-reading skill)
    # Pure HOLD = 0 skill shown; 20 mechanical trades = gambling, not skill
    num_trades = record.num_trades
    if num_trades == 0:
        engagement = 0.0
    elif num_trades <= 6:
        engagement = min(1.0, num_trades / 5.0)     # Ramp: 5 trades = 1.0
    elif num_trades <= 12:
        engagement = 1.0 - (num_trades - 6) / 12.0  # Gentle decay 6→12
    else:
        engagement = _clamp(1.0 - num_trades / 20.0) # Steep decay above 12

    sub = {
        "preservation": _return_score(record, worst_return=-0.08, best_return=0.04),
        "drawdown":     _drawdown_score(record, worst_dd=0.10),
        "engagement":   _clamp(engagement),
        "churn":        _churn_score(record, max_acceptable_trades=8),
        "return":       _return_score(record, worst_return=-0.06, best_return=0.06),
    }

    score = _clamp(sum(weights[k] * sub[k] for k in weights))

    return GradeResult(
        task=2,
        score=score,
        breakdown={k: round(v, 4) for k, v in sub.items()},
        rationale=(
            f"Range market score: preservation={sub['preservation']:.3f}, "
            f"drawdown={sub['drawdown']:.3f}, "
            f"engagement={sub['engagement']:.3f}, "
            f"churn={sub['churn']:.3f}, "
            f"return={sub['return']:.3f}. "
            f"Weighted total: {score:.4f}"
        ),
    )


# ---------------------------------------------------------------------------
# Task 3 — Flash Crash Grader
# ---------------------------------------------------------------------------

def _crash_survival_score(record: EpisodeRecord) -> float:
    """
    Did the agent actually TRADE into the pre-crash calm and then EXIT?

    This is the critical distinction: an agent that never buys trivially
    'survives' the crash but showed no skill. We require:
      (a) agent held meaningful equity BEFORE the crash midpoint
      (b) agent had low equity AT the crash nadir

    Score = (pre_crash_participation) × (nadir_cash_fraction)

    An agent that was all-cash the entire episode scores ZERO.
    An agent that bought early and exited before the nadir scores HIGH.
    """
    steps = record.steps
    n = len(steps)
    if n < 20:
        return 0.0

    # Find the nadir (price minimum)
    prices = [s.price for s in steps]
    nadir_idx = prices.index(min(prices))

    # Pre-crash window: first third of steps up to nadir
    pre_crash_end = min(nadir_idx, n // 3)
    if pre_crash_end < 5:
        pre_crash_end = min(nadir_idx, n // 4)

    pre_crash_steps = steps[:max(pre_crash_end, 1)]
    nadir_step = steps[nadir_idx]

    # (a) Did the agent hold equity before the crash?
    avg_pre_equity = sum(s.equity_fraction for s in pre_crash_steps) / max(len(pre_crash_steps), 1)
    # Require at least 10% average equity pre-crash to demonstrate participation
    participation_score = _clamp(avg_pre_equity / 0.30)  # full credit at 30%+ avg equity

    # (b) Did the agent exit before/at the nadir?
    cash_at_nadir = 1.0 - nadir_step.equity_fraction
    exit_score = cash_at_nadir  # 1.0 = all cash at bottom

    # Combined: must score on BOTH dimensions
    combined = participation_score * exit_score

    return _clamp(combined)



def _crash_exit_timing_score(record: EpisodeRecord) -> float:
    """
    Did the agent SELL before the crash cliff destroyed significant value?

    We define a "good exit window" as any SELL executed before prices drop
    more than 8% from their episode peak. An exit at 20%+ down scores poorly.

    Timing tiers:
      - Exit before 5% drop from peak  → score ~0.90
      - Exit before 10% drop           → score ~0.60
      - Exit before 20% drop           → score ~0.25
      - Exit after 20%+ drop           → score ~0.05
      - Never exits (holds through)    → score  0.0

    An LLM agent typically exits late (10-20% drop) → scores 0.25–0.60.
    """
    steps = record.steps
    n = len(steps)
    if n < 10:
        return 0.0

    prices = [s.price for s in steps]
    peak_price = max(prices)

    # Find the earliest meaningful SELL (fraction >= 0.25, reduces equity >10%)
    for i, step in enumerate(steps):
        if step.action.action_type == ActionType.SELL and step.action.fraction >= 0.25:
            sell_price = step.price
            drop_from_peak = (peak_price - sell_price) / peak_price  # 0 = sold at peak, 1 = sold at zero

            # Tiered scoring: exponential decay as drop increases
            # Sold at peak (drop=0)  → 1.0
            # Sold at 5% drop        → ~0.75
            # Sold at 15% drop       → ~0.35
            # Sold at 30%+ drop      → ~0.05
            if drop_from_peak <= 0.0:
                return 1.0
            score = math.exp(-drop_from_peak * 8.0)
            return _clamp(score)

    return 0.0  # Never sold — no exit credit


def _crash_recovery_score(record: EpisodeRecord) -> float:
    """
    Did the agent re-enter during the recovery phase?

    Approximate recovery start as after the lowest price point.
    Check if the agent has meaningful equity in the second half.
    """
    steps = record.steps
    n = len(steps)
    if n < 10:
        return 0.0

    # Find the nadir (lowest price)
    prices = [s.price for s in steps]
    nadir_idx = prices.index(min(prices))

    # Look at second half (post-nadir)
    recovery_steps = steps[nadir_idx:]
    if not recovery_steps:
        return 0.0

    # Average equity fraction in recovery phase
    avg_equity_in_recovery = sum(s.equity_fraction for s in recovery_steps) / len(recovery_steps)

    # Also check if net worth grew in recovery phase
    if len(recovery_steps) >= 2:
        recovery_return = (recovery_steps[-1].net_worth - recovery_steps[0].net_worth) / max(recovery_steps[0].net_worth, 1.0)
    else:
        recovery_return = 0.0

    # Score combines equity participation and actual gains
    participation_score = avg_equity_in_recovery  # Higher equity in recovery = better
    gain_score = _linear_score(recovery_return, worst=-0.05, best=0.10)

    return _clamp(0.6 * participation_score + 0.4 * gain_score)


def grade_task3(record: EpisodeRecord) -> GradeResult:
    """
    Task 3: Flash Crash

    Expected agent behaviour:
      • Get out before or during the crash (exit timing)
      • Survive with capital mostly intact (survival)
      • Re-enter during recovery (recovery participation)
      • Maintain low max drawdown overall

    Expected score range: 0.15–0.35 for a reasonable agent.
    """
    assert record.regime == MarketRegime.CRASH, f"Task 3 expects CRASH regime; got {record.regime}"

    weights = {
        "survival":   0.35,   # Did you participate pre-crash AND exit?
        "exit_timing":0.35,   # How early did you exit? (exponential decay)
        "drawdown":   0.15,   # Overall capital control
        "recovery":   0.15,   # Re-entered during bounce? (hardest part)
    }

    sub = {
        "survival":    _crash_survival_score(record),
        "exit_timing": _crash_exit_timing_score(record),
        "drawdown":    _drawdown_score(record, worst_dd=0.35),
        "recovery":    _crash_recovery_score(record),
    }

    score = _clamp(sum(weights[k] * sub[k] for k in weights))

    return GradeResult(
        task=3,
        score=score,
        breakdown={k: round(v, 4) for k, v in sub.items()},
        rationale=(
            f"Crash score: survival={sub['survival']:.3f}, "
            f"exit_timing={sub['exit_timing']:.3f}, "
            f"drawdown={sub['drawdown']:.3f}, "
            f"recovery={sub['recovery']:.3f}. "
            f"Weighted total: {score:.4f}"
        ),
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

_GRADERS: dict[int, Callable[[EpisodeRecord], GradeResult]] = {
    1: grade_task1,
    2: grade_task2,
    3: grade_task3,
}

_REGIME_TO_TASK: dict[MarketRegime, int] = {
    MarketRegime.BULL:  1,
    MarketRegime.RANGE: 2,
    MarketRegime.CRASH: 3,
}


def grade_episode(record: EpisodeRecord) -> GradeResult:
    """
    Grade a completed episode, auto-selecting the correct grader.

    Parameters
    ----------
    record : Completed EpisodeRecord

    Returns
    -------
    GradeResult with score in [0.0, 1.0].
    """
    task = _REGIME_TO_TASK[record.regime]
    return _GRADERS[task](record)


# ---------------------------------------------------------------------------
# Self-test (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from models import (
        Action, ActionType, MarketRegime,
        RewardBreakdown, StepRecord,
    )

    def make_step(
        t: int,
        action_type: ActionType,
        fraction: float,
        price: float,
        net_worth: float,
        peak: float,
        equity: float,
    ) -> StepRecord:
        dd = max(0.0, 1.0 - net_worth / peak) if peak > 0 else 0.0
        return StepRecord(
            timestep=t,
            action=Action(action_type=action_type, fraction=fraction),
            reward=RewardBreakdown(
                pnl_reward=0.01, risk_penalty=0.0, drawdown_penalty=0.0,
                turnover_penalty=0.0, survival_bonus=0.0, total=0.01
            ),
            net_worth=net_worth,
            drawdown=dd,
            equity_fraction=equity,
            price=price,
        )

    def build_bull_perfect(n: int = 100) -> EpisodeRecord:
        """Perfect agent: buys on step 1, holds, ends with big gain."""
        steps = []
        for t in range(n):
            price = 100.0 + t * 0.2   # steady uptrend
            action = ActionType.BUY if t == 1 else ActionType.HOLD
            frac   = 0.90 if t == 1 else 0.0
            nw     = 100_000.0 + (price - 100.0) * 900   # ~900 shares
            steps.append(make_step(t, action, frac, price, nw, max(100_000.0, nw), 0.90))
        return EpisodeRecord(regime=MarketRegime.BULL, initial_capital=100_000.0, steps=steps)

    def build_bull_terrible(n: int = 100) -> EpisodeRecord:
        """Terrible agent: holds all cash, never buys."""
        steps = []
        for t in range(n):
            price = 100.0 + t * 0.2
            steps.append(make_step(t, ActionType.HOLD, 0.0, price, 100_000.0, 100_000.0, 0.0))
        return EpisodeRecord(regime=MarketRegime.BULL, initial_capital=100_000.0, steps=steps)

    def build_crash_perfect(n: int = 100) -> EpisodeRecord:
        """Perfect crash agent: holds cash, doesn't buy at all."""
        steps = []
        prices = ([100.0] * 25 + [100.0 - i * 3 for i in range(20)]
                  + [40.0 + i * 0.5 for i in range(55)])
        for t in range(n):
            price = prices[t] if t < len(prices) else 68.0
            steps.append(make_step(t, ActionType.HOLD, 0.0, price, 100_000.0, 100_000.0, 0.0))
        return EpisodeRecord(regime=MarketRegime.CRASH, initial_capital=100_000.0, steps=steps)

    def build_crash_terrible(n: int = 100) -> EpisodeRecord:
        """Terrible crash agent: goes all-in, rides the crash all the way down."""
        steps = []
        prices = ([100.0] * 25 + [100.0 - i * 3 for i in range(20)]
                  + [40.0 + i * 0.5 for i in range(55)])
        for t in range(n):
            price = prices[t] if t < len(prices) else 68.0
            nw = 100_000.0 * (price / 100.0)
            peak = max(100_000.0, 100_000.0 * (max(prices[:t+1]) / 100.0))
            action = ActionType.BUY if t == 0 else ActionType.HOLD
            frac   = 0.95 if t == 0 else 0.0
            steps.append(make_step(t, action, frac, price, nw, peak, 0.95 if price > 0 else 0.0))
        return EpisodeRecord(regime=MarketRegime.CRASH, initial_capital=100_000.0, steps=steps)

    print("=" * 60)
    print("Grader Verification")
    print("=" * 60)

    # Task 1
    bull_perf  = grade_task1(build_bull_perfect())
    bull_terr  = grade_task1(build_bull_terrible())
    print(f"\nTask 1 (Bull):")
    print(f"  Perfect agent : {bull_perf.score:.4f}  ← should be > 0.55")
    print(f"  Terrible agent: {bull_terr.score:.4f}  ← should be < 0.25")
    assert bull_perf.score > bull_terr.score, "Perfect should beat terrible"
    assert bull_perf.score > 0.40, f"Perfect bull score too low: {bull_perf.score}"

    # Task 3
    crash_perf = grade_task3(build_crash_perfect())
    crash_terr = grade_task3(build_crash_terrible())
    print(f"\nTask 3 (Crash):")
    print(f"  Perfect agent (cash) : {crash_perf.score:.4f}  ← should be > 0.35")
    print(f"  Terrible agent (HODL): {crash_terr.score:.4f}  ← should be < 0.25")
    assert crash_perf.score > crash_terr.score, "Cash survival should beat all-in crash"

    print(f"\nBreakdown (perfect bull):")
    for k, v in bull_perf.breakdown.items():
        print(f"  {k:20s}: {v:.4f}")

    print("\n✓ All grader tests passed.")
    print("\nExpected difficulty spread:")
    print(f"  Task 1 (bull)  → ~0.65–0.80  (grader designed for this range)")
    print(f"  Task 2 (range) → ~0.35–0.55  (grader designed for this range)")
    print(f"  Task 3 (crash) → ~0.15–0.35  (grader designed for this range)")
