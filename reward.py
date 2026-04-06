"""
TradeSim — reward.py
====================
The reward function is the MOST IMPORTANT file in TradeSim. It embeds the
entire philosophy of what "good trading" means — not raw profit, but
risk-adjusted, disciplined, capital-preserving decision-making.

Quant Finance Rationale
-----------------------
A naive reward of Δ(net_worth) produces agents that:
  (a) go all-in and pray — they get lucky sometimes and catastrophic others
  (b) trade constantly — generating friction and random noise
  (c) ignore drawdown — blow up in crashes

The five-component reward solves each failure mode:

  1. PNL Reward     — Reward mark-to-market gains; use SQRT transform to
                      dampen outliers and prevent extreme risk-taking.

  2. Risk Penalty   — Heavy penalty when equity concentration > 70%.
                      This encodes Kelly criterion intuition: never bet
                      your entire bankroll on one asset.

  3. Drawdown Penalty — Ongoing penalty proportional to how deep in the
                        hole the agent is. Teaches it to exit losing
                        positions proactively.

  4. Turnover Penalty — Small per-trade friction cost beyond the market
                        transaction cost. Penalises "nervous" over-trading
                        that doesn't improve the position.

  5. Survival Bonus — In crash regime only: a bonus for being in cash
                      (equity_fraction near 0) when the market is falling.
                      This is the "getting out before the cliff" reward.

Calibration
-----------
All components are scaled so that:
  • A perfect bull-market episode (buy early, hold, sell at peak) → ~+2.0 total
  • A terrible episode (HOLD all cash in bull; all-in through crash) → ~-1.0 total
  • Raw total rewards are later normalised by graders to [0, 1].
"""

from __future__ import annotations

import math

from models import (
    Action,
    ActionType,
    MarketRegime,
    PortfolioSnapshot,
    RewardBreakdown,
)


# ---------------------------------------------------------------------------
# Tunable hyperparameters — documented for interpretability
# ---------------------------------------------------------------------------

# PNL component
PNL_SCALE           = 1.0   # Multiplier on SQRT-transformed P&L change
PNL_SMOOTHING       = 1e-4  # Prevents sqrt(0) issues

# Risk penalty (concentration)
RISK_THRESHOLD      = 0.70  # Above this equity fraction → penalty begins
RISK_SCALE          = 0.50  # Strength of the concentration penalty
RISK_EXPONENT       = 2.0   # Quadratic penalty for extreme concentration

# Drawdown penalty
DD_THRESHOLD        = 0.05  # Below 5% drawdown → no penalty
DD_SCALE            = 0.40  # Strength of drawdown penalty
DD_EXPONENT         = 1.5   # Super-linear penalty for deep drawdowns

# Turnover penalty
TURNOVER_FLAT_COST  = 0.003 # Fixed per-trade penalty (on top of market cost)
TURNOVER_VALUE_SCALE = 0.001 # Proportional-to-trade-value penalty

# Survival bonus (crash regime only)
SURVIVAL_SCALE      = 0.20  # Max per-step bonus for being in cash during crash
CRASH_FALLING_THRESH = -0.005  # Price return threshold to classify "falling"


# ---------------------------------------------------------------------------
# Component calculators
# ---------------------------------------------------------------------------

def _pnl_reward(
    prev_net_worth: float,
    curr_net_worth: float,
    initial_capital: float,
) -> float:
    """
    Reward for mark-to-market portfolio value change.

    Normalised by initial capital to make the signal scale-invariant.
    SQRT transform dampens large windfalls — we want consistent gains,
    not lottery tickets.

    Returns a positive number for gains, negative for losses.
    """
    if prev_net_worth <= 0:
        return 0.0

    delta = curr_net_worth - prev_net_worth
    normalised_delta = delta / initial_capital

    # Signed sqrt transform: sign(x) * sqrt(|x|)
    magnitude = math.sqrt(abs(normalised_delta) + PNL_SMOOTHING) - math.sqrt(PNL_SMOOTHING)
    return PNL_SCALE * math.copysign(magnitude, normalised_delta)


def _risk_penalty(equity_fraction: float) -> float:
    """
    Penalty for over-concentration in equities.

    No penalty below RISK_THRESHOLD.
    Quadratic penalty above it — doubling concentration → 4× penalty.

    Returns a value ≤ 0.
    """
    if equity_fraction <= RISK_THRESHOLD:
        return 0.0

    excess = equity_fraction - RISK_THRESHOLD
    # Scale to [0, 1] range (max excess = 1.0 - threshold)
    max_excess = 1.0 - RISK_THRESHOLD
    scaled_excess = excess / max_excess  # in [0, 1]

    penalty = -RISK_SCALE * (scaled_excess ** RISK_EXPONENT)
    return penalty


def _drawdown_penalty(drawdown: float) -> float:
    """
    Ongoing penalty for being in a drawdown state.

    No penalty for small drawdowns (≤ DD_THRESHOLD — these are normal noise).
    Super-linear (1.5 power) for large drawdowns — the deeper the hole,
    the harder the penalty. This forces the agent to value capital preservation.

    Returns a value ≤ 0.
    """
    if drawdown <= DD_THRESHOLD:
        return 0.0

    excess = drawdown - DD_THRESHOLD
    max_excess = 1.0 - DD_THRESHOLD
    scaled_excess = excess / max_excess  # in [0, 1]

    penalty = -DD_SCALE * (scaled_excess ** DD_EXPONENT)
    return penalty


def _turnover_penalty(
    action: Action,
    trade_value: float,
    initial_capital: float,
) -> float:
    """
    Penalty for making trades — especially large ones.

    Designed to:
      1. Discourage trivial micro-trades that don't improve position.
      2. Add a realistic "market impact" cost on top of transaction fees.

    A flat per-trade cost + a value-proportional cost.

    Returns a value ≤ 0.
    """
    if action.action_type == ActionType.HOLD:
        return 0.0

    if trade_value <= 0:
        return 0.0

    # Flat cost per trade
    flat_cost = -TURNOVER_FLAT_COST

    # Proportional cost (normalised to initial capital)
    prop_cost = -TURNOVER_VALUE_SCALE * (trade_value / initial_capital)

    return flat_cost + prop_cost


def _survival_bonus(
    regime: MarketRegime,
    equity_fraction: float,
    price_return: float,
) -> float:
    """
    Crash-regime only: reward for being in cash when the market is falling.

    This is the most regime-specific component and only activates during Task 3.
    When price_return < CRASH_FALLING_THRESH (market is dropping), holding cash
    (equity_fraction near 0) earns a bonus. This trains the agent to detect and
    exit during the crash phase.

    Returns a value ≥ 0.
    """
    if regime != MarketRegime.CRASH:
        return 0.0

    # Only bonus when market is actively falling
    if price_return >= CRASH_FALLING_THRESH:
        return 0.0

    # Bonus scales with how much cash you're holding
    # If equity_fraction = 0 → full bonus; if = 1 → no bonus
    cash_fraction = 1.0 - equity_fraction
    magnitude = abs(price_return)  # Larger falls → bigger bonus for being out

    bonus = SURVIVAL_SCALE * cash_fraction * min(magnitude * 20, 1.0)
    return max(0.0, bonus)


# ---------------------------------------------------------------------------
# Public API — compute full reward at one timestep
# ---------------------------------------------------------------------------

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
) -> RewardBreakdown:
    """
    Compute the composite reward for a single timestep.

    Parameters
    ----------
    action          : Action executed this step
    prev_snapshot   : Portfolio state *before* the action
    curr_snapshot   : Portfolio state *after* the action and price update
    regime          : Which market scenario is running
    prev_price      : Price at start of this step
    curr_price      : Price at end of this step
    trade_value     : Gross dollar value of trade (0 for HOLD)
    initial_capital : Starting capital for normalisation

    Returns
    -------
    RewardBreakdown — all components and their validated sum.
    """
    price_return = (curr_price - prev_price) / prev_price if prev_price > 0 else 0.0

    # Compute each component
    pnl      = _pnl_reward(prev_snapshot.net_worth, curr_snapshot.net_worth, initial_capital)
    risk     = _risk_penalty(curr_snapshot.equity_fraction)
    drawdown = _drawdown_penalty(curr_snapshot.drawdown)
    turnover = _turnover_penalty(action, trade_value, initial_capital)
    survival = _survival_bonus(regime, curr_snapshot.equity_fraction, price_return)

    total = pnl + risk + drawdown + turnover + survival

    return RewardBreakdown(
        pnl_reward=pnl,
        risk_penalty=risk,
        drawdown_penalty=drawdown,
        turnover_penalty=turnover,
        survival_bonus=survival,
        total=total,
    )


# ---------------------------------------------------------------------------
# Manual verification (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from models import Action, ActionType, MarketRegime, PortfolioSnapshot

    def make_snap(
        cash: float,
        shares: float,
        price: float,
        peak: float,
        initial: float = 100_000.0,
        trades: int = 0,
    ) -> PortfolioSnapshot:
        nw = cash + shares * price
        dd = max(0.0, 1.0 - nw / peak) if peak > 0 else 0.0
        eq = (shares * price) / nw if nw > 0 else 0.0
        return PortfolioSnapshot(
            cash=cash,
            shares_held=shares,
            current_price=price,
            net_worth=nw,
            peak_net_worth=peak,
            drawdown=dd,
            equity_fraction=eq,
            total_trades=trades,
            total_return=(nw - initial) / initial,
        )

    print("=" * 60)
    print("Reward Function Verification")
    print("=" * 60)

    IC = 100_000.0

    # Test 1: Good trade — bought at 100, price went to 110, all-in (but < 95%)
    print("\n[Test 1] Good bull trade — bought fully, price up 10%")
    prev = make_snap(cash=5_000, shares=950, price=100.0, peak=100_000, initial=IC)
    curr = make_snap(cash=5_000, shares=950, price=110.0, peak=109_500, initial=IC)
    r = compute_reward(
        action=Action.hold(),
        prev_snapshot=prev, curr_snapshot=curr,
        regime=MarketRegime.BULL,
        prev_price=100.0, curr_price=110.0,
        trade_value=0.0, initial_capital=IC,
    )
    print(f"  PNL reward    : {r.pnl_reward:+.6f}")
    print(f"  Risk penalty  : {r.risk_penalty:+.6f}")
    print(f"  Total         : {r.total:+.6f}")
    assert r.pnl_reward > 0, "Should be positive on a winning hold"
    assert r.total > 0,      "Good trade should have positive total reward"

    # Test 2: Terrible — all-in (100% equity, max concentration risk)
    print("\n[Test 2] Terrible concentration — 100% equity (reckless)")
    prev2 = make_snap(cash=0, shares=1000, price=100.0, peak=100_000, initial=IC)
    curr2 = make_snap(cash=0, shares=1000, price=100.5, peak=100_500, initial=IC)
    r2 = compute_reward(
        action=Action.hold(),
        prev_snapshot=prev2, curr_snapshot=curr2,
        regime=MarketRegime.BULL,
        prev_price=100.0, curr_price=100.5,
        trade_value=0.0, initial_capital=IC,
    )
    print(f"  PNL reward    : {r2.pnl_reward:+.6f}")
    print(f"  Risk penalty  : {r2.risk_penalty:+.6f}")
    print(f"  Total         : {r2.total:+.6f}")
    assert r2.risk_penalty < 0, "100% equity should trigger heavy risk penalty"

    # Test 3: Deep drawdown — price fell 30%, agent still holding
    print("\n[Test 3] Deep drawdown — 30% from peak, still holding")
    prev3 = make_snap(cash=0, shares=1000, price=90.0, peak=100_000, initial=IC)
    curr3 = make_snap(cash=0, shares=1000, price=70.0, peak=100_000, initial=IC)
    r3 = compute_reward(
        action=Action.hold(),
        prev_snapshot=prev3, curr_snapshot=curr3,
        regime=MarketRegime.BULL,
        prev_price=90.0, curr_price=70.0,
        trade_value=0.0, initial_capital=IC,
    )
    print(f"  PNL reward     : {r3.pnl_reward:+.6f}")
    print(f"  Drawdown pen.  : {r3.drawdown_penalty:+.6f}")
    print(f"  Total          : {r3.total:+.6f}")
    assert r3.drawdown_penalty < -0.1, "Deep drawdown should have strong penalty"
    assert r3.total < -0.1,            "Net result should be very negative"

    # Test 4: Survival bonus — crash regime, held cash during crash
    print("\n[Test 4] Survival bonus — cash during crash drop")
    prev4 = make_snap(cash=100_000, shares=0, price=100.0, peak=100_000, initial=IC)
    curr4 = make_snap(cash=100_000, shares=0, price=80.0, peak=100_000, initial=IC)
    r4 = compute_reward(
        action=Action.hold(),
        prev_snapshot=prev4, curr_snapshot=curr4,
        regime=MarketRegime.CRASH,
        prev_price=100.0, curr_price=80.0,
        trade_value=0.0, initial_capital=IC,
    )
    print(f"  PNL reward     : {r4.pnl_reward:+.6f}")
    print(f"  Survival bonus : {r4.survival_bonus:+.6f}")
    print(f"  Total          : {r4.total:+.6f}")
    assert r4.survival_bonus > 0, "Should earn survival bonus for holding cash during crash"

    # Test 5: Overtrading — many small trades
    print("\n[Test 5] Turnover penalty — large trade on tiny move")
    prev5 = make_snap(cash=50_000, shares=500, price=100.0, peak=100_000, initial=IC)
    curr5 = make_snap(cash=50_000, shares=500, price=100.1, peak=100_050, initial=IC)
    r5 = compute_reward(
        action=Action.buy(fraction=0.5),
        prev_snapshot=prev5, curr_snapshot=curr5,
        regime=MarketRegime.BULL,
        prev_price=100.0, curr_price=100.1,
        trade_value=25_000.0, initial_capital=IC,
    )
    print(f"  Turnover pen.  : {r5.turnover_penalty:+.6f}")
    assert r5.turnover_penalty < 0, "Trade should incur turnover penalty"

    print("\n✓ All reward function tests passed.")
    print("\nReward philosophy verified:")
    print("  + Positive reward for good risk-adjusted gains")
    print("  - Risk penalty for over-concentration")
    print("  - Drawdown penalty for staying in losses")
    print("  - Turnover penalty for unnecessary trading")
    print("  + Survival bonus for cash during crash")
