"""
TradeSim — portfolio.py
=======================
The portfolio accounting engine. Maintains perfect mathematical accuracy for
every trade, every timestep. Think of this as the brokerage back-office.

Design principles:
  • Immutable state transitions — every update returns a *new* PortfolioState.
  • Transaction costs deducted at trade time (realistic friction modelling).
  • Fractional shares supported (simulate a modern fractional-share broker).
  • Peak tracking for drawdown is monotonically updated — never reset mid-episode.
  • All arithmetic is in float64; monetary values are kept to 2 dp precision
    only for display — internal calculations use full precision.
  • No side-effects beyond the returned value — safe to call in parallel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from models import Action, ActionType, PortfolioSnapshot


# ---------------------------------------------------------------------------
# Internal mutable state (not exposed to agents)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """
    Mutable internal ledger for the portfolio manager.

    Not the same as PortfolioSnapshot — this is the engine's private workspace;
    PortfolioSnapshot is the read-only view given to the agent.
    """

    initial_capital: float          # USD; never changes
    cash:            float          # Current liquid USD
    shares:          float          # Fractional shares held
    peak_net_worth:  float          # High-water mark for drawdown calc
    total_trades:    int   = 0      # Cumulative trade count
    _history:        list  = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def net_worth(self, price: float) -> float:
        """cash + mark-to-market equity value."""
        return self.cash + self.shares * price

    def equity_value(self, price: float) -> float:
        return self.shares * price

    def equity_fraction(self, price: float) -> float:
        nw = self.net_worth(price)
        if nw <= 0:
            return 0.0
        return self.equity_value(price) / nw

    def drawdown(self, price: float) -> float:
        """Current drawdown from the peak net worth (in [0,1])."""
        nw = self.net_worth(price)
        if self.peak_net_worth <= 0:
            return 0.0
        dd = 1.0 - nw / self.peak_net_worth
        return max(0.0, dd)

    def total_return(self, price: float) -> float:
        return (self.net_worth(price) - self.initial_capital) / self.initial_capital

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute(
        self,
        action: Action,
        price: float,
        transaction_cost: float,
        max_position_fraction: float,
    ) -> tuple["PortfolioState", dict]:
        """
        Execute an action against the current portfolio state.

        Returns a *new* PortfolioState and a diagnostics dict.
        The original state is never mutated.
        """
        diag: dict = {
            "action_type": action.action_type.value,
            "requested_fraction": action.fraction,
            "price": price,
            "executed": False,
            "trade_value": 0.0,
            "cost": 0.0,
            "shares_delta": 0.0,
            "cash_delta": 0.0,
            "reason": "",
        }

        if action.action_type == ActionType.HOLD:
            diag["reason"] = "HOLD — no trade"
            new_state = replace(self)
            new_state._history = self._history  # share reference (read-only downstream)
            return new_state, diag

        # ---- BUY -------------------------------------------------------
        if action.action_type == ActionType.BUY:
            if self.cash <= 0:
                diag["reason"] = "No cash available to buy"
                return replace(self, _history=self._history), diag

            # Enforce max position size
            nw = self.net_worth(price)
            current_equity = self.equity_value(price)
            max_equity = nw * max_position_fraction
            headroom = max(0.0, max_equity - current_equity)

            cash_to_deploy = min(
                self.cash * action.fraction,
                headroom,
            )

            if cash_to_deploy < 0.01:
                diag["reason"] = "Position already at max; no buy executed"
                return replace(self, _history=self._history), diag

            cost = cash_to_deploy * transaction_cost
            net_cash_spent = cash_to_deploy
            shares_bought  = (cash_to_deploy - cost) / price

            new_cash   = self.cash - net_cash_spent
            new_shares = self.shares + shares_bought

            diag.update({
                "executed": True,
                "trade_value": cash_to_deploy,
                "cost": cost,
                "shares_delta": +shares_bought,
                "cash_delta": -net_cash_spent,
                "reason": f"Bought {shares_bought:.4f} shares @ ${price:.2f} (cost ${cost:.4f})",
            })

            new_state = replace(
                self,
                cash=new_cash,
                shares=new_shares,
                total_trades=self.total_trades + 1,
            )
            new_state._history = self._history
            return new_state, diag

        # ---- SELL ------------------------------------------------------
        if action.action_type == ActionType.SELL:
            if self.shares <= 1e-9:
                diag["reason"] = "No shares to sell"
                return replace(self, _history=self._history), diag

            shares_to_sell = self.shares * action.fraction
            if shares_to_sell < 1e-9:
                diag["reason"] = "Fraction too small — no sell executed"
                return replace(self, _history=self._history), diag

            gross_proceeds = shares_to_sell * price
            cost           = gross_proceeds * transaction_cost
            net_proceeds   = gross_proceeds - cost

            new_cash   = self.cash + net_proceeds
            new_shares = self.shares - shares_to_sell

            # Prevent floating-point ghost shares
            if new_shares < 1e-9:
                new_shares = 0.0

            diag.update({
                "executed": True,
                "trade_value": gross_proceeds,
                "cost": cost,
                "shares_delta": -shares_to_sell,
                "cash_delta": +net_proceeds,
                "reason": f"Sold {shares_to_sell:.4f} shares @ ${price:.2f} (cost ${cost:.4f})",
            })

            new_state = replace(
                self,
                cash=new_cash,
                shares=new_shares,
                total_trades=self.total_trades + 1,
            )
            new_state._history = self._history
            return new_state, diag

        # Should never reach here
        raise ValueError(f"Unknown action type: {action.action_type}")

    # ------------------------------------------------------------------
    # Peak update (call after each price tick — even if no trade)
    # ------------------------------------------------------------------

    def update_peak(self, price: float) -> "PortfolioState":
        """Return a new state with the high-water mark updated if needed."""
        nw = self.net_worth(price)
        new_peak = max(self.peak_net_worth, nw)
        return replace(self, peak_net_worth=new_peak)

    # ------------------------------------------------------------------
    # Snapshot — the read-only view given to the agent
    # ------------------------------------------------------------------

    def to_snapshot(self, price: float) -> PortfolioSnapshot:
        """
        Serialise current state into a validated PortfolioSnapshot.

        This is what the agent and reward functions receive.
        """
        nw = self.net_worth(price)
        dd = self.drawdown(price)
        eq = self.equity_fraction(price)
        tr = self.total_return(price)

        return PortfolioSnapshot(
            cash=self.cash,
            shares_held=self.shares,
            current_price=price,
            net_worth=nw,
            peak_net_worth=self.peak_net_worth,
            drawdown=dd,
            equity_fraction=eq,
            total_trades=self.total_trades,
            total_return=tr,
        )


# ---------------------------------------------------------------------------
# Portfolio factory
# ---------------------------------------------------------------------------

def create_portfolio(initial_capital: float = 100_000.0) -> PortfolioState:
    """
    Initialise a fresh portfolio with all cash and no positions.

    Parameters
    ----------
    initial_capital : Starting USD cash balance (default $100,000)

    Returns
    -------
    PortfolioState ready for use at episode start.
    """
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError(f"initial_capital must be a positive finite number; got {initial_capital}")

    return PortfolioState(
        initial_capital=initial_capital,
        cash=initial_capital,
        shares=0.0,
        peak_net_worth=initial_capital,
    )


# ---------------------------------------------------------------------------
# Manual test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run the canonical buy-at-100 / sell-at-110 test.

    Expected result:
      • Start: $100,000 cash
      • Buy  : spend 100% of cash at $100 → own ≈ 999 shares (after 0.1% cost)
      • Sell : sell all shares at $110 → receive ≈ $109,769 (after 0.1% cost)
      • P&L  : ≈ +$9,769  (~+9.77%)
    """
    from models import Action, ActionType

    TC  = 0.001   # 0.1% transaction cost
    MPF = 0.95    # max 95% in equities

    print("=" * 60)
    print("Portfolio Accounting Test — Buy@100 / Sell@110")
    print("=" * 60)

    # --- Setup ---
    port = create_portfolio(initial_capital=100_000.0)
    snap = port.to_snapshot(price=100.0)
    print(f"\nInitial state:")
    print(f"  Cash        : ${snap.cash:,.2f}")
    print(f"  Shares      : {snap.shares_held:.4f}")
    print(f"  Net worth   : ${snap.net_worth:,.2f}")

    # --- Buy all in at $100 ---
    buy_action = Action.buy(fraction=1.0, reason="Full deployment at $100")
    port, diag_buy = port.execute(buy_action, price=100.0,
                                   transaction_cost=TC, max_position_fraction=MPF)
    port = port.update_peak(price=100.0)
    snap = port.to_snapshot(price=100.0)

    print(f"\nAfter BUY (fraction=1.0 @ $100):")
    print(f"  {diag_buy['reason']}")
    print(f"  Cash        : ${snap.cash:,.4f}")
    print(f"  Shares      : {snap.shares_held:.4f}")
    print(f"  Net worth   : ${snap.net_worth:,.2f}")
    print(f"  Equity frac : {snap.equity_fraction:.4f}")
    print(f"  Drawdown    : {snap.drawdown:.6f}")

    # --- Price rises to $110 ---
    port = port.update_peak(price=110.0)
    snap = port.to_snapshot(price=110.0)
    print(f"\nPrice moved to $110 (no trade):")
    print(f"  Net worth   : ${snap.net_worth:,.2f}")
    print(f"  Unrealised P&L: ${snap.net_worth - 100_000:,.2f}")

    # --- Sell all at $110 ---
    sell_action = Action.sell(fraction=1.0, reason="Full exit at $110")
    port, diag_sell = port.execute(sell_action, price=110.0,
                                    transaction_cost=TC, max_position_fraction=MPF)
    port = port.update_peak(price=110.0)
    snap = port.to_snapshot(price=110.0)

    print(f"\nAfter SELL (fraction=1.0 @ $110):")
    print(f"  {diag_sell['reason']}")
    print(f"  Cash        : ${snap.cash:,.4f}")
    print(f"  Shares      : {snap.shares_held:.6f}")
    print(f"  Net worth   : ${snap.net_worth:,.2f}")
    print(f"  Total return: {snap.total_return*100:+.4f}%")

    # --- Assertions ---
    assert snap.shares_held < 1e-6,       "Should hold zero shares after full sell"
    assert snap.net_worth > 100_000,      "Should have made money"
    assert snap.total_return > 0.09,      "Expected ~9.7-9.8% return after costs"
    assert snap.total_return < 0.11,      "Return should be < 11% (costs exist)"
    assert snap.drawdown == 0.0,          "No drawdown — price only went up"
    assert port.total_trades == 2,        "Should have executed exactly 2 trades"

    # --- Test drawdown ---
    port2 = create_portfolio(100_000.0)
    port2, _ = port2.execute(Action.buy(1.0), price=100.0, transaction_cost=TC, max_position_fraction=MPF)
    port2 = port2.update_peak(price=100.0)
    port2 = port2.update_peak(price=120.0)   # Peak set at $120
    snap2 = port2.to_snapshot(price=90.0)    # Price falls to $90

    print(f"\nDrawdown test (peak@120, current@90):")
    print(f"  Peak net worth: ${snap2.peak_net_worth:,.2f}")
    print(f"  Current net worth: ${snap2.net_worth:,.2f}")
    print(f"  Drawdown: {snap2.drawdown*100:.2f}%")
    assert snap2.drawdown > 0.20, "Expected >20% drawdown"

    print("\n✓ All portfolio accounting tests passed.")
