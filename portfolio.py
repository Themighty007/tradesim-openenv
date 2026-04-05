from models import Trade
from typing import List

class PortfolioManager:

    def __init__(self, starting_cash: float = 100_000.0):
        self.starting_cash = starting_cash
        self.reset()

    def reset(self):
        self.cash = self.starting_cash
        self.position = 0.0          # units of stock held
        self.entry_price = 0.0
        self.peak_portfolio_value = self.starting_cash
        self.trade_history: List[Trade] = []
        self._timestep = 0

    def execute_trade(self, decision: str, position_size: float,
                      current_price: float, timestep: int):
        self._timestep = timestep

        if decision == "BUY" and self.cash > 0:
            spend = self.cash * position_size
            units = spend / current_price
            self.cash -= spend
            self.position += units
            self.entry_price = current_price
            self.trade_history.append(Trade(
                timestep=timestep, decision="BUY",
                price=current_price, quantity=units,
                cash_after=self.cash
            ))

        elif decision == "SELL" and self.position > 0:
            proceeds = self.position * current_price
            self.cash += proceeds
            self.trade_history.append(Trade(
                timestep=timestep, decision="SELL",
                price=current_price, quantity=self.position,
                cash_after=self.cash
            ))
            self.position = 0.0
            self.entry_price = 0.0

        # HOLD: do nothing

    def mark_to_market(self, current_price: float) -> float:
        total = self.cash + self.position * current_price
        if total > self.peak_portfolio_value:
            self.peak_portfolio_value = total
        return total

    def get_drawdown(self, current_price: float) -> float:
        current = self.cash + self.position * current_price
        if self.peak_portfolio_value == 0:
            return 0.0
        return (self.peak_portfolio_value - current) / self.peak_portfolio_value

    def get_unrealized_pnl(self, current_price: float) -> float:
        if self.position == 0:
            return 0.0
        return (current_price - self.entry_price) * self.position