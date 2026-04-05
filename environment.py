import pandas as pd
from fastapi import FastAPI, HTTPException  # <--- MAKE SURE THIS IS HERE
from pydantic import BaseModel
from models import Observation, Action, Reward, State, Bar
from market_data import ScenarioBank, IndicatorCalculator
from portfolio import PortfolioManager
from reward import calculate_step_reward, calculate_episode_bonus, calculate_sharpe
from graders import grade_task1_trend, grade_task2_mean_reversion, grade_task3_crash

TASK_META = {
    1: ("Bull trend following",    "Prices are rising steadily. Ride the trend."),
    2: ("Mean reversion — choppy", "Prices bounce $90-$110. Buy low, sell high."),
    3: ("Flash crash survival",    "A sudden crash is coming. Survive and recover."),
}

class TradeSimEnv:

    def __init__(self):
        self.portfolio = PortfolioManager()
        self.data: pd.DataFrame = None
        self.task_id: int = None
        self.timestep: int = 0
        self.pnl_curve: list = []
        self._prev_value: float = 100_000.0
        self._trade_window: list = []   # track last 5 decisions

    def reset(self, task_id: int = 1) -> Observation:
        self.task_id = task_id
        self.timestep = 0
        self.pnl_curve = []
        self._trade_window = []
        self.portfolio.reset()
        self._prev_value = self.portfolio.starting_cash

        if task_id == 1:
            self.data = ScenarioBank.generate_bull_trend()
        elif task_id == 2:
            self.data = ScenarioBank.generate_choppy_range()
        elif task_id == 3:
            self.data = ScenarioBank.generate_flash_crash()
        else:
            raise ValueError(f"task_id must be 1, 2, or 3. Got {task_id}")

        # pre-compute indicators on full series
        close = self.data['close']
        self._ma20    = IndicatorCalculator.moving_average(close, 20)
        self._rsi     = IndicatorCalculator.rsi(close)
        self._macd, _ = IndicatorCalculator.macd(close)
        self._bb_u, self._bb_l = IndicatorCalculator.bollinger_bands(close)

        return self._build_observation()

    def step(self, action: Action):
        price = float(self.data['close'].iloc[self.timestep])
        prev_val = self.portfolio.mark_to_market(price)
        self._prev_value = prev_val

        # execute the trade
        self.portfolio.execute_trade(
            action.decision, action.position_size, price, self.timestep
        )

        # advance time
        self.timestep += 1
        done = self.timestep >= len(self.data) - 1

        # mark to market at new price
        new_price = float(self.data['close'].iloc[self.timestep])
        curr_val  = self.portfolio.mark_to_market(new_price)
        self.pnl_curve.append(curr_val)

        # track trades for overtrading detection
        self._trade_window.append(action.decision)
        if len(self._trade_window) > 5:
            self._trade_window.pop(0)
        trade_count = sum(1 for d in self._trade_window if d != "HOLD")

        drawdown = self.portfolio.get_drawdown(new_price)
        step_reward, breakdown = calculate_step_reward(
            prev_val, curr_val, drawdown, action, trade_count
        )

        info = {"timestep": self.timestep, "portfolio_value": curr_val}

        if done:
            sharpe  = calculate_sharpe(self.pnl_curve)
            max_dd  = max(drawdown, 0)
            bonus   = calculate_episode_bonus(
                sharpe, max_dd, curr_val, self.portfolio.starting_cash
            )
            graders = {1: grade_task1_trend,
                       2: grade_task2_mean_reversion,
                       3: grade_task3_crash}
            final_score = graders[self.task_id](
                self.portfolio.trade_history,
                self.pnl_curve,
                self.portfolio.starting_cash
            )
            info["final_score"] = round(final_score + bonus, 4)
            info["sharpe"]      = round(sharpe, 3)

        obs    = self._build_observation()
        reward = Reward(score=step_reward, feedback=str(breakdown),
                        partial_breakdown=breakdown)
        return obs, reward, done, info

    def state(self) -> State:
        price = float(self.data['close'].iloc[self.timestep]) if self.data is not None else 0.0
        return State(
            task_id=self.task_id or 0,
            current_timestep=self.timestep,
            portfolio_value=self.portfolio.mark_to_market(price),
            peak_value=self.portfolio.peak_portfolio_value,
            trade_history=self.portfolio.trade_history,
            pnl_curve=self.pnl_curve
        )

    def _build_observation(self) -> Observation:
        t = self.timestep
        price = float(self.data['close'].iloc[t])
        start = max(0, t - 4)
        bars  = [
            Bar(timestep=i,
                open=float(self.data['open'].iloc[i]),
                high=float(self.data['high'].iloc[i]),
                low=float(self.data['low'].iloc[i]),
                close=float(self.data['close'].iloc[i]),
                volume=float(self.data['volume'].iloc[i]))
            for i in range(start, t + 1)
        ]
        pval = self.portfolio.mark_to_market(price)
        upnl = self.portfolio.get_unrealized_pnl(price)
        desc, hint = TASK_META.get(self.task_id, ("unknown",""))
        return Observation(
            current_price=round(price, 4),
            last_5_bars=bars,
            indicators={
                "rsi":      round(float(self._rsi.iloc[t]),   2),
                "ma20":     round(float(self._ma20.iloc[t]),  2),
                "macd":     round(float(self._macd.iloc[t]),  4),
                "bb_upper": round(float(self._bb_u.iloc[t]),  2),
                "bb_lower": round(float(self._bb_l.iloc[t]),  2),
            },
            portfolio={
                "cash":           round(self.portfolio.cash, 2),
                "position":       round(self.portfolio.position, 4),
                "unrealized_pnl": round(upnl, 2),
                "total_value":    round(pval, 2),
            },
            timestep=t,
            task_description=desc,
            market_regime_hint=hint
        )
    from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="TradeSim")
_env = TradeSimEnv()

class ResetRequest(BaseModel):
    task_id: int = 1

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest):
    try:
        obs = _env.reset(req.task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done, 
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    return _env.state().model_dump()