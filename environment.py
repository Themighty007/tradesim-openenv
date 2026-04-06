"""
TradeSim — environment.py
=========================
The master loop. This is the game engine — it owns the clock, the market,
the portfolio, the reward signal, and the grading.

Interface Contract (OpenEnv standard):
  env = TradeSimEnv(config)
  obs = env.reset(task_id)           # Start an episode
  obs, reward, done, info = env.step(action)  # Advance one step
  state = env.state()                # Read current episode state

The environment is stateful — it remembers the full episode history
internally, which allows graders to score the complete trajectory.

Design decisions:
  • Strict validation at every boundary (models.py types enforced)
  • Clean separation: environment never touches reward math directly
  • Info dict always populated — diagnostic transparency for debugging
  • Graceful handling of edge cases (end-of-episode, no position, etc.)
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from graders import grade_episode
from market_data import (
    compute_rolling_features,
    generate_prices,
)
from models import (
    Action,
    ActionType,
    EpisodeRecord,
    EnvironmentConfig,
    GradeResult,
    MarketRegime,
    Observation,
    PortfolioSnapshot,
    PriceWindow,
    RewardBreakdown,
    StepRecord,
    StepResult,
    State,
    Trade,
)
from portfolio import PortfolioState, create_portfolio
from reward import compute_reward


# Task ID → MarketRegime mapping
_TASK_REGIMES: dict[int, MarketRegime] = {
    1: MarketRegime.BULL,
    2: MarketRegime.RANGE,
    3: MarketRegime.CRASH,
}

_TASK_DESCRIPTIONS: dict[int, str] = {
    1: (
        "TASK 1 — BULL MARKET\n"
        "The market is trending steadily upward. A reasonable strategy is to "
        "identify the trend early, take a meaningful long position, and hold "
        "through the trend with disciplined risk management. Avoid over-trading "
        "and excessive concentration. Expected score: 0.65–0.80."
    ),
    2: (
        "TASK 2 — CHOPPY RANGE\n"
        "The market is oscillating sideways with no sustained trend. "
        "Trend-following will lose money to friction. Capital preservation is "
        "paramount. Identify the range, trade lightly if at all, and keep "
        "drawdowns tight. Expected score: 0.35–0.55."
    ),
    3: (
        "TASK 3 — FLASH CRASH\n"
        "A sudden violent price drop is coming at some point in the first third "
        "of the episode, followed by a slow partial recovery. Your primary goal "
        "is to SURVIVE — exit before or during the crash, preserve capital, then "
        "cautiously re-enter during recovery. Expected score: 0.15–0.35."
    ),
}

_REGIME_HINTS: dict[MarketRegime, str] = {
    MarketRegime.BULL:  "Trending upward. RSI likely above 50. Momentum is positive.",
    MarketRegime.RANGE: "Oscillating sideways. RSI alternates. No clear trend.",
    MarketRegime.CRASH: "Warning: elevated volatility. Watch for sudden reversal.",
}


class TradeSimEnv:
    """
    TradeSim Reinforcement Learning Environment.

    Implements the standard reset() / step() / state() interface.
    One instance can run multiple episodes sequentially — call reset()
    to start a new one.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialise the environment with an optional config.

        If no config is provided, defaults are used with task_id set on reset().
        """
        self._base_config = config
        self._reset_internals()

    def _reset_internals(self):
        """Zero out all episode-level state."""
        self._prices: Optional[np.ndarray] = None
        self._precomputed_windows: list = []
        self._portfolio: Optional[PortfolioState] = None
        self._config: Optional[EnvironmentConfig] = None
        self._timestep: int = 0
        self._done: bool = False
        self._episode_steps: list[StepRecord] = []
        self._prev_portfolio_snapshot: Optional[PortfolioSnapshot] = None
        self._task_id: Optional[int] = None
        self._episode_start_time: float = 0.0
        self._last_grade: Optional[GradeResult] = None
        self._precomputed_windows: list[PriceWindow] = []

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    def reset(self, task_id: int = 1) -> Observation:
        """
        Start a new episode.
        """
        if task_id not in _TASK_REGIMES:
            raise ValueError(f"task_id must be 1, 2, or 3; got {task_id}")

        regime = _TASK_REGIMES[task_id]

        # Build episode config — merge base config if provided
        if self._base_config is not None:
            config = EnvironmentConfig(
                regime=regime,
                num_steps=self._base_config.num_steps,
                initial_capital=self._base_config.initial_capital,
                window_size=self._base_config.window_size,
                seed=self._base_config.seed,
                transaction_cost=self._base_config.transaction_cost,
                max_position_fraction=self._base_config.max_position_fraction,
            )
        else:
            config = EnvironmentConfig(regime=regime)

        self._config      = config
        self._task_id     = task_id
        self._timestep    = 0
        self._done        = False
        self._episode_steps = []
        self._last_grade  = None
        self._episode_start_time = time.time()

        # Generate price series
        self._prices = generate_prices(
            regime=regime,
            num_steps=config.num_steps,
            seed=config.seed,
        )

        # --- OPTIMIZATION 2: VECTORIZED PRE-COMPUTATION BLOCK ---
        import numpy as np 
        from models import PriceWindow 
        
        self._precomputed_windows = []
        prices = self._prices
        n = config.num_steps
        ws = config.window_size
        
        for t in range(n):
            window_start = max(0, t - ws + 1)
            actual_window_size = t - window_start + 1
            raw_window = prices[window_start:window_start + actual_window_size]
            
            # Pad early timesteps with the initial price
            pad_needed = ws - actual_window_size
            if pad_needed > 0:
                raw_window = np.concatenate((np.full(pad_needed, prices[0]), raw_window))
                
            # Vectorized Log Returns
            log_rets = np.diff(np.log(raw_window)).tolist()
            
            # Vectorized Z-Score Normalization
            mu = np.mean(raw_window)
            std = np.std(raw_window)
            if std < 1e-10:
                norm_window = np.zeros_like(raw_window).tolist()
            else:
                norm_window = ((raw_window - mu) / std).tolist()
                
            self._precomputed_windows.append(
                PriceWindow(
                    raw_prices=raw_window.tolist(),
                    returns=log_rets,
                    normalised_prices=norm_window
                )
            )
        # --- END OPTIMIZATION 2 BLOCK ---

        # Initialise portfolio
        self._portfolio = create_portfolio(initial_capital=config.initial_capital)
        self._portfolio = self._portfolio.update_peak(price=self._prices[0])
        self._prev_portfolio_snapshot = self._portfolio.to_snapshot(price=self._prices[0])
        
        # This line must be indented exactly like the lines above it!
        return self._build_observation(self._timestep)
    
    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Advance the environment by one timestep.

        Parameters
        ----------
        action : Validated Action from the agent.

        Returns
        -------
        StepResult with (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new one.")
        if self._prices is None or self._portfolio is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        config    = self._config
        timestep  = self._timestep
        prices    = self._prices

        prev_price   = prices[timestep]
        prev_snap    = self._prev_portfolio_snapshot

        # Advance to next price (the action is executed, then market moves)
        next_timestep = timestep + 1
        curr_price    = prices[next_timestep] if next_timestep < len(prices) else prices[-1]

        # Execute trade at current_price (before market moves)
        trade_value = self._compute_trade_value(action, prev_price)
        new_portfolio, trade_diag = self._portfolio.execute(
            action,
            price=prev_price,
            transaction_cost=config.transaction_cost,
            max_position_fraction=config.max_position_fraction,
        )

        # Update portfolio to current price (mark-to-market)
        new_portfolio = new_portfolio.update_peak(price=curr_price)
        curr_snap     = new_portfolio.to_snapshot(price=curr_price)

        # Compute reward
        reward = compute_reward(
            action=action,
            prev_snapshot=prev_snap,
            curr_snapshot=curr_snap,
            regime=config.regime,
            prev_price=prev_price,
            curr_price=curr_price,
            trade_value=trade_diag.get("trade_value", 0.0),
            initial_capital=config.initial_capital,
        )

        # Advance timestep
        self._timestep  = next_timestep
        self._portfolio = new_portfolio
        self._prev_portfolio_snapshot = curr_snap

        # Check done
        done = (self._timestep >= config.num_steps - 1)
        self._done = done

        # Record this step
        step_record = StepRecord(
            timestep=timestep,
            action=action,
            reward=reward,
            net_worth=curr_snap.net_worth,
            drawdown=curr_snap.drawdown,
            equity_fraction=curr_snap.equity_fraction,
            price=curr_price,
        )
        self._episode_steps.append(step_record)

        # Build next observation
        obs = self._build_observation(self._timestep)

        # Build info dict
        info: dict = {
            "timestep":        self._timestep,
            "price":           curr_price,
            "net_worth":       curr_snap.net_worth,
            "drawdown":        curr_snap.drawdown,
            "equity_fraction": curr_snap.equity_fraction,
            "total_trades":    curr_snap.total_trades,
            "trade_executed":  trade_diag.get("executed", False),
            "trade_reason":    trade_diag.get("reason", ""),
            "task_id":         self._task_id,
            "regime":          config.regime.value,
            "episode_score":   None,
        }

        # Grade episode at the end
        if done and self._episode_steps:
            grade = self._grade_episode()
            self._last_grade = grade
            info["episode_score"]    = grade.score
            info["episode_breakdown"] = grade.breakdown
            info["episode_rationale"] = grade.rationale
            info["episode_duration_s"] = time.time() - self._episode_start_time

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> State:
        """Return the current episode state (readable summary)."""
        if self._portfolio is None or self._prices is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        curr_price = self._prices[min(self._timestep, len(self._prices) - 1)]
        snap       = self._portfolio.to_snapshot(curr_price)

        trade_history = [
            Trade(
                timestep=s.timestep,
                decision=s.action.action_type.value,
                price=s.price,
                quantity=0.0,  # quantity not tracked at step level
                cash_after=0.0,
            )
            for s in self._episode_steps
            if s.action.action_type != ActionType.HOLD
        ]

        pnl_curve = [s.net_worth for s in self._episode_steps]

        return State(
            task_id=self._task_id or 0,
            current_timestep=self._timestep,
            portfolio_value=snap.net_worth,
            peak_value=snap.peak_net_worth,
            trade_history=trade_history,
            pnl_curve=pnl_curve,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, timestep: int) -> Observation:
        """Construct a fully-validated Observation at the given timestep."""
        config  = self._config
        prices  = self._prices
        n       = config.num_steps
        
        # Clamp to valid range
        t = min(timestep, n - 1)

        # Portfolio snapshot at current timestep
        curr_price = float(prices[t])
        snap = self._portfolio.to_snapshot(curr_price)

        # Time remaining
        time_left = max(0.0, min(1.0, 1.0 - t / n))

        return Observation(
            timestep=t,
            max_steps=n,
            regime=config.regime,
            window=self._precomputed_windows[t],  # <--- THE MAGIC O(1) LOOKUP
            portfolio=snap,
            time_left=time_left,
        )

    def _compute_trade_value(self, action: Action, price: float) -> float:
        """Estimate the gross dollar value of the trade (before execution)."""
        if action.action_type == ActionType.HOLD:
            return 0.0
        if action.action_type == ActionType.BUY:
            return self._portfolio.cash * action.fraction
        if action.action_type == ActionType.SELL:
            return self._portfolio.shares * price * action.fraction
        return 0.0

    def _grade_episode(self) -> GradeResult:
        """Run the appropriate grader on the completed episode history."""
        record = EpisodeRecord(
            regime=self._config.regime,
            initial_capital=self._config.initial_capital,
            steps=self._episode_steps,
        )
        return grade_episode(record)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def current_price(self) -> float:
        if self._prices is None:
            return 0.0
        t = min(self._timestep, len(self._prices) - 1)
        return float(self._prices[t])

    @property
    def last_grade(self) -> Optional[GradeResult]:
        return self._last_grade

    @property
    def task_description(self) -> str:
        if self._task_id is None:
            return "Not initialised"
        return _TASK_DESCRIPTIONS.get(self._task_id, "Unknown task")

    @property
    def regime_hint(self) -> str:
        if self._config is None:
            return "Not initialised"
        return _REGIME_HINTS.get(self._config.regime, "")


# ---------------------------------------------------------------------------
# Self-test (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("TradeSim Environment — 5-Step HOLD Test")
    print("=" * 65)

    env = TradeSimEnv()

    for task_id in [1, 2, 3]:
        print(f"\n{'─'*65}")
        print(f"Task {task_id}: {list(_TASK_REGIMES.values())[task_id-1].value.upper()}")
        print(f"{'─'*65}")

        obs = env.reset(task_id=task_id)

        print(f"  Initial price   : ${obs.portfolio.current_price:.2f}")
        print(f"  Initial cash    : ${obs.portfolio.cash:,.0f}")
        print(f"  Initial equity  : {obs.portfolio.equity_fraction:.2%}")
        print(f"  Time remaining  : {obs.time_left:.2%}")
        print(f"  Window size     : {len(obs.window.raw_prices)} bars")

        total_reward = 0.0
        for step in range(5):
            hold = Action.hold(reason="Test — observing market for 5 steps")
            result = env.step(hold)
            total_reward += result.reward.total

            print(
                f"  Step {step+1:2d}: "
                f"price=${result.observation.portfolio.current_price:.2f}  "
                f"nw=${result.observation.portfolio.net_worth:,.0f}  "
                f"reward={result.reward.total:+.5f}  "
                f"done={result.done}"
            )

        s = env.state()
        print(f"  Cumulative reward (5 steps): {total_reward:+.5f}")
        print(f"  Total trades: {s.current_timestep - 5} episodes processed + 5 steps")

    # Full episode with a buy-and-hold strategy
    print(f"\n{'─'*65}")
    print("Full Episode Test — Buy-and-Hold on Bull Market")
    print(f"{'─'*65}")

    obs = env.reset(task_id=1)
    env.step(Action.buy(fraction=0.90, reason="Buy early and ride the trend"))

    done = False
    while not done:
        result = env.step(Action.hold(reason="Riding trend — no action needed"))
        done = result.done

    grade = env.last_grade
    print(f"  Final net worth  : ${result.observation.portfolio.net_worth:,.2f}")
    print(f"  Total return     : {result.observation.portfolio.total_return:+.2%}")
    print(f"  Max drawdown     : {result.observation.portfolio.drawdown:.2%}")
    print(f"  Episode score    : {grade.score:.4f}")
    print(f"  Score breakdown  : {grade.breakdown}")

    assert grade.score > 0.30, f"Buy-and-hold bull score too low: {grade.score}"
    print(f"\n✓ Environment integration tests passed.")
    print(f"\n{'='*65}")
    print("TradeSim Environment is ready for agent interaction.")
    print(f"{'='*65}")
