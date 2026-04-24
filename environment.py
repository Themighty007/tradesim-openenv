"""
TradeSim v3 — environment.py
============================
FINAL VERSION. Integrates:
  - HMM regime detector (trained at episode start, queries each step)
  - Granger causality test (run once at episode start, stored in observations)
  - Regime alignment bonus in reward
  - Full 4-axis observation space
  - Cascade panic seller with multi-step memory
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from graders import grade_episode
from market_data import (
    DumbAgents,
    HMMRegimeDetector,
    compute_granger_pvalues,
    compute_technical_signals,
    generate_fundamental_series,
    generate_prices,
    generate_psychology_series,
)
from models import (
    Action,
    ActionType,
    EnvironmentConfig,
    EpisodeRecord,
    GradeResult,
    HMMRegimeSignal,
    MarketRegime,
    Observation,
    PortfolioSnapshot,
    PriceWindow,
    RewardBreakdown,
    State,
    StepRecord,
    StepResult,
    Trade,
)
from portfolio import PortfolioState, create_portfolio
from reward import compute_reward


_TASK_REGIMES: dict[int, MarketRegime] = {
    1: MarketRegime.BULL,
    2: MarketRegime.RANGE,
    3: MarketRegime.CRASH,
}

_TASK_DESCRIPTIONS: dict[int, str] = {
    1: (
        "TASK 1 — BULL MARKET\n"
        "Fed is dovish, earnings are beating, institutional flow is positive. "
        "RSI trends above 50, MACD positive, fear/greed elevated. "
        "HMM will detect the bull regime. Credit spreads are tight. "
        "Strategy: enter early, size up, ride the trend, watch for greed extremes."
    ),
    2: (
        "TASK 2 — CHOPPY RANGE\n"
        "Mixed fundamentals, neutral sentiment. RSI oscillates 30-70. "
        "HMM will switch between states frequently. BB breakouts snap back. "
        "Strategy: fade Bollinger extremes, use RSI for timing, "
        "preserve capital — every trade costs 0.1%."
    ),
    3: (
        "TASK 3 — FLASH CRASH (SURVIVAL)\n"
        "A macro shock is coming in the first third. "
        "EARLY WARNING SIGNALS: earnings_surprise will turn negative, "
        "fed_rate_change_bps will spike positive, credit_spread_bps will widen, "
        "yield_curve will invert, VIX will rise, HMM will shift to crash state. "
        "Exit BEFORE the price falls. Preserve capital. Re-enter on recovery."
    ),
}

_REGIME_HINTS = {
    MarketRegime.BULL:  "Bull trend. Watch earnings_surprise, institutional_flow. HMM bull probability rising.",
    MarketRegime.RANGE: "Sideways. Use RSI < 30 to BUY, RSI > 70 to SELL. HMM state uncertain.",
    MarketRegime.CRASH: "HIGH RISK. Monitor credit_spread_bps, yield_curve, VIX. HMM crash probability rising.",
}


class TradeSimEnv:

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self._base_config = config
        self._reset_internals()

    def _reset_internals(self):
        self._prices: Optional[np.ndarray] = None
        self._precomputed_windows: list = []
        self._fundamental_series: list = []
        self._psychology_series: list  = []
        self._hmm_signals: list        = []
        self._portfolio: Optional[PortfolioState] = None
        self._config: Optional[EnvironmentConfig] = None
        self._timestep: int   = 0
        self._done: bool      = False
        self._episode_steps: list[StepRecord] = []
        self._prev_portfolio_snapshot: Optional[PortfolioSnapshot] = None
        self._task_id: Optional[int]    = None
        self._episode_start_time: float = 0.0
        self._last_grade: Optional[GradeResult] = None
        self._dumb_agents: Optional[DumbAgents] = None
        self._hmm_detector: Optional[HMMRegimeDetector] = None
        self._price_multiplier_buffer: float = 1.0
        self._granger_pvals: tuple = (1.0, 1.0)
        self._reward_history: list[float] = []

    def reset(self, task_id: int = 1) -> Observation:
        if task_id not in _TASK_REGIMES:
            raise ValueError(f"task_id must be 1, 2, or 3; got {task_id}")

        regime = _TASK_REGIMES[task_id]

        # CRITICAL: always use the task's regime, never override
        if self._base_config is not None:
            config = EnvironmentConfig(
                regime=regime,
                num_steps=self._base_config.num_steps,
                initial_capital=self._base_config.initial_capital,
                window_size=self._base_config.window_size,
                seed=self._base_config.seed,
                transaction_cost=self._base_config.transaction_cost,
                max_position_fraction=self._base_config.max_position_fraction,
                multi_agent_mode=self._base_config.multi_agent_mode,
                hmm_enabled=self._base_config.hmm_enabled,
            )
        else:
            config = EnvironmentConfig(regime=regime)

        self._config              = config
        self._task_id             = task_id
        self._timestep            = 0
        self._done                = False
        self._episode_steps       = []
        self._last_grade          = None
        self._episode_start_time  = time.time()
        self._price_multiplier_buffer = 1.0
        self._reward_history      = []

        # Generate price series
        self._prices = generate_prices(regime=regime, num_steps=config.num_steps, seed=config.seed)

        # Generate all signal axes for entire episode
        self._fundamental_series = generate_fundamental_series(
            regime=regime, num_steps=config.num_steps, seed=config.seed)
        self._psychology_series = generate_psychology_series(
            regime=regime, num_steps=config.num_steps, seed=config.seed,
            fundamental_series=self._fundamental_series)

        # FIT HMM on the full price series (training = fitting on episode prices)
        if config.hmm_enabled:
            self._hmm_detector = HMMRegimeDetector(n_components=2, seed=config.seed)
            self._hmm_detector.fit(self._prices)
        else:
            self._hmm_detector = None

        # Pre-compute HMM signals for entire episode (O(n) total, O(1) per step)
        self._hmm_signals = self._precompute_hmm_signals(config.num_steps)

        # Granger causality test (run once per episode)
        self._granger_pvals = compute_granger_pvalues(
            self._prices, self._fundamental_series, max_lag=3)

        # Dumb agents
        self._dumb_agents = DumbAgents(seed=config.seed) if config.multi_agent_mode else None

        # Precompute price windows
        self._precomputed_windows = []
        prices = self._prices
        n, ws  = config.num_steps, config.window_size

        for t in range(n):
            window_start = max(0, t - ws + 1)
            actual_ws   = t - window_start + 1
            raw_window  = prices[window_start:window_start + actual_ws]
            pad_needed  = ws - actual_ws
            if pad_needed > 0:
                raw_window = np.concatenate((np.full(pad_needed, prices[0]), raw_window))
            log_rets   = np.diff(np.log(raw_window)).tolist()
            mu, std    = np.mean(raw_window), np.std(raw_window)
            norm_window = ((raw_window - mu) / std).tolist() if std > 1e-10 else [0.0]*len(raw_window)
            self._precomputed_windows.append(
                PriceWindow(raw_prices=raw_window.tolist(), returns=log_rets, normalised_prices=norm_window)
            )

        # Portfolio
        self._portfolio = create_portfolio(initial_capital=config.initial_capital)
        self._portfolio = self._portfolio.update_peak(price=self._prices[0])
        self._prev_portfolio_snapshot = self._portfolio.to_snapshot(price=self._prices[0])

        return self._build_observation(self._timestep, active_agents=[])

    def _precompute_hmm_signals(self, num_steps: int) -> list[HMMRegimeSignal]:
        """Pre-compute HMM probabilities for all timesteps."""
        signals = []
        granger_e, granger_s = self._granger_pvals

        for t in range(num_steps):
            if self._hmm_detector is not None and t >= 10:
                # Use rolling 20-step window of log-returns
                end = t + 1
                start = max(0, end - 20)
                log_returns = np.diff(np.log(self._prices[start:end]))
                if len(log_returns) >= 5:
                    prob_bull, prob_crash = self._hmm_detector.predict_proba(log_returns)
                else:
                    prob_bull, prob_crash = 0.5, 0.5
            else:
                prob_bull, prob_crash = 0.5, 0.5

            current_state = 0 if prob_bull >= prob_crash else 1
            confidence    = max(prob_bull, prob_crash)

            signals.append(HMMRegimeSignal(
                prob_bull=float(prob_bull),
                prob_crash=float(prob_crash),
                current_state=current_state,
                state_confidence=float(confidence),
                granger_earnings_pval=float(granger_e),
                granger_sentiment_pval=float(granger_s),
            ))
        return signals

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode finished. Call reset() first.")
        if self._prices is None or self._portfolio is None:
            raise RuntimeError("Call reset() first.")

        config   = self._config
        timestep = self._timestep
        prices   = self._prices

        prev_price = prices[timestep]
        prev_snap  = self._prev_portfolio_snapshot

        next_timestep = timestep + 1
        raw_next_price = prices[next_timestep] if next_timestep < len(prices) else prices[-1]
        curr_price     = max(raw_next_price * self._price_multiplier_buffer, 0.01)

        # Execute trade
        new_portfolio, trade_diag = self._portfolio.execute(
            action, price=prev_price,
            transaction_cost=config.transaction_cost,
            max_position_fraction=config.max_position_fraction,
        )
        new_portfolio = new_portfolio.update_peak(price=curr_price)
        curr_snap     = new_portfolio.to_snapshot(price=curr_price)

        # Compute dumb-agent impact for NEXT step
        active_agents  = []
        next_multiplier = 1.0
        t_idx = min(next_timestep, len(self._fundamental_series)-1)
        if self._dumb_agents is not None and next_timestep < len(prices) - 1:
            psych = self._psychology_series[min(next_timestep, len(self._psychology_series)-1)]
            fund  = self._fundamental_series[t_idx]
            next_multiplier, active_agents = self._dumb_agents.compute_price_impact(
                current_price=curr_price, prev_price=prev_price,
                psychology=psych, fundamental=fund, regime=config.regime,
            )
        self._price_multiplier_buffer = next_multiplier

        # Get all signal snapshots for THIS timestep
        tech_signals  = compute_technical_signals(prices, timestep, config.window_size)
        fund_signals  = self._fundamental_series[min(timestep, len(self._fundamental_series)-1)]
        psych_signals = self._psychology_series[min(timestep, len(self._psychology_series)-1)]
        hmm_signals   = self._hmm_signals[min(timestep, len(self._hmm_signals)-1)]

        # Reward with regime alignment bonus
        reward = compute_reward(
            action=action,
            prev_snapshot=prev_snap,
            curr_snapshot=curr_snap,
            regime=config.regime,
            prev_price=prev_price,
            curr_price=curr_price,
            trade_value=trade_diag.get("trade_value", 0.0),
            initial_capital=config.initial_capital,
            hmm_signal=hmm_signals,
        )

        self._reward_history.append(reward.total)
        self._timestep  = next_timestep
        self._portfolio = new_portfolio
        self._prev_portfolio_snapshot = curr_snap

        done = (self._timestep >= config.num_steps - 1)
        self._done = done

        step_record = StepRecord(
            timestep=timestep,
            action=action,
            reward=reward,
            net_worth=curr_snap.net_worth,
            drawdown=curr_snap.drawdown,
            equity_fraction=curr_snap.equity_fraction,
            price=curr_price,
            technical=tech_signals,
            fundamental=fund_signals,
            psychology=psych_signals,
            hmm=hmm_signals,
        )
        self._episode_steps.append(step_record)

        obs = self._build_observation(self._timestep, active_agents=active_agents)

        info: dict = {
            "timestep":         self._timestep,
            "price":            curr_price,
            "net_worth":        curr_snap.net_worth,
            "drawdown":         curr_snap.drawdown,
            "equity_fraction":  curr_snap.equity_fraction,
            "total_trades":     curr_snap.total_trades,
            "trade_executed":   trade_diag.get("executed", False),
            "task_id":          self._task_id,
            "regime":           config.regime.value,
            "active_agents":    active_agents,
            "hmm_bull_prob":    hmm_signals.prob_bull,
            "hmm_crash_prob":   hmm_signals.prob_crash,
            "granger_earnings": hmm_signals.granger_earnings_pval,
            "granger_sentiment": hmm_signals.granger_sentiment_pval,
            "episode_score":    None,
            "reward_history":   list(self._reward_history),
            "technical":        tech_signals.model_dump(),
            "fundamental":      fund_signals.model_dump(),
            "psychology":       psych_signals.model_dump(),
        }

        if done and self._episode_steps:
            grade = self._grade_episode()
            self._last_grade = grade
            info["episode_score"]          = grade.score
            info["episode_breakdown"]      = grade.breakdown
            info["episode_rationale"]      = grade.rationale
            info["sharpe_ratio"]           = grade.sharpe_ratio
            info["calmar_ratio"]           = grade.calmar_ratio
            info["technical_score"]        = grade.technical_score
            info["fundamental_score"]      = grade.fundamental_score
            info["psychological_score"]    = grade.psychological_score
            info["hmm_alignment_score"]    = grade.hmm_alignment_score
            info["total_return_pct"]       = grade.total_return_pct
            info["episode_duration_s"]     = time.time() - self._episode_start_time

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> State:
        if self._portfolio is None or self._prices is None:
            raise RuntimeError("Call reset() first.")
        curr_price = self._prices[min(self._timestep, len(self._prices)-1)]
        snap = self._portfolio.to_snapshot(curr_price)
        trade_history = [
            Trade(timestep=s.timestep, decision=s.action.action_type.value,
                  price=s.price, quantity=0.0, cash_after=0.0)
            for s in self._episode_steps if s.action.action_type != ActionType.HOLD
        ]
        # Sharpe so far (rolling)
        sharpe_so_far = 0.0
        if len(self._reward_history) > 5:
            import math as _math
            r = np.array(self._reward_history)
            std = np.std(r)
            if std > 1e-10:
                sharpe_so_far = float(np.mean(r) / std * _math.sqrt(252))

        return State(
            task_id=self._task_id or 0,
            current_timestep=self._timestep,
            portfolio_value=snap.net_worth,
            peak_value=snap.peak_net_worth,
            trade_history=trade_history,
            pnl_curve=[s.net_worth for s in self._episode_steps],
            sharpe_so_far=sharpe_so_far,
        )

    def _build_observation(self, timestep: int, active_agents: list[str]) -> Observation:
        config = self._config
        prices = self._prices
        n      = config.num_steps
        t      = min(timestep, n - 1)
        curr_price = float(prices[t])
        snap       = self._portfolio.to_snapshot(curr_price)
        time_left  = max(0.0, min(1.0, 1.0 - t / n))
        t_idx      = min(t, len(self._fundamental_series)-1)

        return Observation(
            timestep=t,
            max_steps=n,
            regime=config.regime,
            window=self._precomputed_windows[t],
            portfolio=snap,
            time_left=time_left,
            technical=compute_technical_signals(prices, t, config.window_size),
            fundamental=self._fundamental_series[t_idx],
            psychology=self._psychology_series[t_idx],
            hmm=self._hmm_signals[t_idx],
            active_agents=active_agents,
        )

    def _compute_trade_value(self, action: Action, price: float) -> float:
        if action.action_type == ActionType.HOLD:  return 0.0
        if action.action_type == ActionType.BUY:   return self._portfolio.cash * action.fraction
        if action.action_type == ActionType.SELL:  return self._portfolio.shares * price * action.fraction
        return 0.0

    def _grade_episode(self) -> GradeResult:
        record = EpisodeRecord(
            regime=self._config.regime,
            initial_capital=self._config.initial_capital,
            steps=self._episode_steps,
        )
        return grade_episode(record)

    @property
    def is_done(self):      return self._done
    @property
    def current_price(self):
        if self._prices is None: return 0.0
        return float(self._prices[min(self._timestep, len(self._prices)-1)])
    @property
    def last_grade(self):   return self._last_grade
    @property
    def task_description(self): return _TASK_DESCRIPTIONS.get(self._task_id, "Not initialised")
    @property
    def regime_hint(self):
        if self._config is None: return "Not initialised"
        return _REGIME_HINTS.get(self._config.regime, "")
    @property
    def reward_history(self): return list(self._reward_history)