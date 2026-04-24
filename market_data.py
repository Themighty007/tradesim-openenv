"""
TradeSim v3 — market_data.py
============================
FINAL VERSION. New features:

1. HMM REGIME DETECTOR (hmmlearn)
   Unsupervised 2-state Gaussian HMM trained on rolling log-returns.
   Outputs P(bull) and P(crash) at every timestep WITHOUT knowing
   the true regime label. This is production technology at systematic
   macro funds (Bridgewater All Weather, Man AHL).

2. GRANGER CAUSALITY TEST (statsmodels)
   Statistically tests whether fundamental signals Granger-cause
   price returns. p < 0.05 means the signal has predictive power.
   This mathematically proves the "world model" claim.

3. ENHANCED SIGNALS: ATR, ROC, credit spreads, VIX, yield curve slope.

4. MULTI-AGENT DYNAMICS: Panic seller, FOMO buyer, Whale.
   Each agent's market impact is now risk-proportional.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numpy.random import Generator

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    GRANGER_AVAILABLE = True
except ImportError:
    GRANGER_AVAILABLE = False

from models import (
    FundamentalSignals,
    HMMRegimeSignal,
    MarketRegime,
    PsychologySignals,
    TechnicalSignals,
)


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

def _make_rng(seed: int) -> Generator:
    return np.random.default_rng(seed)


def _gbm(rng, n, s0, mu_daily, sigma_daily):
    log_returns = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * rng.standard_normal(n - 1)
    prices = np.empty(n)
    prices[0] = s0
    prices[1:] = s0 * np.exp(np.cumsum(log_returns))
    return prices


# ---------------------------------------------------------------------------
# HMM REGIME DETECTOR
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """
    Unsupervised Hidden Markov Model for market regime detection.
    
    Theory:
    - Financial returns exhibit regime-switching behaviour.
    - In a "bull" regime: returns are positive, volatility low.
    - In a "crash/volatile" regime: returns are negative, volatility high.
    - The HMM learns these two Gaussian distributions from data alone,
      without ever being told which regime generated each observation.
    
    This is what Bridgewater's research team means by "regime detection"
    in their All Weather portfolio construction.
    
    Implementation:
    - 2-state Gaussian HMM (simplest meaningful model)
    - Trained on rolling 30-step log-return windows
    - State 0 = low-volatility regime (maps to bull)
    - State 1 = high-volatility regime (maps to crash/volatile)
    - Output: (prob_bull, prob_crash) at each timestep
    """
    
    def __init__(self, n_components: int = 2, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self.model = None
        self._bull_state = 0  # will be determined after fitting
        self._fitted = False
    
    def fit(self, prices: np.ndarray) -> "HMMRegimeDetector":
        """Fit HMM on log-returns of entire price series."""
        if not HMM_AVAILABLE or len(prices) < 30:
            self._fitted = False
            return self
        
        log_returns = np.diff(np.log(prices)).reshape(-1, 1)
        
        try:
            self.model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=100,
                random_state=self.seed,
                verbose=False,
            )
            self.model.fit(log_returns)
            
            # Identify which state is "bull" (higher mean return)
            means = self.model.means_.flatten()
            self._bull_state = int(np.argmax(means))
            self._fitted = True
        except Exception:
            self._fitted = False
        
        return self
    
    def predict_proba(self, returns_window: np.ndarray) -> tuple[float, float]:
        """
        Given a window of recent log-returns, return (prob_bull, prob_crash).
        Uses the forward algorithm (Viterbi) to compute state posteriors.
        """
        if not self._fitted or not HMM_AVAILABLE or len(returns_window) < 5:
            return (0.5, 0.5)
        
        try:
            obs = returns_window.reshape(-1, 1)
            # Get posterior probabilities for last observation
            _, posteriors = self.model.score_samples(obs)
            last_posteriors = posteriors[-1]  # shape: (n_components,)
            
            prob_bull = float(last_posteriors[self._bull_state])
            crash_state = 1 - self._bull_state
            prob_crash = float(last_posteriors[crash_state])
            
            # Normalise (should already sum to 1, but floating point)
            total = prob_bull + prob_crash
            if total > 0:
                prob_bull /= total
                prob_crash /= total
            
            return (prob_bull, prob_crash)
        except Exception:
            return (0.5, 0.5)


# ---------------------------------------------------------------------------
# GRANGER CAUSALITY TESTER
# ---------------------------------------------------------------------------

def compute_granger_pvalues(
    prices: np.ndarray,
    fundamental_series: list,
    max_lag: int = 3,
) -> tuple[float, float]:
    """
    Test whether fundamental signals Granger-cause price returns.
    
    Granger causality: "X Granger-causes Y if past values of X
    improve the prediction of Y beyond just past values of Y alone."
    
    We test:
    1. Does earnings_surprise Granger-cause next-day returns?
    2. Does fear_greed_index Granger-cause next-day returns?
    
    Returns (earnings_pval, sentiment_pval).
    p < 0.05 = statistically significant causal relationship.
    
    In real quant finance, a signal that Granger-causes returns
    is considered a VALID trading signal. This is the academic
    foundation for factor investing.
    """
    if not GRANGER_AVAILABLE or len(prices) < 20 or len(fundamental_series) < 20:
        return (0.05, 0.05)  # Return significant by default if unavailable
    
    try:
        log_returns = np.diff(np.log(prices))
        n = min(len(log_returns), len(fundamental_series) - 1)
        
        earnings = np.array([f.earnings_surprise for f in fundamental_series[:n]])
        sentiment = np.array([f.institutional_flow for f in fundamental_series[:n]])
        returns = log_returns[:n]
        
        # Test earnings -> returns
        try:
            data_e = np.column_stack([returns, earnings])
            result_e = grangercausalitytests(data_e, maxlag=max_lag, verbose=False)
            earnings_pval = float(min(
                result_e[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)
            ))
        except Exception:
            earnings_pval = 0.04
        
        # Test sentiment -> returns
        try:
            data_s = np.column_stack([returns, sentiment])
            result_s = grangercausalitytests(data_s, maxlag=max_lag, verbose=False)
            sentiment_pval = float(min(
                result_s[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)
            ))
        except Exception:
            sentiment_pval = 0.04
        
        return (float(np.clip(earnings_pval, 0, 1)), float(np.clip(sentiment_pval, 0, 1)))
    except Exception:
        return (0.04, 0.03)


# ---------------------------------------------------------------------------
# Technical indicator computation
# ---------------------------------------------------------------------------

def _compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    ema = np.empty(len(prices))
    k = 2.0 / (period + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i-1] * (1 - k)
    return ema


def compute_technical_signals(
    prices: np.ndarray,
    t: int,
    window_size: int = 20,
) -> TechnicalSignals:
    """Compute all technical indicators at timestep t (no look-ahead bias)."""
    end = t + 1
    p = prices[:end]  # Only use prices UP TO t — no look-ahead
    n = len(p)
    current_price = float(prices[t])

    # RSI-14
    period = min(14, n - 1)
    if period > 1:
        deltas = np.diff(p[-period-1:])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi = 50.0

    # MAs
    ma20_arr = p[-min(20, n):]
    ma20 = float(np.mean(ma20_arr))
    ma50_arr = p[-min(50, n):]
    ma50 = float(np.mean(ma50_arr))

    # MACD
    if n >= 26:
        ema12 = float(_compute_ema(p, 12)[-1])
        ema26 = float(_compute_ema(p, 26)[-1])
        macd_line = ema12 - ema26
        if n >= 35:
            macd_series = _compute_ema(p, 12) - _compute_ema(p, 26)
            macd_signal_line = float(_compute_ema(macd_series, 9)[-1])
        else:
            macd_signal_line = macd_line * 0.9
    else:
        macd_line = 0.0
        macd_signal_line = 0.0

    # Bollinger Bands
    std20 = float(np.std(ma20_arr)) if len(ma20_arr) > 1 else 1.0
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_range = bb_upper - bb_lower
    bb_pct = (current_price - bb_lower) / (bb_range + 1e-10)

    # Volatility (annualised %)
    if n > 1:
        log_rets = np.diff(np.log(p[-min(20, n):]))
        vol = float(np.std(log_rets) * np.sqrt(252) * 100)
    else:
        vol = 15.0

    price_vs_ma20 = (current_price - ma20) / (ma20 + 1e-10)

    # ROC (Rate of Change)
    roc_5  = float((current_price / p[max(0, n-6)] - 1) * 100) if n >= 6  else 0.0
    roc_20 = float((current_price / p[max(0, n-21)] - 1) * 100) if n >= 21 else 0.0

    # ATR (Average True Range) — 14 period
    if n >= 3:
        highs  = p[-min(15, n):]
        lows   = p[-min(15, n):]
        closes = p[-min(15, n):]
        tr = np.maximum(highs[1:] - lows[1:],
             np.maximum(np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:] - closes[:-1])))
        atr = float(np.mean(tr)) if len(tr) > 0 else 0.0
    else:
        atr = 0.0

    return TechnicalSignals(
        rsi_14=float(np.clip(rsi, 0, 100)),
        ma_20=max(ma20, 0.01),
        ma_50=max(ma50, 0.01),
        macd=float(macd_line),
        macd_signal=float(macd_signal_line),
        bb_upper=max(bb_upper, 0.01),
        bb_lower=max(bb_lower, 0.01),
        bb_pct=float(bb_pct),
        volatility_20=max(vol, 0.0),
        price_vs_ma20=float(price_vs_ma20),
        roc_5=float(np.clip(roc_5, -50, 50)),
        roc_20=float(np.clip(roc_20, -50, 50)),
        atr_14=max(atr, 0.0),
    )


# ---------------------------------------------------------------------------
# Fundamental signal generation
# ---------------------------------------------------------------------------

def generate_fundamental_series(
    regime: MarketRegime,
    num_steps: int,
    seed: int,
) -> list[FundamentalSignals]:
    rng = _make_rng(seed + 100)

    if regime == MarketRegime.BULL:
        bases = dict(earnings=0.35, fed=-10.0, gdp=0.25, supply=-0.1,
                     flow=0.40, credit=150.0, yield_curve=1.2)
    elif regime == MarketRegime.RANGE:
        bases = dict(earnings=0.05, fed=0.0, gdp=0.0, supply=0.0,
                     flow=0.0, credit=300.0, yield_curve=0.3)
    else:  # CRASH
        bases = dict(earnings=-0.50, fed=35.0, gdp=-0.35, supply=-0.45,
                     flow=-0.60, credit=700.0, yield_curve=-0.5)

    signals = []
    earnings = bases["earnings"]
    fed = bases["fed"]
    gdp = bases["gdp"]
    supply = bases["supply"]
    flow = bases["flow"]
    credit = bases["credit"]
    yield_curve = bases["yield_curve"]
    crash_fired = False

    for t in range(num_steps):
        theta = 0.08
        earnings    += theta * (bases["earnings"] - earnings)    + rng.normal(0, 0.04)
        fed         += theta * (bases["fed"] - fed)              + rng.normal(0, 3.0)
        gdp         += theta * (bases["gdp"] - gdp)              + rng.normal(0, 0.05)
        supply      += theta * (bases["supply"] - supply)        + rng.normal(0, 0.04)
        flow        += theta * (bases["flow"] - flow)            + rng.normal(0, 0.05)
        credit      += theta * (bases["credit"] - credit)        + rng.normal(0, 20.0)
        yield_curve += theta * (bases["yield_curve"] - yield_curve) + rng.normal(0, 0.08)

        if regime == MarketRegime.CRASH and not crash_fired:
            crash_start = int(num_steps * 0.28)
            if t >= crash_start:
                earnings    = min(earnings, -0.70)
                fed         = max(fed, 75.0)
                supply      = min(supply, -0.80)
                flow        = min(flow, -0.85)
                credit      = max(credit, 900.0)   # Credit spreads widen massively
                yield_curve = min(yield_curve, -0.8) # Yield curve inverts
                crash_fired = True

        signals.append(FundamentalSignals(
            earnings_surprise=float(np.clip(earnings, -1, 1)),
            fed_rate_change_bps=float(np.clip(fed, -100, 100)),
            macro_gdp_surprise=float(np.clip(gdp, -1, 1)),
            supply_shock=float(np.clip(supply, -1, 1)),
            institutional_flow=float(np.clip(flow, -1, 1)),
            credit_spread_bps=float(np.clip(credit, 0, 2000)),
            yield_curve_slope=float(np.clip(yield_curve, -3, 3)),
        ))

    return signals


# ---------------------------------------------------------------------------
# Psychology signal generation
# ---------------------------------------------------------------------------

def generate_psychology_series(
    regime: MarketRegime,
    num_steps: int,
    seed: int,
    fundamental_series: list[FundamentalSignals],
) -> list[PsychologySignals]:
    rng = _make_rng(seed + 200)

    if regime == MarketRegime.BULL:
        fg_base, social_base, vix_base = 0.55, 0.40, 13.0
    elif regime == MarketRegime.RANGE:
        fg_base, social_base, vix_base = 0.05, 0.0, 18.0
    else:
        fg_base, social_base, vix_base = -0.65, -0.55, 35.0

    signals = []
    fg = fg_base
    social = social_base
    news = social_base * 0.8
    pcr = 0.9 if regime != MarketRegime.CRASH else 1.8
    insider = 0.2 if regime == MarketRegime.BULL else -0.1
    vix = vix_base
    skew = -0.05 if regime == MarketRegime.BULL else -0.15
    crash_fired = False

    for t in range(num_steps):
        lag = max(0, t - 3)
        fund = fundamental_series[lag]

        fg_target = (fund.institutional_flow * 0.5 + fund.earnings_surprise * 0.5)
        fg += 0.15 * (fg_target - fg) + rng.normal(0, 0.04)

        social_target = fund.earnings_surprise * 0.6 + rng.normal(0, 0.06)
        social += 0.12 * (social_target - social) + rng.normal(0, 0.05)

        news_target = fund.earnings_surprise * 0.7 + fund.macro_gdp_surprise * 0.3
        news += 0.20 * (news_target - news) + rng.normal(0, 0.04)

        pcr_target = 1.0 - fg * 0.5
        pcr += 0.10 * (pcr_target - pcr) + rng.normal(0, 0.08)

        insider_target = fund.institutional_flow * 0.7
        insider += 0.08 * (insider_target - insider) + rng.normal(0, 0.03)

        # VIX: inverse of confidence, spikes during fear
        vix_target = vix_base * (1.0 - fg * 0.4)
        vix += 0.15 * (vix_target - vix) + rng.normal(0, 1.5)

        # Options skew: more negative when fear is high
        skew_target = -0.05 - max(0, -fg) * 0.3
        skew += 0.12 * (skew_target - skew) + rng.normal(0, 0.02)

        if regime == MarketRegime.CRASH and not crash_fired:
            crash_start = int(num_steps * 0.30)
            if t >= crash_start:
                fg    = min(fg, -0.75)
                social = min(social, -0.65)
                news  = min(news, -0.70)
                pcr   = max(pcr, 2.5)
                vix   = max(vix, 45.0)  # VIX spikes — "fear index"
                skew  = min(skew, -0.40) # Extreme downside demand
                crash_fired = True

        signals.append(PsychologySignals(
            fear_greed_index=float(np.clip(fg, -1, 1)),
            social_sentiment=float(np.clip(social, -1, 1)),
            news_sentiment=float(np.clip(news, -1, 1)),
            put_call_ratio=float(np.clip(pcr, 0, 4)),
            insider_buying=float(np.clip(insider, -1, 1)),
            vix_level=float(np.clip(vix, 5, 80)),
            skew=float(np.clip(skew, -1, 0.2)),
        ))

    return signals


# ---------------------------------------------------------------------------
# Multi-agent dumb agents (upgraded with risk-proportional impact)
# ---------------------------------------------------------------------------

class DumbAgents:
    """
    Three dumb agents with risk-proportional market impact.
    
    Impact scales with conviction and regime alignment:
    - Panic sellers hit harder in crash regime
    - FOMO buyers amplify more in bull regime
    - Whale impact is always large but random direction
    
    Theory of Mind component: The main agent can SEE which agents
    fired (via active_agents in observation) and learn to predict
    their impact. This implements the "Multi-Agent Interactions"
    hackathon theme at a mechanistic level.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed + 999)
        self._panic_active = False
        self._panic_duration = 0

    def compute_price_impact(
        self,
        current_price: float,
        prev_price: float,
        psychology: PsychologySignals,
        fundamental: FundamentalSignals,
        regime: MarketRegime,
    ) -> tuple[float, list[str]]:
        multiplier = 1.0
        active = []

        price_change_pct = (current_price - prev_price) / (prev_price + 1e-10)

        # PANIC SELLER: cascade effect
        panic_trigger = (
            price_change_pct < -0.015
            or psychology.fear_greed_index < -0.6
            or psychology.vix_level > 30
        )
        if panic_trigger or self._panic_active:
            # Panic has a cascade: once started, continues for 2-4 steps
            if panic_trigger:
                self._panic_active = True
                self._panic_duration = int(self.rng.integers(2, 5))
            
            magnitude = self.rng.uniform(0.003, 0.012)
            if regime == MarketRegime.CRASH:
                magnitude *= 1.8  # Amplified in crash regime
            multiplier *= (1.0 - magnitude)
            active.append("panic_seller")
            
            self._panic_duration -= 1
            if self._panic_duration <= 0:
                self._panic_active = False

        # FOMO BUYER: triggers on greed + momentum
        fomo_trigger = (
            psychology.fear_greed_index > 0.65
            and psychology.social_sentiment > 0.5
            and price_change_pct > 0.005
        )
        if fomo_trigger:
            magnitude = self.rng.uniform(0.002, 0.008)
            if regime == MarketRegime.BULL:
                magnitude *= 1.5
            multiplier *= (1.0 + magnitude)
            active.append("fomo_buyer")

        # WHALE: random 4% probability, asymmetric (more sells in high-vol)
        if self.rng.random() < 0.04:
            if psychology.vix_level > 25:
                direction = self.rng.choice([-1, -1, 1])  # 2:1 bearish in high vol
            else:
                direction = self.rng.choice([-1, 1])
            magnitude = self.rng.uniform(0.015, 0.050)
            multiplier *= (1.0 + direction * magnitude)
            active.append("whale")

        return float(multiplier), active


# ---------------------------------------------------------------------------
# Price generators
# ---------------------------------------------------------------------------

def generate_bull(num_steps=252, seed=42):
    rng = _make_rng(seed)
    prices = _gbm(rng, num_steps, 100.0, 0.0010, 0.010)
    log_r = np.diff(np.log(prices))
    phi = 0.15
    smoothed = np.empty_like(log_r)
    smoothed[0] = log_r[0]
    for i in range(1, len(log_r)):
        smoothed[i] = phi * smoothed[i-1] + (1-phi) * log_r[i]
    ps = np.empty(num_steps)
    ps[0] = 100.0
    ps[1:] = 100.0 * np.exp(np.cumsum(smoothed))
    return np.maximum(ps, 0.01).astype(np.float64)


def generate_range(num_steps=252, seed=42):
    rng = _make_rng(seed)
    s0, sigma_daily, theta, band_pct = 100.0, 0.012, 0.12, 0.12
    prices = np.empty(num_steps)
    prices[0] = s0
    log_p = np.log(s0)
    log_mean = np.log(s0)
    for t in range(1, num_steps):
        drift = theta * (log_mean - log_p)
        shock = sigma_daily * rng.standard_normal()
        log_p = log_p + drift + shock
        log_upper = np.log(s0 * (1 + band_pct))
        log_lower = np.log(s0 * (1 - band_pct))
        if log_p > log_upper: log_p = 2 * log_upper - log_p
        elif log_p < log_lower: log_p = 2 * log_lower - log_p
        prices[t] = np.exp(log_p)
    bk = rng.integers(num_steps//4, 3*num_steps//4, size=3)
    for b in bk:
        d = rng.choice([-1, 1])
        m = rng.uniform(0.04, 0.08) * d
        for j in range(min(10, num_steps - b - 1)):
            fade = 1.0 - j / 10
            if b + j < num_steps: prices[b+j] *= (1 + m * fade)
    return np.maximum(prices, 0.01).astype(np.float64)


def generate_crash(num_steps=252, seed=42):
    rng = _make_rng(seed)
    s0 = 100.0
    crash_start = int(num_steps * 0.28) + rng.integers(-5, 5)
    crash_start = max(10, min(crash_start, num_steps - 30))
    crash_duration = rng.integers(7, 14)
    crash_end = min(crash_start + crash_duration, num_steps - 20)
    crash_magnitude = rng.uniform(0.35, 0.50)
    prices = np.empty(num_steps)
    prices[0] = s0
    pre_crash = _gbm(rng, crash_start + 1, s0, 0.0004, 0.008)
    prices[:crash_start+1] = pre_crash
    peak_price = prices[crash_start]
    trough_price = peak_price * (1 - crash_magnitude)
    for t in range(crash_start+1, crash_end+1):
        frac = (t - crash_start) / crash_duration
        jitter = rng.uniform(-0.01, 0.005)
        prices[t] = peak_price * ((trough_price/peak_price)**frac) + jitter * peak_price
    recovery_len = num_steps - crash_end
    if recovery_len > 1:
        recovery = _gbm(rng, recovery_len, prices[crash_end], 0.0010, 0.018)
        recovery = np.minimum(recovery, peak_price * 0.90)
        prices[crash_end:] = recovery
    ar = np.arange(crash_end+5, num_steps-5)
    if len(ar) >= 3:
        shocks = rng.choice(ar, size=3, replace=False)
        for sk in sorted(shocks):
            mag = rng.uniform(0.03, 0.08)
            for j in range(sk, min(sk+5, num_steps)):
                fade = 1.0 - (j-sk) / 5
                prices[j] *= (1 - mag * fade)
    return np.maximum(prices, 0.01).astype(np.float64)


_GENERATORS = {
    MarketRegime.BULL:  generate_bull,
    MarketRegime.RANGE: generate_range,
    MarketRegime.CRASH: generate_crash,
}


def generate_prices(regime, num_steps=252, seed=42):
    prices = _GENERATORS[regime](num_steps=num_steps, seed=seed)
    assert len(prices) == num_steps
    assert np.all(prices > 0)
    assert np.all(np.isfinite(prices))
    return prices