"""
TradeSim — market_data.py
=========================
Generates fake-but-realistic stock price series using geometric Brownian motion
(GBM) for the baseline, with hand-tuned overlays to produce the three canonical
market regimes.

Design principles:
  - Every generator is a *pure function* of (num_steps, seed) → np.ndarray of prices.
  - Parameters are calibrated to realistic US equity statistics:
      • Daily σ ≈ 1-2 %  (annualised ~16-32 %)
      • Daily drift μ ≈ 0.03–0.05 % in bull (≈ 8–12 % annualised)
  - Prices are always strictly positive (no log-domain escapes).
  - All three generators share the same public signature so callers are interchangeable.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from models import MarketRegime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_rng(seed: int) -> Generator:
    return np.random.default_rng(seed)


def _gbm(
    rng: Generator,
    n: int,
    s0: float,
    mu_daily: float,
    sigma_daily: float,
) -> np.ndarray:
    """
    Geometric Brownian Motion discretised in daily steps (Euler–Maruyama).

    S_{t+1} = S_t * exp((μ - σ²/2)*dt + σ*W_t)   where dt=1 (day)
    """
    log_returns = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * rng.standard_normal(n - 1)
    prices = np.empty(n)
    prices[0] = s0
    prices[1:] = s0 * np.exp(np.cumsum(log_returns))
    return prices


# ---------------------------------------------------------------------------
# Regime 1 — Bull Market
# ---------------------------------------------------------------------------

def generate_bull(num_steps: int = 252, seed: int = 42) -> np.ndarray:
    """
    Steady bull market.

    Characteristics:
      • μ ≈ +0.06 % / day  (≈ +16 % annualised)
      • σ ≈ 1.0 % / day    (≈ 16 % annualised vol)
      • Gentle upward channel with no structural breaks
      • Minor momentum overlay: prices trend with slight autocorrelation

    A well-behaved agent should buy early and hold, scoring 0.65–0.80.
    """
    rng = _make_rng(seed)

    mu_daily    = 0.0010   # 0.10 % per day → ~28% annualised (clear bull)
    sigma_daily = 0.010    # 1.0 % per day
    s0          = 100.0

    prices = _gbm(rng, num_steps, s0, mu_daily, sigma_daily)

    # Light momentum overlay (AR(1) on log-returns, φ=0.15)
    log_r = np.diff(np.log(prices))
    phi = 0.15
    smoothed = np.empty_like(log_r)
    smoothed[0] = log_r[0]
    for i in range(1, len(log_r)):
        smoothed[i] = phi * smoothed[i - 1] + (1 - phi) * log_r[i]

    prices_smoothed = np.empty(num_steps)
    prices_smoothed[0] = s0
    prices_smoothed[1:] = s0 * np.exp(np.cumsum(smoothed))

    # Guarantee strictly positive
    prices_smoothed = np.maximum(prices_smoothed, 0.01)
    return prices_smoothed.astype(np.float64)


# ---------------------------------------------------------------------------
# Regime 2 — Choppy Range
# ---------------------------------------------------------------------------

def generate_range(num_steps: int = 252, seed: int = 42) -> np.ndarray:
    """
    Mean-reverting choppy market.

    Characteristics:
      • Net drift ≈ 0 (zero expected return)
      • σ ≈ 1.5 % / day  (higher intraday noise)
      • Prices oscillate within a ±15 % band around S0
      • Ornstein–Uhlenbeck mean-reversion overlay (θ=0.08)
      • Occasional false breakouts that snap back

    Trend-following strategies lose money; mean-reversion earns slightly.
    Expected agent score: 0.35–0.55.
    """
    rng = _make_rng(seed)

    s0          = 100.0
    sigma_daily = 0.012    # 1.2 % per day
    theta       = 0.12     # stronger mean-reversion
    band_pct    = 0.12     # ±12 % corridor

    prices = np.empty(num_steps)
    prices[0] = s0

    log_p = np.log(s0)
    log_mean = np.log(s0)

    for t in range(1, num_steps):
        # OU drift pulls toward log_mean
        drift = theta * (log_mean - log_p)
        shock = sigma_daily * rng.standard_normal()
        log_p = log_p + drift + shock

        # Hard reflecting boundaries to stay in band
        log_upper = np.log(s0 * (1 + band_pct))
        log_lower = np.log(s0 * (1 - band_pct))
        if log_p > log_upper:
            log_p = 2 * log_upper - log_p
        elif log_p < log_lower:
            log_p = 2 * log_lower - log_p

        prices[t] = np.exp(log_p)

    # Inject 3 false breakouts (spikes that quickly revert)
    breakout_steps = rng.integers(num_steps // 4, 3 * num_steps // 4, size=3)
    for bk in breakout_steps:
        direction = rng.choice([-1, 1])
        spike_mag = rng.uniform(0.04, 0.08) * direction
        revert_window = min(10, num_steps - bk - 1)
        for j in range(revert_window):
            fade = 1.0 - j / revert_window
            if bk + j < num_steps:
                prices[bk + j] *= (1 + spike_mag * fade)

    prices = np.maximum(prices, 0.01)
    return prices.astype(np.float64)


# ---------------------------------------------------------------------------
# Regime 3 — Flash Crash
# ---------------------------------------------------------------------------

def generate_crash(num_steps: int = 252, seed: int = 42) -> np.ndarray:
    """
    Flash crash with slow recovery.

    Phases:
      1. Pre-crash calm       (steps 0 – crash_start-1):  μ=+0.04%, σ=0.8%
      2. Crash cliff          (steps crash_start – crash_end):  sharp exponential fall
      3. Post-crash recovery  (steps crash_end+1 – end):  slow GBM with positive drift

    Structural parameters:
      • Crash starts at ~30 % into the episode
      • Crash magnitude: –35 % to –50 % over ~10 days (exponential decay)
      • Recovery: μ = +0.10 % / day, σ = 1.8 % (elevated vol persists)
      • Full recovery is NOT guaranteed (prices may end 10-20 % below peak)

    Survival (exit before crash) is the primary success criterion.
    Expected agent score: 0.15–0.35.
    """
    rng = _make_rng(seed)

    s0              = 100.0
    crash_start     = int(num_steps * 0.28) + rng.integers(-5, 5)
    crash_start     = max(10, min(crash_start, num_steps - 30))
    crash_duration  = rng.integers(7, 14)                     # 7-13 day cliff
    crash_end       = min(crash_start + crash_duration, num_steps - 20)
    crash_magnitude = rng.uniform(0.35, 0.50)                 # 35–50 % drop

    prices = np.empty(num_steps)
    prices[0] = s0

    # Phase 1: Pre-crash calm bull
    pre_crash = _gbm(rng, crash_start + 1, s0, mu_daily=0.0004, sigma_daily=0.008)
    prices[:crash_start + 1] = pre_crash

    # Phase 2: Exponential cliff
    peak_price = prices[crash_start]
    trough_price = peak_price * (1 - crash_magnitude)
    for t in range(crash_start + 1, crash_end + 1):
        frac = (t - crash_start) / crash_duration
        # Exponential decay + jitter (panic selling is lumpy)
        jitter = rng.uniform(-0.01, 0.005)
        prices[t] = peak_price * ((trough_price / peak_price) ** frac) + jitter * peak_price

    # Phase 3: Slow, volatile recovery
    recovery_len = num_steps - crash_end
    if recovery_len > 1:
        # Elevated vol after crash; recovery drift boosted but does NOT fully heal
        recovery = _gbm(rng, recovery_len, prices[crash_end],
                         mu_daily=0.0010, sigma_daily=0.018)
        # Cap recovery so price doesn't exceed pre-crash peak
        recovery = np.minimum(recovery, peak_price * 0.90)
        prices[crash_end:] = recovery

    # Add micro-aftershocks in recovery (3 smaller drops)
    aftershock_region = np.arange(crash_end + 5, num_steps - 5)
    if len(aftershock_region) >= 3:
        shocks = rng.choice(aftershock_region, size=3, replace=False)
        for sk in sorted(shocks):
            mag = rng.uniform(0.03, 0.08)
            end_sk = min(sk + 5, num_steps)
            for j in range(sk, end_sk):
                fade = 1.0 - (j - sk) / (end_sk - sk)
                prices[j] *= (1 - mag * fade)

    prices = np.maximum(prices, 0.01)
    return prices.astype(np.float64)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

_GENERATORS = {
    MarketRegime.BULL:  generate_bull,
    MarketRegime.RANGE: generate_range,
    MarketRegime.CRASH: generate_crash,
}


def generate_prices(
    regime: MarketRegime,
    num_steps: int = 252,
    seed: int = 42,
) -> np.ndarray:
    """
    Public API — generate a price series for the requested regime.

    Parameters
    ----------
    regime    : MarketRegime enum value
    num_steps : Number of price points (including t=0)
    seed      : RNG seed for deterministic reproducibility

    Returns
    -------
    np.ndarray of shape (num_steps,), dtype float64, all values > 0.
    """
    generator_fn = _GENERATORS[regime]
    prices = generator_fn(num_steps=num_steps, seed=seed)

    # Final sanity checks
    assert len(prices) == num_steps,   f"Length mismatch: {len(prices)} != {num_steps}"
    assert np.all(prices > 0),         "Prices must be strictly positive"
    assert np.all(np.isfinite(prices)), "Prices must be finite"

    return prices


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns: r_t = log(P_t / P_{t-1}), length = len(prices) - 1."""
    return np.diff(np.log(prices))


def compute_rolling_features(
    prices: np.ndarray,
    window_start: int,
    window_size: int,
) -> tuple[list[float], list[float], list[float]]:
    """
    Extract (raw_prices, log_returns, z-normalised prices) for a window.

    Parameters
    ----------
    prices       : Full price array
    window_start : First index of the window (inclusive)
    window_size  : Length of the window

    Returns
    -------
    (raw_prices, returns, normalised_prices) — all as Python lists.
    """
    window_end   = window_start + window_size
    window_prices = prices[window_start:window_end].tolist()

    # Log returns for the window
    raw = np.array(window_prices)
    log_rets = np.diff(np.log(raw)).tolist() if len(raw) > 1 else []

    # Z-score normalisation
    mu  = float(np.mean(raw))
    std = float(np.std(raw))
    if std < 1e-10:
        normalised = [0.0] * len(window_prices)
    else:
        normalised = ((raw - mu) / std).tolist()

    return window_prices, log_rets, normalised


# ---------------------------------------------------------------------------
# Quick visual verification (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    N = 252
    configs = [
        (MarketRegime.BULL,  "Task 1 — Bull Market",   "#2ecc71"),
        (MarketRegime.RANGE, "Task 2 — Choppy Range",  "#f39c12"),
        (MarketRegime.CRASH, "Task 3 — Flash Crash",   "#e74c3c"),
    ]

    print("=" * 60)
    print("TradeSim — Market Data Verification")
    print("=" * 60)

    all_prices = {}
    for regime, title, color in configs:
        prices = generate_prices(regime, num_steps=N, seed=42)
        all_prices[regime] = prices
        log_rets = compute_log_returns(prices)
        print(f"\n{title}")
        print(f"  Start price : ${prices[0]:.2f}")
        print(f"  End price   : ${prices[-1]:.2f}")
        print(f"  Min price   : ${prices.min():.2f}")
        print(f"  Max price   : ${prices.max():.2f}")
        print(f"  Total return: {(prices[-1]/prices[0]-1)*100:+.1f}%")
        print(f"  Daily σ     : {log_rets.std()*100:.2f}%")
        print(f"  Skewness    : {float(np.mean(((log_rets-log_rets.mean())/log_rets.std())**3)):.3f}")

    if HAS_MPL:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        for ax, (regime, title, color) in zip(axes, configs):
            prices = all_prices[regime]
            ax.plot(prices, color=color, linewidth=1.5)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel("Price ($)")
            ax.set_xlabel("Day")
            ax.grid(alpha=0.3)
            ax.axhline(prices[0], color="gray", linestyle="--", alpha=0.5, label="Start")
        plt.tight_layout()
        plt.savefig("market_curves.png", dpi=120)
        print("\n✓ Plot saved to market_curves.png")
    else:
        print("\n[matplotlib not installed — skipping plot]")

    print("\n✓ All price series verified: strictly positive, finite, correct length.")
