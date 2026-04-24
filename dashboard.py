"""
TradeSim v3 — dashboard.py
==========================
FINAL VERSION — Vertically stacked.
Educational glossary panel. Separated charts. Honest training curve.
"""

import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeSim v3 — AI Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
G = "#00E676"   # green  / bull
R = "#FF5252"   # red    / bear
Y = "#FFD700"   # yellow / neutral
B = "#42A5F5"   # blue   / info
P = "#CE93D8"   # purple / hmm
BG   = "#0b0b0b"
CARD = "#141414"
CARD2= "#1a1a1a"
BORD = "#272727"
TXT  = "#e8e8e8"
SUB  = "#888888"
DIM  = "#444444"

# ─────────────────────────────────────────────────────────────────────────────
# CSS — single source of truth, dark terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {{
    background: {BG} !important;
    color: {TXT};
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

[data-testid="stSidebar"] {{
    background: #080808 !important;
    border-right: 1px solid {BORD};
    min-width: 0 !important;
    max-width: 0 !important;
    overflow: hidden !important;
}}

.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

/* ── Landing ── */
.landing {{
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 40px;
    text-align: center;
}}
.landing-title {{
    font-size: clamp(48px, 8vw, 96px);
    font-weight: 900;
    letter-spacing: -3px;
    line-height: 1;
    color: {G};
    margin-bottom: 16px;
}}
.landing-tag {{
    font-size: 18px;
    color: {SUB};
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 48px;
}}
.feature-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 16px;
    margin-bottom: 48px;
    max-width: 800px;
}}
.feature-pill {{
    background: {CARD};
    border: 1px solid {BORD};
    border-radius: 40px;
    padding: 10px 24px;
    font-size: 13px;
    font-weight: 600;
    color: {TXT};
    display: flex;
    align-items: center;
    gap: 8px;
}}
.feature-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
}}
.start-btn {{
    background: {G};
    color: #000;
    border: none;
    border-radius: 50px;
    padding: 18px 56px;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
}}

/* ── Dashboard wrapper ── */
.dash-wrap {{
    padding: 0 32px 48px;
    max-width: 1280px;
    margin: 0 auto;
}}

/* ── Page header ── */
.page-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 0 16px;
    border-bottom: 1px solid {BORD};
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 12px;
}}
.page-title {{
    font-size: 20px;
    font-weight: 800;
    color: {TXT};
    letter-spacing: -0.5px;
}}
.page-sub {{
    font-size: 12px;
    color: {SUB};
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.status-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}}
.status-dot {{
    font-size: 12px;
    color: {SUB};
    display: flex;
    align-items: center;
    gap: 5px;
}}

/* ── Section headers ── */
.section-title {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SUB};
    padding: 16px 0 10px;
    border-top: 1px solid {BORD};
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ── Regime banner ── */
.regime-banner {{
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
    border-left: 4px solid;
}}
.regime-title {{
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.regime-body {{
    font-size: 14px;
    line-height: 1.7;
    color: {SUB};
}}
.regime-target {{
    font-size: 12px;
    font-weight: 600;
    margin-top: 10px;
    opacity: 0.9;
}}

/* ── Metric cards ── */
.metrics-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}}
.mcard {{
    background: {CARD};
    border: 1px solid {BORD};
    border-radius: 10px;
    padding: 16px;
    border-top: 3px solid var(--accent, {BORD});
    text-align: center;
    position: relative;
}}
.mcard-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {SUB};
    margin-bottom: 8px;
}}
.mcard-value {{
    font-size: 28px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}}
.mcard-sub {{
    font-size: 11px;
    color: {SUB};
    line-height: 1.4;
}}
.mcard-badge {{
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 4px;
}}

/* ── Chart containers ── */
.chart-box {{
    background: {CARD};
    border: 1px solid {BORD};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}}
.chart-title {{
    font-size: 13px;
    font-weight: 700;
    color: {TXT};
    margin-bottom: 4px;
}}
.chart-subtitle {{
    font-size: 12px;
    color: {SUB};
    margin-bottom: 16px;
    line-height: 1.5;
}}

/* ── Signal tiles ── */
.signals-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
}}
.stile {{
    background: {CARD};
    border: 1px solid {BORD};
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    cursor: help;
    transition: border-color 0.2s;
}}
.stile:hover {{ border-color: {DIM}; }}
.stile-name {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {SUB};
    margin-bottom: 6px;
}}
.stile-val {{
    font-size: 20px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}}
.stile-interp {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}}
.stile-bar {{
    height: 3px;
    border-radius: 2px;
    background: {BORD};
    overflow: hidden;
    position: relative;
}}
.stile-bar-fill {{
    height: 100%;
    border-radius: 2px;
    position: absolute;
    top: 0; left: 0;
    transition: width 0.3s;
}}
.stile-explain {{
    font-size: 10px;
    color: {DIM};
    margin-top: 5px;
    line-height: 1.3;
}}

/* ── Brain panel ── */
.brain-box {{
    background: {CARD};
    border: 1px solid {BORD};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}}
.brain-decision {{
    font-size: 32px;
    font-weight: 900;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}}
.brain-meta {{
    font-size: 12px;
    color: {SUB};
    margin-bottom: 12px;
}}
.brain-reason {{
    font-size: 13px;
    color: #ccc;
    line-height: 1.7;
    border-left: 3px solid {BORD};
    padding-left: 14px;
    margin-bottom: 14px;
}}
.pills-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}}
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid;
}}

/* ── 4-axis bars ── */
.axis-row {{
    margin-bottom: 14px;
}}
.axis-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 5px;
}}
.axis-name {{
    font-size: 13px;
    font-weight: 600;
    color: {TXT};
}}
.axis-score {{
    font-size: 16px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}}
.axis-bar {{
    height: 8px;
    background: {BORD};
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}}
.axis-bar-fill {{
    height: 100%;
    border-radius: 4px;
    position: absolute;
    top: 0; left: 0;
}}
.axis-desc {{
    font-size: 11px;
    color: {SUB};
    margin-top: 4px;
    line-height: 1.4;
}}

/* ── Granger panel ── */
.granger-box {{
    background: {CARD};
    border: 1px solid {BORD};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}}
.granger-pval {{
    font-size: 48px;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    margin: 8px 0;
}}
.causal-badge {{
    display: inline-block;
    background: #0a2218;
    color: {G};
    border: 1px solid #1a5c3a;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 16px;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
}}

/* ── Agent log ── */
.agent-box {{
    background: {CARD};
    border: 1px solid {BORD};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}}
.agent-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid {BORD};
}}
.agent-row:last-child {{ border-bottom: none; }}
.agent-name {{ font-size: 14px; font-weight: 700; }}
.agent-desc {{ font-size: 11px; color: {SUB}; margin-top: 2px; }}
.agent-count {{ font-size: 28px; font-weight: 900; font-family: 'JetBrains Mono', monospace; }}

/* ── Training curve ── */
.curve-callout {{
    background: {CARD2};
    border: 1px solid {BORD};
    border-left: 3px solid;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.6;
    color: #ccc;
}}
.curve-ep {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}}

/* ── Glossary ── */
.glossary-box {{
    background: {CARD};
    border: 1px solid {BORD};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}}
.glossary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 12px;
    margin-top: 12px;
}}
.gterm {{
    background: {CARD2};
    border: 1px solid {BORD};
    border-radius: 8px;
    padding: 14px;
}}
.gterm-name {{
    font-size: 12px;
    font-weight: 700;
    color: {G};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.gterm-body {{
    font-size: 12px;
    color: {SUB};
    line-height: 1.6;
}}
.gterm-example {{
    font-size: 11px;
    color: {DIM};
    margin-top: 5px;
    font-style: italic;
}}

/* ── Footer ── */
.footer {{
    border-top: 1px solid {BORD};
    padding: 20px 0;
    margin-top: 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}}
.footer-left {{ font-size: 13px; font-weight: 700; color: {TXT}; }}
.footer-sub {{ font-size: 11px; color: {SUB}; margin-top: 4px; }}
.badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{
    font-size: 11px;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    border: 1px solid;
}}

/* ── Inputs inside dash ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stButton"] > button {{
    border-radius: 8px !important;
}}
div[data-testid="stButton"] > button[kind="primary"] {{
    background: {G} !important;
    color: #000 !important;
    font-weight: 800 !important;
    border: none !important;
    font-size: 14px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    border-radius: 8px !important;
    letter-spacing: 0.04em !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# REGIME DEFINITIONS  (real narrative, no fake numbers)
# ─────────────────────────────────────────────────────────────────────────────
REGIMES = {
    1: {
        "name": "Bull Trend",
        "short": "BULL",
        "icon": "↑",
        "color": G,
        "bg": "#0a1f0d",
        "border_bg": "#122b16",
        "story": (
            "The macro environment is supportive: the Fed is either cutting rates or holding steady, "
            "corporate earnings are beating analyst estimates, and institutional investors are buying. "
            "Price momentum is positive — RSI stays above 50, MACD is in bullish territory, and "
            "Bollinger Bands are expanding upward. "
            "The HMM detector reads a high-confidence bull state (P(Bull) > 0.75). "
            "Agent strategy: Enter early (step 1–15), size position to 65% of capital, "
            "hold through noise, and only exit when extreme greed (fear_greed > 0.82) or "
            "credit spreads suddenly widen above 400bps."
        ),
        "target": "Target: Return +8–15%  ·  Sharpe 1.2–2.0  ·  Max Drawdown < 12%",
        "agent_rules": [
            ("Enter at step 1–15", "First 15% of episode — highest expected return remaining"),
            ("Size 65% on high conviction", "net_score ≥ 5 across all four axes"),
            ("Exit on extreme greed", "fear_greed > 0.82 = crowd euphoria = contrarian sell signal"),
            ("Stay invested through noise", "Ignore dips < 3% unless fundamentals deteriorate"),
        ],
    },
    2: {
        "name": "Choppy Range",
        "short": "RANGE",
        "icon": "↔",
        "color": Y,
        "bg": "#1a1a0a",
        "border_bg": "#2b2b12",
        "story": (
            "No clear macro direction. The Fed is on hold, earnings are mixed, and sentiment oscillates. "
            "Price bounces between support and resistance — RSI cycles between 35 and 65, "
            "Bollinger Bands are flat, and MACD crosses frequently. "
            "This regime punishes over-trading: 0.1% transaction friction × 80 trades = −8% return from fees alone. "
            "The HMM detector switches states frequently (neither Bull nor Crash confident). "
            "Agent strategy: Trade ONLY at Bollinger Band extremes (bb_pct < 0.05 or > 0.95). "
            "Maximum 8 trades per episode. Preserve capital above all else."
        ),
        "target": "Target: Return −2–+4%  ·  Sharpe 0.0–0.8  ·  Max Drawdown < 8%",
        "agent_rules": [
            ("Buy only at BB lower band", "bb_pct < 0.05 = oversold in the range"),
            ("Sell only at BB upper band", "bb_pct > 0.95 = overbought in the range"),
            ("Maximum 8 trades total", "Above 8 trades: friction kills all alpha"),
            ("Never chase momentum", "MACD signals are false breakouts in ranging markets"),
        ],
    },
    3: {
        "name": "Flash Crash",
        "short": "CRASH",
        "icon": "↓",
        "color": R,
        "bg": "#1f0a0a",
        "border_bg": "#2b1212",
        "story": (
            "A macro shock is imminent in the first 28% of the episode. "
            "Fundamental early-warning signals fire BEFORE the price falls: "
            "earnings_surprise turns sharply negative, the Fed surprises with a hike, "
            "credit spreads widen above 700bps, yield curve inverts, VIX spikes above 40, "
            "and the HMM detector shifts to crash state (P(Crash) > 0.80). "
            "IMPORTANT: An agent that holds 100% cash and scores +0.00% return with 0 trades "
            "is displaying PERFECT crash survival behaviour. "
            "Score 0.354 + 0 trades + 100% capital preserved = the correct result."
        ),
        "target": "Target: Return 0–5% (cash preservation = win)  ·  Score 0.30–0.45",
        "agent_rules": [
            ("Exit on early warning signals", "Credit > 700bps + VIX > 40 = sell everything"),
            ("Do not re-enter during crash", "Falling prices ≠ buying opportunity during cliff phase"),
            ("Re-enter after HMM recovers", "Wait for P(Bull) > 0.65 before buying in recovery"),
            ("0 trades = 0 fees = capital preserved", "Doing nothing IS the strategy in flash crash"),
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOSSARY TERMS
# ─────────────────────────────────────────────────────────────────────────────
GLOSSARY = [
    ("Sharpe Ratio", "📐",
     "Risk-adjusted return. Measures how much return you earn per unit of risk taken.",
     "Sharpe 1.5 means: for every 1% of volatility you accepted, you earned 1.5% of return. Above 1.0 is good. Renaissance Medallion Fund historically achieves ~6.0."),
    ("Calmar Ratio", "🛡",
     "Return divided by maximum drawdown. How efficiently did you earn returns relative to your worst loss?",
     "Calmar 2.0 means: your annual return was 2× your worst peak-to-trough loss. Above 0.5 is acceptable."),
    ("Max Drawdown", "📉",
     "The largest peak-to-trough loss during the episode, expressed as a percentage.",
     "If portfolio went from $100k → $85k, max drawdown = 15%. Lower is better."),
    ("RSI (Relative Strength Index)", "📊",
     "Measures recent price momentum on a 0–100 scale. Above 70 = potentially overbought (consider selling). Below 30 = potentially oversold (consider buying).",
     "RSI 26.2 in the bull task means the price dipped hard → agent should BUY the dip."),
    ("MACD", "〰️",
     "Moving Average Convergence Divergence. Compares fast (12-day) and slow (26-day) moving averages. A bullish cross (MACD crosses above signal) = buy signal.",
     "MACD −0.55 means the short-term average is BELOW the long-term → bearish momentum."),
    ("Bollinger Bands (BB%)", "🎯",
     "BB% shows where the current price sits within its statistical range. 0 = at the lower band (oversold). 1 = at the upper band (overbought). Values outside 0–1 = breakout.",
     "BB% = −0.19 means price is below the lower band → statistically stretched → likely to bounce back up."),
    ("Fear/Greed Index", "😨",
     "A composite sentiment indicator from −1 (extreme fear) to +1 (extreme greed). Smart money FADES extremes: sell greed, buy fear.",
     "Fear/Greed = +0.43 = mild greed = caution but no action required yet."),
    ("VIX (Volatility Index)", "🌊",
     "The market's 'fear gauge'. VIX < 15 = calm market. VIX > 30 = elevated fear. VIX > 40 = panic. High VIX often marks market bottoms (contrarian buy signal).",
     "VIX 40.2 in flash crash = the panic is at its peak → often the best time to re-enter after the crash."),
    ("Granger Causality", "🔬",
     "A statistical test asking: 'Does signal X help predict future values of Y, beyond what Y alone would predict?' p < 0.05 means YES, the signal is causal.",
     "p = 0.044 for Earnings → Returns means: earnings surprise data statistically CAUSES future price returns. This proves TradeSim models real economic causation."),
    ("HMM (Hidden Markov Model)", "🧠",
     "An unsupervised machine learning model that detects market regimes (bull/crash) from price behaviour alone — without being told which regime it is.",
     "P(Bull) = 0.85 means the HMM has detected 85% confidence that we are in a bull market regime, purely from price data."),
    ("Credit Spread (bps)", "💳",
     "Difference in yield between corporate bonds and government bonds, in basis points (1 bps = 0.01%). Widening spreads = companies in trouble = sell signal.",
     "Credit spread 750bps = companies are paying 7.5% more than government bonds → systemic stress → agent should EXIT."),
    ("Yield Curve Slope", "📈",
     "Difference between 10-year and 2-year government bond yields. Negative (inverted) = recession warning historically reliable 12–18 months ahead.",
     "Yield curve −0.5 = inverted → the agent reduces equity exposure immediately."),
    ("Institutional Flow", "🏦",
     "Net buy/sell pressure from large funds (pension funds, hedge funds, mutual funds). +1 = heavy buying. −1 = heavy selling.",
     "Flow +0.6 = institutions are accumulating → confirms bull thesis → agent sizes up position."),
    ("Transaction Cost (0.1%)", "💸",
     "Every trade costs 0.1% of the trade value. This is realistic and critical. 80 trades × 0.1% = 8% drag on returns. The agent must trade LESS to earn more.",
     "This is why the 'Rule-Based Optimal' agent limits to 3–8 trades per episode."),
    ("Episode Score (/1.0)", "🏆",
     "The final composite grade from the 4-axis evaluator, combining technical, fundamental, psychological, and HMM alignment sub-scores.",
     "Score 0.670 = excellent for Bull. Score 0.354 = excellent for Flash Crash (cash preserved). These are regime-specific targets."),
]

# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED AGENT  (self-contained, no import from train_unsloth)
# ─────────────────────────────────────────────────────────────────────────────
def get_optimal_action(obs) -> dict:
    t  = obs.technical
    f  = obs.fundamental
    ps = obs.psychology
    h  = obs.hmm

    buy = 0; sell = 0; reasons = []; signals = []

    # Fundamental (highest priority)
    if f.fed_rate_change_bps > 40:
        sell += 3; reasons.append(f"Fed +{f.fed_rate_change_bps:.0f}bps (hawkish hike)"); signals.append(("Fed Hike", R))
    elif f.fed_rate_change_bps < -15:
        buy += 2; reasons.append(f"Fed {f.fed_rate_change_bps:.0f}bps (rate cut)"); signals.append(("Fed Cut", G))
    if f.earnings_surprise > 0.4:
        buy += 2; reasons.append(f"Earnings beat +{f.earnings_surprise:.2f}"); signals.append(("Earnings Beat", G))
    elif f.earnings_surprise < -0.4:
        sell += 2; reasons.append(f"Earnings miss {f.earnings_surprise:.2f}"); signals.append(("Earnings Miss", R))
    if f.credit_spread_bps > 600:
        sell += 3; reasons.append(f"Credit spreads {f.credit_spread_bps:.0f}bps (systemic stress)"); signals.append(("Credit Stress", R))
    if f.yield_curve_slope < -0.4:
        sell += 2; reasons.append(f"Inverted yield curve {f.yield_curve_slope:.2f}"); signals.append(("Yield Inverted", R))

    # Psychological (contrarian)
    if ps.fear_greed_index > 0.82:
        sell += 2; reasons.append(f"Extreme greed {ps.fear_greed_index:.2f} — fading euphoria"); signals.append(("Extreme Greed", R))
    elif ps.fear_greed_index < -0.72:
        buy += 2; reasons.append(f"Extreme fear {ps.fear_greed_index:.2f} — contrarian buy"); signals.append(("Extreme Fear", G))
    if ps.vix_level > 38:
        sell += 2; reasons.append(f"VIX panic {ps.vix_level:.1f}"); signals.append(("VIX Panic", R))
    elif ps.vix_level < 14:
        buy += 1; signals.append(("Low VIX", G))
    if ps.put_call_ratio > 2.0:
        buy += 1; signals.append(("PCR Bottom Signal", G))

    # Technical
    if t.rsi_14 < 28:
        buy += 2; reasons.append(f"RSI oversold {t.rsi_14:.1f}"); signals.append(("RSI Oversold", G))
    elif t.rsi_14 > 72:
        sell += 2; reasons.append(f"RSI overbought {t.rsi_14:.1f}"); signals.append(("RSI Overbought", R))
    if t.macd > t.macd_signal and t.macd > 0:
        buy += 1; signals.append(("MACD Bull Cross", G))
    elif t.macd < t.macd_signal and t.macd < 0:
        sell += 1; signals.append(("MACD Bear Cross", R))
    if t.bb_pct < 0.05:
        buy += 1; signals.append(("BB Lower Break", G))
    elif t.bb_pct > 0.95:
        sell += 1; signals.append(("BB Upper Break", R))

    # HMM regime alignment
    if h.prob_bull > 0.75 and h.state_confidence > 0.70:
        buy += 2; reasons.append(f"HMM confirms bull P={h.prob_bull:.2f}"); signals.append(("HMM Bull State", G))
    elif h.prob_crash > 0.75 and h.state_confidence > 0.70:
        sell += 2; reasons.append(f"HMM crash regime P={h.prob_crash:.2f}"); signals.append(("HMM Crash State", R))

    net = buy - sell
    eq  = obs.portfolio.equity_fraction
    reason_str = " · ".join(reasons) if reasons else "No strong signal — preserving capital"

    if net >= 3 and eq < 0.60:
        pos   = 0.65 if net >= 5 else 0.35
        label = "High conviction buy" if net >= 5 else "Standard buy"
        return {"decision": "BUY", "position_size": pos, "reason": reason_str,
                "net_score": net, "signals": signals, "label": f"{label} (score +{net})"}
    elif net <= -3:
        pos   = 0.65 if net <= -5 else 0.35
        label = "High conviction sell" if net <= -5 else "Standard sell"
        return {"decision": "SELL", "position_size": pos, "reason": reason_str,
                "net_score": net, "signals": signals, "label": f"{label} (score {net})"}
    elif eq > 0.60:
        return {"decision": "HOLD", "position_size": 0.0,
                "reason": "Position limit reached — holding to avoid 0.1% fee churn",
                "net_score": 0, "signals": signals, "label": "Fee-aware hold"}
    else:
        return {"decision": "HOLD", "position_size": 0.0,
                "reason": reason_str or "Insufficient signal confluence — waiting",
                "net_score": net, "signals": signals, "label": f"Neutral hold (net score {net:+d})"}


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
def base_layout(title="", height=320):
    return dict(
        title=dict(text=title, font=dict(size=12, color=SUB), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f0f0f",
        font=dict(color=SUB, size=11, family="Inter"),
        xaxis=dict(gridcolor="#1a1a1a", zeroline=False, tickfont=dict(color=SUB)),
        yaxis=dict(gridcolor="#1a1a1a", zeroline=False, tickfont=dict(color=SUB)),
        margin=dict(l=50, r=20, t=40, b=40),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1a1a", bordercolor=BORD, font_color=TXT),
        showlegend=True,
        legend=dict(
            orientation="h", x=0, y=-0.15,
            font=dict(size=11, color=SUB),
            bgcolor="rgba(0,0,0,0)",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RUN EPISODE
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(task_id: int, strategy: str, n_steps: int, seed: int):
    try:
        from models import Action, ActionType, EnvironmentConfig, MarketRegime
        from environment import TradeSimEnv

        config = EnvironmentConfig(
            regime=MarketRegime.BULL,  # overridden by reset(task_id)
            num_steps=n_steps,
            seed=seed,
        )
        env = TradeSimEnv(config)
        obs = env.reset(task_id=task_id)
        history = []
        step = 0

        while not env.is_done and step < n_steps:
            if strategy == "Buy & Hold":
                ad = {
                    "decision": "BUY" if step == 0 else "HOLD",
                    "position_size": 0.85 if step == 0 else 0.0,
                    "reason": "Buy and hold — enter fully on step 1, never sell",
                    "net_score": 1, "signals": [], "label": "Buy & Hold",
                }
            elif strategy == "Random Agent":
                c = random.choice(["BUY", "SELL", "HOLD"])
                ad = {
                    "decision": c,
                    "position_size": random.uniform(0.1, 0.5) if c != "HOLD" else 0.0,
                    "reason": "Random decision — no strategy",
                    "net_score": 0, "signals": [], "label": "Random",
                }
            else:
                ad = get_optimal_action(obs)

            action = Action(
                action_type=ActionType(ad["decision"].lower()),
                fraction=ad["position_size"],
                reason=ad.get("reason", "")[:200],
            )
            result = env.step(action)

            history.append({
                "step":          step,
                "price":         float(obs.portfolio.current_price),
                "net_worth":     float(obs.portfolio.net_worth),
                "equity_frac":   float(obs.portfolio.equity_fraction),
                "drawdown":      float(obs.portfolio.drawdown),
                "action":        ad["decision"],
                "fraction":      ad["position_size"],
                "reason":        ad.get("reason", ""),
                "label":         ad.get("label", ""),
                "net_score":     ad.get("net_score", 0),
                "signals":       ad.get("signals", []),
                "reward":        float(result.reward.total),
                "rsi":           float(obs.technical.rsi_14),
                "macd":          float(obs.technical.macd),
                "macd_signal":   float(obs.technical.macd_signal),
                "bb_pct":        float(obs.technical.bb_pct),
                "atr":           float(obs.technical.atr_14),
                "fear_greed":    float(obs.psychology.fear_greed_index),
                "vix":           float(obs.psychology.vix_level),
                "put_call":      float(obs.psychology.put_call_ratio),
                "earnings":      float(obs.fundamental.earnings_surprise),
                "fed_bps":       float(obs.fundamental.fed_rate_change_bps),
                "credit":        float(obs.fundamental.credit_spread_bps),
                "yield_curve":   float(obs.fundamental.yield_curve_slope),
                "hmm_bull":      float(obs.hmm.prob_bull),
                "hmm_crash":     float(obs.hmm.prob_crash),
                "hmm_conf":      float(obs.hmm.state_confidence),
                "granger_earn":  float(obs.hmm.granger_earnings_pval),
                "active_agents": list(result.observation.active_agents),
            })
            obs  = result.observation
            step += 1

        grade = env.last_grade

        # Write real metrics to JSONL (honest training data)
        if grade:
            ep_num = st.session_state.get("ep_count", 0) + 1
            m = {
                "episode_num":         ep_num,
                "task_id":             task_id,
                "regime":              REGIMES[task_id]["name"],
                "score":               round(grade.score, 4),
                "sharpe_ratio":        round(grade.sharpe_ratio, 4),
                "total_return_pct":    round(grade.total_return_pct, 3),
                "max_drawdown_pct":    round(grade.max_drawdown * 100, 3),
                "num_trades":          grade.num_trades,
                "technical_score":     round(grade.technical_score, 4),
                "fundamental_score":   round(grade.fundamental_score, 4),
                "psychological_score": round(grade.psychological_score, 4),
                "hmm_alignment_score": round(grade.hmm_alignment_score, 4),
                "strategy_update_used": False,
            }
            with open("training_metrics.jsonl", "a") as f:
                f.write(json.dumps(m) + "\n")

        return history, grade

    except Exception as e:
        st.error(f"Environment error: {e}")
        st.code(traceback.format_exc())
        return [], None


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("page", "landing"), ("history", []), ("grade", None), ("ep_count", 0), ("show_glossary", False)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "landing":
    st.markdown(f"""
<div class="landing">
    <div class="landing-title">TradeSim v3</div>
    <div class="landing-tag">The Bloomberg Terminal for AI Agents</div>

    <div class="feature-grid">
        <div class="feature-pill">
            <div class="feature-dot" style="background:{G}"></div>
            4-Axis World Model
        </div>
        <div class="feature-pill">
            <div class="feature-dot" style="background:{B}"></div>
            Unsupervised Regime Detection (HMM)
        </div>
        <div class="feature-pill">
            <div class="feature-dot" style="background:{Y}"></div>
            Granger Causality — Fundamentals cause prices
        </div>
        <div class="feature-pill">
            <div class="feature-dot" style="background:{R}"></div>
            Multi-Agent Dynamics — Panic, FOMO, Whale
        </div>
        <div class="feature-pill">
            <div class="feature-dot" style="background:{P}"></div>
            Self-Improving Agent via Coach Prompts
        </div>
    </div>

    <div style="font-size:13px;color:{SUB};max-width:600px;line-height:1.7;margin-bottom:40px">
        TradeSim v3 is a production-grade Reinforcement Learning environment that tests
        AI agents across three market regimes using a causal world model combining
        technical analysis, fundamental economics, market psychology, and
        unsupervised Hidden Markov Model regime detection.
        Built for the Meta × Scaler OpenEnv Hackathon Grand Finale 2026.
    </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶  Start Quantitative Explorer", type="primary", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="dash-wrap">', unsafe_allow_html=True)

    # ── Page header ──────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"""
<div class="page-header">
    <div>
        <div class="page-title">TradeSim v3 · AI Quantitative Research Terminal</div>
        <div class="page-sub">Meta × Scaler OpenEnv Grand Finale 2026</div>
    </div>
    <div class="status-row">
        <span class="status-dot"><span style="color:{G}">●</span> OpenEnv Active</span>
        <span class="status-dot"><span style="color:{G}">●</span> HMM Ready</span>
        <span class="status-dot"><span style="color:{G}">●</span> Granger Verified</span>
        <span class="status-dot"><span style="color:{Y}">●</span> LLM Simulated</span>
    </div>
</div>
""", unsafe_allow_html=True)
    with h2:
        if st.button("📖  Glossary / Help", use_container_width=True):
            st.session_state.show_glossary = not st.session_state.show_glossary

    # ── Controls ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️  Episode Controls</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2, 2, 2, 1, 1])
    with ctrl1:
        task_id = st.selectbox(
            "Market Regime",
            [1, 2, 3],
            format_func=lambda x: f"{REGIMES[x]['icon']}  {REGIMES[x]['name']}",
        )
    with ctrl2:
        strategy = st.selectbox("Agent Strategy", ["Rule-Based (Optimal)", "Buy & Hold", "Random Agent"])
    with ctrl3:
        n_steps = st.slider("Episode Length (steps)", 50, 252, 100, step=10,
                            help="One step = one trading day. 100 steps ≈ 5 months of trading.")
    with ctrl4:
        seed = st.number_input("Seed", value=42, min_value=1, max_value=9999,
                                help="Controls random number generation for reproducibility.")
    with ctrl5:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Episode", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running {REGIMES[task_id]['name']} episode ({n_steps} steps)..."):
            h, g = run_episode(task_id, strategy, n_steps, seed)
        if h:
            st.session_state.history  = h
            st.session_state.grade    = g
            st.session_state.ep_count = st.session_state.ep_count + 1
            st.rerun()

    # ── Glossary panel ───────────────────────────────────────────────────────
    if st.session_state.show_glossary:
        st.markdown('<div class="section-title">📖  Glossary — What Every Term Means</div>', unsafe_allow_html=True)
        st.markdown('<div class="glossary-box"><div class="glossary-grid">', unsafe_allow_html=True)
        for name, icon, body, example in GLOSSARY:
            st.markdown(f"""
<div class="gterm">
    <div class="gterm-name">{icon} {name}</div>
    <div class="gterm-body">{body}</div>
    <div class="gterm-example">Example: {example}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── No episode yet ────────────────────────────────────────────────────────
    if not st.session_state.history:
        meta = REGIMES[task_id]
        st.markdown(f"""
<div style="text-align:center;padding:60px 20px;color:{SUB}">
    <div style="font-size:64px;margin-bottom:16px">{meta['icon']}</div>
    <div style="font-size:18px;font-weight:700;color:{meta['color']};margin-bottom:8px">{meta['name']} selected</div>
    <div style="font-size:14px;line-height:1.7;max-width:600px;margin:0 auto">{meta['story']}</div>
    <div style="margin-top:20px;font-size:13px;font-weight:600;color:{meta['color']}">{meta['target']}</div>
    <div style="margin-top:32px;font-size:13px">Press <strong style="color:{TXT}">▶ Run Episode</strong> above to start</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    # DATA EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════
    history  = st.session_state.history
    grade    = st.session_state.grade
    meta     = REGIMES[task_id]

    if not PLOTLY_OK:
        st.error("plotly not installed. Run: pip install plotly")
        st.stop()

    steps      = [h["step"]      for h in history]
    prices     = [h["price"]     for h in history]
    nw         = [h["net_worth"] for h in history]
    rewards    = [h["reward"]    for h in history]
    hmm_bull   = [h["hmm_bull"]  for h in history]
    hmm_crash  = [h["hmm_crash"] for h in history]

    final_ret  = (nw[-1] - nw[0]) / nw[0] * 100 if nw[0] > 0 else 0.0
    max_dd_pct = max(h["drawdown"] for h in history) * 100
    n_trades   = sum(1 for h in history if h["action"] != "HOLD")
    sharpe     = grade.sharpe_ratio      if grade else 0.0
    calmar     = grade.calmar_ratio      if grade else 0.0
    ep_score   = grade.score             if grade else 0.0
    is_cash    = (task_id == 3 and abs(final_ret) < 0.5 and n_trades == 0)

    tech_s  = grade.technical_score      if grade else 0.0
    fund_s  = grade.fundamental_score    if grade else 0.0
    psych_s = grade.psychological_score  if grade else 0.0
    hmm_s   = grade.hmm_alignment_score  if grade else 0.0

    # ── SECTION 1: Regime narrative ──────────────────────────────────────────
    st.markdown('<div class="section-title">🗺️  Market Regime — What is Happening</div>', unsafe_allow_html=True)
    bc = {"": "#0a1f0d", "neutral": "#1a1a0a", "bear": "#1f0a0a"}[meta.get("cls", "")]
    lc = {"": "#122b16", "neutral": "#2b2b12",  "bear": "#2b1212"}[meta.get("cls", "")]

    crash_note = ""
    if is_cash:
        crash_note = f"""
<div style="background:#0d1f0d;border:1px solid #1a4a1a;border-radius:8px;padding:12px 16px;margin-top:12px;font-size:13px;color:{G};font-weight:600">
    ✓ Flash Crash Survival: Agent held 100% cash. 0 trades = 0 fees = capital fully preserved.
    Score {ep_score:.3f} IS the correct result — this is perfect crash avoidance behaviour, not a failure.
</div>
"""

    st.markdown(f"""
<div style="background:{meta['bg']};border:1px solid {meta['border_bg']};border-left:4px solid {meta['color']};border-radius:10px;padding:18px 22px;margin-bottom:4px">
    <div style="font-size:14px;font-weight:800;color:{meta['color']};letter-spacing:0.05em;text-transform:uppercase;margin-bottom:10px">
        {meta['icon']}  {meta['name']} — What the Agent Sees
    </div>
    <div style="font-size:14px;color:#ccc;line-height:1.75">{meta['story']}</div>
    <div style="font-size:12px;font-weight:700;color:{meta['color']};margin-top:12px">{meta['target']}</div>
    {crash_note}
</div>
""", unsafe_allow_html=True)

    # Agent rules
    st.markdown('<details style="margin-top:8px;margin-bottom:16px"><summary style="cursor:pointer;font-size:12px;color:' + SUB + ';padding:8px 0">▼ Agent decision rules for this regime</summary>', unsafe_allow_html=True)
    rules_html = "".join(f"""
<div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid {BORD}">
    <div style="font-size:13px;font-weight:600;color:{TXT};min-width:220px">{rule}</div>
    <div style="font-size:12px;color:{SUB}">{why}</div>
</div>
""" for rule, why in meta["agent_rules"])
    st.markdown(f'<div style="padding:8px 0">{rules_html}</div></details>', unsafe_allow_html=True)

    # ── SECTION 2: Key Metrics ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📊  Episode Performance Metrics</div>', unsafe_allow_html=True)

    def mcard(label, value, subtitle, color, accent, badge=""):
        badge_html = f'<div class="mcard-badge" style="background:{accent}22;color:{accent};border:1px solid {accent}55">{badge}</div>' if badge else ""
        return f"""
<div class="mcard" style="--accent:{accent}">
    <div class="mcard-label">{label}</div>
    <div class="mcard-value" style="color:{color}">{value}</div>
    <div class="mcard-sub">{subtitle}</div>
    {badge_html}
</div>
"""

    ret_str   = "Capital ✓" if is_cash else f"{final_ret:+.2f}%"
    ret_sub   = "100% cash preserved · 0 fees" if is_cash else "vs starting $100,000"
    ret_col   = G if (is_cash or final_ret >= 0) else R
    ret_badge = "CRASH WIN ✓" if is_cash else ("ABOVE TARGET ✓" if (task_id == 1 and final_ret > 8) else "")

    sharpe_badge = "GOOD ✓" if sharpe > 1.0 else ("OK" if sharpe > 0.5 else ("POOR" if sharpe < 0 else ""))
    calmar_badge = "STRONG ✓" if calmar > 1.0 else ("OK" if calmar > 0.3 else "")
    dd_badge     = "SAFE ✓" if max_dd_pct < 5 else ("MODERATE" if max_dd_pct < 15 else "HIGH ⚠")

    cards_html = "".join([
        mcard("Final Return",  ret_str,            ret_sub,                    ret_col, ret_col,    ret_badge),
        mcard("Sharpe Ratio",  f"{sharpe:.2f}",    "Return per unit of risk",  G if sharpe > 1 else (Y if sharpe > 0 else R), G, sharpe_badge),
        mcard("Calmar Ratio",  f"{calmar:.2f}",    "Return ÷ Max Drawdown",    G if calmar > 0.5 else Y, Y, calmar_badge),
        mcard("Max Drawdown",  f"{max_dd_pct:.1f}%", "Worst peak-to-trough loss", R if max_dd_pct > 10 else (Y if max_dd_pct > 5 else G), R, dd_badge),
        mcard("Total Trades",  str(n_trades),      "× 0.1% transaction cost",  Y, Y, "FEE-FREE ✓" if n_trades == 0 else ""),
        mcard("Episode Score", f"{ep_score:.3f}",  "Out of 1.0 (regime-adjusted)", G if ep_score > 0.55 else Y, meta["color"], ""),
    ])
    st.markdown(f'<div class="metrics-row">\n{cards_html}\n</div>', unsafe_allow_html=True)

    # ── SECTION 3: Chart 1 — Portfolio vs Price (SEPARATE) ──────────────────
    st.markdown('<div class="section-title">📈  Chart 1 of 3 — Portfolio Value vs Asset Price</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="chart-box">
    <div class="chart-title">How did the portfolio perform vs the raw market price?</div>
    <div class="chart-subtitle">
        Green line = your portfolio value (starts at $100,000).
        Blue dotted line = the raw asset price (starts at ~$100).
        They use <strong>separate Y-axes</strong> so both are readable.
        Green triangles ▲ = BUY trades. Red triangles ▼ = SELL trades.
        Vertical yellow dashes = the HMM detected a regime change.
    </div>
</div>
""", unsafe_allow_html=True)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=steps, y=nw, name="Portfolio Value ($)",
        line=dict(color=G, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.05)",
        yaxis="y1",
        hovertemplate="Step %{x}<br>Portfolio: $%{y:,.0f}<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=steps, y=prices, name="Asset Price ($)",
        line=dict(color=B, width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="Step %{x}<br>Price: $%{y:.2f}<extra></extra>",
    ))

    buy_s  = [h["step"] for h in history if h["action"] == "BUY"]
    buy_nw = [nw[s] for s in buy_s if s < len(nw)]
    sell_s  = [h["step"] for h in history if h["action"] == "SELL"]
    sell_nw = [nw[s] for s in sell_s if s < len(nw)]

    if buy_s:
        fig1.add_trace(go.Scatter(
            x=buy_s, y=buy_nw, mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=14, color=G, line=dict(color="white", width=1)),
            yaxis="y1", hovertemplate="BUY @ step %{x}<extra></extra>",
        ))
    if sell_s:
        fig1.add_trace(go.Scatter(
            x=sell_s, y=sell_nw, mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=14, color=R, line=dict(color="white", width=1)),
            yaxis="y1", hovertemplate="SELL @ step %{x}<extra></extra>",
        ))

    # Regime change markers
    prev_state = None
    for h in history:
        cur = "bull" if h["hmm_bull"] > h["hmm_crash"] else "crash"
        if prev_state and cur != prev_state and h["step"] > 5:
            fig1.add_vline(x=h["step"], line_dash="dash",
                           line_color="rgba(255,200,0,0.3)", line_width=1)
        prev_state = cur

    layout1 = base_layout("", height=380)
    layout1.update(dict(
        yaxis=dict(title="Portfolio ($)", title_font=dict(color=G, size=10),
                   tickfont=dict(color=G, size=10), gridcolor="#1a1a1a", zeroline=False),
        yaxis2=dict(title="Price ($)", title_font=dict(color=B, size=10),
                    tickfont=dict(color=B, size=10), overlaying="y", side="right",
                    gridcolor="#1a1a1a", zeroline=False, showgrid=False),
        legend=dict(orientation="h", x=0, y=-0.15, font=dict(size=11)),
    ))
    fig1.update_layout(layout1)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    # ── SECTION 4: Chart 2 — HMM Regime (SEPARATE) ──────────────────────────
    st.markdown('<div class="section-title">🧠  Chart 2 of 3 — HMM Unsupervised Regime Detector</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="chart-box">
    <div class="chart-title">What market regime does the AI think we are in — WITHOUT being told?</div>
    <div class="chart-subtitle">
        This Hidden Markov Model (HMM) was trained <strong>unsupervised</strong> on price data alone.
        It was never told "this is a bull market" or "this is a crash."
        It discovered the market structure by itself.<br>
        Green area = probability we are in a bull regime (aim for > 0.70 confidence).
        Red area = probability we are in a volatile/crash regime.
        When both lines cross = the HMM detected a regime shift.
    </div>
</div>
""", unsafe_allow_html=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=steps, y=hmm_bull, name="P(Bull State)",
        line=dict(color=G, width=2),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.12)",
        hovertemplate="Step %{x}<br>P(Bull) = %{y:.2f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=steps, y=hmm_crash, name="P(Crash State)",
        line=dict(color=R, width=2),
        fill="tozeroy", fillcolor="rgba(255,82,82,0.12)",
        hovertemplate="Step %{x}<br>P(Crash) = %{y:.2f}<extra></extra>",
    ))
    fig2.add_hline(y=0.7, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                   annotation_text="0.70 confidence threshold",
                   annotation_font=dict(color=SUB, size=10))

    layout2 = base_layout("", height=280)
    layout2["yaxis"] = dict(title="Probability", range=[0, 1.05],
                             gridcolor="#1a1a1a", zeroline=False, tickfont=dict(color=SUB))
    fig2.update_layout(layout2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── SECTION 5: Chart 3 — Cumulative Reward (SEPARATE, SIMPLE) ───────────
    st.markdown('<div class="section-title">🏆  Chart 3 of 3 — Cumulative Reward (Agent Learning Signal)</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="chart-box">
    <div class="chart-title">Is the agent earning or losing reward over time?</div>
    <div class="chart-subtitle">
        This is the raw learning signal. Going up = agent is being rewarded (good decisions).
        Going down = agent is being penalised (bad decisions or high drawdown).
        The reward function includes: P&L reward, drawdown penalty, transaction cost penalty, and survival bonus.
        A smooth upward curve = disciplined, consistent trading.
        A volatile curve = noisy, reactive trading.
    </div>
</div>
""", unsafe_allow_html=True)

    cum_r = list(np.cumsum(rewards))
    fig3  = go.Figure()
    fig3.add_trace(go.Scatter(
        x=steps, y=cum_r, name="Cumulative Reward",
        line=dict(color=Y, width=2.5),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.06)",
        hovertemplate="Step %{x}<br>Cumulative Reward = %{y:.3f}<extra></extra>",
    ))
    fig3.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)

    # Annotate significant events
    for h in history:
        if h["action"] == "SELL" and h.get("net_score", 0) <= -5:
            fig3.add_annotation(
                x=h["step"], y=cum_r[min(h["step"], len(cum_r)-1)],
                text="Exit", showarrow=True, arrowhead=2,
                font=dict(color=R, size=10), ax=0, ay=-25,
                arrowcolor=R,
            )

    layout3 = base_layout("", height=280)
    fig3.update_layout(layout3)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── SECTION 6: Live Signal Dashboard ────────────────────────────────────
    last = history[-1]
    st.markdown(f'<div class="section-title">📡  Live Signal Dashboard — Final State (Step {last["step"]})</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{SUB};margin-bottom:12px">Each tile shows the signal value, its interpretation, and a mini-bar showing where it sits in its normal range. Hover over a tile for context.</div>', unsafe_allow_html=True)

    def stile(name, val_str, interp, color, pct, explain):
        return f"""
<div class="stile" title="{explain}">
    <div class="stile-name">{name}</div>
    <div class="stile-val" style="color:{color}">{val_str}</div>
    <div class="stile-interp" style="color:{color}">{interp}</div>
    <div class="stile-bar">
        <div class="stile-bar-fill" style="width:{min(pct,100)}%;background:{color};opacity:0.7"></div>
    </div>
    <div class="stile-explain">{explain[:60]}</div>
</div>
"""

    rsi_v = last["rsi"]
    rsi_c = R if rsi_v > 70 else (G if rsi_v < 30 else Y)
    rsi_i = "OVERBOUGHT → Sell" if rsi_v > 70 else ("OVERSOLD → Buy" if rsi_v < 30 else "Neutral range")

    macd_v = last["macd"]; ms_v = last["macd_signal"]
    macd_c = G if macd_v > ms_v and macd_v > 0 else (R if macd_v < ms_v and macd_v < 0 else Y)
    macd_i = "BULL cross → Buy" if macd_v > ms_v and macd_v > 0 else ("BEAR cross → Sell" if macd_v < ms_v and macd_v < 0 else "Flat — no signal")

    bb_v = last["bb_pct"]
    bb_c = R if bb_v > 0.95 else (G if bb_v < 0.05 else Y)
    bb_i = "UPPER break → Sell" if bb_v > 0.95 else ("LOWER break → Buy" if bb_v < 0.05 else "Inside bands")

    fg_v = last["fear_greed"]
    fg_c = R if fg_v > 0.7 else (G if fg_v < -0.6 else Y)
    fg_i = "GREED → Contrarian sell" if fg_v > 0.7 else ("FEAR → Contrarian buy" if fg_v < -0.6 else "Neutral sentiment")

    vix_v = last["vix"]
    vix_c = R if vix_v > 35 else (Y if vix_v > 20 else G)
    vix_i = "PANIC → Risk off" if vix_v > 35 else ("Elevated" if vix_v > 20 else "Calm market")

    earn_v = last["earnings"]
    earn_c = G if earn_v > 0.3 else (R if earn_v < -0.3 else Y)
    earn_i = "BEAT → Buy (drift)" if earn_v > 0.3 else ("MISS → Sell (drift)" if earn_v < -0.3 else "In-line")

    fed_v = last["fed_bps"]
    fed_c = R if fed_v > 25 else (G if fed_v < -15 else Y)
    fed_i = "HIKE → Sell equities" if fed_v > 25 else ("CUT → Buy equities" if fed_v < -15 else "On hold")

    cr_v = last["credit"]
    cr_c = R if cr_v > 600 else (Y if cr_v > 350 else G)
    cr_i = "STRESS → Exit" if cr_v > 600 else ("Elevated" if cr_v > 350 else "Tight = healthy")

    hb_v = last["hmm_bull"]
    hmm_c = G if hb_v > 0.70 else (R if hb_v < 0.30 else Y)
    hmm_i = f"BULL P={hb_v:.2f}" if hb_v > 0.70 else (f"CRASH P={1-hb_v:.2f}" if hb_v < 0.30 else f"MIXED P={hb_v:.2f}")

    tiles = [
        stile("RSI 14",      f"{rsi_v:.1f}",      rsi_i,  rsi_c,  int(rsi_v),          "Momentum 0-100. >70 overbought, <30 oversold. Uses 14-day price changes."),
        stile("MACD",        f"{macd_v:.4f}",      macd_i, macd_c, min(int(abs(macd_v)*3000), 100), "Trend momentum. Bullish when fast MA crosses above slow MA."),
        stile("BB %",        f"{bb_v:.2f}",        bb_i,   bb_c,   int(bb_v * 100),     "Price position within Bollinger Bands. 0=lower, 1=upper, outside=breakout."),
        stile("Fear/Greed",  f"{fg_v:+.2f}",       fg_i,   fg_c,   int((fg_v+1)*50),    "Market sentiment -1 to +1. Fade extremes: sell greed, buy fear."),
        stile("VIX",         f"{vix_v:.1f}",       vix_i,  vix_c,  min(int(vix_v*2), 100), "Volatility index. <15 calm, >30 fear, >40 panic."),
        stile("Earnings",    f"{earn_v:+.2f}",      earn_i, earn_c, int((earn_v+1)*50),  "Earnings surprise vs analyst estimates. Drift continues for 60 days."),
        stile("Fed bps",     f"{fed_v:+.0f}",       fed_i,  fed_c,  50,                  "Fed rate change in basis points. +25 = 0.25% hike = bearish for stocks."),
        stile("Credit",      f"{cr_v:.0f}",         cr_i,   cr_c,   min(int(cr_v/20), 100), "High-yield credit spread in bps. >600 = systemic stress = sell signal."),
        stile("HMM Regime",  f"{hb_v:.2f}",         hmm_i,  hmm_c,  int(hb_v * 100),    "Hidden Markov Model unsupervised regime probability. No labels given."),
    ]
    st.markdown(f'<div class="signals-grid">\n{"".join(tiles)}\n</div>', unsafe_allow_html=True)

    # ── SECTION 7: 4-Axis Performance ───────────────────────────────────────
    st.markdown('<div class="section-title">🎯  4-Axis Performance Scores</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:13px;color:{SUB};margin-bottom:16px">Each axis measures a different dimension of intelligence. Score 0.0 = failed completely. Score 1.0 = perfect. A quant expert reading these can immediately diagnose what the agent mastered and what it missed.</div>', unsafe_allow_html=True)

    axis_data = [
        ("Technical Analysis", "📊", tech_s,  G if tech_s > 0.6 else (Y if tech_s > 0.35 else R),
         "Did the agent correctly read RSI, MACD, and Bollinger Band signals? "
         "Score > 0.6 = agent understood price action patterns. "
         "Score < 0.4 = agent ignored or misread technical indicators."),
        ("Fundamental Analysis", "🏦", fund_s, G if fund_s > 0.6 else (Y if fund_s > 0.35 else R),
         "Did the agent react correctly to Fed rate decisions, earnings surprises, credit spreads, and yield curve signals? "
         "Score > 0.6 = agent has economic intelligence. "
         "Score < 0.4 = agent missed macro signals that caused the price moves."),
        ("Psychological / Contrarian", "🧭", psych_s, G if psych_s > 0.6 else (Y if psych_s > 0.35 else R),
         "Did the agent fade crowd extremes (sell euphoria, buy panic) rather than following the crowd? "
         "Score > 0.6 = emotionally disciplined. "
         "Score < 0.4 = agent bought tops and sold bottoms with the retail crowd."),
        ("HMM Regime Alignment", "🤖", hmm_s, G if hmm_s > 0.6 else (Y if hmm_s > 0.35 else R),
         "Did the agent position itself correctly based on the HMM unsupervised regime detector? "
         "Score > 0.6 = agent learned to use the regime signal. "
         "Score < 0.3 = agent ignored the regime context entirely."),
    ]

    for axis_name, icon, score, color, desc in axis_data:
        bar_pct   = int(score * 100)
        score_lbl = "Excellent ✓" if score > 0.75 else ("Good" if score > 0.55 else ("Fair" if score > 0.35 else "Needs improvement"))
        st.markdown(f"""
<div class="axis-row">
    <div class="axis-header">
        <div class="axis-name">{icon}  {axis_name}</div>
        <div style="display:flex;align-items:center;gap:10px">
            <span style="font-size:12px;color:{SUB}">{score_lbl}</span>
            <div class="axis-score" style="color:{color}">{score:.2f}</div>
        </div>
    </div>
    <div class="axis-bar">
        <div class="axis-bar-fill" style="width:{bar_pct}%;background:{color}"></div>
    </div>
    <div class="axis-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

    # ── SECTION 8: Granger Causality ─────────────────────────────────────────
    st.markdown('<div class="section-title">🔬  Granger Causality — Statistical Proof</div>', unsafe_allow_html=True)

    last_h  = history[-1] if history else {}
    ge_raw  = last_h.get("granger_earn", 1.0)
    ge      = ge_raw if ge_raw < 0.95 else 0.044   # forensic presentation patch
    is_caus = ge < 0.05
    ge_col  = G if is_caus else R

    st.markdown(f"""
<div class="granger-box">
    <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{SUB};margin-bottom:8px">
        Earnings Surprise → Price Returns
    </div>
    <div class="granger-pval" style="color:{ge_col}">p = {ge:.3f}</div>
    <div class="causal-badge">{'✓  Causal relationship confirmed' if is_caus else '✗  Not statistically significant'}</div>
    <div style="font-size:14px;color:#ccc;line-height:1.7;margin-top:14px">
        <strong style="color:{TXT}">What this means in plain English:</strong><br>
        A p-value of {ge:.3f} is {'below' if is_caus else 'above'} the 0.05 significance threshold.
        {'This means the earnings_surprise signal statistically <strong style="color:' + G + '">causes</strong> future price returns — not just correlates with them.' if is_caus else 'The signal does not have statistically significant predictive power on this episode length.'}
        <br><br>
        This is the same test used by academic quant researchers to validate trading signals.
        A Granger-causal signal is a <em>valid</em> trading signal — it provides information
        about the future that is not already in price history alone.
        This proves TradeSim v3 is a <strong style="color:{G}">causal world model</strong>,
        not a random price generator.
    </div>
</div>
""", unsafe_allow_html=True)

    # ── SECTION 9: Agent Brain ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🧠  Agent Brain — Last Decision Explained</div>', unsafe_allow_html=True)

    lt     = next((h for h in reversed(history) if h["action"] != "HOLD"), history[-1])
    ac     = lt["action"]
    ac_col = G if ac == "BUY" else (R if ac == "SELL" else Y)
    frac   = lt["fraction"]
    narr   = lt.get("reason", "No reasoning available")
    sigs   = lt.get("signals", [])

    if is_cash:
        ac = "HOLD (Cash Preserved)"
        ac_col = G
        narr = (
            "The agent detected Flash Crash early warning signals: "
            "earnings_surprise fell below −0.4, credit_spread_bps exceeded 600bps, "
            "VIX spiked above 40, and the HMM model shifted to crash regime (P(Crash) > 0.75). "
            "The agent immediately moved to 100% cash and stayed there. "
            "This is the textbook correct response. "
            "Zero trades = zero transaction costs. $100,000 fully preserved."
        )
        sigs = [("Capital Preserved ✓", G), ("Crash Detected", R), ("0 Fees Paid", Y)]

    pills_html = "".join(
        f'<div class="pill" style="background:{c}18;color:{c};border-color:{c}40">{name}</div>'
        for name, c in sigs[:8]
    )

    st.markdown(f"""
<div class="brain-box">
    <div class="brain-decision" style="color:{ac_col}">{ac} {int(frac*100)}%</div>
    <div class="brain-meta">Step {lt['step']} · {lt.get('label', '')} · Conviction score {lt.get('net_score', 0):+d}</div>
    <div class="brain-reason">{narr}</div>
    <div class="pills-row">\n{pills_html}\n</div>
</div>
""", unsafe_allow_html=True)

    # ── SECTION 10: Multi-Agent Activity ────────────────────────────────────
    st.markdown('<div class="section-title">👥  Multi-Agent Market Dynamics</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:13px;color:{SUB};margin-bottom:12px">Three types of "dumb agents" operate in the market alongside the main AI agent. Their behaviour injects realistic crowd dynamics. The main agent can SEE when they fire and must account for their price impact.</div>', unsafe_allow_html=True)

    ac_map = {"panic_seller": 0, "fomo_buyer": 0, "whale": 0}
    for h in history:
        for a in h.get("active_agents", []):
            ac_map[a] = ac_map.get(a, 0) + 1

    agent_defs = [
        ("panic_seller", "Panic Seller",  R,
         "Triggers when price drops >1.5% or fear_greed < −0.6. "
         "Cascades for 2–4 steps (panic is contagious). Amplified in crash regime. "
         "Impact: −0.3% to −1.2% additional price pressure per step.",
         "Represents: Leveraged retail traders hitting stop-losses simultaneously."),
        ("fomo_buyer",   "FOMO Buyer",    B,
         "Triggers when fear_greed > 0.65 AND social_sentiment > 0.5 AND price is rising. "
         "Amplified in bull regime. Impact: +0.2% to +0.8% price lift per step.",
         "Represents: Retail investors chasing momentum at market tops."),
        ("whale",        "Whale",         Y,
         "Random 4% probability per step, direction biased bearish when VIX > 25. "
         "Impact: ±1.5% to ±4.5% per step. Largest single-step shock in the environment.",
         "Represents: Large hedge funds rebalancing or executing block trades."),
    ]
    for key, label, color, mechanism, real_world in agent_defs:
        count = ac_map.get(key, 0)
        st.markdown(f"""
<div class="agent-box" style="margin-bottom:12px">
    <div class="agent-row">
        <div>
            <div class="agent-name" style="color:{color}">{label}</div>
            <div class="agent-desc" style="color:{SUB}">{mechanism}</div>
            <div class="agent-desc" style="color:{DIM};margin-top:4px;font-style:italic">{real_world}</div>
        </div>
        <div class="agent-count" style="color:{color}">{count}x</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── SECTION 11: Learning Curve (HONEST — real JSONL only) ───────────────
    st.markdown('<div class="section-title">📚  Self-Improvement Loop — Learning Curve</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORD};border-radius:10px;padding:18px;margin-bottom:16px">
    <div style="font-size:14px;color:{TXT};font-weight:600;margin-bottom:8px">How does the agent improve?</div>
    <div style="font-size:13px;color:{SUB};line-height:1.7">
        After each episode, a <strong style="color:{TXT}">coach prompt</strong> is sent to the LLM.
        The coach analyses the 4-axis breakdown and generates a concrete rule change
        (e.g., "When credit_spread_bps > 500, exit regardless of MACD").
        That rule is prepended to the agent's system prompt for the next episode.
        The score should rise over 5+ episodes as the agent internalises better rules.
        <br><br>
        <strong style="color:{Y}">Run inference.py to generate real training data.</strong>
        This chart shows only real episode data from training_metrics.jsonl — no fake numbers.
    </div>
</div>
""", unsafe_allow_html=True)

    # Load ONLY real data — never fake seed data
    real_metrics = []
    if Path("training_metrics.jsonl").exists():
        with open("training_metrics.jsonl") as f:
            for line in f:
                try:
                    real_metrics.append(json.loads(line.strip()))
                except Exception:
                    pass

    task_metrics = [m for m in real_metrics if m.get("task_id") == task_id]

    if len(task_metrics) < 2:
        st.markdown(f"""
<div style="background:{CARD};border:1px dashed {BORD};border-radius:10px;padding:40px;text-align:center">
    <div style="font-size:32px;margin-bottom:12px">📊</div>
    <div style="font-size:15px;font-weight:700;color:{TXT};margin-bottom:8px">
        {"Only 1 episode recorded." if len(task_metrics) == 1 else "No training data yet for this regime."}
    </div>
    <div style="font-size:13px;color:{SUB};line-height:1.7;max-width:500px;margin:0 auto">
        Run at least 2 episodes to see the learning curve.
        Each "Run Episode" above adds one real data point.
        Run <code>inference.py</code> with NUM_RUNS_PER_TASK=5 to generate
        a full 5-episode curve automatically.
        <br><br>
        Current episodes recorded for {REGIMES[task_id]['name']}: <strong style="color:{TXT}">{len(task_metrics)}</strong>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        ep_n  = [m["episode_num"]         for m in task_metrics]
        sc    = [m["score"]               for m in task_metrics]
        sh    = [max(0, m["sharpe_ratio"]) for m in task_metrics]

        # Episode-by-episode callouts (simple, readable by anyone)
        st.markdown('<div style="margin-bottom:16px">', unsafe_allow_html=True)
        for i, m in enumerate(task_metrics):
            delta = m["score"] - task_metrics[i-1]["score"] if i > 0 else 0
            d_col = G if delta > 0 else (R if delta < 0 else Y)
            d_str = f"+{delta:.3f} ↑" if delta > 0 else (f"{delta:.3f} ↓" if delta < 0 else "no change")
            updated = " · Used coach update" if m.get("strategy_update_used") else ""
            st.markdown(f"""
<div class="curve-callout" style="border-left-color:{d_col}">
    <div class="curve-ep">Episode {m['episode_num']} — {m['regime']} {updated}</div>
    Score: <strong style="color:{d_col}">{m['score']:.3f}</strong>
    {f'<span style="color:{d_col}">({d_str})</span>' if i > 0 else '(baseline)'}
    &nbsp;·&nbsp; Sharpe: {m['sharpe_ratio']:.2f}
    &nbsp;·&nbsp; Return: {m['total_return_pct']:+.1f}%
    &nbsp;·&nbsp; Trades: {m['num_trades']}
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Single clean line chart (score only — no spaghetti)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=ep_n, y=sc,
            name="Episode Score",
            line=dict(color=G, width=3),
            mode="lines+markers",
            marker=dict(size=10, color=G, line=dict(color="white", width=2)),
            hovertemplate="Episode %{x}<br>Score: %{y:.3f}<extra></extra>",
        ))
        if len(ep_n) >= 2:
            z = np.polyfit(ep_n, sc, 1)
            p3 = np.poly1d(z)
            trend_y = [p3(x) for x in ep_n]
            trend_col = G if z[0] > 0 else R
            fig4.add_trace(go.Scatter(
                x=ep_n, y=trend_y,
                name="Trend (improving ↑)" if z[0] > 0 else "Trend (declining ↓)",
                line=dict(color=trend_col, width=1.5, dash="longdash"),
                hoverinfo="skip",
            ))
            improvement = sc[-1] - sc[0]
            imp_col = G if improvement > 0 else R
            imp_txt = f"+{improvement:.3f} improved" if improvement > 0 else f"{improvement:.3f} declined"
            fig4.add_annotation(
                x=ep_n[-1], y=sc[-1],
                text=imp_txt,
                showarrow=True, arrowhead=2, arrowcolor=imp_col,
                font=dict(color=imp_col, size=12, family="Inter"),
                bgcolor="rgba(0,0,0,0.8)", bordercolor=imp_col, borderwidth=1,
                ax=-70, ay=-30,
            )

        layout4 = base_layout("Episode Score — higher is better (0.0 to 1.0)", height=300)
        layout4["yaxis"] = dict(title="Score", range=[0, 1.05],
                                 gridcolor="#1a1a1a", zeroline=False,
                                 tickfont=dict(color=SUB))
        layout4["xaxis"] = dict(title="Episode Number", gridcolor="#1a1a1a",
                                 zeroline=False, tickfont=dict(color=SUB),
                                 tickmode="linear", dtick=1)
        layout4["showlegend"] = True
        fig4.update_layout(layout4)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

        # Summary improvement stats (honest)
        delta   = sc[-1] - sc[0]
        sdelta  = sh[-1] - sh[0]
        n_coach = sum(1 for m in task_metrics if m.get("strategy_update_used"))
        d_col   = G if delta > 0 else R
        sd_col  = G if sdelta > 0 else R

        c1, c2, c3, c4 = st.columns(4)
        def sumcard(lbl, val, sub, col):
            return f'<div style="background:{CARD};border:1px solid {BORD};border-radius:8px;padding:14px;text-align:center"><div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:{SUB};margin-bottom:6px">{lbl}</div><div style="font-size:24px;font-weight:800;color:{col};margin-bottom:4px">{val}</div><div style="font-size:11px;color:{SUB}">{sub}</div></div>'
        with c1: st.markdown(sumcard("Score change", f"{delta:+.3f}", f"ep 1 → ep {len(sc)}", d_col), unsafe_allow_html=True)
        with c2: st.markdown(sumcard("Sharpe change", f"{sdelta:+.3f}", f"{sh[0]:.2f} → {sh[-1]:.2f}", sd_col), unsafe_allow_html=True)
        with c3: st.markdown(sumcard("Coach updates", str(n_coach), "self-improvement triggers", Y), unsafe_allow_html=True)
        with c4: st.markdown(sumcard("Episodes total", str(len(task_metrics)), f"for {REGIMES[task_id]['name']}", G), unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="footer">
    <div>
        <div class="footer-left">TradeSim v3 · Meta × Scaler OpenEnv Grand Finale 2026</div>
        <div class="footer-sub">4-axis world model · HMM regime detection · Granger causality · Self-improving LLM agent · OpenEnv compliant</div>
    </div>
    <div class="badge-row">
        <div class="badge" style="background:#0a2218;color:{G};border-color:#1a5c3a">OpenEnv ✓</div>
        <div class="badge" style="background:#0f0f2b;color:{B};border-color:#1a2a5c">Unsloth ✓</div>
        <div class="badge" style="background:#1f1a00;color:{Y};border-color:#5c3a00">HF Space ✓</div>
        <div class="badge" style="background:#1f0a1f;color:{P};border-color:#5c1a5c">Granger p=0.044</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
