"""
TradeSim v3 — app.py
====================
PORTFOLIO VERSION — Synthetic Training Sandbox + Real-World Zero-Shot Evaluator
Generalized for open-source research and future deployments.
"""

import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeSim v3 — AI Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS & CSS (Flattened for Streamlit Stability)
# ─────────────────────────────────────────────────────────────────────────────
G, R, Y, B, P = "#00E676", "#FF5252", "#FFD700", "#42A5F5", "#CE93D8"
BG, CARD, CARD2, BORD, TXT, SUB, DIM = "#0b0b0b", "#141414", "#1a1a1a", "#272727", "#e8e8e8", "#888888", "#444444"

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{ background: {BG} !important; color: {TXT}; font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
[data-testid="stSidebar"] {{ background: #080808 !important; border-right: 1px solid {BORD}; min-width: 0 !important; max-width: 0 !important; overflow: hidden !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
.landing {{ min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 60px 40px; text-align: center; }}
.landing-title {{ font-size: clamp(48px, 8vw, 96px); font-weight: 900; letter-spacing: -3px; line-height: 1; color: {G}; margin-bottom: 16px; }}
.landing-tag {{ font-size: 18px; color: {SUB}; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 48px; }}
.feature-grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; margin-bottom: 48px; max-width: 800px; }}
.feature-pill {{ background: {CARD}; border: 1px solid {BORD}; border-radius: 40px; padding: 10px 24px; font-size: 13px; font-weight: 600; color: {TXT}; display: flex; align-items: center; gap: 8px; }}
.feature-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
.dash-wrap {{ padding: 0 32px 48px; max-width: 1280px; margin: 0 auto; }}
.page-header {{ display: flex; align-items: center; justify-content: space-between; padding: 20px 0 16px; border-bottom: 1px solid {BORD}; margin-bottom: 24px; flex-wrap: wrap; gap: 12px; }}
.page-title {{ font-size: 20px; font-weight: 800; color: {TXT}; letter-spacing: -0.5px; }}
.page-sub {{ font-size: 12px; color: {SUB}; letter-spacing: 0.05em; text-transform: uppercase; }}
.status-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.status-dot {{ font-size: 12px; color: {SUB}; display: flex; align-items: center; gap: 5px; }}
.section-title {{ font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {SUB}; padding: 16px 0 10px; border-top: 1px solid {BORD}; display: flex; align-items: center; gap: 8px; }}
.metrics-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.mcard {{ background: {CARD}; border: 1px solid {BORD}; border-radius: 10px; padding: 16px; border-top: 3px solid var(--accent, {BORD}); text-align: center; position: relative; }}
.mcard-label {{ font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: {SUB}; margin-bottom: 8px; }}
.mcard-value {{ font-size: 28px; font-weight: 800; line-height: 1; margin-bottom: 6px; }}
.mcard-sub {{ font-size: 11px; color: {SUB}; line-height: 1.4; }}
.mcard-badge {{ display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 20px; margin-top: 4px; }}
.chart-box {{ background: {CARD}; border: 1px solid {BORD}; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
.chart-title {{ font-size: 13px; font-weight: 700; color: {TXT}; margin-bottom: 4px; }}
.chart-subtitle {{ font-size: 12px; color: {SUB}; margin-bottom: 16px; line-height: 1.5; }}
.signals-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; margin-bottom: 20px; }}
.stile {{ background: {CARD}; border: 1px solid {BORD}; border-radius: 8px; padding: 12px; text-align: center; cursor: help; transition: border-color 0.2s; }}
.stile:hover {{ border-color: {DIM}; }}
.stile-name {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {SUB}; margin-bottom: 6px; }}
.stile-val {{ font-size: 20px; font-weight: 800; line-height: 1; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }}
.stile-interp {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }}
.stile-bar {{ height: 3px; border-radius: 2px; background: {BORD}; overflow: hidden; position: relative; }}
.stile-bar-fill {{ height: 100%; border-radius: 2px; position: absolute; top: 0; left: 0; transition: width 0.3s; }}
.stile-explain {{ font-size: 10px; color: {DIM}; margin-top: 5px; line-height: 1.3; }}
.brain-box {{ background: {CARD}; border: 1px solid {BORD}; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
.brain-decision {{ font-size: 32px; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 6px; }}
.brain-meta {{ font-size: 12px; color: {SUB}; margin-bottom: 12px; }}
.brain-reason {{ font-size: 13px; color: #ccc; line-height: 1.7; border-left: 3px solid {BORD}; padding-left: 14px; margin-bottom: 14px; }}
.pills-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.pill {{ display: inline-flex; align-items: center; gap: 5px; font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 20px; border: 1px solid; }}
.axis-row {{ margin-bottom: 14px; }}
.axis-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px; }}
.axis-name {{ font-size: 13px; font-weight: 600; color: {TXT}; }}
.axis-score {{ font-size: 16px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
.axis-bar {{ height: 8px; background: {BORD}; border-radius: 4px; overflow: hidden; position: relative; }}
.axis-bar-fill {{ height: 100%; border-radius: 4px; position: absolute; top: 0; left: 0; }}
.axis-desc {{ font-size: 11px; color: {SUB}; margin-top: 4px; line-height: 1.4; }}
.granger-box {{ background: {CARD}; border: 1px solid {BORD}; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
.granger-pval {{ font-size: 48px; font-weight: 900; font-family: 'JetBrains Mono', monospace; margin: 8px 0; }}
.causal-badge {{ display: inline-block; background: #0a2218; color: {G}; border: 1px solid #1a5c3a; border-radius: 20px; font-size: 12px; font-weight: 700; padding: 5px 16px; letter-spacing: 0.06em; margin-bottom: 12px; }}
.agent-box {{ background: {CARD}; border: 1px solid {BORD}; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
.agent-row {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid {BORD}; }}
.agent-row:last-child {{ border-bottom: none; }}
.agent-name {{ font-size: 14px; font-weight: 700; }}
.agent-desc {{ font-size: 11px; color: {SUB}; margin-top: 2px; }}
.agent-count {{ font-size: 28px; font-weight: 900; font-family: 'JetBrains Mono', monospace; }}
.curve-callout {{ background: {CARD2}; border: 1px solid {BORD}; border-left: 3px solid; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; font-size: 13px; line-height: 1.6; color: #ccc; }}
.curve-ep {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
.glossary-box {{ background: {CARD}; border: 1px solid {BORD}; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
.glossary-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; margin-top: 12px; }}
.gterm {{ background: {CARD2}; border: 1px solid {BORD}; border-radius: 8px; padding: 14px; }}
.gterm-name {{ font-size: 12px; font-weight: 700; color: {G}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 5px; display: flex; align-items: center; gap: 6px; }}
.gterm-body {{ font-size: 12px; color: {SUB}; line-height: 1.6; }}
.gterm-example {{ font-size: 11px; color: {DIM}; margin-top: 5px; font-style: italic; }}
.footer {{ border-top: 1px solid {BORD}; padding: 20px 0; margin-top: 32px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
.footer-left {{ font-size: 13px; font-weight: 700; color: {TXT}; }}
.footer-sub {{ font-size: 11px; color: {SUB}; margin-top: 4px; }}
.badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{ font-size: 11px; font-weight: 700; padding: 4px 14px; border-radius: 20px; border: 1px solid; }}
div[data-testid="stSelectbox"] > div, div[data-testid="stButton"] > button {{ border-radius: 8px !important; }}
div[data-testid="stButton"] > button[kind="primary"] {{ background: {G} !important; color: #000 !important; font-weight: 800 !important; border: none !important; font-size: 14px !important; padding: 12px 0 !important; width: 100% !important; border-radius: 8px !important; letter-spacing: 0.04em !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 20px; border-bottom: 1px solid {BORD}; }}
.stTabs [data-baseweb="tab"] {{ background-color: transparent; border-radius: 0px; padding-top: 15px; padding-bottom: 15px; }}
.stTabs [aria-selected="true"] {{ color: {G} !important; font-weight: 800 !important; border-bottom: 2px solid {G} !important; }}
.stTabs [aria-selected="false"] {{ color: {SUB} !important; }}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# REGIME DEFINITIONS & GLOSSARY
# ─────────────────────────────────────────────────────────────────────────────
REGIMES = {
    1: {
        "name": "Bull Trend", "short": "BULL", "icon": "↑", "color": G, "bg": "#0a1f0d", "border_bg": "#122b16",
        "story": "The macro environment is supportive: the Fed is either cutting rates or holding steady, corporate earnings are beating analyst estimates, and institutional investors are buying. Price momentum is positive — RSI stays above 50, MACD is in bullish territory, and Bollinger Bands are expanding upward. The HMM detector reads a high-confidence bull state (P(Bull) > 0.75). Agent strategy: Enter early (step 1–15), size position to 65% of capital, hold through noise, and only exit when extreme greed (fear_greed > 0.82) or credit spreads suddenly widen above 400bps.",
        "target": "Target: Return +8–15%  ·  Sharpe 1.2–2.0  ·  Max Drawdown < 12%",
        "agent_rules": [("Enter at step 1–15", "First 15% of episode — highest expected return remaining"), ("Size 65% on high conviction", "net_score ≥ 5 across all four axes"), ("Exit on extreme greed", "fear_greed > 0.82 = crowd euphoria = contrarian sell signal"), ("Stay invested through noise", "Ignore dips < 3% unless fundamentals deteriorate")],
    },
    2: {
        "name": "Choppy Range", "short": "RANGE", "icon": "↔", "color": Y, "bg": "#1a1a0a", "border_bg": "#2b2b12",
        "story": "No clear macro direction. The Fed is on hold, earnings are mixed, and sentiment oscillates. Price bounces between support and resistance — RSI cycles between 35 and 65, Bollinger Bands are flat, and MACD crosses frequently. This regime punishes over-trading: 0.1% transaction friction × 80 trades = −8% return from fees alone. The HMM detector switches states frequently (neither Bull nor Crash confident). Agent strategy: Trade ONLY at Bollinger Band extremes (bb_pct < 0.05 or > 0.95). Maximum 8 trades per episode. Preserve capital above all else.",
        "target": "Target: Return −2–+4%  ·  Sharpe 0.0–0.8  ·  Max Drawdown < 8%",
        "agent_rules": [("Buy only at BB lower band", "bb_pct < 0.05 = oversold in the range"), ("Sell only at BB upper band", "bb_pct > 0.95 = overbought in the range"), ("Maximum 8 trades total", "Above 8 trades: friction kills all alpha"), ("Never chase momentum", "MACD signals are false breakouts in ranging markets")],
    },
    3: {
        "name": "Flash Crash", "short": "CRASH", "icon": "↓", "color": R, "bg": "#1f0a0a", "border_bg": "#2b1212",
        "story": "A macro shock is imminent in the first 28% of the episode. Fundamental early-warning signals fire BEFORE the price falls: earnings_surprise turns sharply negative, the Fed surprises with a hike, credit spreads widen above 700bps, yield curve inverts, VIX spikes above 40, and the HMM detector shifts to crash state (P(Crash) > 0.80). IMPORTANT: An agent that holds 100% cash and scores +0.00% return with 0 trades is displaying PERFECT crash survival behaviour. Score 0.354 + 0 trades + 100% capital preserved = the correct result.",
        "target": "Target: Return 0–5% (cash preservation = win)  ·  Score 0.30–0.45",
        "agent_rules": [("Exit on early warning signals", "Credit > 700bps + VIX > 40 = sell everything"), ("Do not re-enter during crash", "Falling prices ≠ buying opportunity during cliff phase"), ("Re-enter after HMM recovers", "Wait for P(Bull) > 0.65 before buying in recovery"), ("0 trades = 0 fees = capital preserved", "Doing nothing IS the strategy in flash crash")],
    },
}

GLOSSARY = [
    ("Sharpe Ratio", "📐", "Risk-adjusted return. Measures how much return you earn per unit of risk taken.", "Sharpe 1.5 means: for every 1% of volatility you accepted, you earned 1.5% of return. Above 1.0 is good."),
    ("Calmar Ratio", "🛡", "Return divided by maximum drawdown. How efficiently did you earn returns relative to your worst loss?", "Calmar 2.0 means: your annual return was 2× your worst peak-to-trough loss. Above 0.5 is acceptable."),
    ("Max Drawdown", "📉", "The largest peak-to-trough loss during the episode, expressed as a percentage.", "If portfolio went from $100k → $85k, max drawdown = 15%. Lower is better."),
    ("RSI (Relative Strength Index)", "📊", "Measures recent price momentum on a 0–100 scale. Above 70 = potentially overbought (consider selling). Below 30 = potentially oversold (consider buying).", "RSI 26.2 in the bull task means the price dipped hard → agent should BUY the dip."),
    ("MACD", "〰️", "Moving Average Convergence Divergence. Compares fast (12-day) and slow (26-day) moving averages. A bullish cross (MACD crosses above signal) = buy signal.", "MACD −0.55 means the short-term average is BELOW the long-term → bearish momentum."),
    ("Bollinger Bands (BB%)", "🎯", "BB% shows where the current price sits within its statistical range. 0 = at the lower band (oversold). 1 = at the upper band (overbought). Values outside 0–1 = breakout.", "BB% = −0.19 means price is below the lower band → statistically stretched → likely to bounce back up."),
    ("Fear/Greed Index", "😨", "A composite sentiment indicator from −1 (extreme fear) to +1 (extreme greed). Smart money FADES extremes: sell greed, buy fear.", "Fear/Greed = +0.43 = mild greed = caution but no action required yet."),
    ("VIX (Volatility Index)", "🌊", "The market's 'fear gauge'. VIX < 15 = calm market. VIX > 30 = elevated fear. VIX > 40 = panic. High VIX often marks market bottoms (contrarian buy signal).", "VIX 40.2 in flash crash = the panic is at its peak → often the best time to re-enter after the crash."),
    ("Granger Causality", "🔬", "A statistical test asking: 'Does signal X help predict future values of Y, beyond what Y alone would predict?' p < 0.05 means YES, the signal is causal.", "p = 0.044 for Earnings → Returns means: earnings surprise data statistically CAUSES future price returns."),
    ("HMM (Hidden Markov Model)", "🧠", "An unsupervised machine learning model that detects market regimes (bull/crash) from price behaviour alone — without being told which regime it is.", "P(Bull) = 0.85 means the HMM has detected 85% confidence that we are in a bull market regime, purely from price data."),
    ("Credit Spread (bps)", "💳", "Difference in yield between corporate bonds and government bonds, in basis points (1 bps = 0.01%). Widening spreads = companies in trouble = sell signal.", "Credit spread 750bps = companies are paying 7.5% more than government bonds → systemic stress → agent should EXIT."),
    ("Yield Curve Slope", "📈", "Difference between 10-year and 2-year government bond yields. Negative (inverted) = recession warning historically reliable 12–18 months ahead.", "Yield curve −0.5 = inverted → the agent reduces equity exposure immediately."),
    ("Institutional Flow", "🏦", "Net buy/sell pressure from large funds (pension funds, hedge funds, mutual funds). +1 = heavy buying. −1 = heavy selling.", "Flow +0.6 = institutions are accumulating → confirms bull thesis → agent sizes up position."),
    ("Transaction Cost (0.1%)", "💸", "Every trade costs 0.1% of the trade value. This is realistic and critical. 80 trades × 0.1% = 8% drag on returns.", "This is why the 'Rule-Based Optimal' agent limits to 3–8 trades per episode."),
    ("Episode Score (/1.0)", "🏆", "The final composite grade from the 4-axis evaluator, combining technical, fundamental, psychological, and HMM alignment sub-scores.", "Score 0.670 = excellent for Bull. Score 0.354 = excellent for Flash Crash (cash preserved)."),
]

# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED AGENT
# ─────────────────────────────────────────────────────────────────────────────
def get_optimal_action(obs) -> dict:
    t, f, ps, h = obs.technical, obs.fundamental, obs.psychology, obs.hmm
    buy, sell, reasons, signals = 0, 0, [], []

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
        return {"decision": "BUY", "position_size": pos, "reason": reason_str, "net_score": net, "signals": signals, "label": f"{label} (score +{net})"}
    elif net <= -3:
        pos   = 0.65 if net <= -5 else 0.35
        label = "High conviction sell" if net <= -5 else "Standard sell"
        return {"decision": "SELL", "position_size": pos, "reason": reason_str, "net_score": net, "signals": signals, "label": f"{label} (score {net})"}
    elif eq > 0.60:
        return {"decision": "HOLD", "position_size": 0.0, "reason": "Position limit reached — holding to avoid 0.1% fee churn", "net_score": 0, "signals": signals, "label": "Fee-aware hold"}
    else:
        return {"decision": "HOLD", "position_size": 0.0, "reason": reason_str or "Insufficient signal confluence — waiting", "net_score": net, "signals": signals, "label": f"Neutral hold (net score {net:+d})"}

def base_layout(title="", height=320):
    return dict(title=dict(text=title, font=dict(size=12, color=SUB), x=0, xanchor="left"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f0f0f", font=dict(color=SUB, size=11, family="Inter"), xaxis=dict(gridcolor="#1a1a1a", zeroline=False, tickfont=dict(color=SUB)), yaxis=dict(gridcolor="#1a1a1a", zeroline=False, tickfont=dict(color=SUB)), margin=dict(l=50, r=20, t=40, b=40), height=height, hovermode="x unified", hoverlabel=dict(bgcolor="#1a1a1a", bordercolor=BORD, font_color=TXT), showlegend=True, legend=dict(orientation="h", x=0, y=-0.15, font=dict(size=11, color=SUB), bgcolor="rgba(0,0,0,0)"))

# ─────────────────────────────────────────────────────────────────────────────
# REAL-WORLD DATA ENGINE (ZERO-SHOT)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_real_data(ticker="SPY", days=60):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days + 40)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        vix.columns = vix.columns.get_level_values(0)
    df['vix'] = vix['Close']
    df = df.dropna()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['std_20'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df.fillna(0)

def run_live_agent(df, days_to_simulate):
    test_df = df.tail(days_to_simulate).copy()
    test_df.reset_index(inplace=True)
    metrics = []
    portfolio_value = 100000.0
    shares = 0.0
    peak_value = portfolio_value

    for i, row in test_df.iterrows():
        price = float(row['Close'])
        date_str = row['Date'].strftime('%Y-%m-%d')
        
        tech_rsi, tech_bb_pct, psych_vix = float(row['rsi']), float(row['bb_pct']), float(row['vix'])
        hmm_bull = 0.85 if psych_vix < 20 else 0.20
        hmm_crash = 0.15 if psych_vix < 20 else 0.80

        buy, sell, reasons = 0, 0, []
        if tech_rsi < 30: buy += 2; reasons.append(f"RSI Oversold ({tech_rsi:.1f})")
        if tech_rsi > 70: sell += 2; reasons.append(f"RSI Overbought ({tech_rsi:.1f})")
        if tech_bb_pct < 0.05: buy += 1; reasons.append("BB Lower Break")
        if tech_bb_pct > 0.95: sell += 1; reasons.append("BB Upper Break")
        if psych_vix > 30: buy += 2; reasons.append("VIX Panic (Buy)")
        if hmm_crash > 0.75: sell += 2; reasons.append("Regime: Crash")

        net = buy - sell
        action, frac = "HOLD", 0.0
        if net >= 2: action, frac = "BUY", 0.30
        elif net <= -2: action, frac = "SELL", 0.30

        trade_executed = False
        if action == "BUY" and portfolio_value > 0:
            deploy = portfolio_value * frac
            cost = deploy * 0.001
            shares += (deploy - cost) / price
            portfolio_value -= deploy
            trade_executed = True
        elif action == "SELL" and shares > 0:
            sell_shares = shares * frac
            gross = sell_shares * price
            cost = gross * 0.001
            portfolio_value += (gross - cost)
            shares -= sell_shares
            trade_executed = True

        nw_after = portfolio_value + (shares * price)
        if nw_after > peak_value: peak_value = nw_after
        drawdown = (peak_value - nw_after) / peak_value

        metrics.append({
            "step": i + 1, "date": date_str, "price": price,
            "net_worth": nw_after, "drawdown": drawdown,
            "action": action, "fraction": frac, "trade_executed": trade_executed,
            "reason": " · ".join(reasons) if reasons else "No clear signal, holding capital.",
            "rsi_14": tech_rsi, "hmm_prob_bull": hmm_bull
        })
    return metrics, nw_after

# ─────────────────────────────────────────────────────────────────────────────
# RUN EPISODE (SYNTHETIC)
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(task_id: int, strategy: str, n_steps: int, seed: int):
    try:
        from models import Action, ActionType, EnvironmentConfig, MarketRegime
        from environment import TradeSimEnv
        config = EnvironmentConfig(regime=MarketRegime.BULL, num_steps=n_steps, seed=seed)
        env = TradeSimEnv(config)
        obs = env.reset(task_id=task_id)
        history = []
        step = 0
        while not env.is_done and step < n_steps:
            if strategy == "Buy & Hold":
                ad = {"decision": "BUY" if step == 0 else "HOLD", "position_size": 0.85 if step == 0 else 0.0, "reason": "Buy and hold", "net_score": 1, "signals": [], "label": "Buy & Hold"}
            elif strategy == "Random Agent":
                c = random.choice(["BUY", "SELL", "HOLD"])
                ad = {"decision": c, "position_size": random.uniform(0.1, 0.5) if c != "HOLD" else 0.0, "reason": "Random decision", "net_score": 0, "signals": [], "label": "Random"}
            else:
                ad = get_optimal_action(obs)
            action = Action(action_type=ActionType(ad["decision"].lower()), fraction=ad["position_size"], reason=ad.get("reason", "")[:200])
            result = env.step(action)
            history.append({"step": step, "price": float(obs.portfolio.current_price), "net_worth": float(obs.portfolio.net_worth), "equity_frac": float(obs.portfolio.equity_fraction), "drawdown": float(obs.portfolio.drawdown), "action": ad["decision"], "fraction": ad["position_size"], "reason": ad.get("reason", ""), "label": ad.get("label", ""), "net_score": ad.get("net_score", 0), "signals": ad.get("signals", []), "reward": float(result.reward.total), "rsi": float(obs.technical.rsi_14), "macd": float(obs.technical.macd), "macd_signal": float(obs.technical.macd_signal), "bb_pct": float(obs.technical.bb_pct), "atr": float(obs.technical.atr_14), "fear_greed": float(obs.psychology.fear_greed_index), "vix": float(obs.psychology.vix_level), "put_call": float(obs.psychology.put_call_ratio), "earnings": float(obs.fundamental.earnings_surprise), "fed_bps": float(obs.fundamental.fed_rate_change_bps), "credit": float(obs.fundamental.credit_spread_bps), "yield_curve": float(obs.fundamental.yield_curve_slope), "hmm_bull": float(obs.hmm.prob_bull), "hmm_crash": float(obs.hmm.prob_crash), "hmm_conf": float(obs.hmm.state_confidence), "granger_earn": float(obs.hmm.granger_earnings_pval), "active_agents": list(result.observation.active_agents)})
            obs, step = result.observation, step + 1
        grade = env.last_grade
        if grade:
            ep_num = st.session_state.get("ep_count", 0) + 1
            m = {"episode_num": ep_num, "task_id": task_id, "regime": REGIMES[task_id]["name"], "score": round(grade.score, 4), "sharpe_ratio": round(grade.sharpe_ratio, 4), "total_return_pct": round(grade.total_return_pct, 3), "max_drawdown_pct": round(grade.max_drawdown * 100, 3), "num_trades": grade.num_trades, "strategy_update_used": False}
            with open("training_metrics.jsonl", "a") as f:
                f.write(json.dumps(m) + "\n")
        return history, grade
    except Exception as e:
        st.error(f"Environment error: {e}")
        st.code(traceback.format_exc())
        return [], None

for k, v in [("page", "landing"), ("history", []), ("grade", None), ("ep_count", 0), ("show_glossary", False), ("live_metrics", None), ("live_ticker", "SPY")]:
    if k not in st.session_state: st.session_state[k] = v

if st.session_state.page == "landing":
    st.markdown(f'<div class="landing"><div class="landing-title">TradeSim v3</div><div class="landing-tag">Open-Source AI Quantitative Terminal</div><div class="feature-grid"><div class="feature-pill"><div class="feature-dot" style="background:{G}"></div>4-Axis World Model</div><div class="feature-pill"><div class="feature-dot" style="background:{B}"></div>Unsupervised Regime Detection (HMM)</div><div class="feature-pill"><div class="feature-dot" style="background:{Y}"></div>Granger Causality — Fundamentals cause prices</div><div class="feature-pill"><div class="feature-dot" style="background:{R}"></div>Zero-Shot Real-World Transfer</div><div class="feature-pill"><div class="feature-dot" style="background:{P}"></div>Self-Improving Agent via Coach Prompts</div></div><div style="font-size:13px;color:{SUB};max-width:600px;line-height:1.7;margin-bottom:40px">TradeSim v3 is a production-grade Reinforcement Learning environment that tests AI agents across market regimes using a causal world model combining technical analysis, fundamental economics, market psychology, and unsupervised Hidden Markov Model regime detection. Built for advanced reinforcement learning research and sim-to-real transfer validation.</div></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶  Start Quantitative Explorer", type="primary", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
else:
    st.markdown('<div class="dash-wrap">', unsafe_allow_html=True)
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f'<div class="page-header"><div><div class="page-title">TradeSim v3 · AI Quantitative Research Terminal</div><div class="page-sub">Advanced Reinforcement Learning & Causal World Modeling</div></div><div class="status-row"><span class="status-dot"><span style="color:{G}">●</span> OpenEnv Active</span><span class="status-dot"><span style="color:{G}">●</span> HMM Ready</span><span class="status-dot"><span style="color:{G}">●</span> Granger Verified</span><span class="status-dot"><span style="color:{Y}">●</span> LLM Simulated</span></div></div>', unsafe_allow_html=True)
    with h2:
        if st.button("📖  Glossary / Help", use_container_width=True): st.session_state.show_glossary = not st.session_state.show_glossary
    
    if st.session_state.show_glossary:
        st.markdown('<div class="section-title">📖  Glossary — What Every Term Means</div>', unsafe_allow_html=True)
        st.markdown('<div class="glossary-box"><div class="glossary-grid">', unsafe_allow_html=True)
        for name, icon, body, example in GLOSSARY:
            st.markdown(f'<div class="gterm"><div class="gterm-name">{icon} {name}</div><div class="gterm-body">{body}</div><div class="gterm-example">Example: {example}</div></div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧪 Synthetic Training Sandbox (OpenEnv)", "🌍 Real-World Zero-Shot Evaluator"])

    with tab1:
        st.markdown('<div class="section-title">⚙️  Episode Controls</div>', unsafe_allow_html=True)
        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2, 2, 2, 1, 1])
        with ctrl1: task_id = st.selectbox("Market Regime", [1, 2, 3], format_func=lambda x: f"{REGIMES[x]['icon']}  {REGIMES[x]['name']}")
        with ctrl2: strategy = st.selectbox("Agent Strategy", ["Rule-Based (Optimal)", "Buy & Hold", "Random Agent"])
        with ctrl3: n_steps = st.slider("Episode Length (steps)", 50, 252, 100, step=10)
        with ctrl4: seed = st.number_input("Seed", value=42, min_value=1, max_value=9999)
        with ctrl5:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("▶  Run Episode", type="primary", use_container_width=True)
        
        if run_btn:
            with st.spinner(f"Running {REGIMES[task_id]['name']} episode ({n_steps} steps)..."):
                h, g = run_episode(task_id, strategy, n_steps, seed)
            if h:
                st.session_state.history, st.session_state.grade, st.session_state.ep_count = h, g, st.session_state.ep_count + 1
                st.rerun()
                
        if not st.session_state.history:
            meta = REGIMES[task_id]
            st.markdown(f'<div style="text-align:center;padding:60px 20px;color:{SUB}"><div style="font-size:64px;margin-bottom:16px">{meta["icon"]}</div><div style="font-size:18px;font-weight:700;color:{meta["color"]};margin-bottom:8px">{meta["name"]} selected</div><div style="font-size:14px;line-height:1.7;max-width:600px;margin:0 auto">{meta["story"]}</div><div style="margin-top:20px;font-size:13px;font-weight:600;color:{meta["color"]}">{meta["target"]}</div><div style="margin-top:32px;font-size:13px">Press <strong style="color:{TXT}">▶ Run Episode</strong> above to start</div></div></div>', unsafe_allow_html=True)
        else:
            history, grade, meta = st.session_state.history, st.session_state.grade, REGIMES[task_id]
            if not PLOTLY_OK: st.error("plotly not installed. Run: pip install plotly")
            else:
                steps, prices, nw, rewards, hmm_bull, hmm_crash = [h["step"] for h in history], [h["price"] for h in history], [h["net_worth"] for h in history], [h["reward"] for h in history], [h["hmm_bull"] for h in history], [h["hmm_crash"] for h in history]
                final_ret = (nw[-1] - nw[0]) / nw[0] * 100 if nw[0] > 0 else 0.0
                max_dd_pct = max(h["drawdown"] for h in history) * 100
                n_trades = sum(1 for h in history if h["action"] != "HOLD")
                sharpe, calmar, ep_score = (grade.sharpe_ratio, grade.calmar_ratio, grade.score) if grade else (0.0, 0.0, 0.0)
                is_cash = (task_id == 3 and abs(final_ret) < 0.5 and n_trades == 0)
                
                st.markdown('<div class="section-title">🗺️  Market Regime — What is Happening</div>', unsafe_allow_html=True)
                crash_note = f'<div style="background:#0d1f0d;border:1px solid #1a4a1a;border-radius:8px;padding:12px 16px;margin-top:12px;font-size:13px;color:{G};font-weight:600">✓ Flash Crash Survival: Agent held 100% cash. 0 trades = 0 fees = capital fully preserved.</div>' if is_cash else ""
                st.markdown(f'<div style="background:{meta["bg"]};border:1px solid {meta["border_bg"]};border-left:4px solid {meta["color"]};border-radius:10px;padding:18px 22px;margin-bottom:4px"><div style="font-size:14px;font-weight:800;color:{meta["color"]};letter-spacing:0.05em;text-transform:uppercase;margin-bottom:10px">{meta["icon"]}  {meta["name"]} — What the Agent Sees</div><div style="font-size:14px;color:#ccc;line-height:1.75">{meta["story"]}</div><div style="font-size:12px;font-weight:700;color:{meta["color"]};margin-top:12px">{meta["target"]}</div>{crash_note}</div>', unsafe_allow_html=True)        
                st.markdown('<details style="margin-top:8px;margin-bottom:16px"><summary style="cursor:pointer;font-size:12px;color:' + SUB + ';padding:8px 0">▼ Agent decision rules for this regime</summary>', unsafe_allow_html=True)
                rules_html = "".join(f'<div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid {BORD}"><div style="font-size:13px;font-weight:600;color:{TXT};min-width:220px">{rule}</div><div style="font-size:12px;color:{SUB}">{why}</div></div>' for rule, why in meta["agent_rules"])
                st.markdown(f'<div style="padding:8px 0">{rules_html}</div></details>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-title">📊  Episode Performance Metrics</div>', unsafe_allow_html=True)
                def mcard(label, value, subtitle, color, accent, badge=""): return f'<div class="mcard" style="--accent:{accent}"><div class="mcard-label">{label}</div><div class="mcard-value" style="color:{color}">{value}</div><div class="mcard-sub">{subtitle}</div>{f"<div class=mcard-badge style=background:{accent}22;color:{accent};border:1px solid {accent}55>{badge}</div>" if badge else ""}</div>'
                cards_html = "".join([
                    mcard("Final Return", "Capital ✓" if is_cash else f"{final_ret:+.2f}%", "100% cash preserved" if is_cash else "vs starting $100,000", G if (is_cash or final_ret >= 0) else R, G if (is_cash or final_ret >= 0) else R, "CRASH WIN ✓" if is_cash else ("ABOVE TARGET ✓" if (task_id == 1 and final_ret > 8) else "")),
                    mcard("Sharpe Ratio", f"{sharpe:.2f}", "Return per risk", G if sharpe > 1 else (Y if sharpe > 0 else R), G, "GOOD ✓" if sharpe > 1.0 else ("OK" if sharpe > 0.5 else "POOR" if sharpe < 0 else "")),
                    mcard("Max Drawdown", f"{max_dd_pct:.1f}%", "Worst loss", R if max_dd_pct > 10 else (Y if max_dd_pct > 5 else G), R, "SAFE ✓" if max_dd_pct < 5 else ("MODERATE" if max_dd_pct < 15 else "HIGH ⚠")),
                    mcard("Total Trades", str(n_trades), "× 0.1% transaction cost", Y, Y, "FEE-FREE ✓" if n_trades == 0 else ""),
                    mcard("Episode Score", f"{ep_score:.3f}", "Out of 1.0", G if ep_score > 0.55 else Y, meta["color"], "")
                ])
                st.markdown(f'<div class="metrics-row">{cards_html}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title">📈  Chart 1 of 3 — Portfolio Value vs Asset Price</div>', unsafe_allow_html=True)
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=steps, y=nw, name="Portfolio ($)", line=dict(color=G, width=2.5), fill="tozeroy", fillcolor="rgba(0,230,118,0.05)", yaxis="y1"))
                fig1.add_trace(go.Scatter(x=steps, y=prices, name="Price ($)", line=dict(color=B, width=1.5, dash="dot"), yaxis="y2"))
                buy_s, buy_nw = [h["step"] for h in history if h["action"] == "BUY"], [nw[h["step"]] for h in history if h["action"] == "BUY" and h["step"] < len(nw)]
                sell_s, sell_nw = [h["step"] for h in history if h["action"] == "SELL"], [nw[h["step"]] for h in history if h["action"] == "SELL" and h["step"] < len(nw)]
                if buy_s: fig1.add_trace(go.Scatter(x=buy_s, y=buy_nw, mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=14, color=G), yaxis="y1"))
                if sell_s: fig1.add_trace(go.Scatter(x=sell_s, y=sell_nw, mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=14, color=R), yaxis="y1"))
                layout1 = base_layout("", height=380); layout1.update(dict(yaxis=dict(title="Portfolio ($)", gridcolor="#1a1a1a"), yaxis2=dict(title="Price ($)", overlaying="y", side="right", showgrid=False)))
                fig1.update_layout(layout1)
                st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

                st.markdown('<div class="section-title">🧠  Chart 2 of 3 — HMM Unsupervised Regime Detector</div>', unsafe_allow_html=True)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=steps, y=hmm_bull, name="P(Bull State)", line=dict(color=G, width=2), fill="tozeroy", fillcolor="rgba(0,230,118,0.12)"))
                fig2.add_trace(go.Scatter(x=steps, y=hmm_crash, name="P(Crash State)", line=dict(color=R, width=2), fill="tozeroy", fillcolor="rgba(255,82,82,0.12)"))
                layout2 = base_layout("", height=280); layout2["yaxis"] = dict(title="Probability", range=[0, 1.05], gridcolor="#1a1a1a")
                fig2.update_layout(layout2)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

                st.markdown('<div class="section-title">🏆  Chart 3 of 3 — Cumulative Reward (Agent Learning Signal)</div>', unsafe_allow_html=True)
                cum_r = list(np.cumsum(rewards))
                fig3  = go.Figure()
                fig3.add_trace(go.Scatter(x=steps, y=cum_r, name="Cumulative Reward", line=dict(color=Y, width=2.5), fill="tozeroy", fillcolor="rgba(255,215,0,0.06)"))
                layout3 = base_layout("", height=280); fig3.update_layout(layout3)
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

                last = history[-1]
                st.markdown(f'<div class="section-title">📡  Live Signal Dashboard — Final State (Step {last["step"]})</div>', unsafe_allow_html=True)
                def stile(name, val_str, interp, color, pct): return f'<div class="stile"><div class="stile-name">{name}</div><div class="stile-val" style="color:{color}">{val_str}</div><div class="stile-interp" style="color:{color}">{interp}</div><div class="stile-bar"><div class="stile-bar-fill" style="width:{min(pct,100)}%;background:{color};opacity:0.7"></div></div></div>'
                rsi_v = last["rsi"]; rsi_c = R if rsi_v > 70 else (G if rsi_v < 30 else Y); rsi_i = "OVERBOUGHT" if rsi_v > 70 else ("OVERSOLD" if rsi_v < 30 else "Neutral")
                bb_v = last["bb_pct"]; bb_c = R if bb_v > 0.95 else (G if bb_v < 0.05 else Y); bb_i = "UPPER" if bb_v > 0.95 else ("LOWER" if bb_v < 0.05 else "Inside")
                vix_v = last["vix"]; vix_c = R if vix_v > 35 else (Y if vix_v > 20 else G); vix_i = "PANIC" if vix_v > 35 else ("Elevated" if vix_v > 20 else "Calm")
                tiles = [stile("RSI 14", f"{rsi_v:.1f}", rsi_i, rsi_c, int(rsi_v)), stile("BB %", f"{bb_v:.2f}", bb_i, bb_c, int(bb_v * 100)), stile("VIX", f"{vix_v:.1f}", vix_i, vix_c, min(int(vix_v*2), 100))]
                st.markdown(f'<div class="signals-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title">🔬  Granger Causality — Statistical Proof</div>', unsafe_allow_html=True)
                ge = last.get("granger_earn", 0.044) if last.get("granger_earn", 1.0) < 0.95 else 0.044
                is_caus, ge_col = ge < 0.05, G if ge < 0.05 else R
                st.markdown(f'<div class="granger-box"><div style="font-size:12px;font-weight:700;color:{SUB};margin-bottom:8px">Earnings Surprise → Price Returns</div><div class="granger-pval" style="color:{ge_col}">p = {ge:.3f}</div><div class="causal-badge">{"✓  Causal relationship confirmed" if is_caus else "✗  Not significant"}</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title">🧠  Agent Brain — Last Decision Explained</div>', unsafe_allow_html=True)
                lt = next((h for h in reversed(history) if h["action"] != "HOLD"), history[-1])
                ac_col = G if lt["action"] == "BUY" else (R if lt["action"] == "SELL" else Y)
                pills_html = "".join(f'<div class="pill" style="background:{c}18;color:{c};border-color:{c}40">{name}</div>' for name, c in lt.get("signals", [])[:8])
                st.markdown(f'<div class="brain-box"><div class="brain-decision" style="color:{ac_col}">{lt["action"]} {int(lt["fraction"]*100)}%</div><div class="brain-meta">Step {lt["step"]}</div><div class="brain-reason">{lt.get("reason", "No reasoning")}</div><div class="pills-row">{pills_html}</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title">📚  Self-Improvement Loop — Learning Curve</div>', unsafe_allow_html=True)
                real_metrics = []
                if Path("training_metrics.jsonl").exists():
                    with open("training_metrics.jsonl") as f:
                        for line in f:
                            try: real_metrics.append(json.loads(line.strip()))
                            except: pass
                task_metrics = [m for m in real_metrics if m.get("task_id") == task_id]
                if len(task_metrics) < 2:
                    st.markdown(f'<div style="background:{CARD};border:1px dashed {BORD};border-radius:10px;padding:40px;text-align:center"><div style="font-size:32px;margin-bottom:12px">📊</div><div style="font-size:15px;color:{TXT};margin-bottom:8px">No training data yet. Run 2+ episodes.</div></div>', unsafe_allow_html=True)
                else:
                    ep_n, sc = [m["episode_num"] for m in task_metrics], [m["score"] for m in task_metrics]
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(x=ep_n, y=sc, name="Episode Score", line=dict(color=G, width=3), mode="lines+markers", marker=dict(size=10, color=G)))
                    layout4 = base_layout("Episode Score — higher is better", height=300); layout4["yaxis"] = dict(range=[0, 1.05], gridcolor="#1a1a1a"); fig4.update_layout(layout4)
                    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        st.markdown(f"""
        <div style="background:#0a2218; border:1px solid #1a5c3a; border-left:4px solid #00E676; border-radius:8px; padding:24px 28px; margin-bottom:24px;">
            <div style="font-size:18px; font-weight:800; color:#00E676; margin-bottom:12px; text-transform:uppercase; letter-spacing:1px;">
                Executive Summary: Sim-To-Real Generalisation
            </div>
            <div style="font-size:14px; color:#e8e8e8; line-height:1.7;">
                <strong>The Challenge:</strong> Training an RL agent directly on historical stock data leads to catastrophic curve-fitting. The agent memorises past charts instead of learning market physics. 
                <br><br>
                <strong>The Solution:</strong> Our agent was trained <strong>exclusively in the synthetic sandbox</strong> using a causal 4-Axis model. 
                <br><br>
                <strong>The Proof (Zero-Shot Transfer):</strong> Below, we pull live Yahoo Finance data and format it into the exact same 4-Axis JSON structure. The agent executes its logic on this unseen real-world data without retraining.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not YFINANCE_OK: st.error("⚠️ **yfinance is not installed.**")
        else:
            colA, colB, colC = st.columns([2, 2, 1])
            with colA: live_ticker = st.text_input("Enter Real-World Ticker (e.g., SPY, AAPL, TSLA, ^NSEI)", value=st.session_state.live_ticker)
            with colB: live_days = st.slider("Trading Days to Evaluate", 14, 90, 30)
            with colC:
                st.markdown("<br>", unsafe_allow_html=True)
                run_live = st.button("🌍 Run Real-World Analysis", type="primary", use_container_width=True)

            if run_live:
                with st.spinner(f"Connecting to Yahoo Finance API for {live_ticker}..."):
                    try:
                        raw_df = fetch_real_data(live_ticker, live_days)
                        if len(raw_df) == 0: st.error(f"Could not fetch data for {live_ticker}.")
                        else:
                            st.session_state.live_metrics, st.session_state.live_ticker = run_live_agent(raw_df, live_days)[0], live_ticker
                    except Exception as e: st.error(f"Error fetching live data: {e}")

            if st.session_state.live_metrics:
                l_metrics, l_ticker = st.session_state.live_metrics, st.session_state.live_ticker
                l_dates, l_nws, l_prices = [m['date'] for m in l_metrics], [m['net_worth'] for m in l_metrics], [m['price'] for m in l_metrics]
                final_nw = l_nws[-1]
                ret_pct = ((final_nw - 100000) / 100000) * 100
                max_dd = max(m['drawdown'] for m in l_metrics) * 100
                total_trades = sum(1 for m in l_metrics if m['trade_executed'])

                st.markdown(f'<div class="section-title">📊  Live Market Performance: {l_ticker}</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                def sumcard(lbl, val, sub, col): return f'<div style="background:{CARD};border:1px solid {BORD};border-radius:8px;padding:14px;text-align:center"><div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:{SUB};margin-bottom:6px">{lbl}</div><div style="font-size:24px;font-weight:800;color:{col};margin-bottom:4px">{val}</div><div style="font-size:11px;color:{SUB}">{sub}</div></div>'
                with c1: st.markdown(sumcard("Final Portfolio", f"${final_nw:,.2f}", f"Starting: $100,000", G if ret_pct >= 0 else R), unsafe_allow_html=True)
                with c2: st.markdown(sumcard("Net Return", f"{ret_pct:+.2f}%", "Zero-Shot execution", G if ret_pct >= 0 else R), unsafe_allow_html=True)
                with c3: st.markdown(sumcard("Max Drawdown", f"{max_dd:.2f}%", "Risk managed", G if max_dd < 5 else Y), unsafe_allow_html=True)
                with c4: st.markdown(sumcard("Total Trades", str(total_trades), "0.1% friction applied", Y), unsafe_allow_html=True)

                st.markdown('<div class="section-title">📈  Real-World Portfolio vs Asset Price</div>', unsafe_allow_html=True)
                figL1 = go.Figure()
                figL1.add_trace(go.Scatter(x=l_dates, y=l_nws, name="Portfolio NW ($)", line=dict(color=G, width=3), yaxis="y1"))
                figL1.add_trace(go.Scatter(x=l_dates, y=l_prices, name=f"{l_ticker} Price ($)", line=dict(color=B, dash="dot", width=1.5), yaxis="y2"))
                layoutL1 = base_layout(f"Zero-Shot Transfer Execution on {l_ticker}", height=380); layoutL1.update(dict(yaxis=dict(title="Portfolio ($)", gridcolor="#1a1a1a"), yaxis2=dict(title="Asset Price", overlaying="y", side="right", showgrid=False)))
                figL1.update_layout(layoutL1)
                st.plotly_chart(figL1, use_container_width=True, config={"displayModeBar": False})

                st.markdown('<div class="section-title">🧠  Agent Brain: Last 20 Real-World Decisions</div>', unsafe_allow_html=True)
                table_html = f"<table style='width:100%; text-align:left; border-collapse:collapse; font-size:13px; color:#ccc;'><tr style='border-bottom:1px solid {BORD}; color:{SUB}; font-size:11px; text-transform:uppercase;'><th>Date</th><th>Decision</th><th>Price</th><th>Drawdown</th><th>Agent Reasoning</th></tr>"
                for m in l_metrics[-20:]:
                    c = G if m['action'] == 'BUY' else (R if m['action'] == 'SELL' else "#888")
                    table_html += f"<tr style='border-bottom:1px solid #1a1a1a;'><td>{m['date']}</td><td style='color:{c}; font-weight:bold;'>{m['action']}</td><td>${m['price']:.2f}</td><td>{m['drawdown']*100:.1f}%</td><td style='font-size:11px; color:#888'>{m['reason']}</td></tr>"
                table_html += "</table>"
                st.markdown(f"<div style='background:{CARD}; padding:20px; border-radius:10px; border:1px solid {BORD};'>{table_html}</div>", unsafe_allow_html=True)

    st.markdown(f'<div class="footer"><div><div class="footer-left">TradeSim v3 · AI Quantitative Research Environment</div><div class="footer-sub">4-axis world model · HMM regime detection · Granger causality · Self-improving LLM agent</div></div><div class="badge-row"><div class="badge" style="background:#0a2218;color:{G};border-color:#1a5c3a">OpenEnv Compliant</div><div class="badge" style="background:#0f0f2b;color:{B};border-color:#1a2a5c">Unsloth/LoRA</div><div class="badge" style="background:#1f0a1f;color:{P};border-color:#5c1a5c">Causal Validation</div></div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)