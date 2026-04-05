---
title: TradeSim
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TradeSim: Risk-Aware RL Trading Environment

TradeSim is a high-fidelity simulated trading environment designed for testing Reinforcement Learning agents. Unlike basic simulators, TradeSim prioritizes **Risk Management** and **Capital Preservation**.

## 🚀 Core Features
- **Synthetic Market Scenarios:** Includes Bull Trends, Choppy Ranges, and Flash Crashes.
- **Advanced Reward Function:** Scores agents on PnL, Drawdown safety, and reasoning.
- **Quant-Grade Metrics:** Built-in Sharpe Ratio and Max Drawdown tracking.
- **FastAPI Integration:** Ready for cloud deployment (Port 7860).

## 📊 Baseline Results (Llama-3-8B)
- **Task 1 (Bull Trend):** 1.0000 (Perfect Efficiency)
- **Task 2 (Mean Reversion):** 0.0000 (High Challenge)
- **Task 3 (Flash Crash):** 0.1306 (Survival Mode)

## 🛠️ Technical Stack
- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Data:** NumPy, Pandas
- **Deployment:** Docker (Debian Slim)