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

Fine Tuned Weights: The Llama-3.1-8B LoRA adapter trained via Unsloth on synthetic Oracle data is available https://drive.google.com/file/d/1ZNo3ZOn1ytIc50fzdud3-2SY1eaK9AZr/view?usp=sharing.
## 📊 Baseline Scores (LLaMA-3.1-8B, zero-shot)
| Task | Score | Note |
|------|-------|------|
| Task 1 — Bull trend | 1.0000 | Perfect trend capture |
| Task 2 — Mean reversion | 0.4200 | (Updating with latest inference) |
| Task 3 — Flash crash | 0.3106 | Survived and executed recovery |

## 🛠️ Technical Stack
- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Data:** NumPy, Pandas
- **Deployment:** Docker (Debian Slim)