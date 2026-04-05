# TradeSim: Risk-Aware RL Trading Environment

TradeSim is a high-fidelity simulated trading environment designed for testing Reinforcement Learning agents. Unlike basic simulators, TradeSim prioritizes **Risk Management** and **Capital Preservation**.

## 🚀 Core Features
- **Synthetic Market Scenarios:** Includes Bull Trends, Choppy Ranges, and Flash Crashes for robust testing.
- **Advanced Reward Function:** Scores agents on PnL, Drawdown safety, and provide bonuses for trading reasoning.
- **Quant-Grade Metrics:** Built-in Sharpe Ratio and Max Drawdown tracking.
- **FastAPI Integration:** Ready for cloud deployment and automated evaluation.

## 🛠️ Technical Stack
- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Data:** NumPy, Pandas
- **AI Integration:** OpenAI/Groq compatible API logic
- **Deployment:** Dockerized for Hugging Face Spaces

## 📊 Evaluation Graders
The environment includes three specific task graders:
1. **Trend Capture:** Measures efficiency in riding upward momentum.
2. **Mean Reversion:** Evaluates performance in sideways, choppy markets.
3. **Crash Survival:** A "stress test" for detecting and exiting during sudden market drops.