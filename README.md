# 📈 TradeSim v3: AI Quantitative Research Terminal

**A production-grade Reinforcement Learning environment designed to train, evaluate, and deploy financial AI agents using causal world models and zero-shot sim-to-real transfer.**

[![Live Application](https://img.shields.io/badge/Live-HuggingFace_Space-blue?style=for-the-badge)](https://huggingface.co/spaces/Themighty007/TradeSim-Terminal)

## 📖 Overview
Current LLM-based trading systems fail in production because they rely on historical data memorization. Training Reinforcement Learning directly on live financial data leads to catastrophic curve-fitting. 

**TradeSim v3** completely reimagines financial RL by rejecting 1D price arrays. Instead, it places the agent inside a **4-Axis Causal Sandbox**. The agent must synthesize multiple dimensions of an economy to survive, learning true risk management before it ever touches real capital.

## ✨ Core Features

### 1. The 4-Axis Causal World Model
Instead of simple price charts, agents navigate a rich, multi-dimensional environment:
* **Technical Axis:** Real-time momentum metrics (RSI, MACD, Bollinger Bands).
* **Fundamental Axis:** Macroeconomic gravity (Fed Rate changes, Yield Curve inversions, Credit Spreads).
* **Psychological Axis:** Market sentiment and contrarian indicators (VIX, Fear/Greed Index).
* **Regime Axis:** An Unsupervised Hidden Markov Model (HMM) that detects the mathematical probability of a "Crash" or "Bull" state purely from variance.

*Statistical Proof:* Validated via **Granger Causality tests**, ensuring synthetic fundamental signals causally predict future returns.

### 2. Hostile Sandbox & Anti-Reward Hacking
RL agents are naturally lazy and tend to exploit standard PnL reward functions. To prevent this, the simulation is engineered to be strictly hostile:
* **0.1% Transaction Friction:** Every trade bleeds capital, mathematically forcing the agent to learn discipline and avoid over-trading.
* **Super-linear Drawdown Penalties:** Severe punishments for risking the portfolio, enforcing strict capital preservation.
* **Multi-Agent Liquidity Shocks:** The agent must survive simulated "Whales" (large block trades) and "Panic Sellers" (cascading retail sell-offs) that disrupt trends.

### 3. Zero-Shot Sim-To-Real Transfer
Because our agent learns causal relationships rather than memorizing raw prices, it achieves successful **Zero-Shot Transfer**.
Through an air-gapped live adapter (`live_data_adapter.py`), the system pulls real-time Yahoo Finance data (e.g., SPY, TSLA), formats it into our 4-Axis JSON state, and allows the trained agent to execute its logic on unseen real-world data without retraining a single weight.

**Real-World Implementation Evaluator (`live_data_adapter.py`)**
We have recently integrated a dedicated real-world evaluator script. It is crucial to emphasize that **while the agent was entirely created and trained in a purely synthetic, simulated environment, this evaluator proves that it works effectively and seamlessly for the real world as well.** It successfully applies causal risk management to live market volatility.

## 🗂️ Project Structure
* `app.py`: The main interactive Streamlit terminal, which safely visualizes the agent's logic and handles live external data.
* `environment.py`: The core physics engine; a robust OpenEnv-compliant reinforcement learning environment.
* `models.py`: AI model logic, handling state processing and agent actions.
* `graders.py`: Evaluation scripts ensuring performance compliance.
* `portfolio.py`: Capital allocation, transaction friction, and risk management logic.
* `market_data.py`: Handles fetching, formatting, and synthesizing historical and real-time market data.
* `live_data_adapter.py`: The integrated real-world evaluator that bridges the synthetic agent to live market data.
* `reward.py`: The hostile reward function definitions punishing drawdowns and transaction costs.
* `Blog.md`: Comprehensive writeup and architectural breakdown.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed. The project relies on standard data science and ML libraries.
```bash
pip install -r requirements.txt
```

### Running the Terminal Locally
To launch the interactive research terminal:
```bash
streamlit run app.py
```

## 🧠 Model Training Pipeline
Training RL from scratch requires millions of compute hours. We utilized a highly efficient Teacher-Student Distillation pipeline:
1. **Data Generation:** Ran a mathematically optimal Oracle through the OpenEnv backend to generate perfect rollouts across all market regimes.
2. **Formatting:** Trajectories formatted using **Hugging Face TRL**.
3. **Quantization:** Data fed into a 4-bit `Llama-3.1-8B` base model using **Unsloth**, successfully compressing institutional risk-management logic into a lightweight LoRA adapter.

## 🔗 Useful Links
* **🔴 2-Minute Pitch Video:** [YouTube Link](https://youtu.be/JDOEUqHMQ6Q)
* **📝 Project Blog / Writeup:** [Blog.md](Blog.md)
* **🚀 Live HF Environment:** [HuggingFace Space](https://huggingface.co/spaces/Themighty007/TradeSim-Terminal)
* **💻 GitHub Repository:** [https://github.com/Themighty007/tradesim-openenv](https://github.com/Themighty007/tradesim-openenv)