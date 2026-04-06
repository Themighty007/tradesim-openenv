"""
TradeSim — inference.py
=======================
Runs an LLM agent against the TradeSim environment.

Reads credentials from environment variables:
  API_BASE_URL  : OpenAI-compatible API endpoint
  MODEL_NAME    : Model identifier (e.g. "gpt-4o-mini", "meta-llama/...")
  HF_TOKEN      : Auth token (HuggingFace or OpenAI key)

Usage:
  API_BASE_URL=https://api.openai.com/v1 \\
  MODEL_NAME=gpt-4o-mini \\
  HF_TOKEN=sk-... \\
  python inference.py

Output:
  Task 1 (Bull Market):  0.72
  Task 2 (Choppy Range): 0.41
  Task 3 (Flash Crash):  0.23

Design:
  • Single-turn prompting per step — agent sees full observation as JSON
  • Structured output parsing with fallback to HOLD on any parse failure
  • Hard 20-minute total timeout with per-task sub-limits
  • Exponential backoff on API errors (max 3 retries)
  • Comprehensive logging for debugging agent behaviour
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import asyncio
from typing import Optional

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI, APIError, RateLimitError

from environment import TradeSimEnv
from models import Action, ActionType, EnvironmentConfig, Observation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tradesim.inference")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "gpt-4o-mini")
HF_TOKEN      = os.environ.get("HF_TOKEN",      os.environ.get("OPENAI_API_KEY", ""))

TOTAL_TIMEOUT_S   = 18 * 60   # 18 minutes hard cap (< 20 min requirement)
PER_TASK_TIMEOUT_S = 5 * 60   # 5 minutes per task
MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 2.0       # seconds

# ---------------------------------------------------------------------------
# System prompt (philosophy embedded)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a disciplined quantitative trading agent operating in a simulated market.

Your goal is NOT to maximise raw profit — it is to make risk-adjusted, disciplined decisions.

PRINCIPLES:
1. SURVIVE FIRST — never risk more than 70% of portfolio in equities at once.
2. TREND AWARENESS — identify the market regime from price history before acting.
3. DRAWDOWN CONTROL — if you are down >10% from peak, reduce exposure.
4. COST CONSCIOUSNESS — every trade costs 0.1%. Over-trading destroys alpha.
5. REASONING REQUIRED — always provide a reason for your decision.
6. LOGICAL CONSISTENCY — Review your "memory". Do NOT flip-flop (e.g., BUY then immediately SELL) unless the market has moved violently against you.

You will receive an observation as JSON. You must respond with ONLY a JSON object.

Response format:
{
  "decision": "BUY" | "SELL" | "HOLD",
  "position_size": <float 0.0 to 1.0>,
  "stop_loss_pct": <float, default 0.05>,
  "reasoning": "<your analysis, min 20 chars>"
}

Rules:
- decision is BUY, SELL, or HOLD (uppercase, exact spelling)
- position_size is fraction of available cash (BUY) or shares (SELL) to trade
- For HOLD: set position_size to 0.0
- reasoning must explain WHY you are making this decision based on the data
- Do NOT include any text outside the JSON object
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(obs: Observation, task_description: str, regime_hint: str, step: int, action_history: list[str]) -> str:
    prices = obs.window.raw_prices
    n = len(prices)
    
    # 1. Extreme Truncation (Rounding to 2-3 decimals max)
    p = round(prices[-1], 2)
    m5 = round(((prices[-1] - prices[-5]) / prices[-5] * 100), 2) if n >= 5 else 0.0
    v = round(float((sum(r**2 for r in obs.window.returns) / len(obs.window.returns)) ** 0.5 * 100), 2) if obs.window.returns else 0.0

    # 2. Key Minification (Abbreviating keys)
    obs_dict = {
        "tsk": task_description[:15] + "...", # Only send the start of the task string
        "mkt": {
            "p": p,       # Price
            "m5": m5,     # 5-bar momentum
            "v": v,       # Volatility
            "h10": round(max(prices[-10:]), 2) if n >= 10 else p,
            "l10": round(min(prices[-10:]), 2) if n >= 10 else p,
        },
        "pf": {           # Portfolio
            "c": round(obs.portfolio.cash, 0),       # Cash (no decimals needed)
            "s": round(obs.portfolio.shares_held, 2),# Shares
            "nw": round(obs.portfolio.net_worth, 0), # Net worth
            "eq": round(obs.portfolio.equity_fraction, 2), # Equity fraction
            "dd": round(obs.portfolio.drawdown * 100, 1),  # Drawdown %
        },
        "t": round(obs.time_left * 100, 0), # Time left %
        "mem": action_history[-3:] if action_history else ["None"]
    }

    # 3. Whitespace Annihilation
    return json.dumps(obs_dict, separators=(',', ':'))

# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Action:
    """
    Parse the LLM's JSON response into a validated Action.

    Fails gracefully — any parse error returns HOLD with explanation.
    """
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse failed: {e}. Raw: {response_text[:200]}")
        return Action.hold(reason=f"Parse error — defaulting to HOLD: {str(e)[:100]}")

    decision_raw = str(data.get("decision", "HOLD")).upper().strip()
    decision_map = {"BUY": ActionType.BUY, "SELL": ActionType.SELL, "HOLD": ActionType.HOLD}
    action_type = decision_map.get(decision_raw, ActionType.HOLD)

    try:
        position_size = float(data.get("position_size", 0.0))
        position_size = max(0.0, min(1.0, position_size))
    except (TypeError, ValueError):
        position_size = 0.0

    reasoning = str(data.get("reasoning", "No reasoning provided"))[:512]

    return Action(
        action_type=action_type,
        fraction=position_size,
        reason=reasoning,
    )


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

async def call_llm(
    client: OpenAI,
    prompt: str,
    step: int,
) -> str:
    """
    Call the LLM with exponential backoff retry.

    Returns the text content of the response.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=150,           # Lowered from 300 since we compressed the keys
                temperature=0.1,          # Lowered to 0.1 for maximum mathematical logic
                timeout=20.0,             # Fast fail
                response_format={"type": "json_object"} # <--- THE ELITE UPGRADE
            
            )
            return response.choices[0].message.content or ""

        except RateLimitError as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning(f"Rate limited on step {step}, attempt {attempt+1}. Waiting {wait}s...")
            await asyncio.sleep(wait)

        except APIError as e:
            log.error(f"API error on step {step}, attempt {attempt+1}: {e}")
            if attempt == MAX_RETRIES - 1:
                return '{"decision": "HOLD", "position_size": 0.0, "reasoning": "API error fallback"}'
            await asyncio.sleep(RETRY_BASE_DELAY)

    return '{"decision": "HOLD", "position_size": 0.0, "reasoning": "Max retries exceeded"}'


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------

async def run_task(
    env: TradeSimEnv,
    client: OpenAI,
    task_id: int,
    task_timeout: float = PER_TASK_TIMEOUT_S,
) -> float:
    """
    Run a complete episode for the given task.

    Returns the final grader score (0.0 – 1.0).
    """
    task_start = time.time()
    log.info(f"Starting Task {task_id}...")

    env = TradeSimEnv()
    obs = env.reset(task_id=task_id)
    task_desc = env.task_description
    regime_hint = env.regime_hint

    step = 0
    final_score = 0.0
    total_reward = 0.0
    action_history = []

    while True:
        elapsed = time.time() - task_start
        if elapsed > task_timeout:
            log.warning(f"Task {task_id} timed out at step {step} ({elapsed:.0f}s). Forcing done.")
            break

        # <--- NEW: Pass action_history into build_prompt
        prompt = build_prompt(obs, task_desc, regime_hint, step, action_history)

        t0 = time.time()
        
        # NOTE: If you haven't implemented Async yet, remove the 'await' keyword below.
        raw_response = await call_llm(client, prompt, step) 
        
        api_latency = time.time() - t0

        action = parse_action(raw_response)
        
        # <--- NEW: Record the agent's decision into memory
        if action.action_type != ActionType.HOLD:
            action_history.append(f"{action.action_type.value} ({action.fraction:.2f}) @ ${obs.portfolio.current_price:.2f}")

        result = env.step(action)
        total_reward += result.reward.total

        if step % 20 == 0 or result.done or action.action_type != ActionType.HOLD:
            log.info(
                f"  Task {task_id} | Step {step:3d} | "
                f"Price ${result.observation.portfolio.current_price:.2f} | "
                f"NW ${result.observation.portfolio.net_worth:,.0f} | "
                f"Action: {action.action_type.value}({action.fraction:.2f}) | "
                f"Reward: {result.reward.total:+.4f} | "
                f"API: {api_latency:.2f}s"
            )

        obs = result.observation

        if result.done:
            final_score = result.info.get("episode_score", 0.0) or 0.0
            log.info(
                f"Task {task_id} complete: "
                f"score={final_score:.4f}, "
                f"total_reward={total_reward:.4f}, "
                f"steps={step+1}, "
                f"duration={time.time()-task_start:.1f}s"
            )
            break

        step += 1

    return task_id, final_score  # Ensure it returns task_id if you are using asyncio.gather
    


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.")
        sys.exit(1)

    print("=" * 65)
    print("TradeSim Inference Runner (ASYNC MODE)")
    print(f"  Model      : {MODEL_NAME}")
    print("=" * 65)

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    global_start = time.time()

    # Launch all 3 tasks simultaneously and wait for them all to finish!
    results = await asyncio.gather(
        run_task(client, 1),
        run_task(client, 2),
        run_task(client, 3)
    )

    # Convert results list of tuples [(1, score), (2, score), (3, score)] into a dictionary
    scores = {task_id: score for task_id, score in results}

    print("\n" + "=" * 65)
    print("FINAL SCORES")
    print("=" * 65)
    print(f"Task 1 (Bull Market) : {scores.get(1, 0.0):.4f}  [expected: 0.65–0.80]")
    print(f"Task 2 (Choppy Range): {scores.get(2, 0.0):.4f}  [expected: 0.35–0.55]")
    print(f"Task 3 (Flash Crash) : {scores.get(3, 0.0):.4f}  [expected: 0.15–0.35]")
    print(f"─" * 65)
    print(f"Total runtime        : {time.time() - global_start:.1f}s")
    print("=" * 65)

# The standard Python entry point now calls asyncio.run()
if __name__ == "__main__":
    asyncio.run(async_main())

