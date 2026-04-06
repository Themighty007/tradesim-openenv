"""
TradeSim — inference.py
=======================
Strict OpenEnv-compliant inference script.
Emits ONLY [START], [STEP], and [END] to stdout for automated validation.
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
# Logging (Strictly routed to STDERR to protect STDOUT validator)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr # CRITICAL: Keeps logs out of the validator's way
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
BENCHMARK_NAME    = "tradesim-openenv"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a disciplined quantitative trading agent operating in a simulated market.

Your goal is NOT to maximise raw profit — it is to make risk-adjusted, disciplined decisions.

PRINCIPLES:
1. SURVIVE FIRST — never risk more than 70% of portfolio in equities at once.
2. TREND AWARENESS — identify the market regime from price history before acting.
3. DRAWDOWN CONTROL — if you are down >10% from peak, reduce exposure.
4. COST CONSCIOUSNESS — every trade costs 0.1%. Over-trading destroys alpha.
5. REASONING REQUIRED — always provide a reason for your decision.
6. LOGICAL CONSISTENCY — Review your "memory". Do NOT flip-flop unless the market has moved violently against you.

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
# MANDATORY STDOUT FORMATTERS
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(obs: Observation, task_description: str, regime_hint: str, step: int, action_history: list[str]) -> str:
    prices = obs.window.raw_prices
    n = len(prices)
    
    p = round(prices[-1], 2)
    m5 = round(((prices[-1] - prices[-5]) / prices[-5] * 100), 2) if n >= 5 else 0.0
    v = round(float((sum(r**2 for r in obs.window.returns) / len(obs.window.returns)) ** 0.5 * 100), 2) if obs.window.returns else 0.0

    obs_dict = {
        "tsk": task_description[:15] + "...", 
        "mkt": {
            "p": p, 
            "m5": m5, 
            "v": v, 
            "h10": round(max(prices[-10:]), 2) if n >= 10 else p,
            "l10": round(min(prices[-10:]), 2) if n >= 10 else p,
        },
        "pf": {
            "c": round(obs.portfolio.cash, 0),
            "s": round(obs.portfolio.shares_held, 2),
            "nw": round(obs.portfolio.net_worth, 0),
            "eq": round(obs.portfolio.equity_fraction, 2),
            "dd": round(obs.portfolio.drawdown * 100, 1),
        },
        "t": round(obs.time_left * 100, 0),
        "mem": action_history[-3:] if action_history else ["None"]
    }
    return json.dumps(obs_dict, separators=(',', ':'))

# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------
def parse_action(response_text: str) -> Action:
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
async def call_llm(client: AsyncOpenAI, prompt: str, step: int) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=150,
                temperature=0.1,
                timeout=20.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content or ""

        except RateLimitError as e:
            wait = 15.0 + (10.0 * attempt) 
            log.warning(f"Rate limited on step {step}. Throttling for {wait}s...")
            await asyncio.sleep(wait)

        except Exception as e:
            log.error(f"API error on step {step}, attempt {attempt+1}: {e}")
            if attempt == MAX_RETRIES - 1:
                return '{"decision": "HOLD", "position_size": 0.0, "reasoning": "API error fallback"}'
            await asyncio.sleep(RETRY_BASE_DELAY)

    return '{"decision": "HOLD", "position_size": 0.0, "reasoning": "Max retries exceeded"}'

# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------
async def run_task(task_id: int, client: AsyncOpenAI, task_timeout: float = PER_TASK_TIMEOUT_S) -> float:
    task_start = time.time()
    log.info(f"Preparing Task {task_id}...")

    env = TradeSimEnv()
    env._base_config = EnvironmentConfig(regime="bull", num_steps=40)
    
    obs = env.reset(task_id=task_id)
    task_desc = env.task_description
    regime_hint = env.regime_hint
    
    # 1. EMIT START LOG
    task_name = f"task_{task_id}_{regime_hint.replace(' ', '_')}"
    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    step = 1
    final_score = 0.0
    action_history = []
    rewards_history = []
    success = False

    while True:
        elapsed = time.time() - task_start
        if elapsed > task_timeout:
            log.warning(f"Task {task_id} timed out at step {step} ({elapsed:.0f}s). Forcing done.")
            log_step(step=step, action="hold(0.0)", reward=0.0, done=True, error="Timeout")
            break

        prompt = build_prompt(obs, task_desc, regime_hint, step, action_history)
        
        raw_response = await call_llm(client, prompt, step) 
        await asyncio.sleep(2.5) # The rate limit pace car
        
        action = parse_action(raw_response)
        
        if action.action_type != ActionType.HOLD:
            action_history.append(f"{action.action_type.value} ({action.fraction:.2f})")

        # Step the environment
        try:
            result = env.step(action)
            error_str = None
        except Exception as e:
            error_str = str(e)
            result = None

        # Format details for the step log
        action_str = f"{action.action_type.value.lower()}({action.fraction:.2f})"
        
        # Handle rewards safely
        if result is not None:
            reward_val = result.reward.total if hasattr(result.reward, 'total') else float(result.reward)
            done = result.done
            obs = result.observation
            if done:
                final_score = result.info.get("episode_score", 0.0) or 0.0
                success = final_score > 0.0
        else:
            reward_val = 0.0
            done = True
            
        rewards_history.append(reward_val)

        # 2. EMIT STEP LOG
        log_step(step=step, action=action_str, reward=reward_val, done=done, error=error_str)

        if done:
            break

        step += 1

    # 3. EMIT END LOG
    log_end(success=success, steps=step, score=final_score, rewards=rewards_history)

    return task_id, final_score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main():
    if not HF_TOKEN:
        log.error("HF_TOKEN environment variable not set.")
        sys.exit(1)

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, max_retries=0)
    
    # Run sequentially
    for task_id in [1, 2, 3]:
        await run_task(task_id, client, task_timeout=900.0)

if __name__ == "__main__":
    asyncio.run(async_main())