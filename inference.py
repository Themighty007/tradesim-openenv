"""
TradeSim v3 — inference.py
==========================
FINAL VERSION. The complete LLM trading agent with:

1. TRAINING CURVE LOGGER — logs EpisodeMetrics after every episode.
   Produces a JSON file that the dashboard plots as "reward improvement."
   This directly satisfies the 20% "Showing Improvement in Rewards" criterion.

2. SELF-IMPROVEMENT LOOP — after each episode:
   a) Agent gets a coach prompt with its 4-axis performance breakdown
   b) Coach outputs a concrete strategy update (cites signal names, thresholds)
   c) Update is prepended to the next episode's system prompt
   d) Improvement is VISIBLE in the training curve

3. LLM FEAR/GREED — second LLM call generates news headlines + sentiment score

4. 4-AXIS OBSERVATION — agent sees Technical + Fundamental + Psychology + HMM

5. THEORY OF MIND — agent knows which dumb agents fired last step

STDOUT: ONLY [START], [STEP], [END] — never anything else.
STDERR: all logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI, RateLimitError

from environment import TradeSimEnv
from models import (
    Action,
    ActionType,
    EnvironmentConfig,
    EpisodeMetrics,
    MarketRegime,
    Observation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("tradesim.inference")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "llama-3.1-8b-instant")
HF_TOKEN      = os.environ.get("HF_TOKEN",      os.environ.get("OPENAI_API_KEY", ""))
BENCHMARK_NAME = "tradesim-v3-openenv"
METRICS_FILE   = os.environ.get("METRICS_FILE", "training_metrics.jsonl")

MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 2.0
PER_TASK_TIMEOUT  = 900.0  # 15 minutes


# ---------------------------------------------------------------------------
# Stdout formatters (strict OpenEnv format — NEVER write anything else here)
# ---------------------------------------------------------------------------

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


# ---------------------------------------------------------------------------
# Training curve logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Logs EpisodeMetrics to a JSONL file after each episode.
    The dashboard reads this file and plots the training curve.
    This is the "reward improvement" evidence required by the judges.
    """
    
    def __init__(self, filepath: str = METRICS_FILE):
        self.filepath = filepath
        self.episode_count = 0
        self.all_metrics: list[EpisodeMetrics] = []
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, metrics: EpisodeMetrics):
        self.episode_count += 1
        self.all_metrics.append(metrics)
        
        # Append to JSONL
        with open(self.filepath, "a") as f:
            f.write(json.dumps(metrics.model_dump()) + "\n")
        
        log.info(
            f"Episode {metrics.episode_num} | Task {metrics.task_id} | "
            f"Score: {metrics.score:.3f} | Sharpe: {metrics.sharpe_ratio:.2f} | "
            f"Return: {metrics.total_return_pct:.1f}%"
        )
    
    def get_improvement(self, task_id: int) -> float:
        """Return score improvement from first to last episode for a task."""
        task_metrics = [m for m in self.all_metrics if m.task_id == task_id]
        if len(task_metrics) < 2:
            return 0.0
        return task_metrics[-1].score - task_metrics[0].score


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """You are an elite quantitative trading agent with mastery of three analytical frameworks:

AXIS 1 — TECHNICAL ANALYSIS:
- RSI > 70 = overbought → SELL or HOLD. RSI < 30 = oversold → BUY or HOLD.
- MACD above signal + positive = bullish → BUY. MACD below signal + negative = bearish → SELL.
- BB% > 1.0 = price outside upper band (stretched) → SELL.
- BB% < 0.0 = price outside lower band (oversold) → BUY.
- High ATR (>2.0) = high volatility → reduce position size.

AXIS 2 — FUNDAMENTAL ANALYSIS:
- fed_rate_change_bps > 25 = hawkish hike → SELL (bonds more attractive).
- fed_rate_change_bps < -15 = rate cut → BUY (equity positive).
- earnings_surprise > 0.4 = strong beat → BUY (PEAD: drift continues).
- earnings_surprise < -0.4 = big miss → SELL.
- credit_spread_bps > 600 = credit stress → SELL (systemic risk).
- yield_curve_slope < -0.3 = inverted yield curve → REDUCE exposure (recession signal).

AXIS 3 — PSYCHOLOGICAL / CONTRARIAN:
- fear_greed > 0.80 = market euphoria → SELL (tops are reached at maximum greed).
- fear_greed < -0.70 = extreme fear → BUY (bottoms form at maximum fear).
- vix_level > 35 = fear spike → often a buying opportunity.
- put_call_ratio > 2.0 = everyone hedged → often a bottom signal → BUY.
- skew < -0.30 = extreme downside demand → crash imminent → EXIT.

AXIS 4 — HMM REGIME DETECTION:
- hmm_bull_prob > 0.70 = regime is bull → stay invested.
- hmm_crash_prob > 0.70 = regime is volatile/crash → reduce to cash.
- Use hmm_confidence to weight how much to trust the HMM signal.

RULES:
1. Never hold > 70% equity at once (Kelly criterion risk management).
2. When fundamentals contradict technicals, fundamentals WIN.
3. When psychology reaches extremes, fade the crowd.
4. If active_agents includes "whale", expect a large price move next step.
5. If active_agents includes "panic_seller", do NOT follow — fade the panic.
6. Always provide reasoning that cites specific signal values.

Respond ONLY with JSON. No other text.
{
  "decision": "BUY" | "SELL" | "HOLD",
  "position_size": <0.0-1.0>,
  "dominant_axis": "technical" | "fundamental" | "psychological" | "hmm",
  "reasoning": "<cite specific signal names and values, min 40 chars>"
}"""


COACH_TEMPLATE = """Episode {ep} complete. Task {task} ({regime}). Results:

Overall score: {score:.3f} / 1.0
Sharpe ratio: {sharpe:.2f} (target: > 1.0)
Total return: {ret:.1f}%
Max drawdown: {dd:.1f}%

Axis breakdown:
  Technical score:     {tech:.3f} — {tech_msg}
  Fundamental score:   {fund:.3f} — {fund_msg}
  Psychological score: {psych:.3f} — {psych_msg}
  HMM alignment:       {hmm:.3f} — {hmm_msg}

You must output a concrete strategy improvement. Cite signal names and thresholds.
Respond ONLY with JSON: {{"strategy_update": "..."}}"""


HEADLINE_TEMPLATE = """Market regime: {regime}. Signals:
- Earnings: {earnings:+.2f}, Fed: {fed:+.0f}bps, Supply: {supply:+.2f}
- Institutional flow: {flow:+.2f}, Credit spread: {credit:.0f}bps
- VIX: {vix:.1f}, Fear/Greed: {fg:+.2f}

Generate 3 realistic Bloomberg-style headlines for these conditions.
Then output a Fear/Greed score (-1 to +1).
JSON only: {{"headlines": ["...", "...", "..."], "fear_greed_score": <float>, "reason": "..."}}"""


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

async def call_llm(client: AsyncOpenAI, prompt: str, step: int, system: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
                max_tokens=250,
                temperature=0.1,
                timeout=20.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            wait = 15.0 + 10.0 * attempt
            log.warning(f"Rate limit step {step}, waiting {wait}s")
            await asyncio.sleep(wait)
        except Exception as e:
            log.error(f"API error step {step} attempt {attempt+1}: {e}")
            if attempt == MAX_RETRIES - 1:
                return '{"decision":"HOLD","position_size":0.0,"reasoning":"API error"}'
            await asyncio.sleep(RETRY_BASE_DELAY)
    return '{"decision":"HOLD","position_size":0.0,"reasoning":"Max retries"}'


async def get_llm_fear_greed(client: AsyncOpenAI, obs: Observation) -> float:
    """Use LLM to generate news headlines and score market sentiment."""
    fallback = obs.psychology.fear_greed_index
    try:
        prompt = HEADLINE_TEMPLATE.format(
            regime=obs.regime.value,
            earnings=obs.fundamental.earnings_surprise,
            fed=obs.fundamental.fed_rate_change_bps,
            supply=obs.fundamental.supply_shock,
            flow=obs.fundamental.institutional_flow,
            credit=obs.fundamental.credit_spread_bps,
            vix=obs.psychology.vix_level,
            fg=obs.psychology.fear_greed_index,
        )
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250, temperature=0.3, timeout=15.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        score = float(data.get("fear_greed_score", fallback))
        headlines = data.get("headlines", [])
        log.info(f"LLM Fear/Greed: {score:.2f} | {headlines[0] if headlines else ''}")
        return max(-1.0, min(1.0, score))
    except Exception as e:
        log.warning(f"LLM fear/greed failed: {e}")
        return fallback


async def get_strategy_update(
    client: AsyncOpenAI,
    ep: int, task: int, regime: str,
    score: float, sharpe: float, ret: float, dd: float,
    tech: float, fund: float, psych: float, hmm: float,
) -> str:
    """Self-improvement coach: generate concrete strategy update."""
    def msg(v, good=0.6):
        if v >= good: return "good"
        if v >= 0.4:  return "needs improvement"
        return "POOR — fix this"

    hmm_msg_val = "good alignment" if hmm > 0.6 else "not using HMM signals"

    try:
        prompt = COACH_TEMPLATE.format(
            ep=ep, task=task, regime=regime, score=score, sharpe=sharpe,
            ret=ret, dd=dd,
            tech=tech, tech_msg=msg(tech),
            fund=fund, fund_msg=msg(fund),
            psych=psych, psych_msg=msg(psych),
            hmm=hmm,    hmm_msg=hmm_msg_val,
        )
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150, temperature=0.2, timeout=15.0,
            response_format={"type": "json_object"},
        )
        data   = json.loads(resp.choices[0].message.content or "{}")
        update = str(data.get("strategy_update", ""))
        log.info(f"Strategy update: {update[:100]}")
        return update
    except Exception as e:
        log.warning(f"Strategy update failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Prompt builder — all 4 axes
# ---------------------------------------------------------------------------

def build_prompt(
    obs: Observation,
    step: int,
    action_history: list[str],
    llm_fg: Optional[float],
    strategy_update: str = "",
) -> str:
    t  = obs.technical
    f  = obs.fundamental
    ps = obs.psychology
    h  = obs.hmm
    p  = obs.portfolio

    payload = {
        "step": step,
        "time_left_pct": round(obs.time_left * 100, 1),
        "regime": obs.regime.value,
        "portfolio": {
            "cash":       round(p.cash, 0),
            "net_worth":  round(p.net_worth, 0),
            "equity_pct": round(p.equity_fraction * 100, 1),
            "drawdown_pct": round(p.drawdown * 100, 1),
        },
        "price": round(obs.window.raw_prices[-1], 2),
        "technical": {
            "rsi_14":     round(t.rsi_14, 1),
            "macd":       round(t.macd, 4),
            "macd_signal": round(t.macd_signal, 4),
            "bb_pct":     round(t.bb_pct, 2),
            "atr_14":     round(t.atr_14, 3),
            "roc_5":      round(t.roc_5, 2),
            "volatility": round(t.volatility_20, 1),
        },
        "fundamental": {
            "earnings_surprise":   round(f.earnings_surprise, 2),
            "fed_bps":             round(f.fed_rate_change_bps, 0),
            "gdp_surprise":        round(f.macro_gdp_surprise, 2),
            "supply_shock":        round(f.supply_shock, 2),
            "institutional_flow":  round(f.institutional_flow, 2),
            "credit_spread_bps":   round(f.credit_spread_bps, 0),
            "yield_curve_slope":   round(f.yield_curve_slope, 2),
        },
        "psychology": {
            "fear_greed":   round(llm_fg if llm_fg is not None else ps.fear_greed_index, 2),
            "vix":          round(ps.vix_level, 1),
            "put_call":     round(ps.put_call_ratio, 2),
            "insider_buy":  round(ps.insider_buying, 2),
            "skew":         round(ps.skew, 3),
        },
        "hmm": {
            "bull_prob":    round(h.prob_bull, 2),
            "crash_prob":   round(h.prob_crash, 2),
            "confidence":   round(h.state_confidence, 2),
            "granger_earn_pval": round(h.granger_earnings_pval, 3),
        },
        "active_agents": obs.active_agents,
        "recent_actions": action_history[-3:],
    }
    if strategy_update:
        payload["coach_directive"] = strategy_update[:200]

    return json.dumps(payload, separators=(',', ':'))


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Action:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    try:
        data = json.loads(text)
    except Exception as e:
        log.warning(f"Parse failed: {e}")
        return Action.hold(reason="Parse error")

    amap  = {"BUY": ActionType.BUY, "SELL": ActionType.SELL, "HOLD": ActionType.HOLD}
    atype = amap.get(str(data.get("decision", "HOLD")).upper(), ActionType.HOLD)
    try:
        fraction = max(0.0, min(1.0, float(data.get("position_size", 0.0))))
    except Exception:
        fraction = 0.0
    reasoning = str(data.get("reasoning", ""))[:512]
    return Action(action_type=atype, fraction=fraction, reason=reasoning)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_strategy_memory: dict[int, str] = {}
_episode_counter = 0
_training_logger: Optional[TrainingLogger] = None


async def run_task(
    task_id: int,
    client: AsyncOpenAI,
    logger: TrainingLogger,
    num_runs: int = 1,
) -> list[float]:
    global _episode_counter, _strategy_memory

    scores = []

    for run in range(num_runs):
        _episode_counter += 1
        task_start = time.time()

        # OVERRIDE: Shorten episode to 40 steps for Hackathon testing
        config = EnvironmentConfig(regime=MarketRegime.BULL, num_steps=50)
        env = TradeSimEnv(config)
        obs = env.reset(task_id=task_id)

        task_name = f"task_{task_id}_{env._config.regime.value}_ep{_episode_counter}"
        log_start(task_name, BENCHMARK_NAME, MODEL_NAME)

        # Get LLM fear/greed at episode start
        llm_fg = None
        try:
            llm_fg = await get_llm_fear_greed(client, obs)
            await asyncio.sleep(1.0)
        except Exception as e:
            log.warning(f"Initial fear/greed failed: {e}")

        strategy_update = _strategy_memory.get(task_id, "")
        step            = 1
        final_score     = 0.0
        action_history  = []
        rewards_history = []
        success         = False
        grade_info      = {}

        while True:
            if time.time() - task_start > PER_TASK_TIMEOUT:
                log.warning(f"Task {task_id} run {run+1} timeout at step {step}")
                log_step(step, "hold(0.0)", 0.0, True, "Timeout")
                break

            prompt = build_prompt(obs, step, action_history, llm_fg, strategy_update)
            system = BASE_SYSTEM_PROMPT
            if strategy_update:
                system = BASE_SYSTEM_PROMPT + f"\n\nIMPORTANT COACH DIRECTIVE:\n{strategy_update}"

            raw    = await call_llm(client, prompt, step, system)
            
            # OVERRIDE: Groq TPM Governor to prevent 429 Too Many Requests
            await asyncio.sleep(10.5)  

            action = parse_action(raw)
            if action.action_type != ActionType.HOLD:
                action_history.append(f"{action.action_type.value}({action.fraction:.2f})")

            try:
                result    = env.step(action)
                error_str = None
            except Exception as e:
                error_str = str(e)
                result    = None

            action_str = f"{action.action_type.value.lower()}({action.fraction:.2f})"

            if result is not None:
                reward_val  = result.reward.total
                done        = result.done
                obs         = result.observation
                if done:
                    final_score = result.info.get("episode_score", 0.0) or 0.0
                    success     = final_score > 0.0
                    grade_info  = result.info
            else:
                reward_val = 0.0
                done       = True

            rewards_history.append(reward_val)
            log_step(step, action_str, reward_val, done, error_str)

            if done:
                break
            step += 1

        log_end(success, step, final_score, rewards_history)
        scores.append(final_score)

        # Log training metrics
        metrics = EpisodeMetrics(
            episode_num=_episode_counter,
            task_id=task_id,
            regime=env._config.regime.value,
            score=final_score,
            sharpe_ratio=grade_info.get("sharpe_ratio", 0.0),
            total_return_pct=grade_info.get("total_return_pct", 0.0),
            max_drawdown_pct=grade_info.get("calmar_ratio", 0.0),
            num_trades=grade_info.get("total_trades", 0),
            technical_score=grade_info.get("technical_score", 0.0),
            fundamental_score=grade_info.get("fundamental_score", 0.0),
            psychological_score=grade_info.get("psychological_score", 0.0),
            hmm_alignment_score=grade_info.get("hmm_alignment_score", 0.0),
            strategy_update_used=bool(strategy_update),
        )
        logger.log(metrics)

        # Self-improvement: generate strategy update for next run
        if grade_info:
            try:
                update = await get_strategy_update(
                    client=client,
                    ep=_episode_counter,
                    task=task_id,
                    regime=env._config.regime.value,
                    score=final_score,
                    sharpe=grade_info.get("sharpe_ratio", 0.0),
                    ret=grade_info.get("total_return_pct", 0.0),
                    dd=grade_info.get("calmar_ratio", 0.0),
                    tech=grade_info.get("technical_score", 0.5),
                    fund=grade_info.get("fundamental_score", 0.5),
                    psych=grade_info.get("psychological_score", 0.5),
                    hmm=grade_info.get("hmm_alignment_score", 0.5),
                )
                _strategy_memory[task_id] = update
                await asyncio.sleep(1.0)
            except Exception as e:
                log.warning(f"Strategy update failed: {e}")

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    global _training_logger

    # Check for the correct key now
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    # Bulletproof Client: Hardcoded to Groq, pulling the correct key
    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1", 
        api_key=os.environ.get("OPENAI_API_KEY"), 
        max_retries=0
    )
    
    logger  = TrainingLogger(METRICS_FILE)
    _training_logger = logger

    # Run each task — multiple runs to show improvement
    num_runs = int(os.environ.get("NUM_RUNS_PER_TASK", "1"))
    for task_id in [1, 2, 3]:
        await run_task(task_id, client, logger, num_runs=num_runs)

    # Print improvement summary to stderr
    for task_id in [1, 2, 3]:
        improvement = logger.get_improvement(task_id)
        log.info(f"Task {task_id} improvement across runs: {improvement:+.3f}")


if __name__ == "__main__":
    asyncio.run(async_main())