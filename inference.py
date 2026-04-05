import os, json, time
from openai import OpenAI
from models import Action
from environment import TradeSimEnv

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "")),
    max_retries=0,
)
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a professional trading agent.
You will receive market data and must respond with a JSON trading decision.
Rules:
- BUY when you see upward momentum or oversold conditions
- SELL when you see downward momentum or overbought conditions
- HOLD when signals are unclear
- Never bet more than 50% of capital at once
- Always explain your reasoning briefly
Respond ONLY with valid JSON — no other text."""

def build_prompt(obs) -> str:
    return f"""Market: ${obs.current_price:.2f} | RSI: {obs.indicators['rsi']:.1f} | MA20: {obs.indicators['ma20']:.2f}
Portfolio: Cash ${obs.portfolio['cash']:,.0f} | Units {obs.portfolio['position']:.2f} | Total ${obs.portfolio['total_value']:,.0f}
Task: {obs.task_description} ({obs.market_regime_hint})
Step: {obs.timestep}/79
JSON: {{"decision": "BUY"|"SELL"|"HOLD", "position_size": 0.0-1.0, "reasoning": "short"}}"""

def run_task(env: TradeSimEnv, task_id: int) -> float:
    obs = env.reset(task_id)
    done = False
    final_score = 0.0
    while not done:
        time.sleep(3.0)
        prompt = build_prompt(obs)
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=200,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            # Clean up markdown blocks if the LLM wraps the JSON
            if raw.startswith("```json"):
                raw = raw[7:-3].strip()
            elif raw.startswith("```"):
                raw = raw[3:-3].strip()
                
            data = json.loads(raw)
            action = Action(**data)
        except Exception as e:
            print(f"  Parse error at t={obs.timestep}: {e} — defaulting to HOLD")
            action = Action(decision="HOLD", reasoning="parse error fallback")
        obs, reward, done, info = env.step(action)
        if done:
            final_score = info.get("final_score", 0.0)
    return final_score

if __name__ == "__main__":
    env = TradeSimEnv()
    print("Running TradeSim baseline inference...")
    scores = {}
    for task_id, name in [(1,"Bull trend"), (2,"Mean reversion"), (3,"Flash crash")]:
        print(f"  Running Task {task_id}: {name}...")
        score = run_task(env, task_id)
        scores[task_id] = score
        print(f"  Task {task_id} score: {score:.4f}")
    print(f"\nFinal scores:")
    print(f"  Task 1 (Bull trend):     {scores[1]:.4f}")
    print(f"  Task 2 (Mean reversion): {scores[2]:.4f}")
    print(f"  Task 3 (Flash crash):    {scores[3]:.4f}")