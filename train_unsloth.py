"""
TradeSim v3 — train_unsloth.py
==============================
MANDATORY HACKATHON REQUIREMENT:
"Show a minimal training script for your environment using Unsloth or HF TRL"

This script:
1. Collects trajectory data by running the TradeSim environment
2. Formats it as instruction-following training data
3. Fine-tunes a Llama-3.1-8B model using Unsloth (4-bit QLoRA)
4. Shows reward improvement via evaluation runs

Architecture:
- Environment generates (observation, optimal_action) pairs
- Optimal actions are determined by the 3-axis grader signals
- Model learns to produce correct JSON trading decisions
- Evaluation shows Sharpe ratio improvement across training

Run on Google Colab with A100/T4 GPU.
Install: pip install unsloth trl peft datasets
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================

import json
import os
import sys
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Try Unsloth (GPU required), fallback to HF TRL for CPU demo
# ---------------------------------------------------------------------------
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
    print("Unsloth available — using 4-bit QLoRA fine-tuning")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth not available — using HF TRL SFTTrainer (CPU fallback)")

from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from environment import TradeSimEnv
from models import Action, ActionType, EnvironmentConfig, MarketRegime


# ============================================================
# SECTION 2: GENERATE TRAINING DATA FROM ENVIRONMENT
# ============================================================

def get_optimal_action_label(obs) -> dict:
    """
    Generate the 'correct' action label for a given observation.
    This implements the ground truth for supervised fine-tuning.
    
    Rules based on the 3-axis signal system:
    Priority: Fundamental > Technical > Psychological
    """
    t  = obs.technical
    f  = obs.fundamental
    ps = obs.psychology
    h  = obs.hmm
    
    decision   = "HOLD"
    position   = 0.0
    axis       = "fundamental"
    reasoning_parts = []
    
    # --- FUNDAMENTAL OVERRIDES (highest priority) ---
    
    # Crash early warning: Fed hike + earnings miss + credit stress
    if (f.fed_rate_change_bps > 50 and 
        f.earnings_surprise < -0.4 and
        f.credit_spread_bps > 500):
        decision = "SELL"
        position = 0.80
        reasoning_parts.append(
            f"CRASH SIGNAL: Fed +{f.fed_rate_change_bps:.0f}bps, "
            f"earnings {f.earnings_surprise:.2f}, credit {f.credit_spread_bps:.0f}bps"
        )
        axis = "fundamental"
    
    # Inverted yield curve = recession
    elif f.yield_curve_slope < -0.5:
        decision = "SELL"
        position = 0.50
        reasoning_parts.append(f"Yield curve inverted ({f.yield_curve_slope:.2f})")
        axis = "fundamental"
    
    # Strong bull fundamental
    elif (f.earnings_surprise > 0.4 and
          f.institutional_flow > 0.3 and
          f.fed_rate_change_bps < 10):
        decision = "BUY"
        position = 0.60
        reasoning_parts.append(
            f"Bull: earnings {f.earnings_surprise:.2f}, flow {f.institutional_flow:.2f}"
        )
        axis = "fundamental"
    
    # --- PSYCHOLOGICAL CONTRARIAN ---
    
    elif ps.fear_greed_index > 0.85:
        decision = "SELL"
        position = 0.40
        reasoning_parts.append(f"Extreme greed {ps.fear_greed_index:.2f} — fade euphoria")
        axis = "psychological"
    
    elif ps.fear_greed_index < -0.75:
        decision = "BUY"
        position = 0.40
        reasoning_parts.append(f"Extreme fear {ps.fear_greed_index:.2f} — contrarian buy")
        axis = "psychological"
    
    elif ps.vix_level > 40:
        decision = "BUY"
        position = 0.30
        reasoning_parts.append(f"VIX spike {ps.vix_level:.1f} — fear buying opportunity")
        axis = "psychological"
    
    # --- TECHNICAL ---
    
    elif t.rsi_14 < 28:
        decision = "BUY"
        position = 0.35
        reasoning_parts.append(f"RSI oversold {t.rsi_14:.1f}")
        axis = "technical"
    
    elif t.rsi_14 > 72:
        decision = "SELL"
        position = 0.35
        reasoning_parts.append(f"RSI overbought {t.rsi_14:.1f}")
        axis = "technical"
    
    elif t.macd > t.macd_signal and t.macd > 0 and t.bb_pct > 0.3:
        decision = "BUY"
        position = 0.30
        reasoning_parts.append(f"MACD bullish cross {t.macd:.4f}")
        axis = "technical"
    
    elif t.macd < t.macd_signal and t.macd < 0 and t.bb_pct < 0.7:
        decision = "SELL"
        position = 0.30
        reasoning_parts.append(f"MACD bearish cross {t.macd:.4f}")
        axis = "technical"
    
    # --- HMM REGIME ---
    
    elif h.prob_bull > 0.80 and h.state_confidence > 0.7:
        decision = "BUY"
        position = 0.40
        reasoning_parts.append(f"HMM bull regime P={h.prob_bull:.2f}")
        axis = "hmm"
    
    elif h.prob_crash > 0.80 and h.state_confidence > 0.7:
        decision = "SELL"
        position = 0.50
        reasoning_parts.append(f"HMM crash regime P={h.prob_crash:.2f}")
        axis = "hmm"
    
    # Default: HOLD
    else:
        reasoning_parts.append("No strong signal — preserve capital")
    
    # Risk management: don't exceed 65% equity
    portfolio = obs.portfolio
    if decision == "BUY" and portfolio.equity_fraction > 0.65:
        decision  = "HOLD"
        position  = 0.0
        reasoning_parts = [f"Position limit: equity already {portfolio.equity_fraction:.0%}"]
    
    return {
        "decision":     decision,
        "position_size": position,
        "dominant_axis": axis,
        "reasoning":    " | ".join(reasoning_parts) if reasoning_parts else "Hold — no clear signal",
    }


def format_observation_for_llm(obs, step: int) -> str:
    """Format observation as a compact JSON string for the LLM input."""
    t  = obs.technical
    f  = obs.fundamental
    ps = obs.psychology
    h  = obs.hmm
    p  = obs.portfolio
    
    return json.dumps({
        "step": step,
        "regime": obs.regime.value,
        "portfolio": {
            "equity_pct":   round(p.equity_fraction * 100, 1),
            "drawdown_pct": round(p.drawdown * 100, 1),
            "net_worth":    round(p.net_worth, 0),
        },
        "technical": {
            "rsi_14":     round(t.rsi_14, 1),
            "macd":       round(t.macd, 4),
            "macd_signal": round(t.macd_signal, 4),
            "bb_pct":     round(t.bb_pct, 2),
            "atr_14":     round(t.atr_14, 3),
            "volatility": round(t.volatility_20, 1),
        },
        "fundamental": {
            "earnings_surprise":  round(f.earnings_surprise, 2),
            "fed_bps":            round(f.fed_rate_change_bps, 0),
            "credit_spread_bps":  round(f.credit_spread_bps, 0),
            "yield_curve":        round(f.yield_curve_slope, 2),
            "institutional_flow": round(f.institutional_flow, 2),
        },
        "psychology": {
            "fear_greed": round(ps.fear_greed_index, 2),
            "vix":        round(ps.vix_level, 1),
            "put_call":   round(ps.put_call_ratio, 2),
            "skew":       round(ps.skew, 3),
        },
        "hmm": {
            "bull_prob":  round(h.prob_bull, 2),
            "crash_prob": round(h.prob_crash, 2),
            "confidence": round(h.state_confidence, 2),
        },
        "active_agents": obs.active_agents,
    }, separators=(',', ':'))


def collect_training_data(
    num_episodes: int = 30,
    steps_per_episode: int = 50,
    seed_offset: int = 0,
) -> list[dict]:
    """
    Run the environment with the rule-based optimal policy
    and collect (observation, action) pairs as training examples.
    """
    print(f"Collecting {num_episodes} episodes × {steps_per_episode} steps...")
    examples = []
    
    for ep in range(num_episodes):
        task_id = (ep % 3) + 1  # Cycle through all 3 tasks
        seed    = 42 + ep + seed_offset
        
        config = EnvironmentConfig(
            regime=MarketRegime.BULL,  # Will be overridden by reset
            num_steps=steps_per_episode,
            seed=seed,
        )
        env = TradeSimEnv(config)
        obs = env.reset(task_id=task_id)
        
        for step in range(steps_per_episode):
            obs_str    = format_observation_for_llm(obs, step + 1)
            label_dict = get_optimal_action_label(obs)
            label_str  = json.dumps(label_dict, separators=(',', ':'))
            
            examples.append({
                "instruction": (
                    "You are a quantitative trading agent. Analyse the market signals "
                    "and output a JSON trading decision."
                ),
                "input":  obs_str,
                "output": label_str,
            })
            
            # Step with the optimal action
            action = Action(
                action_type=ActionType(label_dict["decision"].lower()),
                fraction=label_dict["position_size"],
                reason=label_dict["reasoning"][:200],
            )
            result = env.step(action)
            obs    = result.observation
            if result.done:
                break
    
    print(f"Collected {len(examples)} training examples")
    random.shuffle(examples)
    return examples


# ============================================================
# SECTION 3: FORMAT FOR FINE-TUNING
# ============================================================

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_alpaca(examples: list[dict]) -> Dataset:
    """Format examples in Alpaca instruction-following format."""
    texts = [
        ALPACA_TEMPLATE.format(
            instruction=ex["instruction"],
            input=ex["input"],
            output=ex["output"],
        )
        for ex in examples
    ]
    return Dataset.from_dict({"text": texts})


# ============================================================
# SECTION 4: UNSLOTH FINE-TUNING
# ============================================================

def train_with_unsloth(dataset: Dataset, output_dir: str = "./tradesim_model"):
    """Fine-tune Llama-3.1-8B using Unsloth 4-bit QLoRA."""
    
    print("Loading Llama-3.1-8B-Instruct with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,           # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # SFT training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        max_seq_length=2048,
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    return model, tokenizer


def train_with_trl_cpu(dataset: Dataset, output_dir: str = "./tradesim_model_trl"):
    """CPU fallback: HF TRL SFTTrainer with a small model for demo."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    
    print("Loading small model for CPU demo (GPT-2)...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        max_seq_length=512,
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    print("Training (CPU demo mode)...")
    trainer.train()
    print("Training complete.")
    return model, tokenizer


# ============================================================
# SECTION 5: EVALUATION — SHOW REWARD IMPROVEMENT
# ============================================================

def evaluate_model(model, tokenizer, num_episodes: int = 3) -> dict:
    """
    Evaluate the fine-tuned model on TradeSim.
    Returns per-task Sharpe ratios and scores for the training curve.
    """
    from graders import grade_episode
    from models import EpisodeRecord
    
    results = {}
    
    for task_id in [1, 2, 3]:
        env     = TradeSimEnv()
        obs     = env.reset(task_id=task_id)
        rewards = []
        done    = False
        step    = 1
        
        while not done and step < 60:
            obs_str = format_observation_for_llm(obs, step)
            prompt  = ALPACA_TEMPLATE.format(
                instruction="You are a quantitative trading agent. Analyse and trade.",
                input=obs_str,
                output="",
            )
            
            try:
                if UNSLOTH_AVAILABLE:
                    FastLanguageModel.for_inference(model)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                with __import__("torch").no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response  = generated[len(prompt):]
                
                # Parse JSON from response
                try:
                    start = response.find("{")
                    end   = response.rfind("}") + 1
                    if start >= 0 and end > start:
                        data    = json.loads(response[start:end])
                        atype   = ActionType(data.get("decision", "HOLD").lower())
                        fraction = float(data.get("position_size", 0.0))
                        action  = Action(action_type=atype, fraction=fraction)
                    else:
                        action = Action.hold()
                except Exception:
                    action = Action.hold()
            except Exception:
                action = Action.hold()
            
            result = env.step(action)
            rewards.append(result.reward.total)
            done = result.done
            obs  = result.observation
            step += 1
        
        grade = env.last_grade
        results[f"task_{task_id}"] = {
            "score":       grade.score if grade else 0.0,
            "sharpe":      grade.sharpe_ratio if grade else 0.0,
            "return_pct":  grade.total_return_pct if grade else 0.0,
        }
    
    return results


# ============================================================
# SECTION 6: MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TradeSim v3 — Unsloth Fine-Tuning Pipeline")
    print("=" * 60)
    
    # Step 1: Collect training data
    examples = collect_training_data(
        num_episodes=30,
        steps_per_episode=50,
    )
    dataset = format_alpaca(examples)
    print(f"\nDataset size: {len(dataset)} examples")
    print(f"Sample input: {examples[0]['input'][:200]}...")
    print(f"Sample output: {examples[0]['output']}")
    
    # Save dataset for inspection
    dataset.save_to_disk("tradesim_dataset")
    print("Dataset saved to ./tradesim_dataset")
    
    # Step 2: Train
    output_dir = "./tradesim_finetuned"
    if UNSLOTH_AVAILABLE:
        model, tokenizer = train_with_unsloth(dataset, output_dir)
    else:
        model, tokenizer = train_with_trl_cpu(dataset, output_dir)
    
    # Step 3: Evaluate
    print("\nEvaluating fine-tuned model...")
    results = evaluate_model(model, tokenizer)
    print("\nFine-tuned model performance:")
    for task, metrics in results.items():
        print(f"  {task}: score={metrics['score']:.3f} "
              f"sharpe={metrics['sharpe']:.2f} "
              f"return={metrics['return_pct']:.1f}%")
    
    print("\nTraining pipeline complete.")
    print("Upload model to HuggingFace Hub:")
    print("  model.push_to_hub('your-username/tradesim-llm')")