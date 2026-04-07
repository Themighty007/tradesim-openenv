"""
TradeSim — api.py
=================
FastAPI wrapper exposing TradeSim over HTTP.

Endpoints:
  GET  /              → Health check (must return HTTP 200)
  POST /reset         → Start episode, returns Observation
  POST /step          → Advance one step, returns StepResult
  GET  /state         → Current episode State
  GET  /tasks         → Task descriptions
  POST /run_episode   → Complete episode with simple agent (for demos)

All request/response bodies are the Pydantic models from models.py,
serialised as JSON. FastAPI validates them automatically.

Designed for Hugging Face Spaces (port 7860).
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # <-- FIXED: Was 'from cors'
from fastapi.responses import JSONResponse        # <-- FIXED: Was 'from responses'
from pydantic import BaseModel

from environment import TradeSimEnv, _TASK_DESCRIPTIONS, _TASK_REGIMES
from models import (
    Action,
    ActionType,
    EnvironmentConfig,
    GradeResult,
    MarketRegime,
    Observation,
    State,
    StepResult,
)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Global singleton environment. Thread-unsafe — single-user assumption."""
    env: TradeSimEnv = None
    last_reset_time: float = 0.0
    request_count: int = 0
    start_time: float = time.time()


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise environment on startup."""
    app_state.env = TradeSimEnv()
    app_state.start_time = time.time()
    yield
    # Cleanup (nothing to do)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TradeSim",
    description=(
        "A Reinforcement Learning environment for LLM trading agents. "
        "Three market scenarios: bull market, choppy range, flash crash. "
        "Built to the OpenEnv standard interface."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 1


class ActionInput(BaseModel):
    """Flexible Action input — accepts BUY/SELL/HOLD or buy/sell/hold."""
    action_type: str = "hold"
    fraction: float = 0.0
    reason: str = ""

    def to_action(self) -> Action:
        from models import ActionType
        atype = ActionType(self.action_type.lower())
        return Action(action_type=atype, fraction=max(0.0, min(1.0, self.fraction)), reason=self.reason)


class StepRequest(BaseModel):
    action: ActionInput


class RunEpisodeRequest(BaseModel):
    task_id: int = 1
    strategy: str = "hold"   # "hold", "buy_and_hold", "random"


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    tasks_available: list[int]
    environment_ready: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
async def health():
    """
    Health check endpoint. Judges hit this first.
    Must return HTTP 200.
    """
    app_state.request_count += 1
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - app_state.start_time, 1),
        tasks_available=[1, 2, 3],
        environment_ready=app_state.env is not None,
    )


@app.get("/health")
async def health_alt():
    """Alternative health check path."""
    return {"status": "ok", "service": "TradeSim"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None) -> dict:
    """
    Reset the environment to start a new episode.
    Gracefully handles empty requests from automated graders.
    """
    app_state.request_count += 1

    # THE FIX: If the bot sends an empty body (None), default to task 1
    task_id = request.task_id if request else 1

    if task_id not in [1, 2, 3]:
        raise HTTPException(
            status_code=422,
            detail=f"task_id must be 1, 2, or 3. Got: {task_id}"
        )

    try:
        obs = app_state.env.reset(task_id=task_id)
        app_state.last_reset_time = time.time()
        return {
            "observation": obs.model_dump(),
            "task_description": app_state.env.task_description,
            "regime_hint": app_state.env.regime_hint,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest) -> dict:
    """
    Advance the environment by one step.

    Body: {"action": {"action_type": "BUY", "fraction": 0.5, "reason": "..."}}

    Returns: StepResult as JSON.
    """
    app_state.request_count += 1

    if app_state.env.is_done:
        raise HTTPException(
            status_code=409,
            detail="Episode is done. Call /reset to start a new episode."
        )

    try:
        action = request.action.to_action()
        result = app_state.env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state() -> dict:
    """
    Get the current episode state.

    Returns: State as JSON.
    """
    app_state.request_count += 1
    try:
        s = app_state.env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def list_tasks() -> dict:
    """List all available tasks with descriptions."""
    return {
        "tasks": {
            str(tid): {
                "id": tid,
                "regime": regime.value,
                "description": _TASK_DESCRIPTIONS[tid],
            }
            for tid, regime in _TASK_REGIMES.items()
        }
    }


@app.post("/run_episode")
async def run_episode(request: RunEpisodeRequest) -> dict:
    """
    Run a complete episode with a simple built-in strategy.

    Strategies:
      "hold"         — HOLD every step (baseline)
      "buy_and_hold" — BUY 90% on step 1, HOLD until end
      "random"       — random BUY/SELL/HOLD each step

    Returns final score and episode summary.
    """
    import random as rnd

    app_state.request_count += 1

    obs = app_state.env.reset(task_id=request.task_id)
    step_count = 0
    total_reward = 0.0
    done = False
    final_info = {}

    while not done:
        strategy = request.strategy.lower()

        if strategy == "buy_and_hold":
            if step_count == 0:
                action = Action.buy(fraction=0.90, reason="Buy-and-hold strategy: initial deployment")
            else:
                action = Action.hold(reason="Buy-and-hold strategy: waiting")

        elif strategy == "random":
            choice = rnd.choice(["buy", "sell", "hold"])
            if choice == "buy":
                action = Action.buy(fraction=rnd.uniform(0.1, 0.5), reason="Random strategy")
            elif choice == "sell":
                action = Action.sell(fraction=rnd.uniform(0.1, 0.9), reason="Random strategy")
            else:
                action = Action.hold(reason="Random strategy")

        else:  # hold
            action = Action.hold(reason="Passive hold strategy")

        result = app_state.env.step(action)
        total_reward += result.reward.total
        obs = result.observation
        done = result.done
        final_info = result.info
        step_count += 1

    return {
        "task_id":       request.task_id,
        "strategy":      request.strategy,
        "final_score":   final_info.get("episode_score", 0.0),
        "total_reward":  round(total_reward, 6),
        "total_steps":   step_count,
        "final_net_worth": obs.portfolio.net_worth,
        "total_return_pct": round(obs.portfolio.total_return * 100, 3),
        "breakdown":     final_info.get("episode_breakdown", {}),
    }


@app.get("/info")
async def info() -> dict:
    """Meta-information about this TradeSim instance."""
    return {
        "name": "TradeSim",
        "version": "1.0.0",
        "description": "RL environment for LLM trading agents",
        "interface": "OpenEnv-compatible (reset/step/state)",
        "tasks": 3,
        "action_space": ["BUY", "SELL", "HOLD"],
        "observation_fields": [
            "timestep", "max_steps", "regime", "window", "portfolio", "time_left"
        ],
        "endpoints": {
            "GET  /":            "Health check",
            "POST /reset":       "Start episode",
            "POST /step":        "Advance one step",
            "GET  /state":       "Current state",
            "GET  /tasks":       "Task descriptions",
            "POST /run_episode": "Run full episode with built-in agent",
        },
        "request_count": app_state.request_count,
        "uptime_seconds": round(time.time() - app_state.start_time, 1),
    }


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
