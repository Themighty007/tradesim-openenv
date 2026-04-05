import numpy as np
from models import Action

def calculate_step_reward(
    prev_value: float,
    curr_value: float,
    drawdown: float,
    action: Action,
    trade_count_last_5: int
) -> tuple[float, dict]:

    breakdown = {}
    reward = 0.0

    # 1. PnL signal — did value go up this step?
    pnl_pct = (curr_value - prev_value) / (prev_value + 1e-10)
    pnl_score = float(np.clip(pnl_pct * 10, -1.0, 1.0)) * 0.3
    breakdown['pnl'] = round(pnl_score, 4)
    reward += pnl_score

    # 2. Drawdown safety reward
    if drawdown < 0.05:
        breakdown['safety'] = 0.2
        reward += 0.2
    else:
        breakdown['safety'] = 0.0

    # 3. Reasoning reward — agent explained itself
    if len(action.reasoning.strip()) > 10:
        breakdown['reasoning'] = 0.1
        reward += 0.1
    else:
        breakdown['reasoning'] = 0.0

    # 4. Drawdown danger penalty
    if drawdown > 0.15:
        breakdown['dd_penalty'] = -0.2
        reward -= 0.2
    else:
        breakdown['dd_penalty'] = 0.0

    # 5. Overtrading penalty
    if trade_count_last_5 > 3:
        breakdown['overtrade'] = -0.1
        reward -= 0.1
    else:
        breakdown['overtrade'] = 0.0

    # 6. Reckless all-in penalty (bet everything with no stop)
    if action.position_size >= 0.99 and action.stop_loss_pct == 0:
        breakdown['reckless'] = -0.3
        reward -= 0.3
    else:
        breakdown['reckless'] = 0.0

    final = float(np.clip(reward, -1.0, 1.0))
    return final, breakdown


def calculate_sharpe(pnl_curve: list) -> float:
    if len(pnl_curve) < 2:
        return 0.0
    returns = np.diff(pnl_curve) / (np.array(pnl_curve[:-1]) + 1e-10)
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def calculate_episode_bonus(
    sharpe: float, max_drawdown: float,
    final_value: float, starting_cash: float
) -> float:
    bonus = 0.0
    if sharpe > 1.0:
        bonus += 0.2
    if max_drawdown < 0.08:
        bonus += 0.1
    if final_value < starting_cash * 0.85:
        bonus -= 0.3
    return float(np.clip(bonus, -0.3, 0.3))