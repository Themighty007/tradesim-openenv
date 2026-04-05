import numpy as np
from models import Trade
from reward import calculate_sharpe
from typing import List

def _max_drawdown(pnl_curve: list) -> float:
    peak = pnl_curve[0]
    max_dd = 0.0
    for v in pnl_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-10)
        if dd > max_dd:
            max_dd = dd
    return max_dd

def grade_task1_trend(
    trade_history: List[Trade],
    pnl_curve: list,
    starting_cash: float = 100_000.0
) -> float:
    if not pnl_curve:
        return 0.0
    final_value = pnl_curve[-1]

    # 0.4: how much of the bull trend did agent capture?
    pnl_ratio = np.clip((final_value / starting_cash) - 1.0, 0, 0.2) / 0.2
    score_pnl = float(pnl_ratio) * 0.4

    # 0.3: did agent buy early (before timestep 15)?
    buy_trades = [t for t in trade_history if t.decision == "BUY"]
    early = any(t.timestep < 15 for t in buy_trades)
    score_early = 0.3 if early else 0.1

    # 0.3: consistent position sizing (not erratic)
    if buy_trades:
        sizes = [(t.quantity * t.price) for t in buy_trades]
        cv = np.std(sizes) / (np.mean(sizes) + 1e-10)
        score_size = float(np.clip(1.0 - cv, 0, 1)) * 0.3
    else:
        score_size = 0.0

    return float(np.clip(score_pnl + score_early + score_size, 0.0, 1.0))


def grade_task2_mean_reversion(
    trade_history: List[Trade],
    pnl_curve: list,
    starting_cash: float = 100_000.0
) -> float:
    if not pnl_curve:
        return 0.0

    sharpe = calculate_sharpe(pnl_curve)
    score_sharpe = float(np.clip(sharpe / 2.0, 0, 1)) * 0.4

    max_dd = _max_drawdown(pnl_curve)
    score_dd = float(np.clip(1.0 - max_dd / 0.2, 0, 1)) * 0.3

    sell_trades = [t for t in trade_history if t.decision == "SELL"]
    buy_trades  = [t for t in trade_history if t.decision == "BUY"]
    total_trades = len(sell_trades)
    if total_trades > 0 and buy_trades:
        profitable = sum(
            1 for s in sell_trades
            if any(b.timestep < s.timestep and s.price > b.price
                   for b in buy_trades)
        )
        win_rate = profitable / total_trades
    else:
        win_rate = 0.0
    score_wr = float(win_rate) * 0.3

    return float(np.clip(score_sharpe + score_dd + score_wr, 0.0, 1.0))


def grade_task3_crash(
    trade_history: List[Trade],
    pnl_curve: list,
    starting_cash: float = 100_000.0
) -> float:
    if not pnl_curve:
        return 0.0

    min_value = min(pnl_curve)
    final_value = pnl_curve[-1]

    # 0.4: capital preserved during crash (floor is 80% of start)
    preserved = np.clip(min_value / (starting_cash * 0.8), 0, 1)
    score_pres = float(preserved) * 0.4

    # 0.4: recovery captured after the bottom
    recovery_gain = max(0, final_value - min_value)
    score_rec = float(np.clip(recovery_gain / (starting_cash * 0.1), 0, 1)) * 0.4

    # 0.2: did agent detect crash early (SELL between t=28 and t=35)?
    early_exit = any(
        t.decision == "SELL" and 28 <= t.timestep <= 35
        for t in trade_history
    )
    score_exit = 0.2 if early_exit else 0.0

    return float(np.clip(score_pres + score_rec + score_exit, 0.0, 1.0))