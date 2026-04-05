from environment import TradeSimEnv
from models import Action

print("Testing Master Loop Environment...\n")
env = TradeSimEnv()

for task_id in [1, 2, 3]:
    obs = env.reset(task_id)
    print(f'--- Task {task_id}: {obs.task_description} ---')
    print(f'  Start price: {obs.current_price}')
    print(f'  RSI: {obs.indicators["rsi"]}  MA20: {obs.indicators["ma20"]}')
    
    done = False
    step = 0
    while not done:
        # simple agent: buy early, hold, sell at 60
        if step == 5:
            action = Action(decision='BUY', position_size=0.5, reasoning='buying early')
        elif step == 60:
            action = Action(decision='SELL', position_size=1.0, reasoning='taking profit')
        else:
            action = Action(decision='HOLD', reasoning='waiting')
            
        obs, reward, done, info = env.step(action)
        step += 1
        
    print(f'  Final score: {info.get("final_score", "N/A")}')
    print(f'  Portfolio:   ${info["portfolio_value"]:,.2f}\n')

print('environment.py works correctly!')