import numpy as np
from maze_env import MazeEnv
from agent import QLearningAgent

env = MazeEnv()
agent = QLearningAgent(env)

print("Trenowanie agenta...")
agent.train(episodes=3000, max_steps_per_episode=500)
print("Trening zakończony.")

# Sprawdź ścieżkę
path = agent.find_best_path()
print("Długość znalezionej ścieżki:", len(path))
if env.agent_pos == env.goal_pos:
    print("✅ Agent dotarł do celu.")
else:
    print("❌ Agent NIE dotarł do celu.")

# Zapisz Q-tabelę
np.save("q_table.npy", agent.q_table)
agent.plot_learning_curve()
