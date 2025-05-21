from maze_env import MazeEnv
from agent import QLearningAgent

env = MazeEnv(size=20)
agent = QLearningAgent(env)

agent.train(episodes=3000)
agent.plot_learning_curve()

path = agent.find_best_path()
print(f"Długość znalezionej ścieżki: {len(path)}")
