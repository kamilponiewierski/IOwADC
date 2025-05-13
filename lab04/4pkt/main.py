from maze_env import MazeEnv
from agent import QLearningAgent

def main():
    env = MazeEnv(size=20, seed=42)
    agent = QLearningAgent(env)
    agent.train(episodes=3000, max_steps=250)
    agent.save_q_table()  # zapisujemy nauczoną tabelę
    agent.plot_learning_curve()
    print("Gotowe. Q-table zapisana do 'q_table.pkl'.")

if __name__ == "__main__":
    main()
