import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from maze_env import MazeEnv


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Rate of learning
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.episode_rewards = []  # To store total rewards for plotting
        self.best_path = []  # Store best path for visualization

    def train(self, episodes=1000, max_steps=300):
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < max_steps:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, done = self.env.step(action)
                total_reward += reward

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                self.q_table[state, action] = old_value + self.alpha * (
                        reward + self.gamma * next_max - old_value
                )

                state = next_state
                steps += 1

            self.episode_rewards.append(total_reward)

    def plot_learning_curve(self):
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Curve (Total Reward per Episode)")
        plt.show()

    def find_best_path(self):
        # Start at the initial state
        state, _ = self.env.reset()
        done = False
        path = [state]

        while not done:
            # Choose the best action (exploitation)
            action = np.argmax(self.q_table[state])
            next_state, _, done = self.env.step(action)
            path.append(next_state)
            state = next_state

        # Save the best path
        self.best_path = path
        return path

    def visualize_best_path(self):
        path = self.best_path
        print("Best path taken by the agent:")
        for step in path:
            y, x = divmod(step, self.env.size)
            print(f"Step: Agent at ({y},{x})")

        # Plot the best path on the maze
        maze = np.copy(self.env.maze)
        for step in path:
            y, x = divmod(step, self.env.size)
            maze[y, x] = 2  # Mark the path with '2' (or another unique number)

        plt.imshow(maze, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title("Best Path Taken by the Agent")
        plt.show()

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
