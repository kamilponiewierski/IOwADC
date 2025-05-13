import numpy as np
import random
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.env = env
        self.alpha = alpha          # współczynnik uczenia
        self.gamma = gamma          # współczynnik dyskontowy
        self.epsilon = epsilon      # współczynnik eksploracji
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.episode_rewards = []
        self.best_path = []

    def train(self, episodes=3000, max_steps_per_episode=500):
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < max_steps_per_episode:
                # Eksploracja vs Eksploatacja
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, done = self.env.step(action)

                # Aktualizacja Q-tabeli
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                self.q_table[state, action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

                state = next_state
                total_reward += reward
                steps += 1

            self.episode_rewards.append(total_reward)

            # 🔁 Dynamiczne zmniejszanie epsilonu
            self.epsilon = max(0.05, self.epsilon * 0.995)

            # Co 100 epizodów logujemy postęp
            if episode % 100 == 0:
                print(f"Epizod {episode}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")

    def plot_learning_curve(self):
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Curve")
        plt.grid(True)
        plt.show()

    def find_best_path(self, max_steps=1000):
        state, _ = self.env.reset()
        done = False
        path = [self.env.agent_pos[:]]
        steps = 0

        while not done and steps < max_steps:
            action = np.argmax(self.q_table[state])
            state, _, done = self.env.step(action)
            path.append(self.env.agent_pos[:])
            steps += 1

        self.best_path = path
        return path
