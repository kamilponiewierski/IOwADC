import numpy as np
import random
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.episode_rewards = []
        self.best_path = []

    def train(self, episodes=3000, max_steps_per_episode=300):
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            for _ in range(max_steps_per_episode):
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, done = self.env.step(action)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                self.q_table[state, action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

                state = next_state
                total_reward += reward

                if done:
                    break

            self.episode_rewards.append(total_reward)
            self.epsilon = max(0.05, self.epsilon * 0.995)

            if episode % 100 == 0:
                print(f"Epizod {episode}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")

        np.save("q_table.npy", self.q_table)

    def plot_learning_curve(self):
        plt.plot(self.episode_rewards)
        plt.xlabel("Epizod")
        plt.ylabel("Całkowita nagroda")
        plt.title("Krzywa uczenia")
        plt.grid()
        plt.show()

    def find_best_path(self, max_steps=300):
        state, _ = self.env.reset()
        path = [self.env.agent_pos[:]]
        for _ in range(max_steps):
            action = np.argmax(self.q_table[state])
            state, _, done = self.env.step(action)
            path.append(self.env.agent_pos[:])
            if done:
                break

        self.best_path = path
        print("🔁 Długość ścieżki:", len(path))
        print("📍 Pozycja końcowa agenta:", self.env.agent_pos)
        print("🎯 Czy osiągnięto cel:", self.env.agent_pos == self.env.goal_pos)
        return path
