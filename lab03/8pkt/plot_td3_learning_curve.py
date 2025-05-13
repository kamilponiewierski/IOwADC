import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Rozwiązuje błąd OMP

import gymnasium as gym
from stable_baselines3 import TD3
import numpy as np
import matplotlib.pyplot as plt

# Konfiguracja
gammas = [0.9, 0.95, 0.99]
checkpoints = [1, 2, 3, 4, 5]
rewards_all = {}

def evaluate_model(path, episodes=10):
    model = TD3.load(path)
    env = gym.make("MountainCarContinuous-v0")
    episode_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)

    env.close()
    return np.mean(episode_rewards)

# Ocena każdego checkpointu
for gamma in gammas:
    rewards = []
    for i in checkpoints:
        path = f"td3_mountaincar_v2_gamma_{gamma}_{i}"
        avg_reward = evaluate_model(path, episodes=10)
        print(f"[gamma={gamma}] Checkpoint {i}: avg reward = {avg_reward:.2f}")
        rewards.append(avg_reward)
    rewards_all[gamma] = rewards

# Rysowanie wykresu
plt.figure()
for gamma in gammas:
    plt.plot(checkpoints, rewards_all[gamma], marker='o', label=f"$\\gamma$ = {gamma}")

plt.xlabel("Checkpoint (co 100k timesteps)")
plt.ylabel("Średnia nagroda (10 epizodów)")
plt.title("Krzywa uczenia TD3 dla różnych wartości gamma")
plt.grid()
plt.legend()
plt.tight_layout()

# Zapis i pokazanie wykresu
plt.savefig("learning_curve_training.png")
print("✅ Wykres zapisany jako 'learning_curve_training.png'")
plt.show()
