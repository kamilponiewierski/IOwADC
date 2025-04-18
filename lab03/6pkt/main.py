import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


def train_and_evaluate(gamma: float, total_timesteps=150_000):
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env)  # dodanie wrappera do zapisu statystyk

    model = PPO("MlpPolicy", env, verbose=1, gamma=gamma)
    model.learn(total_timesteps=total_timesteps)

    episode_rewards = env.get_episode_rewards()
    rewards, _ = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True)
    env.close()
    return model, rewards, episode_rewards


def show_final_run(model, attempts=5):
    best_reward = -float("inf")
    best_position = -float("inf")

    for attempt in range(attempts):
        env = gym.make("MountainCarContinuous-v0", render_mode="human")
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Attempt {attempt + 1}: Total reward = {total_reward:.2f}, Final position = {obs[0]:.4f}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_position = obs[0]

        env.close()

    print(f"\nBest run: Total reward = {best_reward:.2f}, Final position = {best_position:.4f}")


def main():
    gammas = [0.8, 0.9, 0.99]
    all_rewards = {}
    all_learning_curves = {}
    models = {}

    for gamma in gammas:
        print(f"Training with gamma={gamma}...")
        model, eval_rewards, episode_rewards = train_and_evaluate(gamma)
        all_rewards[gamma] = eval_rewards
        all_learning_curves[gamma] = episode_rewards
        models[gamma] = model
        model.save(f"ppo_mountaincar_gamma_{gamma}")  # Zapisz model do pliku
        print(f"Mean reward: {np.mean(eval_rewards):.2f}, Std: {np.std(eval_rewards):.2f}\n")

    # Plotting - porównanie współczynników dyskontowych (średnie rewardy)
    plt.figure(figsize=(10, 6))
    for gamma, rewards in all_rewards.items():
        plt.plot(rewards, label=f"gamma={gamma}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Porównanie współczynników dyskontowych - Ewaluacja")
    plt.legend()
    plt.grid()
    plt.savefig("learning_curve_comparison_eval.png")
    plt.show()

    # Plotting - learning curves z treningu
    plt.figure(figsize=(10, 6))
    for gamma, episode_rewards in all_learning_curves.items():
        plt.plot(episode_rewards, label=f"gamma={gamma}")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Krzywe uczenia podczas treningu")
    plt.legend()
    plt.grid()
    plt.savefig("learning_curve_training.png")
    plt.show()

    # Pokaż finałową jazdę najlepszego modelu (gamma=0.99)
    print("\nPokazuję finalny przejazd dla gamma=0.99...")
    show_final_run(models[0.99])


if __name__ == "__main__":
    main()
