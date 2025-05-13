import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from stable_baselines3 import TD3  # <-- UWAGA: zmienione z PPO na TD3

def show_final_run(model_path: str):
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    model = TD3.load(model_path)  # <-- również TD3 zamiast PPO

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Final total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    show_final_run("td3_mountaincar_v2_gamma_0.9_5")  # <-- najlepszy checkpoint TD3
