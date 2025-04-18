import gymnasium as gym
from stable_baselines3 import PPO


def show_final_run(model_path: str):
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Final total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    show_final_run("ppo_mountaincar_gamma_0.99")
