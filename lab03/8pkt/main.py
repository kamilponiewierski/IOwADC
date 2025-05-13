import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameter setup
TOTAL_TIMESTEPS = 500000
NOISE_SIGMA = 0.5
LEARNING_RATE = 3e-4


def train_and_save(GAMMA):
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1)

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=NOISE_SIGMA * np.ones(n_actions),
        theta=0.15,
        dt=1e-2
    )

    model = TD3(
        "MlpPolicy",
        env,
        gamma=GAMMA,
        action_noise=action_noise,
        learning_rate=LEARNING_RATE,
        buffer_size=1_000_000,
        batch_size=256,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_starts=5000,
        train_freq=(1, "episode"),
        gradient_steps=64,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        verbose=0
    )

    rewards_per_stage = []

    for i in range(5):
        model.learn(total_timesteps=TOTAL_TIMESTEPS // 5, reset_num_timesteps=False)
        model.save(f"td3_mountaincar_v2_gamma_{GAMMA}_{i + 1}")
        print(f"Checkpoint {i + 1}/5 saved")

        # Krótkie testowanie (10 epizodów) — średnia nagroda
        eval_env = gym.make("MountainCarContinuous-v0")
        episode_rewards = []

        for _ in range(10):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward)

        avg_reward = np.mean(episode_rewards)
        rewards_per_stage.append(avg_reward)
        eval_env.close()

    env.close()

    # Rysowanie krzywej uczenia
    plt.plot(range(1, 6), rewards_per_stage, marker='o', label=f"$\\gamma$ = {GAMMA}")


def evaluate_model(model_path, num_episodes=50):
    model = TD3.load(model_path)
    env = gym.make("MountainCarContinuous-v0", render_mode='human')

    successes = 0
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                successes += 1
            done = terminated or truncated

    env.close()
    print(f"Success rate: {successes / num_episodes:.2%}")


# Uruchamianie treningu dla różnych gamma i generowanie wspólnego wykresu
if __name__ == "__main__":
    for gamma in [0.9, 0.95, 0.99]:
        train_and_save(gamma)

    # Wyświetlenie wykresu
    plt.xlabel("Etap (co 100k timesteps)")
    plt.ylabel("Średnia nagroda z 10 epizodów")
    plt.title("Krzywa uczenia TD3 dla różnych wartości gamma")
    plt.grid()
    plt.legend()
    plt.savefig("td3_learning_curves_all_gamma.png")
    plt.show()
