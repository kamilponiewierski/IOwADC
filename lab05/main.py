from stable_baselines3 import TD3
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import os
import matplotlib.pyplot as plt

TOTAL_TIMESTEPS = 50000
LEARNING_RATE = 3e-4
NOISE_SIGMA = 0.5


def train_and_save(GAMMA):
    if os.path.exists(f"lab05/models/td3_lunarlander_v2_gamma_{GAMMA}_5"):
        print(f"Model for gamma={GAMMA} already trained and saved. Skipping training.")
        return

    def get_gym(): 
        return gym.make(
            "LunarLander-v3",
            continuous=True,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
        )
    
    env = make_vec_env(get_gym)

    # Improved action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=NOISE_SIGMA * np.ones(n_actions),
        theta=0.15,
        dt=1e-2,
    )

    # Enhanced TD3 configuration
    model = TD3(
        "MlpPolicy",
        env,
        gamma=GAMMA,
        action_noise=action_noise,
        learning_rate=LEARNING_RATE,
        buffer_size=1_000_000,
        batch_size=256,
        policy_kwargs=dict(net_arch=[256, 256]),  # Larger network
        learning_starts=5000,
        train_freq=(1, "episode"),
        gradient_steps=64,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        verbose=1,
    )

    # Progressive training with periodic saving
    for i in range(5):
        model.learn(total_timesteps=TOTAL_TIMESTEPS // 5)
        model.save(f"lab05/models/td3_lunarlander_v2_gamma_{GAMMA}_{i+1}")
        print(f"Checkpoint {i+1}/5 saved")

    env.close()


def evaluate_model(model_path, num_episodes=50):
    model = TD3.load(model_path)
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        # render_mode='human',
    )
    
    if (not os.path.exists(f'lab05/charts/rewards_{model_path}')):
        successes = 0
        episode_rewards = []  # List to store total rewards per episode
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0  # Track total reward for the current episode
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if terminated and info.get("is_success", False):
                    successes += 1
            episode_rewards.append(total_reward)  # Store the total reward for this episode

        create_chart(model_path, num_episodes, episode_rewards)

    env.close()
    print(f"Success rate: {successes/num_episodes:.2%}")

def create_chart(model_path, num_episodes, episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_episodes), episode_rewards, marker="o")
    plt.title(f"Rewards per Episode - {model_path}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(f"lab05/charts/rewards_{model_path.split('/')[-1]}.png")
    plt.show()
    print(f"Model {model_path} evaluated with {num_episodes} episodes.")


for gamma in [0.99, 0.95, 0.90]:
    train_and_save(gamma)
    evaluate_model(f"lab05/models/td3_lunarlander_v2_gamma_{gamma}_5", num_episodes=50)
