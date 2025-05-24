import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

# Hyperparameter setup
GAMMA = 0.99  # Focus on long-term rewards
TOTAL_TIMESTEPS = 500000  # Increased training time
NOISE_SIGMA = 0.5  # More exploration
LEARNING_RATE = 3e-4  # Slower learning for stability

def train_and_save(GAMMA):
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1)
    
    # Improved action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=NOISE_SIGMA * np.ones(n_actions),
        theta=0.15,
        dt=1e-2
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
        verbose=1
    )
    
    # Progressive training with periodic saving
    for i in range(5):
        model.learn(total_timesteps=TOTAL_TIMESTEPS//5)
        model.save(f"td3_mountaincar_v2_gamma_{GAMMA}_{i+1}")
        print(f"Checkpoint {i+1}/5 saved")
    
    env.close()

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
    print(f"Success rate: {successes/num_episodes:.2%}")

# Execute training and evaluation
for gamma in [0.99, 0.95, 0.90]:
    # train_and_save(gamma)
    evaluate_model("td3_mountaincar_v2_gamma_0.99_5")  # Evaluate final checkpoint
