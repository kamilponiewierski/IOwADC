import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from scipy import interpolate

class RewardTrackingCallback(BaseCallback):
    """
    Callback for tracking rewards during training
    """
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.timesteps = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get recent episode rewards
            if hasattr(self.training_env, 'get_episode_rewards'):
                episode_rewards = self.training_env.get_episode_rewards()
                if len(episode_rewards) > 0:
                    # Store average of last few episodes
                    recent_rewards = episode_rewards[-min(10, len(episode_rewards)):]
                    mean_reward = np.mean(recent_rewards)
                    self.rewards.append(mean_reward)
                    self.timesteps.append(self.num_timesteps)
        return True

def run_experiment(hyperparams, run_id, total_timesteps=50000):
    """
    Run single experiment with given hyperparameters
    """
    # Create unique log directory for this run
    log_dir = f"tmp/run_{run_id}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create and wrap environment
    env = gym.make("LunarLanderContinuous-v3")
    env = Monitor(env, log_dir)
    
    # Create action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=hyperparams['noise_sigma'] * np.ones(n_actions)
    )
    
    # Create model with hyperparameters
    model = TD3(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        learning_rate=hyperparams['learning_rate'],
        buffer_size=hyperparams['buffer_size'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        tau=hyperparams['tau'],
        verbose=0
    )
    
    # Create callback for tracking
    callback = RewardTrackingCallback(check_freq=1000)
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Load results from monitor
    try:
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(x) > 0:
            return x, y
    except:
        pass
    
    # Fallback to callback data if monitor fails
    return np.array(callback.timesteps), np.array(callback.rewards)

def interpolate_rewards(timesteps_list, rewards_list, target_timesteps):
    """
    Interpolate all reward curves to common timestep points
    """
    interpolated_rewards = []
    
    for timesteps, rewards in zip(timesteps_list, rewards_list):
        if len(timesteps) > 1 and len(rewards) > 1:
            # Create interpolation function
            f = interpolate.interp1d(timesteps, rewards, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            # Interpolate to target timesteps
            interp_rewards = f(target_timesteps)
            interpolated_rewards.append(interp_rewards)
        else:
            # If not enough data points, use zeros
            interpolated_rewards.append(np.zeros_like(target_timesteps))
    
    return np.array(interpolated_rewards)

# Define three different hyperparameter sets
hyperparameter_sets = {
    'Conservative': {
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005,
        'noise_sigma': 0.1
    },
    'Aggressive': {
        'learning_rate': 1e-3,
        'buffer_size': 200000,
        'batch_size': 128,
        'gamma': 0.95,
        'tau': 0.01,
        'noise_sigma': 0.2
    },
    'Balanced': {
        'learning_rate': 5e-4,
        'buffer_size': 150000,
        'batch_size': 96,
        'gamma': 0.97,
        'tau': 0.007,
        'noise_sigma': 0.15
    }
}

# Number of runs for each hyperparameter set
n_runs = 10
total_timesteps = 50000

# Storage for results
all_results = {}

print("Starting experiments...")

# Run experiments for each hyperparameter set
for hp_name, hp_config in hyperparameter_sets.items():
    print(f"\nRunning experiments for {hp_name} hyperparameters...")
    
    timesteps_list = []
    rewards_list = []
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}")
        
        try:
            timesteps, rewards = run_experiment(hp_config, f"{hp_name}_{run}", total_timesteps)
            timesteps_list.append(timesteps)
            rewards_list.append(rewards)
        except Exception as e:
            print(f"    Error in run {run + 1}: {e}")
            # Add empty arrays to maintain consistency
            timesteps_list.append(np.array([]))
            rewards_list.append(np.array([]))
    
    all_results[hp_name] = (timesteps_list, rewards_list)

# Create common timestep array for interpolation
max_timesteps = total_timesteps
common_timesteps = np.linspace(0, max_timesteps, 100)

# Plot learning curves
plt.figure(figsize=(12, 8))

colors = ['blue', 'red', 'green']
styles = ['-', '--', '-.']

for i, (hp_name, (timesteps_list, rewards_list)) in enumerate(all_results.items()):
    print(f"\nProcessing {hp_name} results...")
    
    # Filter out empty results
    valid_results = [(t, r) for t, r in zip(timesteps_list, rewards_list) 
                    if len(t) > 0 and len(r) > 0]
    
    if len(valid_results) == 0:
        print(f"  No valid results for {hp_name}")
        continue
    
    valid_timesteps, valid_rewards = zip(*valid_results)
    
    # Interpolate to common timesteps
    interpolated_rewards = interpolate_rewards(valid_timesteps, valid_rewards, common_timesteps)
    
    if interpolated_rewards.shape[0] > 0:
        # Calculate mean and standard deviation
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        std_rewards = np.std(interpolated_rewards, axis=0)
        
        # Plot mean curve
        plt.plot(common_timesteps, mean_rewards, 
                color=colors[i], linestyle=styles[i], linewidth=2,
                label=f'{hp_name} (n={len(valid_results)})')
        
        # Plot confidence interval (mean ± std)
        plt.fill_between(common_timesteps, 
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        color=colors[i], alpha=0.2)
        
        print(f"  Final mean reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    else:
        print(f"  No data to plot for {hp_name}")

plt.xlabel('Timesteps')
plt.ylabel('Average Episode Reward')
plt.title('TD3 Learning Curves - LunarLanderContinuous-v3\n(Mean ± Standard Deviation over multiple runs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Print hyperparameter details
print("\n" + "="*60)
print("HYPERPARAMETER CONFIGURATIONS:")
print("="*60)
for hp_name, hp_config in hyperparameter_sets.items():
    print(f"\n{hp_name}:")
    for param, value in hp_config.items():
        print(f"  {param}: {value}")

plt.show()

# Clean up temporary directories
import shutil
try:
    shutil.rmtree("tmp/")
    print("\nCleaned up temporary files.")
except:
    print("\nNote: Some temporary files may remain in 'tmp/' directory.")