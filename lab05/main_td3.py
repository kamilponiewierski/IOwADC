import os
import shutil
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from scipy import interpolate
import torch.nn as nn
from stable_baselines3.td3.policies import TD3Policy

class RewardTrackingCallback(BaseCallback):
    """
    Callback for tracking rewards during training and saving best model
    """
    def __init__(self, check_freq: int = 1000, log_dir: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rewards = []
        self.timesteps = []
        self.best_reward = -np.inf
        self.best_model_path = None
        self.episode_rewards = []
        
    def _init_callback(self) -> None:
        # Create save path
        if self.log_dir is not None:
            self.best_model_path = os.path.join(self.log_dir, "best_model")
            os.makedirs(self.log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Check for completed episodes in infos
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    self.episode_rewards.append(episode_reward)
                    
                    # Save best model immediately when new best is found
                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward
                        if self.best_model_path and self.verbose >= 1:
                            print(f"New best reward: {episode_reward:.2f}, saving model to {self.best_model_path}")
                        if self.best_model_path:
                            self.model.save(self.best_model_path)
        
        # Regular tracking every check_freq steps
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                # Use recent episode rewards for tracking
                recent_rewards = self.episode_rewards[-min(10, len(self.episode_rewards)):]
                mean_reward = np.mean(recent_rewards)
                self.rewards.append(mean_reward)
                self.timesteps.append(self.num_timesteps)
        
        return True

# Define custom network architectures
class SmallNetwork(nn.Module):
    """Small network: 8 -> 64 -> 32 -> output"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class LargeNetwork(nn.Module):
    """Large network: 8 -> 256 -> 128 -> 64 -> output"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Custom TD3 policies for different architectures
class SmallTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = None  # We'll override the networks directly

class LargeTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = None  # We'll override the networks directly

def run_experiment_with_architecture(architecture_name, net_arch, run_id, total_timesteps=100000):
    """
    Run experiment with specific network architecture
    """
    log_dir = f"tmp/{architecture_name}_run_{run_id}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = gym.make("LunarLanderContinuous-v3")
    env = Monitor(env, log_dir)
    
    # Action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Create model with specific architecture
    model = TD3(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        policy_kwargs={"net_arch": net_arch},
        learning_rate=3e-4,
        verbose=0
    )
    
    # Create callback with proper log_dir
    callback = RewardTrackingCallback(check_freq=1000, log_dir=log_dir, verbose=1)
    
    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Also save final model as backup
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    
    # Return best model path along with other data
    best_model_path = os.path.join(log_dir, "best_model")
    
    # Load results
    try:
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(x) > 0:
            return x, y, best_model_path, callback.best_reward
    except:
        pass
    
    return np.array(callback.timesteps), np.array(callback.rewards), best_model_path, callback.best_reward

def draw_detailed_network_schema(architecture_name, net_arch, input_dim, output_dim, ax):
    """
    Draw detailed network architecture schema with individual neurons
    """
    ax.clear()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # Define layers and their properties
    layers = [input_dim] + net_arch + [output_dim]
    if architecture_name == "Small Network":
        layer_names = ["Input Layer\n(State Vector)", "Hidden Layer 1\n(ReLU)", "Hidden Layer 2\n(ReLU)", "Output Layer\n(Actions)"]
        colors = ['#E3F2FD', '#C8E6C9', '#C8E6C9', '#FFCDD2']
    else:
        layer_names = ["Input Layer\n(State Vector)", "Hidden Layer 1\n(ReLU)", "Hidden Layer 2\n(ReLU)", "Hidden Layer 3\n(ReLU)", "Output Layer\n(Actions)"]
        colors = ['#E3F2FD', '#C8E6C9', '#C8E6C9', '#C8E6C9', '#FFCDD2']
    
    x_positions = np.linspace(1.5, 10.5, len(layers))
    max_neurons_to_show = 8  # Maximum neurons to visualize per layer
    
    neuron_positions = []  # Store positions for drawing connections
    
    # Draw each layer
    for layer_idx, (x_pos, layer_size, layer_name, color) in enumerate(zip(x_positions, layers, layer_names, colors)):
        # Determine how many neurons to show
        neurons_to_show = min(layer_size, max_neurons_to_show)
        show_dots = layer_size > max_neurons_to_show
        
        # Calculate vertical positions for neurons
        if neurons_to_show == 1:
            y_positions = [5]
        else:
            y_spacing = min(6 / neurons_to_show, 0.8)
            total_height = (neurons_to_show - 1) * y_spacing
            y_start = 5 - total_height / 2
            y_positions = [y_start + i * y_spacing for i in range(neurons_to_show)]
        
        layer_neuron_positions = []
        
        # Draw neurons
        for i, y_pos in enumerate(y_positions):
            if i < neurons_to_show - (1 if show_dots else 0):
                # Regular neuron
                circle = plt.Circle((x_pos, y_pos), 0.15, facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(circle)
                layer_neuron_positions.append((x_pos, y_pos))
            elif show_dots and i == neurons_to_show - 1:
                # Show dots to indicate more neurons
                for dot_i in range(3):
                    ax.plot(x_pos, y_pos - 0.1 + dot_i * 0.1, 'ko', markersize=3)
                # Add position for the conceptual "last" neuron
                layer_neuron_positions.append((x_pos, y_positions[0]))  # Use first position as representative
        
        neuron_positions.append(layer_neuron_positions)
        
        # Add layer label
        ax.text(x_pos, 1.5, layer_name, ha='center', va='center', fontsize=9, weight='bold')
        ax.text(x_pos, 1, f"{layer_size} neurons", ha='center', va='center', fontsize=8)
        
        # Add activation function
        if layer_idx > 0 and layer_idx < len(layers) - 1:
            ax.text(x_pos, 8.5, "ReLU", ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        elif layer_idx == len(layers) - 1:
            ax.text(x_pos, 8.5, "Linear", ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
    
    # Draw connections between layers
    for layer_idx in range(len(neuron_positions) - 1):
        current_layer = neuron_positions[layer_idx]
        next_layer = neuron_positions[layer_idx + 1]
        
        # Draw sample connections (not all to avoid clutter)
        for i, (x1, y1) in enumerate(current_layer[:3]):  # Show connections from first 3 neurons
            for j, (x2, y2) in enumerate(next_layer[:3]):  # to first 3 neurons of next layer
                ax.plot([x1 + 0.15, x2 - 0.15], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)
    
    # Add title and annotations
    ax.set_title(f"{architecture_name} - Detailed Architecture", fontsize=14, weight='bold', pad=20)
    
    # Add input/output descriptions
    if input_dim == 8:
        input_desc = ["x position", "y position", "x velocity", "y velocity", 
                     "angle", "angular velocity", "left leg contact", "right leg contact"]
        for i, desc in enumerate(input_desc[:min(len(input_desc), max_neurons_to_show)]):
            if i < len(neuron_positions[0]):
                x_pos, y_pos = neuron_positions[0][i]
                ax.text(x_pos - 0.8, y_pos, desc, ha='right', va='center', fontsize=7, style='italic')
    
    if output_dim == 2:
        output_desc = ["Main engine", "Side engines"]
        for i, desc in enumerate(output_desc):
            if i < len(neuron_positions[-1]):
                x_pos, y_pos = neuron_positions[-1][i]
                ax.text(x_pos + 0.8, y_pos, desc, ha='left', va='center', fontsize=7, style='italic')
    
    ax.axis('off')

def create_network_comparison_figure():
    """
    Create a dedicated figure for network architecture comparison
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    
    # Draw both architectures
    draw_detailed_network_schema("Small Network", architectures["Small Network"], 
                                input_dim, output_dim, axes[0])
    draw_detailed_network_schema("Large Network", architectures["Large Network"], 
                                input_dim, output_dim, axes[1])
    
    # Add comparison text
    fig.suptitle("Neural Network Architectures Comparison\nTD3 Actor-Critic Networks", 
                fontsize=16, weight='bold', y=0.95)
    
    # Add technical details
    details_text = """
    Technical Details:
    • Input: 8-dimensional state vector (position, velocity, orientation, contact sensors)
    • Output: 2-dimensional continuous action vector (engine thrusts)
    • Training Algorithm: Twin Delayed Deep Deterministic Policy Gradients (TD3)
    • Both networks use the same architecture for Actor and Critic networks
    • Weights initialized using Xavier/Glorot initialization
    • Optimizer: Adam with learning rate 3e-4
    """
    
    fig.text(0.02, 0.02, details_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def evaluate_saved_agent(model_path, env_name, n_episodes=10):
    """
    Load and evaluate saved agent without exploration
    """
    try:
        # Load the saved model
        model = TD3.load(model_path)
        eval_env = gym.make(env_name)
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Use deterministic action (no exploration)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        eval_env.close()
        return episode_rewards
    
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return []

# Network architectures to test
architectures = {
    "Small Network": [64, 32],      # 8 -> 64 -> 32 -> 2
    "Large Network": [256, 128, 64] # 8 -> 256 -> 128 -> 64 -> 2
}

# Environment info
env_temp = gym.make("LunarLanderContinuous-v3")
input_dim = env_temp.observation_space.shape[0]  # 8 dimensions
output_dim = env_temp.action_space.shape[0]      # 2 dimensions
env_temp.close()

print(f"Environment: LunarLanderContinuous-v3")
print(f"Input dimensions: {input_dim} (state vector)")
print(f"Output dimensions: {output_dim} (continuous actions)")
print(f"State components: [x, y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact]")
print(f"Action components: [main_engine_thrust, side_engine_thrust]")

# Run experiments
n_runs = 5
total_timesteps = 100000
all_results = {}
best_models = {}

print("\nStarting architecture comparison...")

for arch_name, net_arch in architectures.items():
    print(f"\nTesting {arch_name}...")
    
    timesteps_list = []
    rewards_list = []
    models = []
    best_rewards = []
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}")
        
        try:
            timesteps, rewards, best_model_path, best_reward = run_experiment_with_architecture(
                arch_name, net_arch, run, total_timesteps)
            timesteps_list.append(timesteps)
            rewards_list.append(rewards)
            models.append(best_model_path)  # Store path instead of model object
            best_rewards.append(best_reward)
        except Exception as e:
            print(f"    Error in run {run + 1}: {e}")
            timesteps_list.append(np.array([]))
            rewards_list.append(np.array([]))
            models.append(None)
            best_rewards.append(-np.inf)
    
    all_results[arch_name] = (timesteps_list, rewards_list)
    
    # Find best model
    if best_rewards:
        best_idx = np.argmax(best_rewards)
        best_models[arch_name] = {
            'model_path': models[best_idx],  # Store path
            'reward': best_rewards[best_idx]
        }

# Create detailed network architecture visualization
print("\nGenerating detailed network architecture diagrams...")
network_fig = create_network_comparison_figure()
network_fig.show()

# Create visualization for results
fig = plt.figure(figsize=(16, 10))

# Learning curves comparison
ax3 = plt.subplot(2, 1, 1)

# Common timesteps for interpolation
common_timesteps = np.linspace(0, total_timesteps, 100)
colors = ['blue', 'red']

for i, (arch_name, (timesteps_list, rewards_list)) in enumerate(all_results.items()):
    # Filter valid results
    valid_results = [(t, r) for t, r in zip(timesteps_list, rewards_list) 
                    if len(t) > 0 and len(r) > 0]
    
    if len(valid_results) > 0:
        valid_timesteps, valid_rewards = zip(*valid_results)
        
        # Interpolate
        interpolated_rewards = []
        for timesteps, rewards in zip(valid_timesteps, valid_rewards):
            if len(timesteps) > 1:
                f = interpolate.interp1d(timesteps, rewards, kind='linear', 
                                       bounds_error=False, fill_value='extrapolate')
                interp_rewards = f(common_timesteps)
                interpolated_rewards.append(interp_rewards)
        
        if interpolated_rewards:
            interpolated_rewards = np.array(interpolated_rewards)
            mean_rewards = np.mean(interpolated_rewards, axis=0)
            std_rewards = np.std(interpolated_rewards, axis=0)
            
            ax3.plot(common_timesteps, mean_rewards, color=colors[i], linewidth=2,
                    label=f'{arch_name} (training)')
            ax3.fill_between(common_timesteps, mean_rewards - std_rewards,
                           mean_rewards + std_rewards, color=colors[i], alpha=0.2)

ax3.set_xlabel('Timesteps')
ax3.set_ylabel('Episode Reward')
ax3.set_title('Learning Curves Comparison During Training')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Evaluation comparison
ax4 = plt.subplot(2, 1, 2)

eval_results = {}
print("\nEvaluating best models (deterministic policy)...")

for arch_name, model_info in best_models.items():
    if model_info['model_path'] is not None and os.path.exists(model_info['model_path'] + '.zip'):
        print(f"Evaluating {arch_name} (training best: {model_info['reward']:.2f})")
        print(f"Loading model from: {model_info['model_path']}")
        eval_rewards = evaluate_saved_agent(model_info['model_path'], "LunarLanderContinuous-v3", n_episodes=20)
        
        if eval_rewards:  # Only proceed if evaluation was successful
            eval_results[arch_name] = eval_rewards
            
            # Plot evaluation results
            ax4.scatter([arch_name] * len(eval_rewards), eval_rewards, 
                       alpha=0.6, s=50, label=f'{arch_name} evaluation')
            
            # Add mean line
            mean_eval = np.mean(eval_rewards)
            ax4.axhline(y=mean_eval, color=colors[list(best_models.keys()).index(arch_name)], 
                       linestyle='--', alpha=0.8)
            
            print(f"  Evaluation mean: {mean_eval:.2f} ± {np.std(eval_rewards):.2f}")
        else:
            print(f"  Failed to evaluate {arch_name}")
    else:
        print(f"  No saved model found for {arch_name}")

ax4.set_ylabel('Episode Reward')
ax4.set_title('Evaluation Results (Deterministic Policy, No Exploration)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Print detailed comparison
print("\n" + "="*70)
print("NETWORK ARCHITECTURES COMPARISON")
print("="*70)

print(f"\nEnvironment Details:")
print(f"  Input: {input_dim}D state vector")
print(f"    - Position: x, y coordinates")
print(f"    - Velocity: vel_x, vel_y")
print(f"    - Orientation: angle, angular_velocity") 
print(f"    - Contact: leg1_contact, leg2_contact (boolean)")
print(f"  Output: {output_dim}D continuous action vector")
print(f"    - Main engine thrust: [-1, 1]")
print(f"    - Side engines thrust: [-1, 1]")

for arch_name, net_arch in architectures.items():
    total_params = input_dim * net_arch[0]  # First layer
    for i in range(len(net_arch) - 1):
        total_params += net_arch[i] * net_arch[i + 1]
    total_params += net_arch[-1] * output_dim  # Output layer
    
    print(f"\n{arch_name}:")
    print(f"  Architecture: {input_dim} -> {' -> '.join(map(str, net_arch))} -> {output_dim}")
    print(f"  Activation functions: ReLU (hidden layers), Linear (output)")
    print(f"  Approximate parameters: {total_params:,}")
    
    if arch_name in best_models and best_models[arch_name]['model_path'] is not None:
        print(f"  Best training reward: {best_models[arch_name]['reward']:.2f}")
        print(f"  Model saved at: {best_models[arch_name]['model_path']}")
        
        if arch_name in eval_results:
            eval_mean = np.mean(eval_results[arch_name])
            eval_std = np.std(eval_results[arch_name])
            print(f"  Evaluation performance: {eval_mean:.2f} ± {eval_std:.2f}")

# Print saved model locations
print("\n" + "="*70)
print("SAVED MODELS INFORMATION")
print("="*70)
for arch_name, model_info in best_models.items():
    if model_info['model_path'] is not None:
        model_path = model_info['model_path']
        if os.path.exists(model_path + '.zip'):
            print(f"\n✅ {arch_name}:")
            print(f"   Model saved at: {model_path}.zip")
            print(f"   Best reward during training: {model_info['reward']:.2f}")
            print(f"   File size: {os.path.getsize(model_path + '.zip') / 1024:.1f} KB")
        else:
            print(f"\n❌ {arch_name}: Model file not found at {model_path}")

plt.show()

# Clean up - but keep the best models
print(f"\nNote: Best trained models are saved in 'tmp/' directory and will NOT be deleted.")
print("You can load them later using: model = TD3.load('path_to_model')")

# # Only clean up non-best model files
# try:
#     for arch_name in architectures.keys():
#         for run in range(n_runs):
#             run_dir = f"tmp/{arch_name}_run_{run}/"
#             if os.path.exists(run_dir):
#                 # Keep only the best model directories
#                 is_best = False
#                 if arch_name in best_models:
#                     best_path = best_models[arch_name]['model_path']
#                     if best_path and run_dir in best_path:
#                         is_best = True
#                         print(f"Keeping best model directory: {run_dir}")
                
#                 if not is_best:
#                     shutil.rmtree(run_dir)
    
#     print("Cleaned up non-essential temporary files.")
# except Exception as e:
#     print(f"Note: Some temporary files may remain: {e}")