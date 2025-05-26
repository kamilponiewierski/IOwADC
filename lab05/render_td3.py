import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
from typing import List, Tuple, Optional
import glob
import json
from datetime import datetime

class TD3ResultsRenderer:
    """
    A comprehensive class for rendering and analyzing TD3 model training results.
    """
    
    def __init__(self, log_dir: str = "tmp/", model_path: str = None):
        """
        Initialize the results renderer.
        
        Args:
            log_dir: Directory containing training logs
            model_path: Path to saved model (optional)
        """
        self.log_dir = log_dir
        self.model_path = model_path or os.path.join(log_dir, "best_model")
        self.env_name = "LunarLanderContinuous-v3"
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_training_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training results from monitor logs."""
        try:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            return x, y
        except Exception as e:
            print(f"Error loading training results: {e}")
            return np.array([]), np.array([])
    
    def plot_training_progress(self, save_fig: bool = True, show_plot: bool = True):
        """Plot training progress showing reward over time."""
        x, y = self.load_training_results()
        
        if len(x) == 0:
            print("No training data found!")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot raw rewards
        ax1.plot(x, y, alpha=0.6, color='lightblue', linewidth=0.5)
        
        # Plot moving average
        window_size = min(100, len(y) // 10)
        if window_size > 1:
            moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
            moving_x = x[window_size-1:]
            ax1.plot(moving_x, moving_avg, color='red', linewidth=2, 
                    label=f'Moving Average ({window_size} episodes)')
        
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('TD3 Training Progress - Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot learning curve (cumulative average)
        cumulative_avg = np.cumsum(y) / np.arange(1, len(y) + 1)
        ax2.plot(x, cumulative_avg, color='green', linewidth=2)
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Cumulative Average Reward')
        ax2.set_title('TD3 Learning Curve - Cumulative Average')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.log_dir, 'training_progress.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def evaluate_model(self, n_eval_episodes: int = 100, render: bool = False) -> dict:
        """
        Evaluate the trained model and return performance metrics.
        
        Args:
            n_eval_episodes: Number of episodes to evaluate
            render: Whether to render the environment during evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not os.path.exists(self.model_path + ".zip"):
            print(f"Model not found at {self.model_path}.zip")
            return {}
        
        # Load the model
        model = TD3.load(self.model_path + '.zip')
        
        # Create evaluation environment
        eval_env = gym.make(self.env_name, render_mode="human" if render else None)
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, 
            deterministic=True, return_episode_rewards=False
        )
        
        # Get episode rewards for more detailed analysis
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            if render:
                print(f"Episode {episode + 1}/{n_eval_episodes}")
            
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if render:
                print(f"Episode reward: {episode_reward:.2f}")
        
        eval_env.close()
        
        # Calculate additional metrics
        metrics = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'success_rate': np.mean(np.array(episode_rewards) > 200),  # LunarLander success threshold
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        return metrics
    
    def plot_evaluation_results(self, metrics: dict, save_fig: bool = True, show_plot: bool = True):
        """Plot evaluation results."""
        if not metrics:
            print("No evaluation metrics to plot!")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Episode rewards histogram
        ax1.hist(metrics['episode_rewards'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(metrics['mean_reward'], color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {metrics["mean_reward"]:.2f}')
        ax1.axvline(metrics['median_reward'], color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {metrics["median_reward"]:.2f}')
        ax1.set_xlabel('Episode Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode rewards over evaluation episodes
        ax2.plot(metrics['episode_rewards'], color='green', alpha=0.7)
        ax2.axhline(metrics['mean_reward'], color='red', linestyle='--', 
                   label=f'Mean: {metrics["mean_reward"]:.2f}')
        ax2.set_xlabel('Evaluation Episode')
        ax2.set_ylabel('Episode Reward')
        ax2.set_title('Episode Rewards During Evaluation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Episode lengths histogram
        ax3.hist(metrics['episode_lengths'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.axvline(metrics['mean_episode_length'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {metrics["mean_episode_length"]:.1f}')
        ax3.set_xlabel('Episode Length')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Episode Lengths')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Box plot of rewards
        ax4.boxplot([metrics['episode_rewards']], labels=['Rewards'])
        ax4.set_ylabel('Episode Reward')
        ax4.set_title('Episode Rewards Box Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.log_dir, 'evaluation_results.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def print_evaluation_summary(self, metrics: dict):
        """Print a summary of evaluation metrics."""
        if not metrics:
            print("No evaluation metrics available!")
            return
        
        print("\n" + "="*50)
        print("TD3 MODEL EVALUATION SUMMARY")
        print("="*50)
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Median Reward: {metrics['median_reward']:.2f}")
        print(f"Min/Max Reward: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
        print(f"Success Rate: {metrics['success_rate']*100:.1f}% (rewards > 200)")
        print(f"Mean Episode Length: {metrics['mean_episode_length']:.1f} ± {metrics['std_episode_length']:.1f}")
        print("="*50)
    
    def compare_training_runs(self, log_dirs: List[str], labels: List[str] = None):
        """Compare multiple training runs."""
        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(log_dirs))]
        
        plt.figure(figsize=(12, 8))
        
        for i, log_dir in enumerate(log_dirs):
            try:
                x, y = ts2xy(load_results(log_dir), "timesteps")
                if len(x) > 0:
                    # Plot moving average
                    window_size = min(50, len(y) // 10)
                    if window_size > 1:
                        moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                        moving_x = x[window_size-1:]
                        plt.plot(moving_x, moving_avg, linewidth=2, label=labels[i])
            except Exception as e:
                print(f"Error loading results from {log_dir}: {e}")
        
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward (Moving Average)')
        plt.title('Training Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_results_summary(self, metrics: dict, save_path: str = None):
        """Save evaluation results to a JSON file."""
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'evaluation_summary.json')
        
        # Prepare data for JSON serialization
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'environment': self.env_name,
            'metrics': {
                'mean_reward': float(metrics['mean_reward']),
                'std_reward': float(metrics['std_reward']),
                'min_reward': float(metrics['min_reward']),
                'max_reward': float(metrics['max_reward']),
                'median_reward': float(metrics['median_reward']),
                'success_rate': float(metrics['success_rate']),
                'mean_episode_length': float(metrics['mean_episode_length']),
                'std_episode_length': float(metrics['std_episode_length'])
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results summary saved to {save_path}")
    
    def watch_agent_play(self, n_episodes: int = 5, delay: float = 0.01):
        """
        Watch the trained agent play in real-time with pygame rendering.
        
        Args:
            n_episodes: Number of episodes to watch
            delay: Delay between steps (in seconds) to control playback speed
        """
        import time
        
        if not os.path.exists(self.model_path + ".zip"):
            print(f"Model not found at {self.model_path}.zip")
            return
        
        print(f"Loading model from {self.model_path}.zip...")
        model = TD3.load(self.model_path + ".zip")
        
        # Create environment with human rendering
        env = gym.make(self.env_name, render_mode="human")
        
        print(f"\nWatching agent play for {n_episodes} episodes...")
        print("Close the pygame window to stop early.")
        
        episode_rewards = []
        
        try:
            for episode in range(n_episodes):
                print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
                obs, _ = env.reset()
                episode_reward = 0
                step_count = 0
                done = False
                
                while not done:
                    # Get action from trained model
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Take step in environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    done = terminated or truncated
                    
                    # Render the environment
                    env.render()
                    
                    # Add small delay to make it watchable
                    time.sleep(delay)
                    
                    # Print step info occasionally
                    if step_count % 50 == 0:
                        print(f"Step {step_count}, Reward so far: {episode_reward:.2f}")
                
                episode_rewards.append(episode_reward)
                
                # Episode summary
                if terminated:
                    if episode_reward > 200:
                        print(f"✅ SUCCESS! Episode reward: {episode_reward:.2f} in {step_count} steps")
                    else:
                        print(f"❌ Episode ended. Reward: {episode_reward:.2f} in {step_count} steps")
                else:
                    print(f"⏱️  Episode truncated. Reward: {episode_reward:.2f} in {step_count} steps")
                
                print(f"Average reward so far: {np.mean(episode_rewards):.2f}")
                
                # Brief pause between episodes
                time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(f"Error during rendering: {e}")
        finally:
            env.close()
            
        if episode_rewards:
            print(f"\n🎯 Final Summary:")
            print(f"Episodes played: {len(episode_rewards)}")
            print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Success rate: {np.mean(np.array(episode_rewards) > 200)*100:.1f}%")
    
    def interactive_demo(self):
        """
        Interactive demo that lets user control playback and evaluation.
        """
        while True:
            print("\n" + "="*50)
            print("TD3 AGENT DEMO - INTERACTIVE MODE")
            print("="*50)
            print("1. Watch agent play (5 episodes)")
            print("2. Watch agent play (custom episodes)")
            print("3. Evaluate model performance (no rendering)")
            print("4. Evaluate model performance (with rendering)")
            print("5. Show training progress plots")
            print("6. Complete analysis")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                self.watch_agent_play(n_episodes=5)
            
            elif choice == '2':
                try:
                    n_eps = int(input("Enter number of episodes to watch: "))
                    speed = input("Playback speed (fast/normal/slow) [normal]: ").strip().lower()
                    delay = {'fast': 0.005, 'normal': 0.01, 'slow': 0.02}.get(speed, 0.01)
                    self.watch_agent_play(n_episodes=n_eps, delay=delay)
                except ValueError:
                    print("Invalid input. Using default values.")
                    self.watch_agent_play()
            
            elif choice == '3':
                try:
                    n_eps = int(input("Number of evaluation episodes [100]: ") or "100")
                    metrics = self.evaluate_model(n_eval_episodes=n_eps, render=False)
                    if metrics:
                        self.print_evaluation_summary(metrics)
                except ValueError:
                    print("Invalid input. Using default value.")
                    metrics = self.evaluate_model(render=False)
                    if metrics:
                        self.print_evaluation_summary(metrics)
            
            elif choice == '4':
                try:
                    n_eps = int(input("Number of episodes to evaluate with rendering [5]: ") or "5")
                    metrics = self.evaluate_model(n_eval_episodes=n_eps, render=True)
                    if metrics:
                        self.print_evaluation_summary(metrics)
                except ValueError:
                    print("Invalid input. Using default value.")
                    metrics = self.evaluate_model(n_eval_episodes=5, render=True)
                    if metrics:
                        self.print_evaluation_summary(metrics)
            
            elif choice == '5':
                self.plot_training_progress()
            
            elif choice == '6':
                self.render_complete_analysis()
            
            elif choice == '7':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def render_complete_analysis(self, n_eval_episodes: int = 100):
        """Perform complete analysis of the trained model."""
        print("Starting complete TD3 model analysis...")
        
        # Plot training progress
        print("1. Plotting training progress...")
        self.plot_training_progress()
        
        # Evaluate model
        print(f"2. Evaluating model over {n_eval_episodes} episodes...")
        metrics = self.evaluate_model(n_eval_episodes=n_eval_episodes)
        
        if metrics:
            # Print summary
            self.print_evaluation_summary(metrics)
            
            # Plot evaluation results
            print("3. Plotting evaluation results...")
            self.plot_evaluation_results(metrics)
            
            # Save results
            print("4. Saving results summary...")
            self.save_results_summary(metrics)
            
            # Ask if user wants to watch the agent
            watch = input("\nWould you like to watch the agent play? (y/n) [y]: ").strip().lower()
            if watch != 'n':
                self.watch_agent_play(n_episodes=3)
        
        print("Analysis complete!")

def main():
    """Main function to demonstrate usage."""
    # Initialize the renderer
    renderer = TD3ResultsRenderer(log_dir="tmp/")
    
    # Check if model exists
    if not os.path.exists(renderer.model_path + ".zip"):
        print("No trained model found! Please train a model first using your TD3 training script.")
        return
    
    print("TD3 Model found! Choose your option:")
    print("1. Quick demo - Watch agent play")
    print("2. Interactive demo - Full control")
    print("3. Complete analysis - All visualizations and metrics")
    
    choice = input("Enter choice (1-3) [2]: ").strip() or "2"
    
    if choice == "1":
        print("🚀 Quick demo - Watching agent play...")
        renderer.watch_agent_play(n_episodes=3)
    elif choice == "2":
        renderer.interactive_demo()
    elif choice == "3":
        renderer.render_complete_analysis(n_eval_episodes=50)
    else:
        print("Invalid choice, starting interactive demo...")
        renderer.interactive_demo()

if __name__ == "__main__":
    main()