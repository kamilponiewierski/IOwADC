import os
import random
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.butterfly import cooperative_pong_v5

import supersuit as ss


@dataclass
class Args:
    exp_name: str = "ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = True
    total_timesteps: int = 50000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space("paddle_0").n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        # Fix shape: (batch, H, W, C) or (H, W, C) -> (batch, C, H, W)
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.ndim == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def make_env():
    env = cooperative_pong_v5.env(render_mode="rgb_array")
    env = ss.color_reduction_v0(env, mode='B')  # Convert to grayscale
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize observations
    env = ss.frame_stack_v1(env, 4)  # Stack 4 frames
    return env


if __name__ == "__main__":
    args = Args()
    run_name = f"cooperative_pong__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Environment setup
    env = make_env()
    
    # Create agents for both players
    agents = {
        "paddle_0": Agent(env).to(device),
        "paddle_1": Agent(env).to(device)
    }
    optimizers = {
        agent_id: optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        for agent_id, agent in agents.items()
    }

    # Training metrics
    episode_returns = {agent_id: [] for agent_id in agents.keys()}
    learning_curves = {agent_id: [] for agent_id in agents.keys()}

    # Training loop
    global_step = 0
    start_time = time.time()
    next_obs = None
    next_done = None

    while global_step < args.total_timesteps:
        env.reset(seed=args.seed)
        episode_reward = {agent_id: 0 for agent_id in agents.keys()}
        
        for agent_id in env.agent_iter():
            if global_step >= args.total_timesteps:
                break
                
            global_step += 1
            
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            if obs is not None:
                obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, logprob, _, value = agents[agent_id].get_action_and_value(obs)
                
                action = action.item() if hasattr(action, "item") else int(action)
                env.step(action)
                episode_reward[agent_id] += reward
                
            if done:
                for aid, reward in episode_reward.items():
                    episode_returns[aid].append(reward)
                    if len(episode_returns[aid]) % 100 == 0:  # Record every 100 episodes
                        learning_curves[aid].append(np.mean(episode_returns[aid][-100:]))
                episode_reward = {agent_id: 0 for agent_id in agents.keys()}

    # Plot and save learning curves
    plt.figure(figsize=(10, 5))
    for agent_id, rewards in learning_curves.items():
        plt.plot(rewards, label=agent_id)
    plt.xlabel('Episodes (x100)')
    plt.ylabel('Average Return (last 100 episodes)')
    plt.title('Learning Curves for Cooperative Pong Agents')
    plt.legend()
    plt.savefig('cooperative_pong_learning_curves.png')
    plt.close()

    env.close()

    