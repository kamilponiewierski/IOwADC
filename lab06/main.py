from pettingzoo.butterfly import cooperative_pong_v5
import supersuit as ss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

# --- Simple CNN Policy for each agent ---
class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(obs_shape[-1], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)
    def forward(self, x):
        x = x / 255.0
        features = self.net(x)
        return self.actor(features), self.critic(features)
    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

# --- Helper: preprocess obs to torch tensor ---
def obs_to_tensor(obs, device):
    x = torch.tensor(obs, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)
    elif x.ndim == 4:
        x = x.permute(0, 3, 1, 2)
    return x

# --- Main training loop ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = cooperative_pong_v5.env(render_mode=None)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env.reset(seed=42)
    agent_ids = env.agents
    obs_space = env.observation_space(agent_ids[0])
    act_space = env.action_space(agent_ids[0])
    obs_shape = obs_space.shape
    n_actions = act_space.n
    # Create separate policy/optimizer for each agent
    policies = {aid: CNNPolicy(obs_shape, n_actions).to(device) for aid in agent_ids}
    optimizers = {aid: optim.Adam(policies[aid].parameters(), lr=2.5e-4) for aid in agent_ids}
    # Storage for learning curves
    episode_rewards = {aid: [] for aid in agent_ids}
    # --- Training ---
    total_episodes = 300
    gamma = 0.99
    for ep in range(total_episodes):
        obs = env.reset(seed=ep)
        done = {aid: False for aid in agent_ids}
        rewards = {aid: 0.0 for aid in agent_ids}
        log_probs = {aid: [] for aid in agent_ids}
        values = {aid: [] for aid in agent_ids}
        rewards_list = {aid: [] for aid in agent_ids}
        actions = {aid: [] for aid in agent_ids}
        # --- Collect episode ---
        for agent in env.agent_iter():
            ob, reward, termination, truncation, info = env.last()
            d = termination or truncation
            if not d:
                x = obs_to_tensor(ob, device)
                action, logprob, _, value = policies[agent].act(x)
                action = action.item()
            else:
                action = None
            env.step(action)
            if not d:
                log_probs[agent].append(logprob)
                values[agent].append(value)
                actions[agent].append(action)
            rewards[agent] += reward
            rewards_list[agent].append(reward)
            done[agent] = d
        # --- Compute returns and update ---
        for aid in agent_ids:
            # Fix: ensure rewards_list and values are the same length
            if len(rewards_list[aid]) > len(values[aid]):
                rewards_list[aid] = rewards_list[aid][:len(values[aid])]
            elif len(values[aid]) > len(rewards_list[aid]):
                values[aid] = values[aid][:len(rewards_list[aid])]
            R = 0
            returns = []
            for r in reversed(rewards_list[aid]):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            if values[aid]:
                values_t = torch.cat(values[aid]).squeeze(-1)
                log_probs_t = torch.cat(log_probs[aid])
                advantage = returns - values_t.detach()
                policy_loss = -(log_probs_t * advantage).mean()
                value_loss = (returns - values_t).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss
                optimizers[aid].zero_grad()
                loss.backward()
                optimizers[aid].step()
        for aid in agent_ids:
            episode_rewards[aid].append(rewards[aid])
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}: " + ", ".join([f"{aid}: {np.mean(episode_rewards[aid][-10:]):.2f}" for aid in agent_ids]))
    # --- Plot learning curves ---
    plt.figure()
    for aid in agent_ids:
        plt.plot(episode_rewards[aid], label=aid)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Learning Curves (Separate Policies)')
    plt.legend()
    plt.savefig('multiagent_separate_policies_learning_curve.png')
    plt.close()
    # --- Evaluation: render last 3 episodes ---
    env = cooperative_pong_v5.env(render_mode="human")
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    for ep in range(3):
        obs = env.reset(seed=1000+ep)
        done = {aid: False for aid in agent_ids}
        while not all(done.values()):
            for agent in env.agent_iter():
                ob, reward, termination, truncation, info = env.last()
                d = termination or truncation
                if not d:
                    x = obs_to_tensor(ob, device)
                    action, _, _, _ = policies[agent].act(x)
                    action = action.item()
                else:
                    action = None
                env.step(action)
                done[agent] = d
    env.close()

if __name__ == "__main__":
    main()