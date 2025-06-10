import torch
import numpy as np
from pettingzoo.atari import space_invaders_v2
import supersuit
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Dodaj sieć wartości (baseline) do agenta
class SimplePPOAgent(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)
    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

def select_action(agent, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = agent(obs_t)
    probs = torch.softmax(logits, dim=1)
    action = torch.multinomial(probs, num_samples=1).item()
    return action

# Przygotowanie środowiska
env = space_invaders_v2.parallel_env()
env = supersuit.color_reduction_v0(env, mode='full')
env = supersuit.resize_v1(env, x_size=84, y_size=84)
env = supersuit.frame_stack_v1(env, 4)

agents = env.possible_agents
agent_models = {agent: SimplePPOAgent(env.observation_space(agent).shape, env.action_space(agent).n) for agent in agents}
reward_history = {agent: [] for agent in agents}
# Dodaj optymalizatory i prosty krok uczenia (policy gradient, nie pełne PPO)
optimizers = {agent: optim.Adam(agent_models[agent].parameters(), lr=1e-3) for agent in agents}

for episode in range(100):
    obs = env.reset()
    # PettingZoo parallel_env.reset() may return (obs, info) tuple
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _ = obs
    done = {agent: False for agent in agents}
    total_rewards = {agent: 0 for agent in agents}
    # Zbieranie trajektorii dla prostego policy gradient
    trajectories = {agent: {'obs': [], 'actions': [], 'rewards': []} for agent in agents}
    while not all(done.values()):
        actions = {}
        for agent in agents:
            if not done[agent]:
                action = select_action(agent_models[agent], obs[agent])
                actions[agent] = action
                trajectories[agent]['obs'].append(obs[agent])
                trajectories[agent]['actions'].append(action)
            else:
                actions[agent] = 0  # no-op
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        if isinstance(next_obs, tuple) and len(next_obs) == 2:
            next_obs, _ = next_obs
        for agent in agents:
            trajectories[agent]['rewards'].append(rewards[agent])
            total_rewards[agent] += rewards[agent]
            done[agent] = terminations[agent] or truncations[agent]
        obs = next_obs
    # Policy gradient update (REINFORCE z baseline)
    for agent in agents:
        R = 0
        returns = []
        for r in reversed(trajectories[agent]['rewards']):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        obs_batch = torch.tensor(np.array(trajectories[agent]['obs']), dtype=torch.float32)
        action_batch = torch.tensor(trajectories[agent]['actions'], dtype=torch.int64)
        logits, values = agent_models[agent](obs_batch)
        log_probs = torch.log_softmax(logits, dim=1)
        chosen_log_probs = log_probs[range(len(action_batch)), action_batch]
        # Advantage = returns - value
        values = values.squeeze(-1)
        advantage = returns - values.detach()
        # Policy loss (z baseline)
        policy_loss = -(chosen_log_probs * advantage).mean()
        # Value loss
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = policy_loss + value_loss
        optimizers[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()
    for agent in agents:
        reward_history[agent].append(total_rewards[agent])
    print(f"Episode {episode+1}: {[f'{agent}: {total_rewards[agent]}' for agent in agents]}")

# Wykres sumy nagród dla każdego agenta
for agent in agents:
    plt.plot(reward_history[agent], label=agent)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Learning curve: Simple PPO-like agent on Space Invaders (multi-agent)")
plt.legend()
plt.show()