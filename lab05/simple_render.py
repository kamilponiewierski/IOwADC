import gymnasium as gym
from stable_baselines3 import TD3
if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    model = TD3.load('tmp/Large Network_run_0/final_model', env=env, device='cpu')
    obs, _ = env.reset()
    done = False
    rewards = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        rewards += reward
        
    print(f"Total rewards: {rewards}")