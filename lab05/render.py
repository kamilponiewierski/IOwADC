import gymnasium as gym
from stable_baselines3 import PPO, TD3
import time


def evaluate_and_render_high_rewards(
    model_path, reward_threshold=200, num_episodes=50, max_renders=5
):
    """
    Evaluate model and render episodes that achieve rewards above threshold

    Args:
        model_path: Path to trained model
        reward_threshold: Minimum reward to trigger rendering
        num_episodes: Total episodes to evaluate
        max_renders: Maximum number of episodes to render (to avoid too many windows)
    """
    model = PPO.load(model_path, device="cpu")

    # Environment for evaluation (no rendering initially)
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    # Environment for rendering
    render_env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode="human",
    )

    high_reward_episodes = []
    rendered_count = 0

    print(f"Evaluating {num_episodes} episodes...")
    print(f"Will render episodes with reward > {reward_threshold}")
    print("-" * 50)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_actions = []
        episode_observations = []

        # First, run episode without rendering to check reward
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            episode_observations.append(obs.copy())
            episode_actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep+1:3d}: Reward = {total_reward:7.2f}", end="")

        # If reward is above threshold and we haven't rendered too many, render it
        if total_reward > reward_threshold and rendered_count < max_renders:
            print(f" -> RENDERING (High reward!)")
            high_reward_episodes.append((ep + 1, total_reward))

            # Replay the episode with rendering
            render_episode_replay(
                render_env,
                model,
                episode_observations,
                episode_actions,
                ep + 1,
                total_reward,
            )
            rendered_count += 1

        else:
            if total_reward > reward_threshold:
                print(f" -> High reward (not rendering, limit reached)")
                high_reward_episodes.append((ep + 1, total_reward))
            else:
                print()

    env.close()
    render_env.close()

    print("-" * 50)
    print(f"Episodes with reward > {reward_threshold}:")
    for ep_num, reward in high_reward_episodes:
        print(f"  Episode {ep_num}: {reward:.2f}")
    print(f"Total high-reward episodes: {len(high_reward_episodes)}/{num_episodes}")
    print(f"Rendered episodes: {rendered_count}")


def render_episode_replay(
    render_env, model, observations, actions, episode_num, total_reward
):
    """Replay an episode with rendering"""
    print(f"\n=== RENDERING EPISODE {episode_num} (Reward: {total_reward:.2f}) ===")
    print("Press any key to start rendering...")
    input()  # Wait for user input

    obs, _ = render_env.reset()

    for i, (stored_obs, stored_action) in enumerate(zip(observations, actions)):
        # Use stored action for exact replay
        obs, reward, terminated, truncated, info = render_env.step(stored_action)
        time.sleep(0.05)  # Slow down for better viewing

        if terminated or truncated:
            break

    print("Episode finished. Press Enter to continue...")
    input()


def render_live_evaluation(model_path, reward_threshold=200, num_episodes=20):
    """
    Alternative: Render episodes live and continue only if reward is high
    This is more efficient but you might miss some high-reward episodes
    """
    model = PPO.load(model_path)

    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode="human",
    )

    high_reward_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"\nEpisode {ep+1}: ", end="", flush=True)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # Early check - if reward is getting very negative, skip rendering rest
            if step_count > 100 and total_reward < -100:
                print(f"Low reward ({total_reward:.1f}) - skipping rest", end="")
                # Reset without rendering
                break

        print(f" Final reward: {total_reward:.2f}")

        if total_reward > reward_threshold:
            high_reward_count += 1
            print(f"HIGH REWARD EPISODE! ({high_reward_count} so far)")
            time.sleep(2)  # Pause to appreciate the success

    env.close()
    print(
        f"\nCompleted evaluation: {high_reward_count}/{num_episodes} high-reward episodes"
    )
