

import sys
import os

# Allow running this script directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment import RobotArmEnv


def run_random_episode(env, episode_number):
    observation, info = env.reset()
    total_reward = 0.0
    total_energy = 0.0
    step = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()  # completely random action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_energy += info["energy_used"]
        step += 1

    print(
        f"Episode {episode_number}: steps={step}, "
        f"total_reward={total_reward:.2f}, total_energy={total_energy:.4f}, "
        f"final_distance={info['distance']:.3f}, task_done={info['task_done']}"
    )


if __name__ == "__main__":
    env = RobotArmEnv(render_mode=None)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print("-" * 60)

    for ep in range(3):
        run_random_episode(env, ep + 1)

    env.close()
    print("-" * 60)
    print("Environment ran successfully with no crashes.")
