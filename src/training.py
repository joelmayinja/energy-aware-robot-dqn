"""Train the project teacher DQN and report simulator metrics."""

import csv
import random
import time
from pathlib import Path
from threading import Event
from typing import Callable


StepCallback = Callable[[dict], None]
EpisodeCallback = Callable[[dict], None]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _wait_for_training(stop_event, pause_event, speed):
    while not pause_event.wait(timeout=0.1):
        if stop_event.is_set():
            return False
    if stop_event.is_set():
        return False
    time.sleep(0.005 / max(1, speed))
    return not stop_event.is_set()


def _run_constant_speed_baseline(env, seed, episode, stop_event, pause_event, speed, on_step):
    """Measure fixed-speed inverse-kinematics control without using the buffer."""
    import numpy as np
    import pybullet as p

    env.buffer_enabled = False
    try:
        observation, _ = env.reset(seed=seed)
        total_grid_draw = 0.0
        peak_grid_draw = 0.0
        steps = []
        terminated = False
        truncated = False

        while not (terminated or truncated) and not stop_event.is_set():
            if not _wait_for_training(stop_event, pause_event, speed()):
                break
            target_angles = p.calculateInverseKinematics(
                env.robot_id,
                env.end_effector_link,
                env.target_position.tolist(),
            )
            action = np.asarray(
                [
                    np.sign(target_angles[joint] - p.getJointState(env.robot_id, joint)[0])
                    for joint in env.controlled_joints
                ],
                dtype=np.float32,
            )
            observation, _, terminated, truncated, info = env.step(action)
            total_grid_draw += info["grid_draw"]
            peak_grid_draw = max(peak_grid_draw, info["grid_draw"])
            steps.append(
                {
                    "grid_draw": info["grid_draw"],
                    "total_grid_draw": total_grid_draw,
                    "buffer_level": info["buffer_level"],
                    "distance": info["distance"],
                }
            )
            on_step(
                {
                    "stage": "baseline",
                    "episode": episode,
                    "episode_step": len(steps),
                    "max_steps": env.MAX_STEPS,
                    "baseline_grid_draw": info["grid_draw"],
                    "agent_grid_draw": None,
                    "baseline_total_grid_draw": total_grid_draw,
                    "total_grid_draw": total_grid_draw,
                    "buffer_level": info["buffer_level"],
                    "distance": info["distance"],
                    "phase": info["task_phase"],
                    "joint_positions": observation[:6].tolist(),
                }
            )

        return {
            "steps": steps,
            "total_grid_draw": total_grid_draw,
            "peak_grid_draw": peak_grid_draw,
        }
    finally:
        env.buffer_enabled = True


def train(
    episodes: int,
    stop_event: Event,
    pause_event: Event,
    speed_provider: Callable[[], int],
    on_step: StepCallback,
    on_episode: EpisodeCallback,
) -> None:
    """Train a 256×256 teacher DQN using the project's 500-episode plan."""
    import numpy as np
    import torch

    from src.dqn_teacher import DQNAgent
    from src.environment import RobotArmEnv

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = RobotArmEnv(render_mode=None)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        num_joints=env.action_space.shape[0],
        buffer_capacity=10_000,
    )
    training_history = PROJECT_ROOT / "training_history.csv"
    csv_fields = (
        "episode",
        "reward",
        "grid_energy_ws",
        "baseline_grid_energy_ws",
        "peak_grid_draw",
        "baseline_peak_grid_draw",
        "peak_reduction_percent",
        "task_done",
        "distance_m",
        "buffer_used_ws",
        "epsilon",
        "mean_loss",
    )
    global_step = 0

    try:
        for episode in range(1, episodes + 1):
            if stop_event.is_set():
                break

            baseline = _run_constant_speed_baseline(
                env, episode, episode, stop_event, pause_event, speed_provider, on_step
            )
            if stop_event.is_set():
                break

            observation, _ = env.reset(seed=episode)
            episode_reward = 0.0
            episode_steps = 0
            episode_losses = []
            last_info = {}
            terminated = False
            truncated = False
            epsilon = max(0.01, 0.995 ** (episode - 1))

            while not (terminated or truncated) and not stop_event.is_set():
                if not _wait_for_training(stop_event, pause_event, speed_provider()):
                    break

                action_index, action = agent.select_action(observation, epsilon)
                next_observation, reward, terminated, truncated, info = env.step(action)
                agent.store(
                    observation,
                    action_index,
                    reward,
                    next_observation,
                    terminated,
                )
                loss = agent.learn(batch_size=64)
                if loss is not None:
                    episode_losses.append(loss)
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                global_step += 1
                last_info = info

                baseline_step = min(episode_steps - 1, len(baseline["steps"]) - 1)
                baseline_grid_draw = (
                    baseline["steps"][baseline_step]["grid_draw"]
                    if baseline_step >= 0
                    else 0.0
                )
                on_step(
                    {
                        "stage": "teacher",
                        "episode": episode,
                        "episode_step": episode_steps,
                        "max_steps": env.MAX_STEPS,
                        "reward": episode_reward,
                        "agent_grid_draw": info["grid_draw"],
                        "baseline_grid_draw": baseline_grid_draw,
                        "total_grid_draw": info["total_grid_draw"],
                        "baseline_total_grid_draw": baseline["total_grid_draw"],
                        "buffer_level": info["buffer_level"],
                        "buffer_used": info["total_buffer_used"],
                        "distance": info["distance"],
                        "phase": info["task_phase"],
                        "epsilon": epsilon,
                        "loss": episode_losses[-1] if episode_losses else None,
                        "joint_positions": observation[:6].tolist(),
                    }
                )

            if episode_steps and not stop_event.is_set():
                agent_peak = env.peak_grid_draw
                baseline_peak = baseline["peak_grid_draw"]
                peak_reduction = (
                    (baseline_peak - agent_peak) / baseline_peak * 100.0
                    if baseline_peak > 0
                    else 0.0
                )
                result = {
                    "episode": episode,
                    "steps": episode_steps,
                    "reward": round(episode_reward, 4),
                    "grid_energy": round(last_info["total_grid_draw"], 4),
                    "baseline_grid_energy": round(baseline["total_grid_draw"], 4),
                    "peak_grid_draw": round(agent_peak, 4),
                    "baseline_peak_grid_draw": round(baseline_peak, 4),
                    "peak_reduction_percent": round(peak_reduction, 2),
                    "buffer_used": round(last_info["total_buffer_used"], 4),
                    "buffer_level": round(last_info["buffer_level"], 4),
                    "distance": round(last_info["distance"], 4),
                    "task_done": bool(last_info["task_done"]),
                    "epsilon": round(epsilon, 4),
                    "mean_loss": round(float(np.mean(episode_losses)), 6) if episode_losses else None,
                }
                on_episode(result)
                with training_history.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=csv_fields)
                    if file.tell() == 0:
                        writer.writeheader()
                    writer.writerow(
                        {
                            "episode": result["episode"],
                            "reward": result["reward"],
                            "grid_energy_ws": result["grid_energy"],
                            "baseline_grid_energy_ws": result["baseline_grid_energy"],
                            "peak_grid_draw": result["peak_grid_draw"],
                            "baseline_peak_grid_draw": result["baseline_peak_grid_draw"],
                            "peak_reduction_percent": result["peak_reduction_percent"],
                            "task_done": result["task_done"],
                            "distance_m": result["distance"],
                            "buffer_used_ws": result["buffer_used"],
                            "epsilon": result["epsilon"],
                            "mean_loss": result["mean_loss"],
                        }
                    )
                print(
                    f"Episode {episode}/{episodes}: reward={result['reward']}, "
                    f"teacher grid={result['grid_energy']} Ws, "
                    f"constant-speed baseline={result['baseline_grid_energy']} Ws, "
                    f"task_done={result['task_done']}"
                )

            if episode % 10 == 0:
                agent.update_target_network()
    finally:
        env.close()

    if global_step:
        model_path = PROJECT_ROOT / "models" / "teacher.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(model_path)
        replay_path = PROJECT_ROOT / "data" / "processed" / "teacher_replay.npz"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay = list(agent.replay_buffer.buffer)
        if replay:
            np.savez_compressed(
                replay_path,
                states=np.asarray([item[0] for item in replay], dtype=np.float32),
                actions=np.asarray([item[1] for item in replay], dtype=np.int64),
            )
        print(f"Saved trained teacher weights to {model_path}")
        print(f"Saved replay states and actions to {replay_path}")
