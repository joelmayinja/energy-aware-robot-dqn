"""Train the project teacher DQN and report simulator metrics."""

import csv
import copy
import random
import time
from pathlib import Path
from threading import Event
from typing import Callable


StepCallback = Callable[[dict], None]
EpisodeCallback = Callable[[dict], None]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IK_DEMO_EPISODES = 100
IK_BEHAVIOR_UPDATES = 3_000
IK_IMITATION_WEIGHT = 0.10
SELECTION_SEED = 70_000
EVALUATION_SEED = 80_000
EVALUATION_EPISODES = 50


def _wait_for_training(stop_event, pause_event, speed):
    while not pause_event.wait(timeout=0.1):
        if stop_event.is_set():
            return False
    if stop_event.is_set():
        return False
    time.sleep(0.005 / max(1, speed))
    return not stop_event.is_set()


def _run_constant_speed_baseline(env, seed, episode, stop_event, pause_event, speed, on_step):
    """Measure inverse-kinematics control under the same buffer model as the teacher."""
    import numpy as np
    import pybullet as p

    env.buffer_enabled = True
    try:
        observation, _ = env.reset(seed=seed)
        total_grid_draw = 0.0
        peak_grid_draw = 0.0
        steps = []
        last_info = {}
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
            last_info = info
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
            "task_done": bool(last_info.get("task_done", False)),
            "distance": last_info.get("distance"),
        }
    finally:
        env.buffer_enabled = True


def _collect_ik_demonstrations(env, agent, episodes, stop_event, pause_event, speed):
    """Collect expert state/action labels using the simulator's IK controller."""
    import numpy as np
    import pybullet as p

    lookup = {tuple(action.tolist()): index for index, action in enumerate(agent.action_table)}
    states = []
    actions = []
    successes = 0
    for episode in range(episodes):
        if stop_event.is_set():
            break
        observation, _ = env.reset(seed=50_000 + episode)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated) and not stop_event.is_set():
            if not _wait_for_training(stop_event, pause_event, speed()):
                break
            target_angles = p.calculateInverseKinematics(
                env.robot_id, env.end_effector_link, env.target_position.tolist()
            )
            action = np.asarray(
                [
                    np.sign(target_angles[joint] - p.getJointState(env.robot_id, joint)[0])
                    for joint in env.controlled_joints
                ],
                dtype=np.float32,
            )
            action_index = lookup[tuple(action.tolist())]
            states.append(observation.copy())
            actions.append(action_index)
            observation, _, terminated, truncated, info = env.step(action)
        successes += int(bool(info.get("task_done", False)))

    if not states:
        return None, successes
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    ), successes


def _behavior_clone(agent, expert_data, updates, stop_event):
    """Warm-start the Q network to predict successful IK joint commands."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    states, actions = expert_data
    last_loss = None
    for _ in range(updates):
        if stop_event.is_set():
            break
        indices = np.random.randint(0, len(states), size=min(256, len(states)))
        state_batch = torch.as_tensor(states[indices], dtype=torch.float32, device=agent.device)
        action_batch = torch.as_tensor(actions[indices], dtype=torch.long, device=agent.device)
        loss = F.cross_entropy(agent.q_network(state_batch), action_batch)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        last_loss = float(loss.detach().cpu())
    agent.update_target_network()
    return last_loss


def _score_checkpoint(env, agent, state_dict, first_seed, count=20):
    """Score a greedy policy on held-out targets; prioritize success, then energy."""
    import numpy as np

    agent.q_network.load_state_dict(state_dict)
    successes = 0
    successful_energies = []
    for offset in range(count):
        observation, _ = env.reset(seed=first_seed + offset)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _action_index, action = agent.select_action(observation, epsilon=0.0)
            observation, _, terminated, truncated, info = env.step(action)
        if info.get("task_done", False):
            successes += 1
            successful_energies.append(info["total_grid_draw"])
    mean_success_energy = (
        float(np.mean(successful_energies)) if successful_energies else float("inf")
    )
    return successes, mean_success_energy


def _run_ik_episode(env, seed):
    """Run one deterministic IK baseline episode and return its measurements."""
    import numpy as np
    import pybullet as p

    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        target_angles = p.calculateInverseKinematics(
            env.robot_id, env.end_effector_link, env.target_position.tolist()
        )
        action = np.asarray(
            [
                np.sign(target_angles[joint] - p.getJointState(env.robot_id, joint)[0])
                for joint in env.controlled_joints
            ],
            dtype=np.float32,
        )
        observation, _, terminated, truncated, info = env.step(action)
    return {
        "task_done": bool(info["task_done"]),
        "grid_energy": float(info["total_grid_draw"]),
        "peak_grid_draw": float(env.peak_grid_draw),
        "distance": float(info["distance"]),
    }


def _write_final_evaluation(env, agent):
    """Write matched-seed IK-assisted DQN and IK baseline evaluation results."""
    import csv
    import numpy as np

    summary_path = PROJECT_ROOT / "evaluation_summary.csv"
    paired_path = PROJECT_ROOT / "paired_evaluation.csv"
    paired_rows = []
    for offset in range(EVALUATION_EPISODES):
        seed = EVALUATION_SEED + offset
        baseline = _run_ik_episode(env, seed)
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _action_index, action = agent.select_action(observation, epsilon=0.0)
            observation, _, terminated, truncated, info = env.step(action)
        teacher = {
            "task_done": bool(info["task_done"]),
            "grid_energy": float(info["total_grid_draw"]),
            "peak_grid_draw": float(env.peak_grid_draw),
            "distance": float(info["distance"]),
        }
        comparable = teacher["task_done"] and baseline["task_done"]
        paired_rows.append(
            {
                "seed": seed,
                "teacher_task_done": teacher["task_done"],
                "baseline_task_done": baseline["task_done"],
                "comparison_valid": comparable,
                "teacher_energy_ws": round(teacher["grid_energy"], 4),
                "baseline_energy_ws": round(baseline["grid_energy"], 4),
                "teacher_peak_ws": round(teacher["peak_grid_draw"], 4),
                "baseline_peak_ws": round(baseline["peak_grid_draw"], 4),
                "teacher_distance_m": round(teacher["distance"], 4),
                "baseline_distance_m": round(baseline["distance"], 4),
                "energy_savings_percent": (
                    round(
                        (baseline["grid_energy"] - teacher["grid_energy"])
                        / baseline["grid_energy"] * 100.0,
                        2,
                    )
                    if comparable and baseline["grid_energy"] > 0
                    else ""
                ),
                "peak_reduction_percent": (
                    round(
                        (baseline["peak_grid_draw"] - teacher["peak_grid_draw"])
                        / baseline["peak_grid_draw"] * 100.0,
                        2,
                    )
                    if comparable and baseline["peak_grid_draw"] > 0
                    else ""
                ),
            }
        )

    rows_by_agent = {
        "IK-assisted DQN": [
            {
                "task_done": row["teacher_task_done"],
                "grid_energy": row["teacher_energy_ws"],
                "peak_grid_draw": row["teacher_peak_ws"],
                "distance": row["teacher_distance_m"],
            }
            for row in paired_rows
        ],
        "IK baseline": [
            {
                "task_done": row["baseline_task_done"],
                "grid_energy": row["baseline_energy_ws"],
                "peak_grid_draw": row["baseline_peak_ws"],
                "distance": row["baseline_distance_m"],
            }
            for row in paired_rows
        ],
    }
    summary_fields = (
        "agent",
        "evaluation_episodes",
        "completed_episodes",
        "completion_rate_percent",
        "avg_energy_all_episodes_ws",
        "avg_energy_completed_ws",
        "avg_peak_completed_ws",
        "avg_final_distance_m",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for name, rows in rows_by_agent.items():
            completed = [row for row in rows if row["task_done"]]
            writer.writerow(
                {
                    "agent": name,
                    "evaluation_episodes": len(rows),
                    "completed_episodes": len(completed),
                    "completion_rate_percent": round(len(completed) / len(rows) * 100, 2),
                    "avg_energy_all_episodes_ws": round(
                        float(np.mean([row["grid_energy"] for row in rows])), 4
                    ),
                    "avg_energy_completed_ws": (
                        round(float(np.mean([row["grid_energy"] for row in completed])), 4)
                        if completed
                        else ""
                    ),
                    "avg_peak_completed_ws": (
                        round(float(np.mean([row["peak_grid_draw"] for row in completed])), 4)
                        if completed
                        else ""
                    ),
                    "avg_final_distance_m": round(
                        float(np.mean([row["distance"] for row in rows])), 4
                    ),
                }
            )
    paired_fields = (
        "seed",
        "teacher_task_done",
        "baseline_task_done",
        "comparison_valid",
        "teacher_energy_ws",
        "baseline_energy_ws",
        "energy_savings_percent",
        "teacher_peak_ws",
        "baseline_peak_ws",
        "peak_reduction_percent",
    )
    with paired_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=paired_fields)
        writer.writeheader()
        for row in paired_rows:
            writer.writerow({key: row[key] for key in paired_fields})
    return [
        {
            "agent": name,
            "evaluation_episodes": len(rows),
            "completed_episodes": sum(bool(row["task_done"]) for row in rows),
            "completion_rate_percent": round(
                sum(bool(row["task_done"]) for row in rows) / len(rows) * 100, 2
            ),
            "avg_energy_completed_ws": (
                round(
                    float(np.mean([row["grid_energy"] for row in rows if row["task_done"]])),
                    4,
                )
                if any(row["task_done"] for row in rows)
                else None
            ),
        }
        for name, rows in rows_by_agent.items()
    ]


def _append_training_history(path, fields, result):
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = tuple(reader.fieldnames or ())
            previous_results = list(reader) if existing_fields != fields else []
        if existing_fields != fields:
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            with temporary_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for previous_result in previous_results:
                    previous_result.setdefault("energy_comparison_valid", "False")
                    previous_result.setdefault("baseline_task_done", "")
                    previous_result.setdefault("baseline_distance_m", "")
                    writer.writerow(previous_result)
            temporary_path.replace(path)

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(result)


def train(
    episodes: int,
    stop_event: Event,
    pause_event: Event,
    speed_provider: Callable[[], int],
    on_step: StepCallback,
    on_episode: EpisodeCallback,
    resume_model: bool = False,
    on_evaluation: Callable[[list[dict]], None] | None = None,
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
    model_path = PROJECT_ROOT / "models" / "teacher.pt"
    checkpoint_path = PROJECT_ROOT / "models" / "teacher_training_checkpoint.pt"
    training_history = PROJECT_ROOT / "training_history.csv"
    csv_fields = (
        "episode",
        "reward",
        "grid_energy_ws",
        "baseline_grid_energy_ws",
        "peak_grid_draw",
        "baseline_peak_grid_draw",
        "peak_reduction_percent",
        "energy_comparison_valid",
        "task_done",
        "baseline_task_done",
        "distance_m",
        "baseline_distance_m",
        "buffer_used_ws",
        "epsilon",
        "mean_loss",
    )
    global_step = 0
    resume_metadata = {}
    full_checkpoint_available = resume_model and checkpoint_path.is_file() and (
        not model_path.is_file()
        or checkpoint_path.stat().st_mtime_ns >= model_path.stat().st_mtime_ns
    )
    start_episode = 1
    episodes_completed = 0
    expert_data = None
    behavior_cloning_state = None
    behavior_cloning_optimizer_state = None

    try:
        if full_checkpoint_available:
            resume_metadata = agent.load_checkpoint(checkpoint_path)
            global_step = int(resume_metadata.get("global_step", 0))
            episodes_completed = int(resume_metadata.get("episodes_completed", 0))
            start_episode = episodes_completed + 1
            expert_data = resume_metadata.get("expert_data")
            behavior_cloning_state = resume_metadata.get("behavior_cloning_state")
            behavior_cloning_optimizer_state = resume_metadata.get(
                "behavior_cloning_optimizer_state"
            )
            if behavior_cloning_state is None:
                behavior_cloning_state = copy.deepcopy(agent.q_network.state_dict())
            print(
                f"Restored full training checkpoint after episode {episodes_completed}; "
                f"replay entries={len(agent.replay_buffer)}"
            )
        else:
            expert_data, demo_successes = _collect_ik_demonstrations(
                env,
                agent,
                IK_DEMO_EPISODES,
                stop_event,
                pause_event,
                speed_provider,
            )
        if resume_model and not full_checkpoint_available:
            agent.load(model_path)
            behavior_cloning_state = copy.deepcopy(agent.q_network.state_dict())
            behavior_cloning_optimizer_state = copy.deepcopy(
                agent.optimizer.state_dict()
            )
        elif not resume_model and expert_data is not None and not stop_event.is_set():
            bc_loss = _behavior_clone(
                agent, expert_data, IK_BEHAVIOR_UPDATES, stop_event
            )
            behavior_cloning_state = copy.deepcopy(agent.q_network.state_dict())
            behavior_cloning_optimizer_state = copy.deepcopy(
                agent.optimizer.state_dict()
            )
            print(
                f"IK demonstrations: {demo_successes}/{IK_DEMO_EPISODES} completed; "
                f"warm-start loss={bc_loss}"
            )
        for episode in range(start_episode, start_episode + episodes):
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
            epsilon = max(0.01, 0.10 * (0.995 ** (episode - 1)))

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
                loss = agent.learn(
                    batch_size=64,
                    expert_data=expert_data,
                    imitation_weight=(
                        IK_IMITATION_WEIGHT if expert_data is not None else 0.0
                    ),
                )
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
                energy_comparison_valid = bool(
                    last_info["task_done"] and baseline["task_done"]
                )
                peak_reduction = (
                    (baseline_peak - agent_peak) / baseline_peak * 100.0
                    if energy_comparison_valid and baseline_peak > 0
                    else None
                )
                result = {
                    "episode": episode,
                    "steps": episode_steps,
                    "reward": round(episode_reward, 4),
                    "grid_energy": round(last_info["total_grid_draw"], 4),
                    "baseline_grid_energy": round(baseline["total_grid_draw"], 4),
                    "peak_grid_draw": round(agent_peak, 4),
                    "baseline_peak_grid_draw": round(baseline_peak, 4),
                    "peak_reduction_percent": (
                        round(peak_reduction, 2) if peak_reduction is not None else None
                    ),
                    "energy_comparison_valid": energy_comparison_valid,
                    "baseline_task_done": baseline["task_done"],
                    "baseline_distance": round(baseline["distance"], 4),
                    "buffer_used": round(last_info["total_buffer_used"], 4),
                    "buffer_level": round(last_info["buffer_level"], 4),
                    "distance": round(last_info["distance"], 4),
                    "task_done": bool(last_info["task_done"]),
                    "epsilon": round(epsilon, 4),
                    "mean_loss": round(float(np.mean(episode_losses)), 6) if episode_losses else None,
                }
                on_episode(result)
                _append_training_history(
                    training_history,
                    csv_fields,
                    {
                        "episode": result["episode"],
                        "reward": result["reward"],
                        "grid_energy_ws": result["grid_energy"],
                        "baseline_grid_energy_ws": result["baseline_grid_energy"],
                        "peak_grid_draw": result["peak_grid_draw"],
                        "baseline_peak_grid_draw": result["baseline_peak_grid_draw"],
                        "peak_reduction_percent": result["peak_reduction_percent"],
                        "energy_comparison_valid": result["energy_comparison_valid"],
                        "task_done": result["task_done"],
                        "baseline_task_done": result["baseline_task_done"],
                        "distance_m": result["distance"],
                        "baseline_distance_m": result["baseline_distance"],
                        "buffer_used_ws": result["buffer_used"],
                        "epsilon": result["epsilon"],
                        "mean_loss": result["mean_loss"],
                    },
                )
                episodes_completed = episode
                print(
                    f"Episode {episode} ({episode - start_episode + 1}/{episodes} this run): "
                    f"reward={result['reward']}, "
                    f"teacher grid={result['grid_energy']} Ws, "
                    f"constant-speed baseline={result['baseline_grid_energy']} Ws, "
                    f"task_done={result['task_done']}"
                )

            if episode % 5 == 0:
                agent.update_target_network()
            if episodes_completed == episode:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save_checkpoint(
                    checkpoint_path,
                    metadata={
                        "episodes_completed": episodes_completed,
                        "global_step": global_step,
                        "expert_data": expert_data,
                        "behavior_cloning_state": behavior_cloning_state,
                        "behavior_cloning_optimizer_state": behavior_cloning_optimizer_state,
                    },
                )
    finally:
        env.close()

    if global_step:
        if behavior_cloning_state is not None:
            finetuned_state = copy.deepcopy(agent.q_network.state_dict())
            validation_env = RobotArmEnv(render_mode=None)
            try:
                warm_start_score = _score_checkpoint(
                    env=validation_env,
                    agent=agent,
                    state_dict=behavior_cloning_state,
                    first_seed=SELECTION_SEED,
                    count=20,
                )
                fine_tuned_score = _score_checkpoint(
                    env=validation_env,
                    agent=agent,
                    state_dict=finetuned_state,
                    first_seed=SELECTION_SEED,
                    count=20,
                )
            finally:
                validation_env.close()
            if fine_tuned_score[0] < warm_start_score[0] or (
                fine_tuned_score[0] == warm_start_score[0]
                and fine_tuned_score[1] > warm_start_score[1]
            ):
                agent.q_network.load_state_dict(behavior_cloning_state)
                if behavior_cloning_optimizer_state is not None:
                    agent.optimizer.load_state_dict(behavior_cloning_optimizer_state)
                selected_checkpoint = "IK demonstration warm-start"
            else:
                agent.q_network.load_state_dict(finetuned_state)
                selected_checkpoint = "DQN fine-tuned"
            agent.update_target_network()
            print(
                f"Held-out checkpoint scores: warm-start={warm_start_score}, "
                f"fine-tuned={fine_tuned_score}; selected={selected_checkpoint}"
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(model_path)
        agent.save_checkpoint(
            checkpoint_path,
            metadata={
                "episodes_completed": episodes_completed,
                "global_step": global_step,
                "expert_data": expert_data,
                "behavior_cloning_state": behavior_cloning_state,
                "behavior_cloning_optimizer_state": behavior_cloning_optimizer_state,
            },
        )
        replay_path = PROJECT_ROOT / "data" / "processed" / "teacher_replay.npz"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay = list(agent.replay_buffer.buffer)
        if replay:
            np.savez_compressed(
                replay_path,
                states=np.asarray([item[0] for item in replay], dtype=np.float32),
                actions=np.asarray([item[1] for item in replay], dtype=np.int64),
            )
        evaluation_env = RobotArmEnv(render_mode=None)
        try:
            final_evaluation = _write_final_evaluation(evaluation_env, agent)
        finally:
            evaluation_env.close()
        if on_evaluation is not None:
            on_evaluation(final_evaluation)
        print(f"Saved trained teacher weights to {model_path}")
        print(f"Saved replay states and actions to {replay_path}")
        print(
            f"Saved matched-seed evaluation to "
            f"{PROJECT_ROOT / 'evaluation_summary.csv'} and {PROJECT_ROOT / 'paired_evaluation.csv'}"
        )
