"""Bounded V1.1 policy distillation and independent evaluation utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dqn_teacher import DQNAgent, QNetwork, build_discrete_action_table
from experiments.v1_1.environment_v1_1 import RobotArmEnvV11


def _atomic_torch_save(payload, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


@dataclass
class RunConfig:
    mode: str = "deadline"
    seed: int = 20261126
    trajectories: int = 64
    max_steps: int = 200
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.0005
    temperature: float = 2.0
    q_regression_weight: float = 0.05
    validation_fraction: float = 0.20
    patience: int = 12
    evaluation_episodes: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_agent(device: str | None = None) -> DQNAgent:
    env = RobotArmEnvV11()
    try:
        return DQNAgent(env.observation_space.shape[0], len(env.controlled_joints), device=device)
    finally:
        env.close()


def collect_teacher_trajectories(agent: DQNAgent, count: int, seed: int,
                                 max_steps: int, progress=None) -> dict[str, np.ndarray]:
    """Collect greedy policy examples; every split is grouped by episode ID."""
    env = RobotArmEnvV11()
    obs_rows, q_rows, action_rows, traj_rows, success_rows, energy_rows = [], [], [], [], [], []
    try:
        for trajectory_id in range(count):
            obs, _ = env.reset(seed=seed + trajectory_id)
            episode_energy = 0.0
            success = False
            for _ in range(max_steps):
                state = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = agent.q_network(state).squeeze(0).cpu().numpy()
                action_idx = int(np.argmax(q_values))
                obs_rows.append(obs.copy())
                q_rows.append(q_values.copy())
                action_rows.append(action_idx)
                traj_rows.append(trajectory_id)
                obs, _, terminated, truncated, info = env.step(agent.action_table[action_idx])
                episode_energy += float(info["grid_draw"])
                success |= bool(terminated)
                if terminated or truncated:
                    break
            # Metadata is constant for this trajectory and joined by ID later.
            success_rows.append(success)
            energy_rows.append(episode_energy)
            if progress:
                progress(trajectory_id + 1, count, success, episode_energy)
    finally:
        env.close()
    return {
        "observations": np.asarray(obs_rows, dtype=np.float32),
        "q_values": np.asarray(q_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.int64),
        "trajectory_ids": np.asarray(traj_rows, dtype=np.int64),
        "trajectory_success": np.asarray(success_rows, dtype=np.bool_),
        "trajectory_energy_j": np.asarray(energy_rows, dtype=np.float32),
    }


def collect_corrective_trajectories(teacher: DQNAgent, student: nn.Module,
                                    count: int, seed: int, max_steps: int,
                                    device: str) -> dict[str, np.ndarray]:
    """Collect teacher Q/action labels on states visited by the student policy."""
    env = RobotArmEnvV11()
    student.eval()
    obs_rows, q_rows, action_rows, traj_rows, success_rows, energy_rows = [], [], [], [], [], []
    table = build_discrete_action_table(len(env.controlled_joints))
    first_id = 1_000_000 + int(seed)
    try:
        for episode in range(count):
            trajectory_id = first_id + episode
            obs, _ = env.reset(seed=seed + episode)
            energy, success = 0.0, False
            for _ in range(max_steps):
                state = torch.as_tensor(obs, dtype=torch.float32, device=teacher.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = teacher.q_network(state).squeeze(0).cpu().numpy()
                    student_state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action_idx = int(student(student_state).argmax(1).item())
                obs_rows.append(obs.copy()); q_rows.append(q_values.copy())
                action_rows.append(int(np.argmax(q_values))); traj_rows.append(trajectory_id)
                obs, _, terminated, truncated, info = env.step(table[action_idx])
                energy += float(info["grid_draw"])
                success |= bool(terminated)
                if terminated or truncated: break
            success_rows.append(success); energy_rows.append(energy)
    finally:
        env.close()
    return {"observations":np.asarray(obs_rows,dtype=np.float32),
            "q_values":np.asarray(q_rows,dtype=np.float32),
            "actions":np.asarray(action_rows,dtype=np.int64),
            "trajectory_ids":np.asarray(traj_rows,dtype=np.int64),
            "trajectory_success":np.asarray(success_rows,dtype=np.bool_),
            "trajectory_energy_j":np.asarray(energy_rows,dtype=np.float32)}


def _group_split(ids: np.ndarray, validation_fraction: float, seed: int):
    groups = np.unique(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * validation_fraction)))
    val_ids = set(groups[:n_val].tolist())
    mask = np.asarray([int(x) in val_ids for x in ids])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def successful_trajectory_subset(data: dict[str, np.ndarray], minimum_successes: int = 5):
    """Prefer successful greedy-expert trajectories; retain fallback transparently."""
    successful_ids = np.flatnonzero(np.asarray(data["trajectory_success"], dtype=bool))
    use_success_only = len(successful_ids) >= minimum_successes
    if not use_success_only:
        return data, {"successful_trajectories": int(len(successful_ids)),
                      "used_success_only": False,
                      "note": "Too few successful expert trajectories for a trajectory-disjoint split; retained all greedy trajectories and recorded success labels."}
    trajectory_order = np.unique(data["trajectory_ids"])
    successful_trajectory_ids = trajectory_order[np.asarray(data["trajectory_success"], dtype=bool)]
    keep = np.isin(data["trajectory_ids"], successful_trajectory_ids)
    filtered = {k: v[keep] if k in {"observations", "q_values", "actions", "trajectory_ids"} else v
                for k, v in data.items()}
    filtered["trajectory_success"] = np.ones(len(successful_trajectory_ids), dtype=np.bool_)
    filtered["trajectory_energy_j"] = np.asarray(data["trajectory_energy_j"])[np.asarray(data["trajectory_success"], dtype=bool)]
    return filtered, {"successful_trajectories": int(len(successful_ids)),
                      "used_success_only": True, "note": "Distillation transitions restricted to successful greedy teacher trajectories."}


def distillation_data_signature(data: dict[str, np.ndarray]) -> str:
    """Stable checksum of the observations and expert labels used for distillation."""
    signature_hash = hashlib.sha256()
    for key in ("observations", "q_values", "actions", "trajectory_ids"):
        arr = np.ascontiguousarray(data[key])
        signature_hash.update(key.encode("utf-8"))
        signature_hash.update(str(arr.shape).encode("ascii"))
        signature_hash.update(arr.tobytes())
    return signature_hash.hexdigest()


def distill_student(data: dict[str, np.ndarray], student: nn.Module,
                    config: RunConfig, output: Path, progress=None,
                    closed_loop_check=None, evaluate_every: int = 10,
                    resume: bool = True) -> list[dict]:
    """Train a student over complete epochs with trajectory-disjoint validation."""
    device = torch.device(config.device)
    student.to(device)
    train_idx, val_idx = _group_split(data["trajectory_ids"], config.validation_fraction, config.seed)
    obs = torch.as_tensor(data["observations"], dtype=torch.float32)
    qs = torch.as_tensor(data["q_values"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.long)
    train_ds = TensorDataset(obs[train_idx], qs[train_idx], actions[train_idx])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    opt = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    kl = nn.KLDivLoss(reduction="batchmean")
    ce = nn.CrossEntropyLoss()
    output.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, bad, history, start_epoch = float("inf"), 0, 0, [], 0
    resume_path = output / "student_resume_checkpoint.pt"
    data_signature = distillation_data_signature(data)
    if resume_path.exists() and not resume:
        raise RuntimeError("A student checkpoint already exists. Use resume=True or a fresh output directory; existing checkpoints are not overwritten.")
    if resume_path.exists() and resume:
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        old_config = dict(resume_state.get("config", {}))
        new_config = asdict(config)
        old_config.pop("device", None)
        new_config.pop("device", None)
        if old_config != new_config or resume_state.get("data_signature") != data_signature:
            raise RuntimeError("Student resume checkpoint does not match this configuration/data. Use a fresh output directory; stale checkpoints are never reused.")
        student.load_state_dict(resume_state["state_dict"])
        opt.load_state_dict(resume_state["optimizer"])
        best_val, best_epoch, bad = resume_state["best_val"], resume_state["best_epoch"], resume_state["bad_epochs"]
        history, start_epoch = resume_state["history"], resume_state["epoch"]
        if "python_rng_state" in resume_state:
            random.setstate(resume_state["python_rng_state"])
            np.random.set_state(resume_state["numpy_rng_state"])
            torch.set_rng_state(resume_state["torch_rng_state"].cpu())
            if resume_state.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in resume_state["cuda_rng_state_all"]])
    for epoch in range(start_epoch + 1, config.epochs + 1):
        student.train()
        total, n = 0.0, 0
        for xb, qb, ab in train_loader:
            xb, qb, ab = xb.to(device), qb.to(device), ab.to(device)
            logits = student(xb)
            temp = config.temperature
            kd = kl(torch.log_softmax(logits / temp, dim=1), torch.softmax(qb / temp, dim=1)) * temp**2
            loss = kd + ce(logits, ab) + config.q_regression_weight * nn.functional.smooth_l1_loss(logits, qb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        student.eval()
        with torch.no_grad():
            vx, vq, va = obs[val_idx].to(device), qs[val_idx].to(device), actions[val_idx].to(device)
            vl = student(vx)
            temp = config.temperature
            val_loss = float((kl(torch.log_softmax(vl / temp, dim=1), torch.softmax(vq / temp, dim=1)) * temp**2
                              + ce(vl, va)
                              + config.q_regression_weight * nn.functional.smooth_l1_loss(vl, vq)).cpu())
            agreement = float((vl.argmax(1) == va).float().mean().cpu())
        row = {"epoch": epoch, "train_loss": total / max(n, 1), "validation_loss": val_loss,
               "validation_action_agreement": agreement, "closed_loop_validation_success": np.nan}
        if closed_loop_check is not None and epoch % evaluate_every == 0:
            row["closed_loop_validation_success"] = float(closed_loop_check(student, epoch))
        history.append(row)
        if progress:
            progress(row)
        if val_loss < best_val:
            best_val, best_epoch, bad = val_loss, epoch, 0
            _atomic_torch_save({"state_dict": student.state_dict(), "epoch": epoch,
                                "validation_loss": val_loss, "config": asdict(config)}, output / "student_best.pt")
        else:
            bad += 1
            if bad >= config.patience:
                break
        _atomic_torch_save({"state_dict": student.state_dict(), "optimizer": opt.state_dict(),
                            "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch,
                            "bad_epochs": bad, "history": history, "config": asdict(config),
                            "data_signature": data_signature,
                            "python_rng_state":random.getstate(), "numpy_rng_state":np.random.get_state(),
                            "torch_rng_state":torch.get_rng_state(),
                            "cuda_rng_state_all":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, resume_path)
    _atomic_torch_save({"state_dict": student.state_dict(), "epoch": best_epoch,
                        "validation_loss": best_val, "config": asdict(config)}, output / "student_last.pt")
    with (output / "student_training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output / "student_training_complete.json").write_text(json.dumps({
        "completed_epochs": len(history), "best_epoch": best_epoch,
        "best_validation_loss": best_val, "data_signature": data_signature,
        "config": asdict(config)}, indent=2), encoding="utf-8")
    return history


def evaluate_policy(agent: DQNAgent, episodes: int, seed_start: int,
                    max_steps: int, label: str = "policy") -> list[dict]:
    """Greedy closed-loop target-reaching evaluation on independent seeds."""
    env = RobotArmEnvV11()
    results = []
    try:
        for offset in range(episodes):
            obs, _ = env.reset(seed=seed_start + offset)
            episode_reward, peak_w, steps = 0.0, 0.0, 0
            success, grid_j, mechanical_j = False, 0.0, 0.0
            max_buffer_j = float(env.energy_buffer)
            components = {"reward_energy":0.0,"reward_peak":0.0,"reward_progress":0.0,"reward_completion":0.0}
            for steps in range(1, max_steps + 1):
                action_idx, action = agent.select_action(obs, epsilon=0.0)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                grid_j += float(info["grid_draw"])
                mechanical_j += float(info["energy_raw"])
                peak_w = max(peak_w, float(info["grid_power_w"]))
                max_buffer_j = max(max_buffer_j, float(info["buffer_level"]))
                for key in components: components[key] += float(info[key])
                success |= bool(terminated)
                if terminated or truncated:
                    break
            results.append({"policy": label, "seed": seed_start + offset, "success": int(success),
                            "steps": steps, "episode_reward": episode_reward,
                            "grid_energy_j": grid_j, "peak_grid_power_w": peak_w,
                            "mechanical_energy_j": mechanical_j,
                            "completion_time_s": steps * float(info["timestep_s"]),
                            "position_error_m": float(info["distance"]),
                            "buffer_max_level_j": max_buffer_j,
                            "buffer_capacity_fraction": max_buffer_j / env.BUFFER_CAPACITY,
                            "buffer_discharged_j": float(env.total_buffer_used),
                            "energy_per_success_j": grid_j if success else np.nan,
                            "failure_energy_j": grid_j if not success else np.nan, **components})
    finally:
        env.close()
    return results


def evaluate_student_validation(student: nn.Module, seeds: list[int],
                                max_steps: int, device: str) -> float:
    """Closed-loop validation helper using only the student network outputs."""
    env = RobotArmEnvV11()
    student.eval()
    successes = 0
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            for _ in range(max_steps):
                with torch.no_grad():
                    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action_idx = int(student(tensor).argmax(1).item())
                obs, _, terminated, truncated, _ = env.step(env_action_from_index(env, action_idx))
                if terminated:
                    successes += 1
                    break
                if truncated:
                    break
    finally:
        env.close()
    return successes / max(1, len(seeds))


def env_action_from_index(env: RobotArmEnvV11, action_idx: int) -> np.ndarray:
    """Map the original 89-action index through the preserved action table."""
    return build_discrete_action_table(len(env.controlled_joints))[action_idx]


def new_student(hidden_size: int = 64, state_dim: int = 23, actions: int = 89) -> QNetwork:
    return QNetwork(state_dim, actions, hidden_size=hidden_size)


def run_ik_baseline(controller: str, episodes: int, seed_start: int,
                    max_steps: int, fixed_speed: float = 2.0) -> list[dict]:
    """Matched-seed IK baselines; target reaching only, with shared env ledger."""
    import pybullet as p

    if controller not in {"fixed_original", "fixed_slow", "proportional", "adaptive"}:
        raise ValueError(f"Unknown IK baseline: {controller}")
    env = RobotArmEnvV11()
    results = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed_start + episode)
            success, energy_j, mechanical_j, reward_sum, peak_w = False, 0.0, 0.0, 0.0, 0.0
            max_buffer_j = float(env.energy_buffer)
            components = {"reward_energy":0.0,"reward_peak":0.0,"reward_progress":0.0,"reward_completion":0.0}
            for step in range(max_steps):
                target = env.target_position
                joint_targets = p.calculateInverseKinematics(
                    env.robot_id, env.end_effector_link, target,
                    maxNumIterations=100, residualThreshold=1e-4)
                current = np.asarray([p.getJointState(env.robot_id, j)[0] for j in env.controlled_joints])
                delta = np.asarray(joint_targets[:len(env.controlled_joints)]) - current
                if controller in {"fixed_original", "fixed_slow"}:
                    speed = fixed_speed if controller == "fixed_original" else fixed_speed * 0.5
                    desired = np.where(np.abs(delta) > 1e-3, np.sign(delta) * speed, 0.0)
                elif controller == "proportional":
                    desired = np.clip(delta * 3.0, -0.8, 0.8)
                else:
                    # Lower speed for larger joint motion; conservative near target.
                    distance = float(np.linalg.norm(obs[12:15] - target))
                    scale = float(np.clip(distance / 0.25, 0.12, 1.0))
                    desired = np.sign(delta) * np.minimum(np.abs(delta) * 6.0, 0.9 * scale)
                action = np.clip(desired / env.MAX_JOINT_VELOCITY, -1.0, 1.0).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                energy_j += float(info["grid_draw"])
                mechanical_j += float(info["energy_raw"])
                peak_w = max(peak_w, float(info["grid_power_w"]))
                max_buffer_j = max(max_buffer_j, float(info["buffer_level"]))
                for key in components: components[key] += float(info[key])
                success |= bool(terminated)
                if terminated or truncated:
                    break
            results.append({"policy": controller, "seed": seed_start + episode,
                            "success": int(success), "steps": step + 1,
                            "episode_reward": reward_sum, "grid_energy_j": energy_j,
                            "mechanical_energy_j": mechanical_j,
                            "completion_time_s": (step + 1) * float(info["timestep_s"]),
                            "position_error_m": float(info["distance"]),
                            "peak_grid_power_w": peak_w,
                            "buffer_max_level_j": max_buffer_j,
                            "buffer_capacity_fraction": max_buffer_j / env.BUFFER_CAPACITY,
                            "buffer_discharged_j": float(env.total_buffer_used),
                            "energy_per_success_j": energy_j if success else np.nan,
                            "failure_energy_j": energy_j if not success else np.nan, **components})
    finally:
        env.close()
    return results


class DQNAgentV11(DQNAgent):
    """Double-DQN update with Huber loss and bounded gradients for V1.1 runs."""

    def learn(self, batch_size=64, expert_data=None, imitation_weight=0.0):
        if len(self.replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, terminal = self.replay_buffer.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        terminal = torch.as_tensor(terminal, dtype=torch.float32, device=self.device)
        current = self.q_network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            selected = self.q_network(next_states).argmax(1, keepdim=True)
            next_value = self.target_network(next_states).gather(1, selected).squeeze(1)
            target = rewards + self.gamma * next_value * (1.0 - terminal)
        loss = nn.functional.smooth_l1_loss(current, target)
        if expert_data is not None and imitation_weight > 0:
            expert_states, expert_actions = expert_data
            take = np.random.randint(0, len(expert_states), size=min(batch_size, len(expert_states)))
            xs = torch.as_tensor(expert_states[take], dtype=torch.float32, device=self.device)
            ys = torch.as_tensor(expert_actions[take], dtype=torch.long, device=self.device)
            loss = loss + float(imitation_weight) * nn.functional.cross_entropy(self.q_network(xs), ys)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.detach().cpu())


def train_teacher(agent: DQNAgentV11, episodes: int, seed_start: int,
                  max_steps: int, output: Path, progress=None, resume: bool = True) -> list[dict]:
    """Bounded online fine-tuning; timeout is terminal for this finite horizon."""
    env = RobotArmEnvV11()
    history = []
    epsilon = 0.35
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "teacher_resume_checkpoint.pt"
    history_path = output / "teacher_training_history.csv"
    run_signature = {"episodes": int(episodes), "seed_start": int(seed_start),
                     "max_steps": int(max_steps), "architecture": "23-256-256-89"}
    start_episode = 0
    if resume and checkpoint_path.exists():
        metadata = agent.load_checkpoint(checkpoint_path)
        if metadata.get("run_signature") != run_signature:
            raise RuntimeError("Teacher resume checkpoint is stale or belongs to another run. Use a fresh output directory.")
        start_episode = int(metadata.get("episode", 0))
        epsilon = float(metadata.get("epsilon", epsilon))
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as existing:
                history = list(csv.DictReader(existing))
    elif checkpoint_path.exists() and not resume:
        raise RuntimeError("A teacher checkpoint already exists. Use resume=True or a fresh output directory; existing checkpoints are not overwritten.")
    try:
        for episode in range(start_episode, episodes):
            obs, _ = env.reset(seed=seed_start + episode)
            reward_sum = energy_j = 0.0
            components = {"reward_energy":0.0,"reward_peak":0.0,"reward_progress":0.0,"reward_completion":0.0}
            success = False
            for step in range(max_steps):
                action_idx, action = agent.select_action(obs, epsilon)
                next_obs, reward, terminated, truncated, info = env.step(action)
                # A finite-horizon timeout ends this episode and must not bootstrap.
                agent.store(obs, action_idx, reward, next_obs, bool(terminated or truncated))
                loss = agent.learn(batch_size=64)
                if (step + 1) % 250 == 0:
                    agent.update_target_network()
                obs = next_obs
                reward_sum += float(reward)
                energy_j += float(info["grid_draw"])
                for key in components: components[key] += float(info[key])
                success |= bool(terminated)
                if terminated or truncated:
                    break
            epsilon = max(0.05, epsilon * 0.992)
            history.append({"episode": episode + 1, "reward": reward_sum, "success": int(success),
                            "grid_energy_j": energy_j, "epsilon": epsilon,
                            "loss": float(loss) if loss is not None else np.nan, **components})
            if progress:
                progress(history[-1])
            if (episode + 1) % 10 == 0:
                agent.save_checkpoint(checkpoint_path,
                                      metadata={"episode": episode + 1, "epsilon": epsilon,
                                                "run_signature": run_signature})
                with history_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(history[0]))
                    writer.writeheader(); writer.writerows(history)
    finally:
        env.close()
    agent.save_checkpoint(checkpoint_path, metadata={
        "episode": (int(history[-1]["episode"]) if history else start_episode),
        "epsilon": epsilon, "run_signature": run_signature})
    output.mkdir(parents=True, exist_ok=True)
    agent.save(output / "teacher_finetuned.pt")
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    (output / "teacher_training_complete.json").write_text(json.dumps({
        "completed_episodes": len(history),
        "episodes_requested": int(episodes),
        "successful_episodes": int(sum(int(row["success"]) for row in history)),
        "final_epsilon": float(epsilon), "run_signature": run_signature,
        "weights_file": "teacher_finetuned.pt"}, indent=2), encoding="utf-8")
    return history
