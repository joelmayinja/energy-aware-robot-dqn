"""Run a single reproducible PyBullet V1.1 target-reaching demonstration."""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from experiments.v1_1.environment_v1_1 import RobotArmEnvV11
from experiments.v1_1.pipeline import env_action_from_index
from src.dqn_teacher import QNetwork


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "DEFENCE_2026" / "models"
CONTROLLERS = ("ik", "teacher_original", "student_original", "student_improved")
MODEL_FILES = {
    "teacher_original": ("teacher_original.pt", 256, 94809),
    "student_original": ("student_original.pt", 64, 11481),
    "student_improved": ("student_final.pt", 64, 11481),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, default="student_improved")
    parser.add_argument("--seed", type=int, default=140000,
                        help="Scenario seed; use the same seed for each controller to match its scenario.")
    parser.add_argument("--headless", action="store_true", help="Use PyBullet DIRECT without a GUI.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--slowdown", type=float, default=5.0,
                        help="GUI-only wall-clock pacing multiplier; does not change simulated time or physics.")
    parser.add_argument("--log-file", type=Path, default=ROOT / "DEFENCE_2026" / "demo_runs.csv")
    parser.add_argument("--no-realtime", action="store_true", help="Run the GUI simulation without wall-clock pacing.")
    return parser.parse_args()


def load_controller(name: str, device: torch.device):
    if name == "ik":
        return None, 0
    filename, hidden, parameter_count = MODEL_FILES[name]
    path = MODEL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Defence checkpoint is missing: {path}. Run experiments/v1_1/export_defence_models.py first.")
    model = QNetwork(23, 89, hidden_size=hidden).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.eval()
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != parameter_count:
        raise ValueError(f"Unexpected parameter count for {name}: {actual}; expected {parameter_count}.")
    return model, actual


def action_for(controller: str, model, obs: np.ndarray, env: RobotArmEnvV11,
               device: torch.device) -> np.ndarray:
    if controller == "ik":
        joint_targets = p.calculateInverseKinematics(
            env.robot_id, env.end_effector_link, env.target_position,
            maxNumIterations=100, residualThreshold=1e-4,
            physicsClientId=env.physics_client)
        current = np.asarray([p.getJointState(env.robot_id, j, physicsClientId=env.physics_client)[0]
                              for j in env.controlled_joints])
        delta = np.asarray(joint_targets[:len(env.controlled_joints)]) - current
        desired = np.where(np.abs(delta) > 1e-3, np.sign(delta) * 2.0, 0.0)
        return np.clip(desired / env.MAX_JOINT_VELOCITY, -1, 1).astype(np.float32)
    with torch.no_grad():
        state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_index = int(model(state).argmax(1).item())
    return env_action_from_index(env, action_index)


def run(args: argparse.Namespace) -> dict:
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    device = torch.device("cpu")
    model, parameter_count = load_controller(args.controller, device)
    env = RobotArmEnvV11(render_mode=None if args.headless else "human")
    peak_w = total_grid_j = total_mech_j = 0.0
    success = False
    info = {"task_phase": 0, "distance": float("nan"), "timestep_s": 1/240,
            "grid_power_w": 0.0, "grid_draw": 0.0}
    hud_id = target_id = -1
    try:
        obs, _ = env.reset(seed=int(args.seed))
        client = env.physics_client
        if not args.headless:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
        print(f"DEMONSTRATION ONLY — controller={args.controller}; seed={args.seed}; "
              f"sample=1 scenario, not the 50-scenario evaluation")
        print(f"Initial light target (m): {env.light_target.tolist()}; heavy target (m): {env.heavy_target.tolist()}")
        for step in range(1, args.max_steps + 1):
            action = action_for(args.controller, model, obs, env, device)
            obs, _, terminated, truncated, info = env.step(action)
            total_grid_j += float(info["grid_draw"])
            total_mech_j += float(info["energy_raw"])
            peak_w = max(peak_w, float(info["grid_power_w"]))
            success = success or bool(terminated)
            if not args.headless:
                target_id = p.addUserDebugText("TARGET", env.target_position.tolist(),
                    textColorRGB=[1, 0.25, 0.15], textSize=1.2, lifeTime=0.15,
                    replaceItemUniqueId=target_id, physicsClientId=client)
                hud = (f"{args.controller} | phase {info['task_phase']}/1 | "
                       f"done={success} | t={step*info['timestep_s']:.2f}s | "
                       f"grid={total_grid_j:.1f}J | peak={peak_w:.1f}W | "
                       f"params={parameter_count if parameter_count else 'IK'}")
                hud_id = p.addUserDebugText(hud, [0.0, 0.0, 1.35],
                    textColorRGB=[1, 1, 1], textSize=1.0, lifeTime=0.15,
                    replaceItemUniqueId=hud_id, physicsClientId=client)
                if not args.no_realtime:
                    time.sleep(min(float(info["timestep_s"])*max(0.0,args.slowdown), 0.05))
            if step % 20 == 0 or terminated or truncated:
                print(f"step={step}/{args.max_steps} controller={args.controller} "
                      f"phase={info['task_phase']} target={env.target_position.round(3).tolist()} "
                      f"error={info['distance']:.3f}m done={success} "
                      f"sim_time={step*info['timestep_s']:.3f}s grid={total_grid_j:.3f}J "
                      f"peak={peak_w:.3f}W params={parameter_count if parameter_count else 'IK'}")
            if terminated or truncated:
                break
        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "controller": args.controller, "seed": int(args.seed),
            "demonstration_episodes": 1, "is_full_evaluation": False,
            "success": bool(success), "steps": step,
            "completion_time_s": step*float(info["timestep_s"]),
            "final_position_error_m": float(info["distance"]),
            "grid_energy_j": total_grid_j, "mechanical_energy_estimate_j": total_mech_j,
            "peak_simulated_power_w": peak_w, "parameter_count": parameter_count,
            "task": "two-phase target reaching; no grasp/place",
            "energy_note": "V1.1 simulated joint mechanical-work ledger; not measured electrical consumption",
        }
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        existed = args.log_file.exists()
        with args.log_file.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            if not existed:
                writer.writeheader()
            writer.writerow(result)
        print("DEMONSTRATION SUMMARY:", result)
        print("Logged to:", args.log_file)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    from experiments.v1_1.defence_visualizer import main as defence_main
    defence_main()
