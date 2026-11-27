"""Run a tiny local end-to-end V1.1 smoke test; outputs are labelled smoke only."""
from __future__ import annotations

import csv
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from experiments.v1_1.environment_v1_1 import RobotArmEnvV11
from experiments.v1_1.pipeline import (
    DQNAgentV11, RunConfig, build_agent, collect_teacher_trajectories,
    distill_student, evaluate_policy, evaluate_student_validation,
    new_student, run_ik_baseline, seed_everything,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    run_root = root / "experiments" / "v1_1" / "runs" / "2026-11-26"
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / f"smoke_local_v1_1_{datetime.now().strftime('%H%M%S_%f')}"
    output.mkdir(parents=True, exist_ok=True)
    RobotArmEnvV11.MAX_STEPS = 20
    cfg = RunConfig(mode="smoke", trajectories=3, epochs=2, batch_size=32,
                    patience=2, max_steps=20, evaluation_episodes=2, device="cpu")
    seed_everything(cfg.seed)

    env = RobotArmEnvV11()
    try:
        obs, _ = env.reset(seed=410)
        obs2, reward, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        assert obs.shape == obs2.shape == (23,)
        assert np.isfinite(reward) and info["grid_draw"] >= 0 and info["grid_power_w"] >= 0
        assert not term and not trunc
    finally:
        env.close()

    teacher = build_agent(device="cpu")
    teacher.q_network.load_state_dict(torch.load(root / "models" / "teacher.pt", map_location="cpu", weights_only=True))
    teacher.target_network.load_state_dict(teacher.q_network.state_dict())
    assert teacher.num_actions == 89
    data = collect_teacher_trajectories(teacher, 3, 500, 20)
    np.savez_compressed(output / "smoke_demonstrations.npz", **data)
    student = new_student()
    history = distill_student(data, student, cfg, output,
        closed_loop_check=lambda model, epoch: evaluate_student_validation(model, [800+epoch], 20, "cpu"),
        evaluate_every=1)
    best = torch.load(output / "student_best.pt", map_location="cpu", weights_only=False)
    student.load_state_dict(best["state_dict"])
    actor = build_agent(device="cpu")
    actor.q_network = student
    actor.target_network = copy.deepcopy(student)
    student_results = evaluate_policy(actor, 2, 900, 20, "student_smoke")
    baseline_results = run_ik_baseline("fixed_original", 2, 900, 20)
    with (output / "smoke_evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(student_results[0]))
        w.writeheader(); w.writerows(student_results)
    agent = DQNAgentV11(23, 6, device="cpu")
    for i in range(64):
        s = np.full(23, i / 64, dtype=np.float32)
        agent.store(s, i % 89, 1.0, s * 0.9, i == 63)
    assert np.isfinite(agent.learn(batch_size=64))
    agent.save_checkpoint(output / "checkpoint_roundtrip.pt", metadata={"smoke": True})
    agent2 = DQNAgentV11(23, 6, device="cpu")
    meta = agent2.load_checkpoint(output / "checkpoint_roundtrip.pt")
    assert meta["smoke"] and len(agent2.replay_buffer) == 64
    print({"smoke_only": True, "demonstration_transitions": len(data["actions"]),
           "student_epochs": len(history), "student_eval_episodes": len(student_results),
           "baseline_eval_episodes": len(baseline_results), "checkpoint_replay_size": len(agent2.replay_buffer),
           "output": str(output)})


if __name__ == "__main__":
    main()
