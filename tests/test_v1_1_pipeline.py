"""Smoke and regression checks for the isolated V1.1 implementation."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.v1_1.environment_v1_1 import RobotArmEnvV11, account_step_energy
from experiments.v1_1.pipeline import (DQNAgentV11, RunConfig, build_agent,
                                       distill_student, new_student, train_teacher)


class V11EnvironmentTests(unittest.TestCase):
    def test_buffer_efficiency_and_energy_balance(self):
        charged = account_step_energy(2.0, 10.0, 0.0, 50.0, .8, .9, True, 0)
        self.assertAlmostEqual(charged["stored_j"], 8.0)
        self.assertAlmostEqual(charged["buffer_j"], 8.0)
        used = account_step_energy(10.0, 0.0, 5.0, 50.0, .8, .9, True, 1)
        self.assertAlmostEqual(used["grid_j"] + used["delivered_j"], 10.0)
        self.assertAlmostEqual(used["removed_j"], 5.0)
        self.assertAlmostEqual(used["buffer_j"], 0.0)
        self.assertGreaterEqual(used["grid_j"], 0.0)

    def test_buffer_capacity_spill_is_not_created_energy(self):
        result = account_step_energy(0.0, 10.0, 49.0, 50.0, .8, .9, True, 0)
        self.assertAlmostEqual(result["stored_j"], 1.0)
        self.assertAlmostEqual(result["spilled_j"], 7.0)
        self.assertEqual(result["buffer_j"], 50.0)

    def test_one_step_has_finite_nonnegative_dimensioned_energy(self):
        env = RobotArmEnvV11()
        try:
            obs, _ = env.reset(seed=811)
            self.assertEqual(obs.shape, (23,))
            next_obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
            self.assertEqual(next_obs.shape, (23,))
            self.assertTrue(np.isfinite(reward))
            self.assertAlmostEqual(reward, sum(info[k] for k in ("reward_energy", "reward_peak", "reward_progress", "reward_completion")))
            self.assertGreaterEqual(info["grid_draw"], 0.0)
            self.assertGreaterEqual(info["grid_power_w"], 0.0)
            self.assertGreaterEqual(info["buffer_level"], 0.0)
            self.assertLessEqual(info["buffer_level"], env.BUFFER_CAPACITY)
            self.assertGreater(info["timestep_s"], 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        finally:
            env.close()

    def test_teacher_checkpoint_has_original_architecture(self):
        agent = build_agent(device="cpu")
        state = torch.load("models/teacher.pt", map_location="cpu", weights_only=True)
        agent.q_network.load_state_dict(state)
        self.assertEqual(agent.q_network.net[0].in_features, 23)
        self.assertEqual(agent.q_network.net[-1].out_features, 89)

    def test_v11_double_dqn_update_runs_with_terminal_replay(self):
        agent = DQNAgentV11(23, 6, device="cpu")
        for idx in range(8):
            s = np.full(23, idx / 10, dtype=np.float32)
            agent.store(s, idx % agent.num_actions, 1.0, s + .01, idx == 7)
        loss = agent.learn(batch_size=8)
        self.assertTrue(np.isfinite(loss))

    def test_teacher_training_writes_teacher_completion_metadata(self):
        agent = DQNAgentV11(23, 6, device="cpu")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            history = train_teacher(agent, episodes=1, seed_start=99001,
                                    max_steps=1, output=output, resume=False)
            self.assertEqual(len(history), 1)
            marker = json.loads((output / "teacher_training_complete.json").read_text())
            self.assertEqual(marker["completed_episodes"], 1)
            self.assertEqual(marker["episodes_requested"], 1)
            self.assertTrue((output / "teacher_finetuned.pt").is_file())
            self.assertFalse((output / "student_training_complete.json").exists())

    def test_teacher_resume_rejects_a_stale_run_signature(self):
        agent = DQNAgentV11(23, 6, device="cpu")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            train_teacher(agent, episodes=1, seed_start=99002,
                          max_steps=1, output=output, resume=False)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                train_teacher(DQNAgentV11(23, 6, device="cpu"), episodes=2,
                              seed_start=99002, max_steps=1, output=output, resume=True)

    def test_student_resume_records_completion_and_rejects_changed_data(self):
        rng = np.random.default_rng(42)
        data = {"observations": rng.normal(size=(8, 23)).astype(np.float32),
                "q_values": rng.normal(size=(8, 89)).astype(np.float32),
                "actions": rng.integers(0, 89, size=8, dtype=np.int64),
                "trajectory_ids": np.arange(8, dtype=np.int64)}
        cfg = RunConfig(mode="test", seed=42, epochs=1, batch_size=4,
                        patience=1, device="cpu")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            history = distill_student(data, new_student(), cfg, output,
                                      evaluate_every=1, resume=False)
            self.assertEqual(len(history), 1)
            marker = json.loads((output / "student_training_complete.json").read_text())
            self.assertEqual(marker["completed_epochs"], 1)
            changed = dict(data)
            changed["q_values"] = data["q_values"].copy()
            changed["q_values"][0, 0] += 1
            with self.assertRaisesRegex(RuntimeError, "configuration/data"):
                distill_student(changed, new_student(), cfg, output, resume=True)


if __name__ == "__main__":
    unittest.main()
