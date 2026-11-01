import sys
import tempfile
import unittest
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from src import training


class FakeRobotArmEnv:
    MAX_STEPS = 2
    BUFFER_CAPACITY = 50.0

    def __init__(self, render_mode=None, buffer_enabled=True):
        self.observation_space = SimpleNamespace(shape=(23,))
        self.action_space = SimpleNamespace(shape=(6,))
        self.controlled_joints = list(range(6))
        self.target_position = np.zeros(3)
        self.buffer_enabled = buffer_enabled
        self.step_count = 0
        self.total_grid_draw = 0.0
        self.peak_grid_draw = 0.0
        self.robot_id = 1
        self.end_effector_link = 6

    def reset(self, seed=None):
        self.step_count = 0
        self.total_grid_draw = 0.0
        self.peak_grid_draw = 0.0
        return np.zeros(23, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        grid_draw = 0.5
        self.total_grid_draw += grid_draw
        self.peak_grid_draw = max(self.peak_grid_draw, grid_draw)
        observation = np.full(23, self.step_count / 10, dtype=np.float32)
        info = {
            "grid_draw": grid_draw,
            "total_grid_draw": self.total_grid_draw,
            "peak_grid_draw": self.peak_grid_draw,
            "buffer_level": 6.0,
            "total_buffer_used": 0.0,
            "distance": 0.4,
            "task_phase": int(self.step_count > 1),
            "task_done": False,
        }
        return observation, -0.5, False, self.step_count >= self.MAX_STEPS, info

    def close(self):
        pass


class TrainingLoopTests(unittest.TestCase):
    def test_training_compares_baseline_saves_outputs_and_records_results(self):
        fake_environment = ModuleType("src.environment")
        fake_environment.RobotArmEnv = FakeRobotArmEnv
        fake_pybullet = ModuleType("pybullet")
        fake_pybullet.calculateInverseKinematics = lambda *args: [0.0] * 7
        fake_pybullet.getJointState = lambda *args: [0.0]
        progress = []
        completed = []
        pause_event = Event()
        pause_event.set()

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.dict(
                    sys.modules,
                    {"src.environment": fake_environment, "pybullet": fake_pybullet},
                ),
                patch.object(training, "PROJECT_ROOT", training.Path(temporary_directory)),
            ):
                training.train(
                    1,
                    Event(),
                    pause_event,
                    lambda: 5,
                    progress.append,
                    completed.append,
                )

            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["grid_energy"], 1.0)
            self.assertEqual(completed[0]["baseline_grid_energy"], 1.0)
            self.assertEqual(completed[0]["peak_reduction_percent"], 0.0)
            self.assertEqual([item["stage"] for item in progress], ["baseline"] * 2 + ["teacher"] * 2)
            self.assertTrue((training.Path(temporary_directory) / "models" / "teacher.pt").is_file())
            self.assertTrue((training.Path(temporary_directory) / "training_history.csv").is_file())
            self.assertTrue((training.Path(temporary_directory) / "data" / "processed" / "teacher_replay.npz").is_file())


if __name__ == "__main__":
    unittest.main()
