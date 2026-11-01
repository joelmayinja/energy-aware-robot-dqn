import unittest
import importlib.util

import numpy as np

if importlib.util.find_spec("pybullet") is None:
    raise unittest.SkipTest("Install PyBullet to run KUKA environment tests.")

from src.environment import RobotArmEnv


class RobotArmEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = RobotArmEnv(render_mode=None)
        self.addCleanup(self.env.close)

    def test_reset_returns_expected_observation(self):
        observation, info = self.env.reset(seed=7)

        self.assertEqual(observation.shape, (23,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(info, {})
        self.assertGreaterEqual(self.env.energy_buffer, 5.0)
        self.assertLessEqual(self.env.energy_buffer, 15.0)

    def test_step_returns_energy_metrics_and_valid_episode_flags(self):
        self.env.reset(seed=7)
        observation, reward, terminated, truncated, info = self.env.step(
            np.zeros(6, dtype=np.float32)
        )

        self.assertEqual(observation.shape, (23,))
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        for key in (
            "grid_draw",
            "total_grid_draw",
            "buffer_level",
            "distance",
            "task_done",
            "peak_penalty",
            "peak_grid_draw",
        ):
            self.assertIn(key, info)
        self.assertGreaterEqual(info["buffer_level"], 0.0)
        self.assertLessEqual(info["buffer_level"], self.env.BUFFER_CAPACITY)
        self.assertAlmostEqual(info["peak_grid_draw"], info["grid_draw"])

    def test_constant_speed_baseline_does_not_use_energy_buffer(self):
        baseline = RobotArmEnv(render_mode=None, buffer_enabled=False)
        self.addCleanup(baseline.close)

        baseline.reset(seed=7)
        self.assertEqual(baseline.energy_buffer, 0.0)
        _, _, _, _, info = baseline.step(np.ones(6, dtype=np.float32))

        self.assertEqual(info["buffer_level"], 0.0)
        self.assertEqual(info["buffer_deployed"], 0.0)
        self.assertEqual(info["grid_draw"], info["energy_raw"])


if __name__ == "__main__":
    unittest.main()
