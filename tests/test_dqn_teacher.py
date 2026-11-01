import unittest

import numpy as np

from src.dqn_teacher import DQNAgent, build_discrete_action_table


class DQNAgentTests(unittest.TestCase):
    def test_action_table_matches_six_joint_environment(self):
        actions = build_discrete_action_table(6)

        self.assertEqual(len(actions), 25)
        self.assertEqual(actions[0].shape, (6,))
        self.assertTrue(all(np.all(action >= -1.0) and np.all(action <= 1.0) for action in actions))

    def test_teacher_uses_project_network_and_replay_sizes(self):
        agent = DQNAgent(state_dim=23, num_joints=6, device="cpu")

        self.assertEqual(agent.replay_buffer.buffer.maxlen, 10_000)
        self.assertEqual(agent.q_network.net[0].out_features, 256)
        self.assertEqual(agent.q_network.net[2].out_features, 256)
        self.assertEqual(agent.q_network.net[-1].out_features, 25)

    def test_agent_selects_environment_compatible_action(self):
        agent = DQNAgent(state_dim=23, num_joints=6, device="cpu")
        action_index, action = agent.select_action(np.zeros(23, dtype=np.float32), epsilon=0.0)

        self.assertIn(action_index, range(agent.num_actions))
        self.assertEqual(action.shape, (6,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_agent_learns_after_replay_buffer_fills(self):
        agent = DQNAgent(state_dim=23, num_joints=6, device="cpu")
        observation = np.zeros(23, dtype=np.float32)
        for _ in range(64):
            agent.store(observation, 0, 1.0, observation, False)

        loss = agent.learn(batch_size=64)

        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
