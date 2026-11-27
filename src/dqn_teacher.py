"""
src/dqn_teacher.py

The DQN "Teacher" model - the core learning algorithm from the proposal.

Contains:
    - QNetwork: a small neural network that estimates how good each
      discrete action is, given the current state.
    - ReplayBuffer: a memory of past (state, action, reward, next_state, done)
      experiences, sampled randomly during training for stability.
    - DQNAgent: ties the two together - chooses actions, learns from batches,
      and keeps a slow-moving "target network" for stable learning targets.

Note on action space: our environment (src/environment.py) uses a
CONTINUOUS action space (6 joint velocities, each -1 to 1) because that's
how PyBullet's velocity control works. Classic DQN expects a small DISCRETE
set of actions. To bridge this, we discretize each joint into a small set
of velocity levels (e.g. "fast negative", "slow negative", "stop",
"slow positive", "fast positive") and let the agent pick one combination.
This keeps DQN's actual learning rule simple and matches the proposal's
"Deep Q-Network" framing, at the cost of a coarser range of motion than
true continuous control. A logical future improvement (not required for
the proposal) would be DDPG/SAC for continuous actions.
"""

import random
import os
import tempfile
from itertools import product
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Action discretization -------------------------------------------------
# Each of the 6 joints can move at one of these velocity levels.
VELOCITY_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]
NUM_LEVELS = len(VELOCITY_LEVELS)


def build_discrete_action_table(num_joints):
    """
    Builds a compact set containing the original single-joint actions plus
    coordinated full-speed directions for all joints. The coordinated
    actions let the DQN change several joints in one simulation step, as the
    IK baseline does, without enumerating all 5**num_joints combinations.

    Returns a list of 6-length velocity arrays, one per discrete action.
    """
    actions = [np.zeros(num_joints, dtype=np.float32)]  # action 0: stay still
    for joint_idx in range(num_joints):
        for level in VELOCITY_LEVELS:
            if level == 0.0:
                continue  # already covered by the "stay still" action
            action = np.zeros(num_joints, dtype=np.float32)
            action[joint_idx] = level
            actions.append(action)
    for directions in product((-1.0, 1.0), repeat=num_joints):
        actions.append(np.asarray(directions, dtype=np.float32))
    return actions


class QNetwork(nn.Module):
    """Small feedforward network: state -> one Q-value per discrete action."""

    def __init__(self, state_dim, num_actions, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    """Fixed-size memory of past experiences, sampled randomly for training."""

    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Ties QNetwork + ReplayBuffer together: chooses actions (epsilon-greedy),
    stores experiences, and runs the actual learning update.
    """

    def __init__(self, state_dim, num_joints, lr=1e-3, gamma=0.99,
                 buffer_capacity=10_000, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_table = build_discrete_action_table(num_joints)
        self.num_actions = len(self.action_table)
        self.gamma = gamma

        self.q_network = QNetwork(state_dim, self.num_actions).to(self.device)
        self.target_network = QNetwork(state_dim, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target net is never trained directly

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon):
        """Epsilon-greedy: random action with probability epsilon, otherwise
        the action the network currently thinks is best."""
        if random.random() < epsilon:
            action_idx = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.q_network(state_t)
                action_idx = int(torch.argmax(q_values, dim=1).item())
        return action_idx, self.action_table[action_idx]

    def store(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.push(state, action_idx, reward, next_state, done)

    def update_target_network(self):
        """Copies the live network's weights into the slow-moving target
        network. Called periodically (every C steps), not every step -
        that's what keeps the learning target stable."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def learn(self, batch_size=64, expert_data=None, imitation_weight=0.0):
        """Run a DQN update, optionally retaining an IK demonstration policy.

        expert_data is a pair of NumPy arrays (states, action_indices). The
        auxiliary cross-entropy term is used for behavior-cloning warm starts
        and fine-tuning; set imitation_weight=0 for ordinary DQN learning.
        """
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Q-value the network currently assigns to the action actually taken
        q_values = self.q_network(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: reward + gamma * best Q-value of the next state
        # (using the target network, not the live one, for stability)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = F.mse_loss(current_q, target_q)

        if expert_data is not None and imitation_weight > 0:
            expert_states, expert_actions = expert_data
            sample_indices = np.random.randint(
                0, len(expert_states), size=min(batch_size, len(expert_states))
            )
            expert_states_t = torch.as_tensor(
                expert_states[sample_indices], dtype=torch.float32, device=self.device
            )
            expert_actions_t = torch.as_tensor(
                expert_actions[sample_indices], dtype=torch.long, device=self.device
            )
            imitation_loss = F.cross_entropy(
                self.q_network(expert_states_t), expert_actions_t
            )
            loss = loss + float(imitation_weight) * imitation_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, path, metadata=None):
        """Save all learner state needed to continue training after a restart.

        ``save`` above remains the weights-only format used for inference and
        model uploads. This full checkpoint is a local training artifact.
        """
        from pathlib import Path

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            torch.save(
                {
                    "format_version": 1,
                    "q_network": self.q_network.state_dict(),
                    "target_network": self.target_network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "gamma": self.gamma,
                    "replay_capacity": self.replay_buffer.buffer.maxlen,
                    "replay_buffer": list(self.replay_buffer.buffer),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "metadata": metadata or {},
                },
                temporary_path,
            )
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_checkpoint(self, path):
        """Restore learner, optimizer, replay memory, and RNG states.

        Only load checkpoints generated by this local training pipeline; unlike
        ``load`` this method is not intended for arbitrary uploaded model files.
        Returns the checkpoint metadata dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported DQN training checkpoint format.")
        if len(checkpoint["q_network"]["net.0.weight"]) != self.q_network.net[0].out_features:
            raise ValueError("Checkpoint architecture does not match this DQN agent.")

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gamma = float(checkpoint["gamma"])
        self.replay_buffer = ReplayBuffer(capacity=int(checkpoint["replay_capacity"]))
        self.replay_buffer.buffer.extend(checkpoint["replay_buffer"])

        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_rng_state = checkpoint.get("cuda_rng_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
        return checkpoint.get("metadata", {})
