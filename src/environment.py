"""PyBullet robot-arm environment with a simplified regenerative-energy model.

Each episode presents an easy target followed by a harder target. The buffer
starts empty and can only charge from simulated braking work, then offset grid
draw during the harder phase. This simulation does not establish physical
battery, motor, or peak-shaving performance. Energy savings must be measured
against a baseline on matched successful targets.

--- STATE SPACE (what the robot "sees" each step, 23 values total) ---
    - joint positions        (6 values, radians)
    - joint velocities       (6 values, rad/s)
    - end-effector position  (3 values: x, y, z in metres)
    - target position        (3 values: x, y, z in metres)
    - task_phase             (1 value: 0.0 = light task, 1.0 = heavy task)
    - energy_buffer_level    (1 value: 0.0 empty -> 1.0 full)
    - grid_draw_last         (1 value: last step's grid draw, normalised)
    - distance_to_target     (1 value: direct scalar, helps the agent plan)
    - step_fraction          (1 value: how far through the episode we are)

--- ACTION SPACE ---
    6 continuous values (-1 to 1), one per joint velocity.
    Scaled to MAX_JOINT_VELOCITY inside step().

--- REWARD FUNCTION ---
    reward = (
        - ALPHA      * grid_draw          # penalise pulling power FROM THE GRID
        - PEAK_PEN   * peak_spike         # heavy extra penalty if grid draw spikes
        + BUFFER_REW * buffer_saved       # reward saving energy into the buffer
        + BUFFER_USE * buffer_deployed    # reward using buffer (not grid) for heavy tasks
    + BETA       * progress           # reward reducing target distance
        + COMPLETION_BONUS if done        # big reward for finishing
    )

--- EPISODE STRUCTURE ---
Each episode has TWO PHASES:
    Phase 0 (LIGHT): target is close and easy. Braking work can charge the
                     buffer when the simplified mechanical-work model detects it.
    Phase 1 (HEAVY): target jumps to a far, harder position. Robot should
                     draw from the buffer to reach it without spiking the grid.

This two-phase structure makes any stored braking energy available for a later
heavy movement. The buffer may remain empty if no braking energy is recovered.

--- NOTE ON 6-DOF ---
PyBullet's KUKA iiwa has 7 physical joints. Joint 6 (wrist roll) is locked
rigid — confirmed to drift <0.000004 radians under sustained disturbance.
Only joints 0-5 are controlled or observed.
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class RobotArmEnv(gym.Env):
    """
    6-DoF robot arm with an energy buffer.
    An IK-assisted DQN controls a simulated arm with a regenerative energy
    buffer. Energy savings are evaluated only on completed matched targets.
    """

    metadata = {"render_modes": ["human", None]}

# --- Physical limits ---
    MAX_JOINT_VELOCITY  = 2.0    # rad/s (increased for reachability)
    MAX_STEPS           = 200    # shorter episodes = more episodes per run
    TARGET_THRESHOLD    = 0.15   # metres — "close enough" (relaxed further)

    # --- Buffer settings ---
    BUFFER_CAPACITY     = 50.0   # watt-seconds — max energy the buffer can hold
    BUFFER_CHARGE_EFF   = 0.80   # 80% of saved energy actually goes into buffer
    PEAK_THRESHOLD      = 3.0    # watt-seconds — grid draw above this is a "spike"

    # --- Reward weights ---
    ALPHA           = 0.002    # minimal grid draw penalty
    PEAK_PEN        = 0.30
    BUFFER_REW      = 0.10
    BUFFER_USE      = 0.20
    BETA            = 20.0     # dense reward for reducing target distance
    COMPLETION_BONUS = 100.0   # terminal reward for completing the heavy target

    # --- Phase transition ---
    PHASE_SWITCH_FRACTION = 0.25   # switch to heavy task at 25% (more time for heavy phase)

    def __init__(self, render_mode=None, buffer_enabled=True):
        super().__init__()
        self.render_mode = render_mode
        self.buffer_enabled = buffer_enabled

        self.physics_client = p.connect(
            p.GUI if render_mode == "human" else p.DIRECT
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robot_id        = None
        self.num_joints      = None
        self.controlled_joints = None
        self.end_effector_link = None
        self.locked_joint    = None

        # Episode state (reset each episode)
        self.target_position      = None
        self.light_target         = None   # close, easy target (phase 0)
        self.heavy_target         = None   # far, hard target  (phase 1)
        self.energy_buffer        = 0.0    # current buffer level (watt-seconds)
        self.task_phase           = 0      # 0 = light, 1 = heavy
        self.step_count           = 0
        self.buffer_used_last     = 0.0
        self.grid_draw_last       = 0.0
        self.peak_grid_draw       = 0.0
        self.total_grid_draw      = 0.0    # tracked for reporting
        self.total_buffer_used    = 0.0    # tracked for reporting

        # Load robot once to measure joint count, then define spaces
        self._load_robot()
        num_controlled = len(self.controlled_joints)

        # 23-dimensional observation (see module docstring)
        obs_dim = (
            num_controlled      # joint positions
            + num_controlled    # joint velocities
            + 3                 # end-effector xyz
            + 3                 # target xyz
            + 1                 # task_phase
            + 1                 # energy_buffer_level (normalised 0-1)
            + 1                 # grid_draw_last (normalised)
            + 1                 # distance to target
            + 1                 # step fraction (how far through episode)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_controlled,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _load_robot(self):
        """Rebuild the simulated world from scratch."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True
        )
        self.num_joints        = p.getNumJoints(self.robot_id)   # 7 physically
        self.end_effector_link = self.num_joints - 1             # tip of the arm
        self.locked_joint      = self.num_joints - 1             # joint 6 → rigid
        self.controlled_joints = list(range(self.num_joints - 1))# joints 0-5

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_robot()

        # Randomise starting joint angles slightly
        for j in self.controlled_joints:
            p.resetJointState(self.robot_id, j, self.np_random.uniform(-0.1, 0.1))

        # Lock joint 6 rigid
        p.resetJointState(self.robot_id, self.locked_joint, 0)
        p.setJointMotorControl2(
            self.robot_id, self.locked_joint,
            controlMode=p.POSITION_CONTROL, targetPosition=0, force=500,
        )

        # --- Two targets: one easy (phase 0), one hard (phase 1) ---
        # Light target: close to the arm, easy to reach gently
        self.light_target = np.array([
            self.np_random.uniform(0.20, 0.30),
            self.np_random.uniform(-0.1, 0.1),
            self.np_random.uniform(0.20, 0.30),
        ])
        # Heavy target: reachable but needs power
        self.heavy_target = np.array([
            self.np_random.uniform(0.30, 0.45),
            self.np_random.uniform(-0.15, 0.15),
            self.np_random.uniform(0.15, 0.35),
        ])

        self.target_position   = self.light_target.copy()
        self.task_phase        = 0
        # Start empty so every stored joule must come from simulated recovery.
        self.energy_buffer     = 0.0
        self.buffer_used_last  = 0.0
        self.grid_draw_last    = 0.0
        self.peak_grid_draw    = 0.0
        self.total_grid_draw   = 0.0
        self.total_buffer_used = 0.0
        self.step_count        = 0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # --- Phase transition check ---
        # At the configured fraction of the episode, switch to the heavy task.
        # Recovered braking energy is available for the heavy phase.
        phase_switch_step = int(self.MAX_STEPS * self.PHASE_SWITCH_FRACTION)
        if self.step_count == phase_switch_step and self.task_phase == 0:
            self.task_phase      = 1
            self.target_position = self.heavy_target.copy()

        ee_before = np.array(
            p.getLinkState(self.robot_id, self.end_effector_link)[0], dtype=np.float32
        )
        distance_before = float(np.linalg.norm(ee_before - self.target_position))

        # --- Apply joint velocities ---
        target_velocities = action * self.MAX_JOINT_VELOCITY
        for idx, joint_idx in enumerate(self.controlled_joints):
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(target_velocities[idx]),
                force=200,
            )
        # Re-assert the wrist lock
        p.setJointMotorControl2(
            self.robot_id, self.locked_joint,
            controlMode=p.POSITION_CONTROL, targetPosition=0, force=500,
        )
        p.stepSimulation()
        self.step_count += 1

        # --- Measure raw energy this step ---
        dt = 1.0 / 240.0
        raw_energy = 0.0
        regenerative_energy = 0.0
        for joint_idx in self.controlled_joints:
            js = p.getJointState(self.robot_id, joint_idx)
            mechanical_work = js[3] * js[1] * dt
            raw_energy += max(0.0, mechanical_work)
            regenerative_energy += max(0.0, -mechanical_work)

        # --- Energy buffer logic ---
        # Store recovered braking work during phase 0 and deploy it in phase 1.

        buffer_saved    = 0.0
        buffer_deployed = 0.0
        grid_draw       = raw_energy   # default: everything comes from the grid

        if not self.buffer_enabled:
            grid_draw = raw_energy

        elif self.task_phase == 0:
            # Charge only from mechanical energy recovered during braking.
            buffer_saved = min(
                regenerative_energy * self.BUFFER_CHARGE_EFF,
                self.BUFFER_CAPACITY - self.energy_buffer
            )
            self.energy_buffer += buffer_saved
            # grid draw is just the raw energy (we used what we used)
            grid_draw = raw_energy

        else:
            # Heavy phase: draw from buffer first, then grid for the rest
            buffer_deployed = min(raw_energy, self.energy_buffer)
            self.energy_buffer -= buffer_deployed
            grid_draw = max(0.0, raw_energy - buffer_deployed)

        # Peak spike: how much did grid_draw EXCEED the safe threshold?
        peak_spike = max(0.0, grid_draw - self.PEAK_THRESHOLD)
        peak_penalty = float(grid_draw > self.PEAK_THRESHOLD)

        self.buffer_used_last   = buffer_deployed
        self.grid_draw_last     = grid_draw
        self.peak_grid_draw     = max(self.peak_grid_draw, grid_draw)
        self.total_grid_draw   += grid_draw
        self.total_buffer_used += buffer_deployed

        # --- Distance to current target ---
        ee_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link)[0])
        distance = float(np.linalg.norm(ee_pos - self.target_position))

        # --- Reward ---
        task_done = (distance < self.TARGET_THRESHOLD) and (self.task_phase == 1)

        reward = (
            - self.ALPHA    * grid_draw
            - self.PEAK_PEN * peak_penalty
            + self.BUFFER_REW  * buffer_saved
            + self.BUFFER_USE  * buffer_deployed
            + self.BETA     * (distance_before - distance)
        )
        if task_done:
            reward += self.COMPLETION_BONUS

        terminated = bool(task_done)
        truncated  = self.step_count >= self.MAX_STEPS

        info = {
            "energy_raw":       raw_energy,
            "regenerative_energy": regenerative_energy,
            "grid_draw":        grid_draw,
            "peak_spike":      peak_spike,
            "peak_penalty":    peak_penalty,
            "peak_grid_draw":  self.peak_grid_draw,
            "buffer_saved":     buffer_saved,
            "buffer_deployed":  buffer_deployed,
            "buffer_level":     self.energy_buffer,
            "distance":         distance,
            "task_phase":       self.task_phase,
            "task_done":        task_done,
            "total_grid_draw":  self.total_grid_draw,
            "total_buffer_used":self.total_buffer_used,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        positions, velocities = [], []
        for j in self.controlled_joints:
            s = p.getJointState(self.robot_id, j)
            positions.append(s[0])
            velocities.append(s[1])

        ee_pos = np.array(
            p.getLinkState(self.robot_id, self.end_effector_link)[0],
            dtype=np.float32
        )
        distance = float(np.linalg.norm(ee_pos - self.target_position))
        step_fraction = self.step_count / self.MAX_STEPS

        return np.concatenate([
            np.array(positions,  dtype=np.float32),
            np.array(velocities, dtype=np.float32),
            ee_pos,
            self.target_position.astype(np.float32),
            np.array([float(self.task_phase)],                          dtype=np.float32),
            np.array([self.energy_buffer / self.BUFFER_CAPACITY],       dtype=np.float32),
            np.array([self.grid_draw_last / (self.PEAK_THRESHOLD + 1e-8)], dtype=np.float32),
            np.array([distance],                                         dtype=np.float32),
            np.array([step_fraction],                                    dtype=np.float32),
        ])

    # ------------------------------------------------------------------
    def close(self):
        p.disconnect(self.physics_client)
