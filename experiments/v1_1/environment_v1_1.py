"""Version 1.1 energy-accounting wrapper around the preserved V1.0 task.

The original environment remains unchanged. This subclass retains its robot,
23-value observation, 89-action-compatible task and 0.15 m success tolerance,
while correcting the documented fixed-step energy and peak-penalty behavior.
The model is still simplified mechanical-work accounting, not measured mains
electricity consumption.
"""

from __future__ import annotations

import numpy as np

from src.environment import RobotArmEnv as _V10RobotArmEnv


def account_step_energy(positive_work_j: float, regenerative_work_j: float,
                       buffer_j: float, capacity_j: float, charge_efficiency: float,
                       discharge_efficiency: float, enabled: bool, phase: int) -> dict:
    """Pure per-step energy ledger, with explicit storage and conversion losses."""
    if not 0.0 < charge_efficiency <= 1.0 or not 0.0 < discharge_efficiency <= 1.0:
        raise ValueError("Buffer efficiencies must be in (0, 1].")
    if min(positive_work_j, regenerative_work_j, buffer_j, capacity_j) < 0.0:
        raise ValueError("Energy values must be nonnegative.")
    stored_j = delivered_j = removed_j = spilled_j = 0.0
    grid_j = positive_work_j
    next_buffer = buffer_j
    if enabled and phase == 0:
        potential = regenerative_work_j * charge_efficiency
        stored_j = min(potential, max(0.0, capacity_j - buffer_j))
        spilled_j = max(0.0, potential - stored_j)
        next_buffer += stored_j
    elif enabled and phase == 1:
        delivered_j = min(positive_work_j, buffer_j * discharge_efficiency)
        removed_j = delivered_j / discharge_efficiency
        next_buffer -= removed_j
        grid_j = positive_work_j - delivered_j
    next_buffer = float(np.clip(next_buffer, 0.0, capacity_j))
    if grid_j < -1e-10 or delivered_j > positive_work_j + 1e-10:
        raise ArithmeticError("Energy ledger produced energy or over-discharged the load.")
    return {"grid_j": max(0.0, float(grid_j)), "stored_j": float(stored_j),
            "delivered_j": float(delivered_j), "removed_j": float(removed_j),
            "spilled_j": float(spilled_j), "buffer_j": next_buffer}


class RobotArmEnvV11(_V10RobotArmEnv):
    """Same target-reaching task with explicit, dimensioned energy accounting."""

    BUFFER_DISCHARGE_EFF = 0.90  # assumed parameter; sensitivity analysis required
    PEAK_THRESHOLD_W = 720.0     # 3 J / (1/240 s), preserves legacy threshold scale
    PEAK_REWARD_WEIGHT = 0.30
    BUFFER_REW = 0.0              # no reward for charging/cycling energy by itself
    BUFFER_USE = 0.0              # avoid rewarding needless buffer discharge

    def step(self, action):
        """Run V1.0 dynamics and replace its energy/reward accounting per step.

        This implementation intentionally reproduces the existing joint action
        and state transitions. It updates the energy ledger after one physics
        tick and recalculates the reward from the same task progress.
        """
        # Capture state needed to reconstruct the progress reward and preserve
        # the original simulation/action mechanics without modifying V1.0.
        import pybullet as p

        action = np.clip(action, -1.0, 1.0)
        phase_switch_step = int(self.MAX_STEPS * self.PHASE_SWITCH_FRACTION)
        if self.step_count == phase_switch_step and self.task_phase == 0:
            self.task_phase = 1
            self.target_position = self.heavy_target.copy()
        before = np.asarray(p.getLinkState(self.robot_id, self.end_effector_link)[0], dtype=np.float32)
        distance_before = float(np.linalg.norm(before - self.target_position))
        dt = float(p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0 / 240.0))
        for i, joint_idx in enumerate(self.controlled_joints):
            p.setJointMotorControl2(self.robot_id, joint_idx, p.VELOCITY_CONTROL,
                                    targetVelocity=float(action[i] * self.MAX_JOINT_VELOCITY), force=200)
        p.setJointMotorControl2(self.robot_id, self.locked_joint, p.POSITION_CONTROL,
                                targetPosition=0, force=500)
        p.stepSimulation()
        self.step_count += 1

        positive_j = 0.0
        negative_j = 0.0
        for joint_idx in self.controlled_joints:
            state = p.getJointState(self.robot_id, joint_idx)
            work_j = float(state[3]) * float(state[1]) * dt
            positive_j += max(0.0, work_j)
            negative_j += max(0.0, -work_j)

        ledger = account_step_energy(positive_j, negative_j, self.energy_buffer,
                                     self.BUFFER_CAPACITY, self.BUFFER_CHARGE_EFF,
                                     self.BUFFER_DISCHARGE_EFF, self.buffer_enabled,
                                     self.task_phase)
        grid_j = ledger["grid_j"]
        stored_j = ledger["stored_j"]
        delivered_j = ledger["delivered_j"]
        spilled_recovery_j = ledger["spilled_j"]
        self.energy_buffer = ledger["buffer_j"]
        grid_power_w = grid_j / dt if dt > 0 else 0.0
        peak_excess_w = max(0.0, grid_power_w - self.PEAK_THRESHOLD_W)
        normalized_peak_excess = peak_excess_w / self.PEAK_THRESHOLD_W
        self.grid_draw_last = grid_j
        self.buffer_used_last = delivered_j
        self.peak_grid_draw = max(self.peak_grid_draw, grid_j)
        self.total_grid_draw += grid_j
        self.total_buffer_used += delivered_j
        self.peak_grid_power_w = max(getattr(self, "peak_grid_power_w", 0.0), grid_power_w)

        after = np.asarray(p.getLinkState(self.robot_id, self.end_effector_link)[0], dtype=np.float32)
        distance = float(np.linalg.norm(after - self.target_position))
        task_done = bool(distance < self.TARGET_THRESHOLD and self.task_phase == 1)
        reward_energy = -self.ALPHA * grid_j
        reward_peak = -self.PEAK_REWARD_WEIGHT * normalized_peak_excess
        reward_progress = self.BETA * (distance_before - distance)
        reward_completion = self.COMPLETION_BONUS if task_done else 0.0
        reward = reward_energy + reward_peak + reward_progress + reward_completion
        terminated = task_done
        truncated = bool(self.step_count >= self.MAX_STEPS)
        info = {
            "energy_raw": positive_j, "regenerative_energy": negative_j,
            "grid_draw": grid_j, "grid_power_w": grid_power_w,
            "peak_excess_w": peak_excess_w, "peak_grid_power_w": self.peak_grid_power_w,
            "buffer_saved": stored_j, "buffer_deployed": delivered_j,
            "buffer_removed": ledger["removed_j"],
            "buffer_level": self.energy_buffer, "buffer_spilled_recovery": spilled_recovery_j,
            "distance": distance, "task_phase": self.task_phase,
            "task_done": task_done, "total_grid_draw": self.total_grid_draw,
            "total_buffer_used": self.total_buffer_used, "timestep_s": dt,
            "energy_model": "joint_mechanical_work_v1_1_assumed_efficiencies",
            "reward_energy": reward_energy, "reward_peak": reward_peak,
            "reward_progress": reward_progress, "reward_completion": reward_completion,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.peak_grid_power_w = 0.0
        return super().reset(*args, **kwargs)
