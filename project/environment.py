"""Evaluation snapshots before Isaac Gym's automatic reset."""
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot

class CourseG1(LeggedRobot):
    def step(self, actions):
        # Hold the command that generated this action, before resampling at t+1.
        self.metric_commands = self.commands.clone()
        return super().step(actions)

    def check_termination(self):
        super().check_termination()
        if getattr(self.cfg.env, 'capture_metrics', False):
            height = torch.max(self.root_states[:, 2:3] - self.feet_pos[:, :, 2], dim=1).values
            height += self.cfg.asset.ankle_sole_distance
            self.metric_snapshot = {
                'linear_velocity_error_m_s': torch.linalg.norm(self.base_lin_vel[:, :2] - self.metric_commands[:, :2], dim=1).clone(),
                'yaw_velocity_error_rad_s': torch.abs(self.base_ang_vel[:, 2] - self.metric_commands[:, 2]).clone(),
                'height_error_m': torch.abs(height - self.metric_commands[:, 4]).clone(),
            }
            # Count physical termination even when it coincides with a time limit.
            contact = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
            self.metric_failed = (contact | self.gravity_termination_buf).clone()
