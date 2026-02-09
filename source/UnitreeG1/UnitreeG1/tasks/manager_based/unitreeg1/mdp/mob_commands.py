from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
import re
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationCfg
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from .curriculums import RewardThresholdCurriculum
if TYPE_CHECKING:
    from .mob_commands_cfg import MoBCommandCfg

from isaaclab.envs.mdp import *  # noqa: F401, F403
from .curriculums import RewardThresholdCurriculum

def torch_rand_float(lower, upper, shape, device):

    return (upper - lower) * torch.rand(shape, device=device) + lower

class MoBCommand(UniformVelocityCommand):
    cfg: MoBCommandCfg

    def __init__(self, cfg: MoBCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        self.commands = torch.zeros(self.num_envs, 10, device= self.device)

        self.category_names = ['walking', 'jumping']        
        self.curricula = []
        for category in self.category_names:
            self.curricula += [RewardThresholdCurriculum(seed=self.cfg.seed,
                                                        x_vel=(self.cfg.limit_vel_x[0],
                                                               self.cfg.limit_vel_x[1],
                                                               self.cfg.num_bins_vel_x),
                                                        yaw_vel=(self.cfg.limit_vel_yaw[0],
                                                               self.cfg.limit_vel_yaw[1],
                                                               self.cfg.num_bins_vel_yaw)
                                                        )]

        self.env_command_bins = torch.zeros(self.num_envs, device= self.device)
        self.env_command_categories = torch.zeros(self.num_envs, device= self.device)
        low = np.array([self.cfg.limit_vel_x[0], self.cfg.limit_vel_yaw[0], ], )
        high = np.array([self.cfg.limit_vel_x[1], self.cfg.limit_vel_yaw[1], ], )
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_indices = torch.tensor([0, 1], dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False, )
        self.velocity_level = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.curriculum_thresholds = {
            "tracking_lin_vel": 0.8,
            "tracking_ang_vel": 0.4,
            "tracking_contacts_shaped_force": 0.8,
            "tracking_contacts_shaped_vel": 0.8,
        }
        self.reward_scales = {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 3.0,
            "tracking_contacts_shaped_force": 2.0,
            "tracking_contacts_shaped_vel": 4.0,
        }

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "MoBCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.all_commands.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def get_foot_indices(self) -> torch.Tensor:
        """Get the desired contact states."""
        if not hasattr(self, "desired_contact_states"):
            raise AttributeError("desired_contact_states is not set. Please ensure it is initialized.")
        return self.foot_indices

    def get_desired_contact_states(self) -> torch.Tensor:
        """Get the desired contact states."""
        if not hasattr(self, "desired_contact_states"):
            raise AttributeError("desired_contact_states is not set. Please ensure it is initialized.")
        # print("get")   
        return self.desired_contact_states
    
    def get_clock_inputs(self) -> torch.Tensor:
        """Get the clock inputs."""
        if not hasattr(self, "clock_inputs"):
            raise AttributeError("clock_inputs is not set. Please ensure it is initialized.")
        return self.clock_inputs
    def get_foot_indices(self) -> torch.Tensor:
        """Get the foot indices."""
        if not hasattr(self, "foot_indices"):
            raise AttributeError("foot_indices is not set. Please ensure it is initialized.")
        return self.foot_indices
    
    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.commands

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0: 
            return

        self.commands[env_ids, 0] = torch_rand_float(self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.lin_vel_y[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1], (len(env_ids), 1), device=self.device).squeeze(1)

        #========================

        timesteps = int(self.cfg.resampling_time_range[0] / self.env.step_dt)
        ep_len = min(self.env.max_episode_length, timesteps)
        
        self.command_sums = self.env.reward_manager.command_sums
        # print("commands:", self.all_commands)
        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]
            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55]))      
        
        
        # # assign resampled environments to new categories
        # random_env_floats = torch.rand(len(env_ids), device=self.device)
        # probability_per_category = 1. / len(self.category_names)
        # category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
        #                                               random_env_floats < probability_per_category * (i + 1))] for i in
        #                     range(len(self.category_names))]

        jumping_mask = self.commands[env_ids, 4] == 0
        walking_mask = self.commands[env_ids, 4] == 0.5
        category_env_ids = [env_ids[walking_mask], env_ids[jumping_mask]]
        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category] = torch.as_tensor(
                new_bin_inds, device=self.env_command_bins.device, dtype=self.env_command_bins.dtype
            )
            self.env_command_categories[env_ids_in_category] = torch.as_tensor(
                i, device=self.env_command_categories.device, dtype=self.env_command_categories.dtype
            )
            
            # self.commands[env_ids_in_category, :] = torch.as_tensor(
            #     new_commands[:, :],
            #     device=self.device,
            #     dtype=self.commands.dtype
            # )                       
                        
            self.commands[env_ids_in_category, 0] = torch.Tensor(new_commands[:, 0]).to(self.device)
            self.commands[env_ids_in_category, 1] = torch.rand(len(env_ids_in_category), device=self.device) - 0.5
            self.commands[env_ids_in_category, 2] = torch.Tensor(new_commands[:, 1]).to(self.device)
        #========================
        # print("env id", env_ids)
        # print("reset command", self.env.reward_manager.command_sums)
                # random_env_floats = torch.rand(len(env_ids), device=self.device)
        # probability_per_category = 1. / len(self.category_names)
        # category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
        #                                               random_env_floats < probability_per_category * (i + 1))] for i in
        #                     range(len(self.category_names))]
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.commands[standing_env_ids, :3] = 0.0

        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.15).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > 0.15)

        self.velocity_level[env_ids] = torch.clip(1.0*torch.norm(self.commands[env_ids, :2], dim=-1)+0.5*torch.abs(self.commands[env_ids, 2]), min=1)

        # update gait commands
        self.commands[env_ids, 3] = torch_rand_float(self.cfg.ranges.gait_frequency[0], self.cfg.ranges.gait_frequency[1], (len(env_ids), 1), device=self.device).squeeze(1)  # Frequency
        phases = torch.tensor([0, 0.5], device=self.device)
        random_indices = torch.randint(0, len(phases), (len(env_ids), ), device=self.device)
        self.commands[env_ids, 4] = phases[random_indices] # phases
        self.commands[env_ids, 5] = 0.5  # durations
        self.commands[env_ids, 6] = torch_rand_float(self.cfg.ranges.foot_swing_height[0], self.cfg.ranges.foot_swing_height[1], (len(env_ids), 1), device=self.device).squeeze(1)  # swing_heights

        # clip commands for high speed envs
        high_speed_env_mask = self.velocity_level[env_ids] > 1.8
        self.commands[env_ids[high_speed_env_mask], 3] = self.commands[env_ids[high_speed_env_mask], 3].clip(min=2.0)  # Frequency

        # clip swing height for high frequency
        high_frequency_env_mask = self.commands[env_ids, 3] > 2.5
        self.commands[env_ids[high_frequency_env_mask], 6] = self.commands[env_ids[high_frequency_env_mask], 6].clip(max=0.20)

        jumping_mask = self.commands[env_ids, 4] == 0
        walking_mask = self.commands[env_ids, 4] == 0.5
        jumping_env_ids = env_ids[jumping_mask]
        walking_env_ids = env_ids[walking_mask]

        # Body height command
        self.commands[env_ids, 7] = torch_rand_float(self.cfg.ranges.body_height[0], self.cfg.ranges.body_height[1], (len(env_ids), 1), device=self.device).squeeze(1)

        # clip swing height for low body height
        low_height_env_mask = self.commands[env_ids, 7] < -0.15
        self.commands[env_ids[low_height_env_mask], 6] = self.commands[env_ids[low_height_env_mask], 6].clip(max=0.20)
    
        # Body pitch command
        self.commands[env_ids, 8] = torch_rand_float(self.cfg.ranges.body_pitch[0], self.cfg.ranges.body_pitch[1], (len(env_ids), 1), device=self.device).squeeze(1)

        # clip body_pitch for low body height
        low_height_env_mask = self.commands[env_ids, 7] < -0.2
        self.commands[env_ids[low_height_env_mask], 8] = self.commands[env_ids[low_height_env_mask], 8].clip(max=0.3)
        self.commands[env_ids[high_speed_env_mask], 8] = self.commands[env_ids[high_speed_env_mask], 8].clip(max=0.3)      
        
        # Waist roll command
        self.commands[env_ids, 9] = torch_rand_float(self.cfg.ranges.waist_roll[0], self.cfg.ranges.waist_roll[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids[high_speed_env_mask], 9] = self.commands[env_ids[high_speed_env_mask], 9].clip(min=-0.15, max=0.15)

        # clip commands for jumping envs
        self.commands[jumping_env_ids, 6] = self.commands[jumping_env_ids, 6].clip(max=0.2)
        self.commands[jumping_env_ids, 8] = self.commands[jumping_env_ids, 8].clip(max=0.3)
        
        # print("command: ", self.commands[0:3, :])
        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.all_commands[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        # standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        # self.all_commands[standing_env_ids, :] = 0.0
        self._step_contact_targets()

    def _step_contact_targets(self):
        frequencies = self.commands[:, 3]
        phases = self.commands[:, 4]
        durations = self.commands[:, 5]

        self.gait_indices = torch.remainder(self.gait_indices + frequencies * self.env.step_dt, 1.0)
        # print(self.env.step_dt)
        foot_indices = [self.gait_indices + phases,
                        self.gait_indices]

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        for idxs in foot_indices:
            idxs[standing_env_ids] = 0.0

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(2)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (0.5 / (1 - durations[swing_idxs]))
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])

        # von mises distribution
        kappa = 0.05
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        
        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        # print("in step" ,self.desired_contact_states)
        # print(smoothing_multiplier_FL[0], smoothing_multiplier_FR[0], smoothing_multiplier_RL[0], smoothing_multiplier_RR[0])
        self.desired_footswing_height = self.commands[:, 6]
    
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
