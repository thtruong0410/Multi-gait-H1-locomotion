from __future__ import annotations
import gymnasium as gym

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence
import re
from typing import Any, ClassVar
import math
import inspect
import omni.log
from isaacsim.core.version import get_version
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ActionManager, ObservationManager, EventManager, RecorderManager
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from prettytable import PrettyTable
from isaaclab.utils import configclass
from isaaclab.managers import ManagerBase, EventTermCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

class CusObservationManager(ObservationManager):

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        if cfg is None:
            raise ValueError("Observation manager configuration is None. Please provide a valid configuration.")

        super().__init__(cfg, env) 
        self.env = env
        print("[INFO] Custom Observation Manager: created.")
        # print(self.env)
    def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute the observations per group for all groups.

        The method computes the observations for all the groups handled by the observation manager.
        Please check the :meth:`compute_group` on the processing of observations per group.

        Returns:
            A dictionary with keys as the group names and values as the computed observations.
            The observations are either concatenated into a single tensor or returned as a dictionary
            with keys corresponding to the term's name.
        """
        # create a buffer for storing obs from all the groups
        obs_buffer = dict()
        # iterate over all the terms in each group
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name)
        # otherwise return a dict with observations of all groups
        # Cache the observations.
        self._obs_buffer = obs_buffer
        return obs_buffer
    
    def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the observations for a given group.

        The observations for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        The following steps are performed for each observation term:

        1. Compute observation term by calling the function
        2. Apply custom modifiers in the order specified in :attr:`ObservationTermCfg.modifiers`
        3. Apply corruption/noise model based on :attr:`ObservationTermCfg.noise`
        4. Apply clipping based on :attr:`ObservationTermCfg.clip`
        5. Apply scaling based on :attr:`ObservationTermCfg.scale`

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world. If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.

        Args:
            group_name: The name of the group for which to compute the observations. Defaults to None,
                in which case observations for all the groups are computed and returned.

        Returns:
            Depending on the group's configuration, the tensors for individual observation terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """

        group_cfg = self.env.observation_manager.cfg.__dict__.items() #.history_length
        num_history = None
        for group_n, group_cfg in group_cfg:
            if group_n != group_name:
                continue
            num_history = group_cfg.history_length
        if num_history is None:
            num_history = 1
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])
        device = self._env.device
        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for term_name, term_cfg in obs_terms:
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            obs = obs.to(device)
            # apply post-processing
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if term_cfg.noise:
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            # Update the history buffer if observation term has history enabled
            if term_cfg.history_length > 0:
                self._group_obs_term_history_buffer[group_name][term_name].append(obs)
                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = self._group_obs_term_history_buffer[group_name][term_name].buffer.reshape(
                        self._env.num_envs, -1
                    )
                else:
                    group_obs[term_name] = self._group_obs_term_history_buffer[group_name][term_name].buffer
            else:
                group_obs[term_name] = obs

        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            # Collect as (num_envs, history_length, feature_dim) for each term
            term_histories = []
            for term_name, obs in group_obs.items():
                obs = obs.view(self.env.num_envs, num_history, -1)
                if obs.ndim == 2:  # no history, expand to match shape
                    obs = obs.unsqueeze(1)  # (num_envs, 1, feature_dim)
                term_histories.append(obs)
            # Stack terms along feature dimension for each timestep
            # Result shape: (num_envs, history_length, sum_of_feature_dims)
            stacked = torch.cat(term_histories, dim=2)
            # Finally flatten history dimension into feature dimension
            return stacked.reshape(self._env.num_envs, -1)
        else:
            return group_obs