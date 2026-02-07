from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import re
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationCfg
from ..custom_manager import CusManagerBasedRLEnv
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedEnv

def disturbance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    force = asset._external_force_b[:, 0, :]
    return force

def friction(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties().cpu()
    friction = (materials[:, 0, 0].unsqueeze(1) - 0.5) * 2
    return friction

def restitution(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties().cpu()
    restitution = (materials[:, 0, 2].unsqueeze(1) - 0.5) * 2
    return restitution


def mass(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    payload = asset.root_physx_view.get_masses() - asset.data.default_mass # check
    return payload
def com(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_com_pos_w.cpu()

def clock_inputs(
    env: CusManagerBasedRLEnv, command_name: str | None = None
) -> torch.Tensor:
    # print(env.command_manager.get_clock_inputs.shape)
    return env.command_manager.get_clock_inputs

def last_last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_prev_actions

