# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from . import custom_manager

##
# Register Gym environments.
##


gym.register(
    id="Template-Unitreeg1-v0",
    entry_point=f"{custom_manager.__name__}.cus_manager_based_RL_env:CusManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitreeg1_env_cfg:Unitreeg1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Unitreeg1-v0-Play",
    entry_point=f"{custom_manager.__name__}.cus_manager_based_RL_env:CusManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitreeg1_env_cfg:Unitreeg1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)