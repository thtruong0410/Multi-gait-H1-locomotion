# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import Literal, Dict, Any

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

PROPRIOCEPTION_DIM = 63 
CMD_DIM = 3 + 4 + 1 + 2
TERRAIN_DIM = 221 
PRIVILEGED_DIM = 3 + 1 + 2 + 1 + 6 + 11  # 24
CLOCK_INPUT = 2

@configclass
class MlpAdaptModelCfg:
    proprioception_dim: int = MISSING
    cmd_dim: int = MISSING
    privileged_dim: int = MISSING
    terrain_dim: int = MISSING

    latent_dim: int = 32
    privileged_recon_dim: int = 3

    actor_hidden_dims: list[int] = MISSING
    mlp_hidden_dims: list[int] = MISSING

@configclass
class CustomRslRlPpoActorCriticCfg:
    class_name: str = "ActorCritic"

    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"

    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING

    activation: str = MISSING
    output_activation: str | None = None

    model_name: str = "MlpAdaptModel"
    NetModel: Dict[str, Any] = MISSING   

    critic_obs_dim: int = MISSING

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "g1_rough"  # same as task name
    empirical_normalization = False
    policy = CustomRslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 32],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        output_activation=None,

        model_name="MlpAdaptModel",

        NetModel={
            "MlpAdaptModel": {
                "proprioception_dim": PROPRIOCEPTION_DIM,
                "cmd_dim": CMD_DIM,
                "privileged_dim": PRIVILEGED_DIM,
                "terrain_dim": TERRAIN_DIM,
                "latent_dim": 32,
                "privileged_recon_dim": 3,
                "actor_hidden_dims": [256, 128, 32],
                "mlp_hidden_dims": [256, 128],
            }
        },

        critic_obs_dim=PROPRIOCEPTION_DIM + CMD_DIM + PRIVILEGED_DIM + TERRAIN_DIM,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name = "PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

