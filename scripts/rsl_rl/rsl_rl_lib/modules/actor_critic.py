# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from .net_model import *

from rsl_rl_lib.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 output_activation=None,
                 model_name="MlpAdaptModel",
                 NetModel=None,
                 init_noise_std=1.0,
                 max_std = 1.0,
                 min_std = 0.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        self.is_recurrent = False
        self.max_std = max_std
        self.min_std = min_std
        self.model_name = model_name
        print("teacher model name: ", model_name)
        teacher_net_class = eval(self.model_name)
        print("model_name", self.model_name)

        self.actor = teacher_net_class(obs_dim=num_actor_obs,
                                       act_dim=num_actions,
                                       activation=activation,
                                       output_activation=output_activation,
                                       **(NetModel[self.model_name]))

        # activation = resolve_nn_activation(activation)

        # mlp_input_dim_a = num_actor_obs
        # mlp_input_dim_c = num_critic_obs
        # # Policy
        # actor_layers = []
        # actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # actor_layers.append(activation)
        # for layer_index in range(len(actor_hidden_dims)):
        #     if layer_index == len(actor_hidden_dims) - 1:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
        #     else:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
        #         actor_layers.append(activation)
        # self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, **kwargs):
        mean = self.actor(observations, **kwargs)
        # std = torch.clamp(self.std, min=self.min_std, max=self.max_std)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations, **kwargs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None