import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardManager
from isaaclab.envs import ManagerBasedRLEnv

class CusRewardManager(RewardManager):
    _env: ManagerBasedRLEnv
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command_sums = {
            name: torch.zeros(env.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self._term_names) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}
        self._reward_buf_pos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._reward_buf_neg = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.env = env
        print("[INFO] Custom Reward Manager: created.")

    def compute(self, dt: float) -> torch.Tensor:
        # reset computation
        self._reward_buf[:] = 0.0
        self._reward_buf_pos[:] = 0.0
        self._reward_buf_neg[:] = 0.0
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            # print(name)
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            if torch.sum(value) >= 0:
                self._reward_buf_pos += value
            elif torch.sum(value) <= 0:
                self._reward_buf_neg += value

            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += term_cfg.weight + value
            else:
                self.command_sums[name] += value
            # self._reward_buf[:] = self._reward_buf_pos[:] * torch.exp(self._reward_buf_neg[:] / 0.02)
            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt

            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
            asset: RigidObject = self.env.scene[asset_cfg.name]

            self.command_sums["lin_vel_raw"] += asset.data.root_lin_vel_b[:, 0]
            self.command_sums["ang_vel_raw"] += asset.data.root_lin_vel_b[:, 2]
            self.command_sums["lin_vel_residual"] += (self.env.command_manager.get_command("base_velocity")[:, 0] - asset.data.root_lin_vel_b[:, 0]) ** 2
            self.command_sums["ang_vel_residual"] += (self.env.command_manager.get_command("base_velocity")[:, 2] - asset.data.root_lin_vel_b[:, 2]) ** 2
            self.command_sums["ep_timesteps"] += 1

        return self._reward_buf