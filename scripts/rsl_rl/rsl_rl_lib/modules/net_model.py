import torch
import torch.nn as nn

def get_activation(act_name: str):
    act_name = act_name.lower()
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Invalid activation: {act_name}")

def MLP(input_dim, output_dim, hidden_dims, activation, output_activation=None):
    act = get_activation(activation)
    layers = [nn.Linear(input_dim, hidden_dims[0]), act]
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[i], output_dim))
            if output_activation is not None:
                layers.append(get_activation(output_activation))
        else:
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(act)
    return layers


class BaseAdaptModel(nn.Module):
    def __init__(
        self,
        act_dim=19,
        proprioception_dim=63,
        cmd_dim=10,
        privileged_dim=24,
        terrain_dim=221,
        latent_dim=32,
        privileged_recon_dim=3,
        actor_hidden_dims=(256, 128, 32),
        activation="elu",
        output_activation=None,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.proprioception_dim = proprioception_dim
        self.cmd_dim = cmd_dim
        self.privileged_dim = privileged_dim
        self.terrain_dim = terrain_dim
        self.latent_dim = latent_dim
        self.privileged_recon_dim = privileged_recon_dim

        self.privileged_recon_loss = torch.tensor(0.0)

        # predict privileged pieces from latent
        self.state_estimator = nn.Sequential(
            *MLP(latent_dim, privileged_recon_dim, [64, 32], activation)
        )

        # low-level policy takes (latent + pred_priv + pro_now + cmd_now)
        ll_in = latent_dim + privileged_recon_dim + proprioception_dim + cmd_dim
        self.low_level_net = nn.Sequential(
            *MLP(ll_in, act_dim, list(actor_hidden_dims), activation, output_activation)
        )

    def forward(self, x_flat, privileged_obs=None, sync_update=False, **kwargs):
        raise NotImplementedError

    def compute_adaptation_pred_loss(self, metrics: dict):
        if self.privileged_recon_loss is not None:
            metrics["privileged_recon_loss"] += float(self.privileged_recon_loss.item())
        return self.privileged_recon_loss


class MlpAdaptModel(BaseAdaptModel):
    """
    Compatible with flattened history:
      x_flat: (N, history_len * step_dim)
    where step_dim = proprio + cmd + privileged + terrain (you can adjust dims).
    Newest timestep is at the end.
    """

    def __init__(
        self,
        obs_dim,  # total flat dim from env = history_len * step_dim
        act_dim=19,
        proprioception_dim=63,
        cmd_dim=10,
        privileged_dim=24,
        terrain_dim=221,
        latent_dim=32,
        privileged_recon_dim=3,
        actor_hidden_dims=(256, 128, 32),
        activation="elu",
        output_activation=None,
        history_len=5,                # T
        mlp_hidden_dims=(256, 128),   # encoder hidden
        **kwargs,
    ):
        super().__init__(
            act_dim=act_dim,
            proprioception_dim=proprioception_dim,
            cmd_dim=cmd_dim,
            privileged_dim=privileged_dim,
            terrain_dim=terrain_dim,
            latent_dim=latent_dim,
            privileged_recon_dim=privileged_recon_dim,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            output_activation=output_activation,
        )

        self.history_len = int(history_len)
        self.step_dim = proprioception_dim + cmd_dim + 2 # 2 = clock

        # sanity check: obs_dim must be T * step_dim
        if obs_dim != self.history_len * self.step_dim:
            raise ValueError(
                f"obs_dim mismatch: got obs_dim={obs_dim}, but expected history_len*step_dim="
                f"{self.history_len}*{self.step_dim}={self.history_len*self.step_dim}. "
                "Fix history_len or step_dim (dims of proprio/cmd/priv/terrain) to match your ObservationManager."
            )

        # encoder takes flattened proprio history: (N, T*proprio)
        self.mem_encoder = nn.Sequential(
            *MLP(proprioception_dim * self.history_len, latent_dim, list(mlp_hidden_dims), activation)
        )

    def forward(self, x_flat, privileged_obs=None, sync_update=False, **kwargs):
        """
        x_flat: (N, T*step_dim) where newest timestep is at the end.
        """
        if x_flat.dim() != 2:
            raise ValueError(f"Expected x_flat (N, T*D), got {x_flat.shape}")

        N = x_flat.shape[0]
        x = x_flat.view(N, self.history_len, self.step_dim)  # (N, T, step_dim)

        # split per-timestep
        pro_seq = x[:, :, : self.proprioception_dim]  # (N, T, pro)
        cmd_now = x[:, -1, self.proprioception_dim : self.proprioception_dim + self.cmd_dim]  # newest
        pro_now = pro_seq[:, -1, :]  # newest proprio

        # encode memory from whole proprio history (oldest..newest)
        pro_hist_flat = pro_seq.reshape(N, -1)  # (N, T*pro)
        mem = self.mem_encoder(pro_hist_flat)   # (N, latent)

        privileged_pred_now = self.state_estimator(mem)  # (N, priv_recon)

        # final action
        act_in = torch.cat([mem, privileged_pred_now, pro_now, cmd_now], dim=-1)
        action = self.low_level_net(act_in)  # (N, act_dim)

        # optional privileged recon loss
        if sync_update and privileged_obs is not None:
            # privileged_obs assumed to be "current frame" layout (N, something)
            # If yours is different, change slicing here.
            privileged_now = privileged_obs[
                :,
                self.proprioception_dim + self.cmd_dim :
                self.proprioception_dim + self.cmd_dim + self.privileged_recon_dim
            ]
            self.privileged_recon_loss = 2.0 * (privileged_pred_now - privileged_now.detach()).pow(2).mean()

        return action
