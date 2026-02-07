import torch
from abc import abstractmethod
from prettytable import PrettyTable
from collections.abc import Sequence
from isaaclab.managers import CommandManager, CommandTerm, CommandTermCfg
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

class CusCommandTerm(CommandTerm):
    def __init__(self, cfg, CommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
    
    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        # update the metrics based on current state
        self._update_metrics()
        # reduce the time left before resampling
        self.time_left -= dt
        # resample the command if necessary
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)        
        # update the command
        # self._step_contact_targets()
        self._update_command()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)

        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0

        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # resample the command
        self._resample(env_ids)
        self.gait_indices[env_ids] = 0

        return extras

class CusCommandManager(CommandManager):
    _env: ManagerBasedRLEnv
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        if cfg is None:
            raise ValueError("Command manager configuration is None. Please provide a valid configuration.")
        self._terms: dict[str, CusCommandTerm] = dict()
        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        print("[INFO] Custom Command Manager: created.")
        self._commands = dict()
        if self.cfg:
            self.cfg.debug_vis = False
            for term in self._terms.values():
                self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for the command manager."""
        msg = f"<CusCommandManager> contains {len(self._terms.values())} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Command Terms"
        table.field_names = ["Index", "Name", "Type"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.__class__.__name__])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    @property
    def get_foot_indices(self) -> torch.Tensor:
        """Get the desired contact states.
        
        This function calls the get_desired_contact_states method of the first command term.
        It is used in the rewards to compute the reward based on the desired contact states.
        """
        return self._terms["base_velocity"].get_foot_indices()
    
    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())
    
    @property
    def get_desired_contact_states(self) -> torch.Tensor:
        """Get the desired contact states.
        
        This function calls the get_desired_contact_states method of the first command term.
        It is used in the rewards to compute the reward based on the desired contact states.
        """
        return self._terms["base_velocity"].get_desired_contact_states()
    
    @property
    def get_clock_inputs(self) -> torch.Tensor:
        """Get the desired contact states.
        
        This function calls the get_clock_inputs method of the first command term.
        It is used in the rewards to compute the reward based on the desired contact states.
        """
        return self._terms["base_velocity"].get_clock_inputs()
    
    @property
    def get_foot_indices(self) -> torch.Tensor:
        """Get the foot indices.
        
        This function calls the get_foot_indices method of the first command term.
        It is used in the rewards to compute the reward based on the foot indices.
        """
        return self._terms["base_velocity"].get_foot_indices()
    
