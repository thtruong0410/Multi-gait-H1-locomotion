import math
from dataclasses import MISSING

from isaaclab.utils import configclass

from . import UniformVelocityCommandCfg
from .mob_commands import MoBCommand, MoBCommandPlay

@configclass
class MoBCommandCfg(UniformVelocityCommandCfg):
    class_type: type = MoBCommand | MoBCommandPlay
    seed: int = 100
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING


        lin_vel_y: tuple[float, float] = MISSING

        ang_vel_z: tuple[float, float] = MISSING
        # MoB Command
        gait_frequency: tuple[float, float] = MISSING
        foot_swing_height: tuple[float, float] = MISSING
        body_height: tuple[float, float] = MISSING
        body_pitch: tuple[float, float] = MISSING
        waist_roll: tuple[float, float] = MISSING
        heading: tuple[float, float] = MISSING
        abs_vel: tuple[float, float] = MISSING

    limit_vel_x = [-0.6, 2.0]
    limit_vel_yaw = [-1.0, 1.0]

    num_bins_vel_x: int = 12
    num_bins_vel_yaw: int = 10
    num_bins_ang_vel_z: int = 21
    num_bins_body_height: int = 1
    num_bins_offset: int = 1
    num_bins_foot_height: int = 1
    num_bins_period: int = 1

    num_commands: int = 10
    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


