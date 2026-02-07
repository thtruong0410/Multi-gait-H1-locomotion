# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from . import mdp
from .mdp.mob_commands import MoBCommand, MoBCommandPlay

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from isaaclab_assets import G1_MINIMAL_CFG, H1_MINIMAL_CFG  # isort: skip
from .unitreeg1_robot_cfg import H1_NEW_CFG

##
# Scene definition
##


@configclass
class Unitreeg1SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = H1_NEW_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    base_velocity = mdp.MoBCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        class_type = MoBCommand,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        ranges=mdp.MoBCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.3, 0.3),
            gait_frequency=(1.5, 3.5),
            foot_swing_height=(0.1, 0.35),
            body_height=(-0.1, 0.0),
            body_pitch=(0.0, 0.2),
            waist_roll=(-0.0, 0.0),
            heading=(-math.pi, math.pi),
            abs_vel=(0.35, 0.6),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_position = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25) #3
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25, 1.0, 1.0, 1.0, 0.15, 2.0, 0.3, 0.3))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), scale = 1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5), scale = 0.05)
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_position"})
        # last_actions = ObsTerm(func=mdp.last_last_action, params={"action_name": "joint_position"})
        clock_inputs = ObsTerm(func=mdp.clock_inputs, params={"command_name": "base_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25) #3
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25, 1.0, 1.0, 1.0, 0.15, 2.0, 0.3, 0.3))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), scale = 1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5), scale = 0.05)
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_position"})
        # last_actions = ObsTerm(func=mdp.last_last_action, params={"action_name": "joint_position"})
        clock_inputs = ObsTerm(func=mdp.clock_inputs, params={"command_name": "base_velocity"})

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2) #3
        friction = ObsTerm(func=mdp.friction)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
            scale=5.0
        ) #187
        
        # def __post_init__(self) -> None:
        #     self.enable_corruption = True
        #     self.concatenate_terms = True
            # self.history_length = 2
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    # # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    ) #
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    ) #

    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1) #
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5) # 

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    ) #

    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    # )

    # energy = RewTerm(func=mdp.energy, weight=-2e-5)

    alive = RewTerm(func=mdp.is_alive, weight=0.2) # 

    joint_deviation_elbow = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch",
                    ".*_elbow",
                ],
            )
        },
    ) #

    joint_deviation_arms_shoulder = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll",
                    ".*_shoulder_yaw",
                ],
            )
        },
    ) #

    # joint_deviation_waists = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "waist.*",
    #             ],
    #         )
    #     },
    # )

    no_fly = RewTerm(
        func=mdp.no_fly,
        weight=0.25,
            params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
            "threshold": 1,
        }    
    ) # 
    joint_pos_limit = RewTerm(func=mdp.joint_pos_limits,weight=-10.0) #
    # joint_vel_limit = RewTerm(func=mdp.joint_vel_limits, weight=-2) #
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # 

    # -- robot
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.8, #0.2 - 0.8
    #         "offset": [0.0, 0.0],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
        },
    ) #

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.2, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
        },
    )#

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-5, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
        }
    ) #
    # contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.2,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
    #     },
    # ) 

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5e-6)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) #

    # _reward_orientation_control
    orientation_control = RewTerm(
        func=mdp.orientation_control,
        weight = -20.0,
        params={
            "command_name": "base_velocity",
        }
    ) #

    waist_control = RewTerm(
        func=mdp.waist_control,
        weight = -2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "command_name": "base_velocity",
        }
    ) #

    base_height = RewTerm(
        func=mdp.base_height,
        weight= -40.0,
        params={
            "command_name": "base_velocity",
            "target_height": 0.98,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        }
    ) #

    # _reward_tracking_contacts_shaped_force
    tracking_contacts_shaped_force = RewTerm(
        func=mdp.tracking_contacts_shaped_force,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
            "gait_force_sigma": 50.0,
        }
    ) #

    # _reward_tracking_contacts_shaped_vel
    tracking_contacts_shaped_vel = RewTerm(
        func=mdp.tracking_contacts_shaped_vel,
        weight=4.0,
        params={
            "gait_vel_sigma": 5.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*")
        }
    ) #

    # _reward_feet_clearance_cmd_linear
    foot_clearance_cmd_linear = RewTerm(
        func=mdp.foot_clearance_cmd_linear,
        weight = -30.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            }
    ) #
    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.06, # 0.02 - 0.1
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )

    # -- other
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1,
    #     params={
    #         "threshold": 1,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
    #     },
    # )

    hopping_symmetry = RewTerm(
        func=mdp.hopping_symmetry,
        weight=-5.0,   # penalty -> weight âm
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
        },
    ) #




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

##
# Environment configuration
##


@configclass
class Unitreeg1EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Unitreeg1SceneCfg = Unitreeg1SceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
        # else:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = False


        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-1.0, 1.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Rewards
        # self.rewards.lin_vel_z_l2.weight = 0.0
        # self.rewards.undesired_contacts = None
        # self.rewards.flat_orientation_l2.weight = -1.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

@configclass
class Unitreeg1EnvCfg_PLAY(Unitreeg1EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # # spawn the robot randomly in the grid (instead of their terrain levels)
        # self.scene.terrain.max_init_terrain_level = None
        # # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 5
        #     self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
